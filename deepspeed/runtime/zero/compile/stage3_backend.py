# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
from collections import defaultdict
from typing import Dict, List
import time

import torch
from torch.fx import Graph, GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
import torch.utils._pytree as pytree
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .fx import get_output_node, add_gather_and_release, add_gather_and_reduce, register_and_add_wait_allgather, add_free_activations
from .graph_param import DSGraphParamManager
from .profilers import ProfilingResult
from .profilers.graph_profile import ProfilingInterpreter, MemoryProfilingInterpreter
from .passes import run_opt_passes
from .passes.offload_activation import offload_activation_fwd, reload_activation_bwd
from .passes.offload_adam_states import insert_offload_opt_states, offload_adam_states_sync
from .patch_compiled_func import patch_compiled_func, unpatch_compiled_func
from .list_schedule import simple_prefetch, fast_free_schedule
from .util import get_input_nodes, get_param_nodes, count_inflight_values, exclude_from_act_offload, get_activation_node_names, add_mem_profile_nodes
from .partitioner import get_wrapped_partitioner

graph_counts = defaultdict(int)
param_manager: Dict[int, DSGraphParamManager] = {}
output_names: Dict[int, List[str]] = {}
graph_order = []
last_graph_order = None


def reset_graph_order():
    global graph_order
    global last_graph_order
    last_graph_order = graph_order
    graph_order = []


def _set_time_and_tensor_size(graph_id, graph: Graph, bwd, profiling_results):
    node_time = []
    tensor_sizes = []

    for n in graph.nodes:
        node_time.append((n.name, n.meta["device_time"] if "device_time" in n.meta else 0.0,
                          n.meta["wall_time"] if "wall_time" in n.meta else 0.0))
        tensor_sizes.append((n.name, n.meta["tensor_size"] if "tensor_size" in n.meta else 0))

    if bwd:
        profiling_results[graph_id].bwd_graph = graph
        profiling_results[graph_id].bwd_time = node_time
        profiling_results[graph_id].bwd_tensor_sizes = tensor_sizes
    else:
        profiling_results[graph_id].fwd_graph = graph
        profiling_results[graph_id].fwd_time = node_time
        profiling_results[graph_id].fwd_tensor_sizes = tensor_sizes


def dump_graph(graph: GraphModule, name: str, skip=False):
    if not skip and dist.get_rank() == 0:
        global graph_counts
        fname = f"{name}_{graph_counts[name]}"
        # graph.graph.print_tabular()

        g = FxGraphDrawer(graph, fname)
        with open(f"{fname}.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())

        graph_counts[name] += 1


def get_index_by_graph_id(graph_order, target_graph_id):
    for index, (graph_id, _) in enumerate(graph_order):
        if graph_id == target_graph_id:
            return index
    return -1


profiling_results: Dict[int, ProfilingResult] = {}
remaining_bwd_compile_count = 0

enable_opt_passes = False
opt_pass_times = []


def launch_opt_passes():
    global enable_opt_passes
    enable_opt_passes = True
    reset_graph_order()
    profiling_results.clear()
    param_manager.clear()


def make_stage3_backend(opt_passes,
                        scheduler,
                        free_activation=True,
                        offload_activation=False,
                        offload_opt_states=False,
                        dump_graphs=False,
                        profile_memory=False,
                        debug_log=False):
    from deepspeed.ops.op_builder import NativeZ3Builder
    nz3 = NativeZ3Builder().load()
    rank = dist.get_rank()

    bwd_inputs_stack = patch_compiled_func()

    if scheduler == "simple_prefetch":
        scheduler_fn = simple_prefetch
    elif scheduler == "fast_free":
        scheduler_fn = fast_free_schedule
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")

    def stage3_backend(gm: GraphModule, real_inputs):
        graph_id = id(gm.graph)

        needs_backward = pytree.tree_any(lambda x: x.requires_grad if torch.is_tensor(x) else False, real_inputs)
        num_original_outputs = len(get_output_node(gm.graph).args[0])

        global graph_order
        graph_order.append((graph_id, needs_backward))

        if len(list(gm.named_parameters())) == 0:
            param_indices = [(i, input_val.ds_id, input_val.ds_shape) for i, input_val in enumerate(real_inputs)
                             if hasattr(input_val, 'ds_id')]
        else:
            # < v2.5
            param_indices = [(i, param.ds_id, param.ds_shape) for i, (n, param) in enumerate(gm.named_parameters())]

        global profiling_results
        if graph_id not in profiling_results:
            profiling_results[graph_id] = ProfilingResult()
            profiling_results[graph_id].param_indices = param_indices
            profiling_results[graph_id].needs_backward = needs_backward

        def fw(gm, sample_inputs):
            time_start = time.time()
            graph_index = len(graph_order) - 1
            if rank == 0 and debug_log:
                print(
                    f"Fwd initial graph {graph_index} graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()}"
                )

            param_manager[graph_id] = DSGraphParamManager(gm.graph, real_inputs, param_indices)
            output_names[graph_id] = [n.name for n in get_output_node(gm.graph).args[0]]

            gm.graph = add_gather_and_release(graph_id, gm.graph, param_manager[graph_id],
                                              get_param_nodes(gm.graph, param_indices))

            if needs_backward and offload_activation:
                outputs = get_output_node(gm.graph).args[0]
                output_node_with_original_names = [(name, n) for name, n in zip(output_names[graph_id], outputs)]
                nodes_to_offload = [(name, node)
                                    for name, node in output_node_with_original_names[num_original_outputs:]
                                    if not exclude_from_act_offload(node)]
                gm.graph = offload_activation_fwd(gm.graph, graph_id, nodes_to_offload, graph_order,
                                                  get_accelerator().available_memory(), param_manager[graph_id])

            if needs_backward:
                global remaining_bwd_compile_count
                remaining_bwd_compile_count += 1

            nz3.register_graph(graph_id, [v[1] for v in param_indices])  # Need this before profiling

            def create_fwd_inputs():
                return real_inputs

            if offload_opt_states and len(graph_order) == 1:
                offload_adam_states_sync()

            profiler = ProfilingInterpreter(nz3, gm, debug_log=False)
            profiler.run(*create_fwd_inputs())
            del profiler
            gc.collect()
            get_accelerator().empty_cache()

            if rank == 0 and debug_log:
                print(f"Fwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

            gm.graph = scheduler_fn(
                gm.graph,
                get_accelerator().available_memory(),
                0,  # unused
                debug_log=debug_log)

            if rank == 0 and debug_log:
                count_inflight_values(gm.graph, f"fwd_{graph_id}_inflight_values.csv")

            _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, False)
            nz3.register_graph_ops(graph_id, [n.name for n in ag_wait_nodes], [len(n.args) for n in ag_wait_nodes])

            if rank == 0 and debug_log:
                print(f"Fwd after scheduling {graph_index} graph_id={graph_id} {gm.graph}")

            dump_graph(gm, f"forward_aot_scheduled_{graph_id}", skip=not dump_graphs)

            if offload_opt_states:
                gm.graph = insert_offload_opt_states(gm.graph, graph_id, graph_order, profiling_results,
                                                     get_accelerator().available_memory(), param_manager[graph_id],
                                                     False)

            mem_prof = MemoryProfilingInterpreter(nz3, gm)
            mem_prof.run(*create_fwd_inputs())

            if rank == 0 and debug_log:
                mem_prof.dump(f"mem_prof_fwd_{graph_index}_{graph_id}.csv")
            profiling_results[graph_id].fwd_mem = mem_prof.mem_record
            del mem_prof

            _set_time_and_tensor_size(graph_id, gm.graph, False, profiling_results)

            gc.collect()
            get_accelerator().empty_cache()

            if enable_opt_passes:
                gm = run_opt_passes(nz3, graph_id, gm, create_fwd_inputs, opt_passes, graph_order, profiling_results,
                                    param_manager, False, debug_log and rank == 0)

            if profile_memory:
                add_mem_profile_nodes(gm.graph, f"mem_prof fwd {graph_index} {graph_id}")

            if rank == 0 and debug_log:
                print(f"Fwd end graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()}")

            opt_pass_times.append(("fwd", graph_index, graph_id, time.time() - time_start))

            return make_boxed_func(gm.forward)

        def bw(gm, sample_inputs):
            time_start = time.time()
            graph_index = get_index_by_graph_id(graph_order, graph_id)

            if rank == 0 and debug_log:
                print(f"Bwd initial graph graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()}")

            if len(bwd_inputs_stack) == 0:
                # dynamo calls bw compiler ahead of time when symints are saved for backward. See the details for aot_dispatch_autograd in jit_compile_runtime_wrappers.
                # As we currently use actually bwd input values in bw compiler, we return None to skip the compilation there.
                # This would need be handled properly in the future.
                return None

            assert graph_id in param_manager, f"Graph {graph_id} not found in param_manager"
            param_nodes_bw, param_name_to_grad = param_manager[graph_id].get_bwd_mapping(gm.graph)

            gm.graph = add_gather_and_reduce(graph_id, gm.graph, param_manager[graph_id], param_nodes_bw,
                                             param_name_to_grad)
            if offload_activation:
                gm.graph = reload_activation_bwd(gm.graph, graph_id, graph_order,
                                                 get_accelerator().available_memory(), param_manager[graph_id])

            input_nodes = get_input_nodes(gm.graph)
            assert len(input_nodes) == len(
                sample_inputs), f"Expected {len(sample_inputs)} inputs, got {len(input_nodes)}"

            bwd_real_inputs = bwd_inputs_stack.pop()

            def create_bwd_inputs():
                return tuple(bwd_real_inputs)

            real_outputs = ProfilingInterpreter(nz3, gm, debug_log=False).run(*create_bwd_inputs())

            del real_outputs
            gc.collect()
            get_accelerator().empty_cache()

            if rank == 0 and debug_log:
                print(f"Bwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

            gm.graph = scheduler_fn(gm.graph, get_accelerator().available_memory(), 0, debug_log=debug_log)

            if rank == 0 and debug_log:
                count_inflight_values(gm.graph, f"bwd_{graph_id}_inflight_values.csv")

            _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, True)
            nz3.register_bwd_graph_ops(graph_id, [n.name for n in ag_wait_nodes], [len(n.args) for n in ag_wait_nodes])

            dump_graph(gm, f"backward_aot_scheduled_{graph_index}_{graph_id}", skip=not dump_graphs)

            if offload_opt_states:
                gm.graph = insert_offload_opt_states(gm.graph, graph_id, graph_order, profiling_results,
                                                     get_accelerator().available_memory(), param_manager[graph_id],
                                                     True)

            mem_prof = MemoryProfilingInterpreter(nz3, gm)
            mem_prof.run(*create_bwd_inputs())
            if rank == 0 and debug_log:
                mem_prof.dump(f"mem_prof_bwd_{graph_index}_{graph_id}.csv")
            profiling_results[graph_id].bwd_mem = mem_prof.mem_record

            del mem_prof
            gc.collect()
            get_accelerator().empty_cache()

            _set_time_and_tensor_size(graph_id, gm.graph, True, profiling_results)

            global enable_opt_passes
            if enable_opt_passes:
                gm = run_opt_passes(nz3, graph_id, gm, create_bwd_inputs, opt_passes, graph_order, profiling_results,
                                    param_manager, True, debug_log and rank == 0)

            if free_activation:
                param_names = [n.name for n in param_nodes_bw]
                non_param_input_names = [n.name for n in get_input_nodes(gm.graph) if n.name not in param_names]
                add_free_activations(graph_id, gm.graph,
                                     get_activation_node_names(gm.graph, param_nodes_bw, non_param_input_names))

            global remaining_bwd_compile_count
            remaining_bwd_compile_count -= 1
            if remaining_bwd_compile_count == 0:
                unpatch_compiled_func()

            output_names.pop(graph_id)

            if rank == 0 and debug_log:
                print(
                    f"Bwd end {graph_index} graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()} {gm.graph}"
                )

            if profile_memory:
                add_mem_profile_nodes(gm.graph, f"mem_prof bwd {graph_index} {graph_id}")

            gm.recompile()

            opt_pass_times.append(("bwd", graph_index, graph_id, time.time() - time_start))

            return make_boxed_func(gm.forward)

        # Call AOTAutograd
        aot_mod = aot_module_simplified(gm,
                                        real_inputs,
                                        fw_compiler=fw,
                                        bw_compiler=bw,
                                        partition_fn=get_wrapped_partitioner(param_indices))
        aot_mod = torch._dynamo.optimize()(aot_mod)

        return aot_mod

    return stage3_backend
