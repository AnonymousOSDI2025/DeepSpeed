# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Callable, Any, List, Dict
from collections import defaultdict

import torch
from torch.fx import Node, Graph

from .util import get_last_uses


def get_output_node(graph: Graph):
    for v in graph.nodes:
        if v.target == "output":
            return v
    raise ValueError("No output node found")


def move_primals_to_head(graph: Graph):

    # Move primals to the head of the graph
    primals = [n for n in graph.nodes if n.op == "placeholder"]
    non_primals = [n for n in graph.nodes if n.op != "placeholder"]
    all_nodes = primals + non_primals

    new_graph = Graph()
    env = {}
    for node in all_nodes:
        new_node = new_graph.node_copy(node, lambda n: env[n.name])
        env[node.name] = new_node
    new_graph.lint()

    return new_graph


def add_args_process(graph: Graph,
                     node: Node,
                     fn: Callable[..., Any],
                     extra_args: List[int] = [],
                     name=None,
                     meta={}) -> List[Node]:
    # Apply fn to all args of node
    new_nodes = []
    with graph.inserting_before(node):
        target_args = [arg for arg in node.args if isinstance(arg, Node)]

        for arg in target_args:
            new_node = graph.create_node('call_function', fn, (arg, ) + tuple(extra_args), name=name)
            for k, v in meta.items():
                new_node.meta[k] = v
            node.replace_input_with(arg, new_node)
            new_nodes.append(new_node)

    return new_nodes


def add_postprocess(graph: Graph,
                    node: Node,
                    fn: Callable[..., Any],
                    extra_args: List[int] = [],
                    name=None,
                    meta={}) -> Node:
    # https://github.com/pytorch/examples/blob/main/fx/wrap_output_dynamically.py
    with graph.inserting_after(node):
        args = (node, )
        for a in extra_args:  # To add ds_id
            args += (a, )

        node_users = node.users.keys()
        new_node = graph.create_node('call_function', fn, args, {}, name=name)
        users = {}
        for u in node_users:
            if u != new_node:
                users[u] = (node, new_node)
        for u, (old_in, new_in) in users.items():
            u.replace_input_with(old_in, new_in)

    for k, v in meta.items():
        new_node.meta[k] = v

    return new_node


ops_no_wait = [torch.ops.aten.sym_size.int]


def _make_node_meta(node: Node, ds_id: int, comm: bool):
    meta = {"param_name": node.name, "ds_id": ds_id, "comm": comm}
    if "tensor_meta" in node.meta:
        meta["tensor_meta"] = node.meta["tensor_meta"]
    return meta


def add_allgather(graph_id: int, graph: Graph, node: Node, ds_id: int):
    new_node = add_postprocess(graph,
                               node,
                               torch.ops.native_z3.allgather_param,
                               extra_args=[graph_id, ds_id],
                               name=f"allgather_ds_param_{node.target}_{ds_id}",
                               meta=_make_node_meta(node, ds_id, True))

    # Set the previous node back to output
    # We don't want to change the output node to allgather
    output_node = get_output_node(graph)
    output_node.replace_input_with(new_node, node)
    return new_node


def add_release(graph_id: int, graph: Graph, node: Node, release_node: Node, ds_id: int):
    add_postprocess(graph,
                    node,
                    torch.ops.native_z3.release_param,
                    extra_args=[graph_id, ds_id],
                    name=f"release_ds_param_{release_node.target}_{node.name}_{ds_id}",
                    meta=_make_node_meta(node, ds_id, False))


def add_wait_allgather(graph_id: int, graph: Graph, node: Node, ds_id: int, user: str, n_args: int, bwd: bool):
    add_args_process(graph,
                     node,
                     torch.ops.native_z3.wait_allgather,
                     extra_args=[graph_id, ds_id, user, n_args, bwd],
                     name=f"wait_allgather_ds_param_{ds_id}",
                     meta=_make_node_meta(node, ds_id, False))


def add_reduce(graph_id: int, graph: Graph, grad_node: Node, param_name: str, ds_id: int):
    add_postprocess(graph,
                    grad_node,
                    torch.ops.native_z3.reduce_grad,
                    extra_args=[graph_id, ds_id],
                    name=f"reduce_ds_param_{param_name}",
                    meta=_make_node_meta(grad_node, ds_id, True))


def register_and_add_wait_allgather(graph_id: int, graph: Graph, bwd: bool):

    ds_ids = []
    ag_wait_nodes = []

    for node in graph.nodes:
        ag_args = [
            arg for arg in node.args if isinstance(arg, Node) and arg.target == torch.ops.native_z3.allgather_param
        ]
        if len(ag_args) > 0:
            if node.target in ops_no_wait:
                continue

            assert len(ag_args) == 1, f"Node {node.name} takes multiple allgathered params"
            ag_wait_nodes.append(node)

            ds_id = ag_args[0].meta["ds_id"]
            add_wait_allgather(graph_id, graph, node, ds_id, node.name, len(node.args), bwd)
            ds_ids.append(ds_id)

    return ds_ids, ag_wait_nodes


def add_gather_and_release(graph_id: int, graph: Graph, param_manager, param_nodes: List[Node]) -> Graph:
    ag_nodes = []
    for pn in param_nodes:
        ag_node = add_allgather(graph_id, graph, pn, param_manager.ds_ids[pn.name])
        ag_nodes.append((pn, ag_node))

    node_to_last_use, _ = get_last_uses(graph)
    for pn, ag in ag_nodes:
        last_use = node_to_last_use[ag]
        ds_id = param_manager.ds_ids[pn.name]
        add_release(graph_id, graph, last_use, pn, ds_id)

    return move_primals_to_head(graph)


def add_gather_and_reduce(graph_id: int, graph: Graph, param_manager, param_nodes_bw: List[Node],
                          param_name_to_grad: Dict[str, Node]) -> Graph:

    add_gather_and_release(graph_id, graph, param_manager, param_nodes_bw)

    for param_name in param_manager.param_names:
        add_reduce(graph_id, graph, param_name_to_grad[param_name], param_name, param_manager.ds_ids[param_name])

    return move_primals_to_head(graph)


def add_free_activations(graph_id: int, graph: Graph, activation_node_names: List[str]):
    node_to_last_use, _ = get_last_uses(graph)
    activation_nodes_set = set([n for n in graph.nodes if n.op == "placeholder" and n.name in activation_node_names])

    offload_id_to_node = {}
    node_to_wait_reload = {}
    for node in graph.nodes:
        if node.target == torch.ops.native_z3.reload_tensor:
            offload_act = node.args[0]
            # node_to_offload_id[offload_act] = node.args[2]
            offload_id_to_node[node.args[2]] = offload_act
        elif node.target == torch.ops.native_z3.wait_reload:
            offload_id = node.args[2]
            node_to_wait_reload[offload_id_to_node[offload_id]] = node

    activation_nodes_set = set(node_to_wait_reload[n] if n in node_to_wait_reload else n for n in activation_nodes_set)

    last_user_to_uses = defaultdict(list)
    for node, last_user in node_to_last_use.items():
        last_user_to_uses[last_user].append(node)

    def _should_free(node: Node) -> bool:
        if not hasattr(node, "meta"):
            return False
        if not "tensor_meta" in node.meta:
            return False
        return True

    def free_tensors(tensors: List[torch.Tensor]):
        for a in tensors:
            if a.numel() > 10_000_000:
                a.data = torch.empty([0], device=a.device, dtype=a.dtype)

    for last_user, used_nodes in last_user_to_uses.items():
        activation_args = [an for an in used_nodes if an in activation_nodes_set and _should_free(an)]

        if len(activation_args) == 0:
            continue

        node_name = f"free_activations_{[n.name for n in used_nodes]}"
        with graph.inserting_after(last_user):
            args = (activation_args, )
            # graph.create_node('call_function', torch.ops.native_z3.free_tensors, args, {}, name=node_name)

            # Python version for debugging
            graph.create_node('call_function', free_tensors, args, {}, name=node_name)
