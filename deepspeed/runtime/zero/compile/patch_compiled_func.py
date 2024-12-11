# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List

import torch
from torch._prims_common import CUDARngStateHelper
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from torch._functorch._aot_autograd.schemas import (
    OutputType,
    SubclassCreationMeta,
)
from torch._functorch._aot_autograd.subclass_utils import (
    get_types_for_subclass,
    unwrap_tensor_subclasses,
)
from torch._functorch._aot_autograd.runtime_wrappers import AOTDispatchAutograd
from torch._subclasses import FakeTensor

backward_inputs = []


# Copied from torch._functorch._aot_autograd.runtime_wrappers
def make_backward_input(CompiledFunction, ctx, flat_args):
    num_intermediate_bases = (CompiledFunction.metadata.num_intermediate_bases)
    num_mutated_runtime_inps = (CompiledFunction.metadata.num_mutated_inp_runtime_indices)
    expected_grad_outs = (CompiledFunction.metadata.num_outputs + num_mutated_runtime_inps + num_intermediate_bases)
    deterministic = CompiledFunction.metadata.deterministic
    global_deterministic = torch.are_deterministic_algorithms_enabled()
    if deterministic is not None:
        torch._check(
            not (not deterministic and global_deterministic),
            lambda: ("This compiled backward function is being run with "
                     "torch.use_deterministic_algorithms(True), "
                     "but it was previously generated during the forward function while "
                     "torch.use_deterministic_algorithms(False) was set."),
        )

    assert len(flat_args) == expected_grad_outs
    out_info = CompiledFunction.metadata.output_info

    inp_tangents, out_tangents, intermediate_base_tangents = (
        flat_args[:num_mutated_runtime_inps],
        flat_args[num_mutated_runtime_inps:num_mutated_runtime_inps + CompiledFunction.metadata.num_outputs],
        flat_args[num_mutated_runtime_inps + CompiledFunction.metadata.num_outputs:],
    )
    # input_info contains info on *every* input,
    # But in the backward(), we are only given grad outputs for every mutated input
    # We then need to filter out the grad outputs that correspond to metadata-only mutations or don't require grad
    input_info = CompiledFunction.metadata.input_info
    inp_tangents_filtered = [
        x for x, info_idx in zip(
            inp_tangents,
            CompiledFunction.metadata.mutated_inp_runtime_indices,
        ) if input_info[info_idx].mutates_data and input_info[info_idx].requires_grad
    ]
    # We also need to filter out grad outputs that correspond to outputs aliasing inputs/intermediates
    out_tangents_filtered = [
        x for x, info in zip(out_tangents, out_info) if info.output_type in [
            OutputType.non_alias,
            OutputType.unsafe_view_alias,
            OutputType.custom_function_view,
        ] and issubclass(info.raw_type, torch.Tensor) and info.requires_grad
    ]
    # intermediate bases always require gradients, and always participate in the backward graph.
    flat_bw_args_with_grads = [
        *inp_tangents_filtered,
        *out_tangents_filtered,
        *intermediate_base_tangents,
    ]
    num_flat_bw_args_with_grads = len(flat_bw_args_with_grads)

    # sanity asserts
    # metadata_only_inps = [
    #     x for x, info_idx in zip(inp_tangents, mutated_inp_indices)
    #     if not input_info[info_idx].mutates_data
    # ]
    # aliased_outputs = [
    #     x for x, info in zip(out_tangents, out_info) if info.output_type != OutputType.non_alias]
    # assert all(x is None for x in metadata_only_inps)
    # assert all(x is None for x in aliased_outputs)
    # TODO: replace this with FunctionalizedRngRuntimeWrapper
    rng_args = []
    if CompiledFunction.metadata.is_rng_op_functionalized:
        # Add the seed and offset to args
        rng_args = CUDARngStateHelper.get_torch_state_as_tuple()

    bw_tokens = [None] * CompiledFunction.metadata.num_backward_tokens

    # - note: donated buffer logic requires (*ctx.symints, *ctx.saved_tensors) showing up first
    #   in the bw output order.

    # Every dereference of ctx.saved_tensors incurs saved_tensors_hooks calls
    # There are tests that count these calls, saving to var.
    ctx_saved_tensors = ctx.saved_tensors
    num_ctx_saved_tensors = len(ctx_saved_tensors)
    all_args = [
        *ctx.symints,
        *ctx_saved_tensors,
        *flat_bw_args_with_grads,
        *bw_tokens,
        *rng_args,
    ]

    del ctx_saved_tensors

    # Note: [AOTAutograd Backward Guards]
    # During AOTDispatch, we eagerly create and trace out a joint fw-bw graph.
    # Doing so requires us to "guess" about some of the metadata of our grad_outputs.
    #
    # In particular: if an output to the forward is a plain tensor or a subclass,
    # its corresponding grad_output in the backward **may or may not** be
    # a plain tensor or a subclass. The main cases are:
    # (1) If an output is a plain tensor, its grad_out will also be a plain tensor,
    #     *unless* the output is used in some subclass compute later in the forward graph,
    #     which will cause its grad_output to become a subclass
    # (2) If an output is a subclass, its grad_out will also be a subclass,
    #     *unless* the output of the forward did not actually participate in the gradient computation,
    #     in which case autograd will insert a plain tensor of zeros for the grad_output.
    #     We could avoid this case with `torch.autograd.Function.set_materialize_grads`,
    #     although this is not turned on today in AOTAutgrad and would require more work.
    #
    # Today, we make a guess on subclass-ness based on the above examples,
    # and hard-error in the backward if we guessed wrong.
    #
    # In the future, we should add backward guards that would allow us to
    # properly handle this case instead of erroring: we would need to retrace the backward graph,
    # since we might produce an entirely different trace if our grad_outputs are subclass or not.
    assert (len(CompiledFunction.metadata.output_types) == num_flat_bw_args_with_grads)

    grad_output_types = [type(x) for x in flat_bw_args_with_grads]
    # In general, we can add more asserts/guards here for when we partitioned
    # with incorrect assumptions about the grad_outputs.
    # Normalize FakeTensor -> torch.Tensor
    # - during tracing our types are FakeTensor
    # - at runtime in the backward our types are torch.Tensor...
    # - unless we're running compiled backward, in which case they are also FakeTensor
    grad_output_types_ = [torch.Tensor if x is FakeTensor else x for x in grad_output_types]
    assert (grad_output_types_ == CompiledFunction.metadata.output_types), f"""\
We incorrectly attempted to compile the backward with incorrect subclass metadata.
If you run into this error, please file an issue.
Expected grad_output types: {str(CompiledFunction.metadata.output_types)}
Got grad_output types: {str(grad_output_types)}"""

    del flat_bw_args_with_grads

    tangents_start_idx = (len(all_args) - num_flat_bw_args_with_grads - len(rng_args) - len(bw_tokens))
    assert tangents_start_idx == len(ctx.symints) + num_ctx_saved_tensors
    tangents_end_idx = len(all_args) - len(rng_args) - len(bw_tokens)

    # TODO: figure out how to refactor the backward properly
    # so I can use aot_dispatch_subclass_wrapper() here.
    if CompiledFunction.maybe_subclass_metadata is not None:
        tangents = all_args[tangents_start_idx:tangents_end_idx]

        def get_types_for_tangents(tangents):
            infos = []
            idx = 0
            for a in tangents:
                if isinstance(a, torch.Tensor) and is_traceable_wrapper_subclass(a):
                    infos.append(get_types_for_subclass(a))
                else:
                    infos.append(idx)
                idx += 1
            return infos

        runtime_subclass_info = get_types_for_tangents(tangents)

        if len(runtime_subclass_info) != len(CompiledFunction.metadata.subclass_tangent_meta):
            raise RuntimeError("The grad inputs should be same number as forward output tangents")
        for a, b in zip(
                runtime_subclass_info,
                CompiledFunction.metadata.subclass_tangent_meta,
        ):
            # Types should match between runtime and traced tangents.
            # TODO (tmanlaibaatar) Should actually call coerce_runtime_tangent
            if isinstance(a, List) and (isinstance(b, SubclassCreationMeta) and b.subclass_type):
                if not a == b.subclass_type:
                    raise RuntimeError("The grad inputs should be same tensor subclass type as forward output")

        # Get the number of tangents after unwrapping
        len_tangents = len(unwrap_tensor_subclasses(
            tangents,
            is_joint_structure=False,
        ))
        assert CompiledFunction.metadata.traced_tangent_metas is not None
        all_args = [(AOTDispatchAutograd.coerce_runtime_tangent(
            t,
            CompiledFunction.metadata.traced_tangent_metas[i - tangents_start_idx],
        ) if tangents_start_idx <= i < tangents_end_idx else t) for i, t in enumerate(all_args)]
        all_args = unwrap_tensor_subclasses(all_args, is_joint_structure=False)
        tangents_start_idx = (len(all_args) - len_tangents - len(rng_args) - len(bw_tokens))
        tangents_end_idx = tangents_start_idx + len_tangents

    # Make the tangents contiguous. Note that we must do this after subclass desugaring
    # because inputs to inductor have to be contiguous
    all_args = [(AOTDispatchAutograd._force_contiguous(t) if (tangents_start_idx <= i < tangents_end_idx) else t)
                for i, t in enumerate(all_args)]

    return all_args


enabled_patched_func = False
original_grad_fn = None


def patch_compiled_func():
    base_meta = type(torch.autograd.Function)

    global enabled_patched_func
    enabled_patched_func = True

    class FunctionMeta(base_meta):

        def __new__(cls, name, bases, dct):
            if name == "CompiledFunction":
                original_backward = dct.get("backward", None)

                def wrapped_backward(ctx, *grad_outputs):

                    assert original_backward is not None

                    if enabled_patched_func:
                        all_args = make_backward_input(wrapped_backward.owner_class, ctx, grad_outputs)
                        backward_inputs.append(all_args)

                    return original_backward(ctx, *grad_outputs)

                wrapped_backward.owner_class = None
                dct["backward"] = staticmethod(wrapped_backward)
                new_class = super().__new__(cls, name, bases, dct)
                wrapped_backward.owner_class = new_class

                return new_class

            return super().__new__(cls, name, bases, dct)

    class PatchedFunction(torch.autograd.Function, metaclass=FunctionMeta):
        pass

    global original_grad_fn
    original_grad_fn = torch.autograd.Function

    torch.autograd.Function = PatchedFunction

    return backward_inputs


def unpatch_compiled_func():
    global enabled_patched_func
    enabled_patched_func = False

    global original_grad_fn
    torch.autograd.Function = original_grad_fn
