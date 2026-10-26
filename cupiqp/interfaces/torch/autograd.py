"""Private PyTorch autograd bridge of the solver layers.

One ``torch.autograd.Function`` serves every backend. It owns the lifetime
of the backward state of one differentiable call: it leases that state from
the layer in its forward, keeps what its backward needs on the context, and
releases the lease after the backward, on a failed forward or backward, or
when the graph is dropped without a backward. The solver work itself (the
core calls, stream ordering, gradient extraction) stays in the layer.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import torch
from torch.autograd.function import FunctionCtx, once_differentiable


if TYPE_CHECKING:
    from .base import SolverLayerBase


def _release_lease_of(layer_ref: weakref.ReferenceType, backward_lease) -> None:
    """Finalizer callback: release ``backward_lease`` on the layer if it still
    exists. It holds the layer weakly, so a pending finalizer never keeps a
    layer (and its core solver) alive."""
    layer = layer_ref()
    if layer is not None:
        layer._release_backward_lease(backward_lease)


class QPSolveFunction(torch.autograd.Function):
    """Differentiable solve of the QP currently held by a solver layer.

    ``data_field_inputs`` are the tensors passed to the layer's ``forward()``,
    one per data field, ``None`` for fields left unchanged;
    ``input_is_shared`` says, per field, whether that tensor has the
    single-problem shape or carries the batch axis. The outputs are the
    present solution fields in the layer's order, followed by the status and
    iteration tensors, which are not differentiable.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: FunctionCtx,
        solver_layer: SolverLayerBase,
        input_is_shared: tuple[bool | None, ...],
        *data_field_inputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        backward_lease = solver_layer._acquire_backward_lease()
        lease_finalizer = None
        try:
            all_outputs = solver_layer._run_forward_solve(data_field_inputs, backward_lease)
            solution_count = len(solver_layer._solution_fields)
            ctx.mark_non_differentiable(*all_outputs[solution_count:])
            # Held weakly: the layer's result holds these outputs, which hold
            # this context; a strong reference back would make the layer
            # collectable only by the cyclic garbage collector, at an
            # arbitrary time, together with its core solver.
            ctx.solver_layer_ref = weakref.ref(solver_layer)
            ctx.backward_lease = backward_lease
            ctx.input_is_shared = input_is_shared
            # needs_input_grad covers (solver_layer, input_is_shared, *data_field_inputs).
            ctx.input_needs_grad = tuple(ctx.needs_input_grad[2:])
            ctx.provided_field_names = tuple(
                name for name, tensor in zip(solver_layer._FORWARD_FIELDS, data_field_inputs)
                if tensor is not None
            )
            ctx.solution_count = solution_count
            ctx.backward_consumed = False
            if backward_lease.uses_saved_state:
                # The backward restores this call's data before refactorizing:
                # every passed tensor, also those that need no gradient.
                ctx.save_for_backward(*(t for t in data_field_inputs if t is not None))
            # Releases the lease at most once: after the backward, or when the
            # graph is dropped without one.
            lease_finalizer = weakref.finalize(
                ctx, _release_lease_of, weakref.ref(solver_layer), backward_lease
            )
            ctx.lease_finalizer = lease_finalizer
            return all_outputs
        except BaseException:
            if lease_finalizer is None:
                solver_layer._release_backward_lease(backward_lease)
            else:
                lease_finalizer()
            raise

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: Any, *output_cotangents: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        if ctx.backward_consumed:
            raise RuntimeError(
                "A cuPIQP solve cannot be differentiated twice. Run the layer "
                "again instead of using retain_graph=True."
            )
        ctx.backward_consumed = True
        solver_layer = ctx.solver_layer_ref()
        if solver_layer is None:
            raise RuntimeError(
                "The solver layer of this solve no longer exists; keep the layer "
                "alive until backward() has run."
            )
        restored_data = None
        if ctx.backward_lease.uses_saved_state:
            restored_data = {
                name: tensor.detach()
                for name, tensor in zip(ctx.provided_field_names, ctx.saved_tensors)
            }
        try:
            input_gradients = solver_layer._compute_input_gradients(
                solution_cotangents=output_cotangents[:ctx.solution_count],
                input_needs_grad=ctx.input_needs_grad,
                input_is_shared=ctx.input_is_shared,
                backward_lease=ctx.backward_lease,
                restored_data=restored_data,
            )
            # No gradient for solver_layer and input_is_shared, then one per data field.
            return (None, None, *input_gradients)
        finally:
            ctx.lease_finalizer()
