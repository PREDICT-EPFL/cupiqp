"""Private PyTorch autograd bridge of the solver layers.

One ``torch.autograd.Function`` serves every backend. Its forward runs the
layer's core update and solve on the tensors passed to ``forward()`` and
returns zero-copy views of the solution; its backward hands the upstream
cotangents to the core implicit-differentiation ``backward()`` and maps the
returned data gradients onto the input tensors. The function knows nothing
about how a backend stores its matrices: the layer does the mapping.
"""

from __future__ import annotations

import weakref

import torch
from torch.autograd.function import once_differentiable


class _SolveFunction(torch.autograd.Function):
    """Differentiable solve of the QP currently held by a solver layer.

    ``flat_inputs`` are the tensors passed to the layer's ``forward()``, one
    per data slot, ``None`` for slots left unchanged. ``shared`` says, per
    slot, whether that tensor has the single-problem shape (shared by the
    batch) or carries the batch axis. The outputs are the present result
    fields, in the layer's order.

    While a solve awaits its backward the layer refuses further
    differentiable calls (its "lease"). Backward releases it; if the graph is
    dropped without a backward, the finalizer on this context releases it.
    """

    @staticmethod
    def forward(ctx, layer, shared, *flat_inputs):
        needs_grad = tuple(ctx.needs_input_grad[2:])
        outputs = layer._solve_core(flat_inputs)
        ctx.layer = layer
        ctx.shared = shared
        ctx.needs_grad = needs_grad
        ctx.has_inputs = tuple(t is not None for t in flat_inputs)
        ctx.consumed = False
        ctx.token = None
        if any(needs_grad):
            ctx.token = layer._acquire_lease()
            weakref.finalize(ctx, layer._release_lease, ctx.token)
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        if ctx.token is None:
            return (None, None) + (None,) * len(ctx.has_inputs)
        if ctx.consumed:
            raise RuntimeError(
                "A cuPIQP solve cannot be differentiated twice. Run the layer "
                "again instead of using retain_graph=True."
            )
        ctx.consumed = True
        try:
            gradients = ctx.layer._backward_core(
                grad_outputs, ctx.has_inputs, ctx.needs_grad, ctx.shared
            )
            return (None, None, *gradients)
        finally:
            ctx.layer._release_lease(ctx.token)
