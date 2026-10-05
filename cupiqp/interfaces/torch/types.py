"""Public PyTorch-facing result types and errors."""

from __future__ import annotations

from typing import Optional

import torch

from ...results import Info


_SOLUTION_FIELDS = (
    "x", "y", "z_u", "z_l", "z_bu", "z_bl", "s_u", "s_l", "s_bu", "s_bl"
)


class StructureMismatchError(ValueError):
    """A tensor passed to a layer call does not fit the structure fixed by
    ``setup()``: a block that was not set up, or a shape other than the
    single-problem or batched shape. Call ``reset()`` and ``setup()`` again,
    or use another layer instance, to change the structure."""


class TorchSolution:
    """Solution of the latest call of a solver layer, as CUDA tensors.

    Everything stays on the device. ``x`` has shape ``(B, n)``; the dual and
    slack fields are ``(B, k)`` tensors, or ``None`` when the corresponding
    constraint block was not given at ``setup()``. ``status`` and
    ``iterations`` are int32 ``(B,)`` tensors with one entry per problem.
    All of them are views of the solver's buffers, written in stream order:
    they can be used in further Torch work on the same stream without
    synchronization, and the next call of the layer overwrites them. Use
    :meth:`clone` to keep a result, and ``.cpu()`` on a tensor when you
    need its values on the host.

    ``info`` is the core solver's per-problem diagnostics (residuals,
    objectives, step sizes, ...) as Warp arrays on the device; its
    ``to_host()`` copies them all at once.

    Example::

        x = layer(c=c_k)
        solved = layer.result.status == cupiqp.Status.CUPIQP_SOLVED  # (B,) bool, on device
        if not bool(solved.all()):            # explicit host read
            print(layer.result.status.cpu())
    """

    x: torch.Tensor
    y: Optional[torch.Tensor]
    z_u: Optional[torch.Tensor]
    z_l: Optional[torch.Tensor]
    z_bu: Optional[torch.Tensor]
    z_bl: Optional[torch.Tensor]
    s_u: Optional[torch.Tensor]
    s_l: Optional[torch.Tensor]
    s_bu: Optional[torch.Tensor]
    s_bl: Optional[torch.Tensor]

    def __init__(
        self,
        tensors: dict,
        status: torch.Tensor,
        iterations: torch.Tensor,
        info: Info
    ):
        for name in _SOLUTION_FIELDS:
            setattr(self, name, tensors.get(name))
        self.status = status
        self.iterations = iterations
        self.info = info

    def clone(self, *, detach: bool = False) -> "TorchSolution":
        """Return a solution whose tensors own independent storage, so that
        later calls of the layer do not change it. Allocates; meant for
        keeping a result, not for the hot path."""

        def copy_tensor(value: Optional[torch.Tensor]):
            if value is None:
                return None
            if detach:
                value = value.detach()
            return value.clone()

        return TorchSolution(
            {name: copy_tensor(getattr(self, name)) for name in _SOLUTION_FIELDS},
            self.status.clone(),
            self.iterations.clone(),
            self.info,
        )

    def __repr__(self) -> str:
        present = [name for name in _SOLUTION_FIELDS if getattr(self, name) is not None]
        return f"TorchSolution(fields={present}, batch_size={self.status.numel()})"
