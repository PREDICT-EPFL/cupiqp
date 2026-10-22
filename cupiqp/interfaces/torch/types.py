"""Public PyTorch-facing result types and errors."""

from __future__ import annotations

from typing import Mapping

import torch

from ...results import Info
from .utils import SOLUTION_FIELDS


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
    y: torch.Tensor | None
    z_u: torch.Tensor | None
    z_l: torch.Tensor | None
    z_bu: torch.Tensor | None
    z_bl: torch.Tensor | None
    s_u: torch.Tensor | None
    s_l: torch.Tensor | None
    s_bu: torch.Tensor | None
    s_bl: torch.Tensor | None
    status: torch.Tensor
    iterations: torch.Tensor
    info: Info

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        status: torch.Tensor,
        iterations: torch.Tensor,
        info: Info
    ) -> None:
        for name in SOLUTION_FIELDS:
            setattr(self, name, tensors.get(name))
        self.status = status
        self.iterations = iterations
        self.info = info

    def clone(self, *, detach: bool = False) -> TorchSolution:
        """Return a solution whose tensors own independent storage, so that
        later calls of the layer do not change it. Allocates; meant for
        keeping a result, not for the hot path."""

        def copy_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            if detach:
                value = value.detach()
            return value.clone()

        tensors = {
            name: copied
            for name in SOLUTION_FIELDS
            if (copied := copy_tensor(getattr(self, name))) is not None
        }
        return TorchSolution(tensors, self.status.clone(), self.iterations.clone(), self.info)

    def __repr__(self) -> str:
        present = [name for name in SOLUTION_FIELDS if getattr(self, name) is not None]
        return f"TorchSolution(fields={present}, batch_size={self.status.numel()})"
