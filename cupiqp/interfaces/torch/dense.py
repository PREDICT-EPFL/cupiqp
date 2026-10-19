"""Differentiable dense QP layer for PyTorch."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from ...data import Data
from ...dense.dense_solver import DenseSolver
from .base import SolverLayerBase
from .utils import MATRIX_FIELDS, validate_dense_tensor


class DenseSolverLayer(SolverLayerBase):
    """Differentiable QP layer backed by ``cupiqp.DenseSolver``.

    Solves, for one problem or a batch,

        minimize    0.5 x^T P x + c^T x
        subject to  A x = b
                    h_l <= G x <= h_u
                    x_l <= x <= x_u

    with dense CUDA tensors. ``setup()`` fixes the dimensions and which
    blocks exist and allocates all GPU memory; every later call updates the
    values it is given, solves, and returns ``x``::

        layer = cupiqp.torch.DenseSolverLayer()
        layer.setup(P, c, x_l=x_l, x_u=x_u)
        x = layer(c=c_k)                    # solve with a new linear cost
        x.sum().backward()                  # gradient for c_k

    ``setup()`` takes the arguments of ``cupiqp.DenseSolver.setup`` as dense
    CUDA tensors of the layer dtype: each matrix is ``(rows, cols)`` or
    ``(B, rows, cols)`` and each vector ``(k,)`` or ``(B, k)``; the batch
    size is read from the batched arguments. See :class:`cupiqp.DenseSolver`
    for the meaning of every argument and for ``+/-inf`` bounds, and
    :class:`SolverLayerBase` for ``max_pending_solves`` (several solves
    awaiting one backward, as in a rollout).

    Parameters
    ----------
    dtype : torch.dtype, default: torch.float64
        ``torch.float32`` or ``torch.float64``, used for all data and results.
    stream : torch.cuda.Stream, optional
        Torch stream to order the solver with. By default the layer orders
        itself with Torch's current stream at every call.
    device : torch.device or str, optional
        CUDA device to solve on. Defaults to the stream's device, else
        Torch's current CUDA device. All tensors must live on it.
    """

    _core_solver_cls = DenseSolver

    def _prepare_setup_inputs(self, inputs: dict[str, torch.Tensor | None]) -> dict[str, Any]:
        core_setup_args = {}
        for name, tensor in inputs.items():
            if tensor is None:
                core_setup_args[name] = None
                continue
            validate_dense_tensor(name, tensor, self._device, self._dtype)
            # A matrix is (rows, cols) or batched (B, rows, cols); a vector
            # (k,) or batched (B, k).
            low, high = (2, 3) if name in MATRIX_FIELDS else (1, 2)
            if tensor.dim() not in (low, high):
                raise ValueError(
                    f"'{name}' must be {low}-D or {high}-D (batched); got shape "
                    f"{tuple(tensor.shape)}."
                )
            if name == "P" and tensor.shape[-1] != tensor.shape[-2]:
                raise ValueError(f"'P' must be square; got shape {tuple(tensor.shape)}.")
            core_setup_args[name] = tensor.detach().contiguous()
        return core_setup_args

    def _matrix_unbatched_shape(self, name: str) -> tuple[int, ...]:
        data = self._core_solver.data
        rows = {"P": data.n, "A": data.p, "G": data.m}[name]
        return (rows, data.n)

    def _matrix_data_gradient_array(self, name: str, data_gradients: Data) -> wp.array:
        return getattr(data_gradients, name)
