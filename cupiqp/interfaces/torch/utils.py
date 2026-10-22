"""Constants, the dtype mapping and input checks shared by the modules of the PyTorch interface."""

from __future__ import annotations

from typing import Final

import torch
import warp as wp


#: The nine data arguments (fields) of ``setup()`` and ``forward()``, in order.
DATA_FIELDS: Final[tuple[str, ...]] = ("P", "c", "A", "b", "G", "h_u", "h_l", "x_u", "x_l")
#: The data fields that are matrices; their format depends on the layer.
MATRIX_FIELDS: Final[frozenset[str]] = frozenset(("P", "A", "G"))
#: The primal, dual and slack variables of a solution, in the core's order.
SOLUTION_FIELDS: Final[tuple[str, ...]] = (
    "x", "y", "z_u", "z_l", "z_bu", "z_bl", "s_u", "s_l", "s_bu", "s_bl"
)


def torch_to_warp_dtype(dtype: torch.dtype) -> type:
    """The Warp dtype the core solver uses for the Torch ``dtype`` of a
    layer: ``wp.float32`` for ``torch.float32``, ``wp.float64`` for
    ``torch.float64``. Any other dtype raises ``TypeError``."""
    if dtype == torch.float32:
        return wp.float32
    if dtype == torch.float64:
        return wp.float64
    raise TypeError(f"dtype must be torch.float32 or torch.float64; got {dtype!r}.")


def validate_cuda_tensor(name: str, tensor: object, device: torch.device, dtype: torch.dtype) -> None:
    """Check that the input ``name`` is a Torch tensor on the CUDA ``device``
    with the floating ``dtype``; raise naming the input otherwise. Any
    layout is accepted."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"'{name}' must be a torch.Tensor; got {type(tensor).__name__}.")
    if tensor.device.type != "cuda" or (
        tensor.device.index is not None and tensor.device.index != device.index
    ):
        raise ValueError(f"'{name}' must be on {device}; got {tensor.device}.")
    if tensor.dtype != dtype:
        raise ValueError(f"'{name}' has dtype {tensor.dtype}; this layer uses {dtype}.")


def validate_dense_tensor(name: str, tensor: object, device: torch.device, dtype: torch.dtype) -> None:
    """Like :func:`validate_cuda_tensor`, and the tensor must also have the
    dense (strided) layout."""
    validate_cuda_tensor(name, tensor, device, dtype)
    if tensor.layout != torch.strided:
        raise TypeError(f"'{name}' must be a dense (strided) tensor.")
