"""Shared helpers of the PyTorch interface tests.

Importing this module skips the importing test module when PyTorch is not
installed; ``requires_cuda`` skips it when no CUDA device is available.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cupiqp import Status

DEVICE = "cuda"
DTYPE = torch.float64

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def tensor(values, dtype=DTYPE):
    """``values`` (array-like) as a new CUDA tensor of ``dtype``; None passes
    through, so optional problem fields can be mapped blindly."""
    if values is None:
        return None
    return torch.tensor(np.asarray(values), dtype=dtype, device=DEVICE)


def layer_statuses(layer) -> tuple:
    """Per-problem ``Status`` of the latest call of a solver layer (an
    explicit device-to-host read)."""
    return tuple(Status(int(v)) for v in layer.result.status.cpu().tolist())
