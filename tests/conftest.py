"""Shared pytest configuration and fixtures of the test suite.

Layout: ``linalg/`` (dense and sparse kernels and factorizations), ``kkt/``
(KKT assembly and solves), ``problem_data/`` (Data containers), ``solver/``
(end-to-end solves and input handling), ``runtime/`` (streams, CUDA graphs,
memory and lifetime), ``gradients/`` (backward passes), ``torch_interface/``
(the PyTorch interface), ``jax_interface/`` (the JAX interface) and
``experimental/`` (APIs not yet public).
Shared problem builders and conversions live in ``helpers.py``.

Run GPU tests sequentially: concurrent pytest processes on one GPU produce
spurious cuSOLVER errors.
"""
import numpy as np
import pytest
import warp as wp

wp.config.log_level = wp.LOG_WARNING


@pytest.fixture
def poison_uninitialized_memory(monkeypatch):
    """Fill every buffer from ``wp.empty`` with ``0xFF`` bytes (NaN floats,
    -1 integers) for the duration of a test.

    Fresh GPU memory is usually zero, so a kernel that reads a buffer before
    writing it (e.g. ``out = alpha * acc + beta * out`` with ``beta == 0``)
    passes until it gets recycled memory holding inf or NaN. Poisoning turns
    such a read into a failed solve immediately and deterministically.
    """
    real_empty = wp.empty

    def poisoned_empty(*args, **kwargs):
        a = real_empty(*args, **kwargs)
        if a.device.is_cuda and a.size > 0:
            # Byte view of the whole allocation; the fill is issued on the
            # current stream, so it is ordered before the buffer's first use.
            wp.array(ptr=a.ptr, dtype=wp.uint8, shape=(a.capacity,),
                     device=a.device, copy=False).fill_(255)
        return a

    monkeypatch.setattr(wp, "empty", poisoned_empty)
    yield
    wp.synchronize()
