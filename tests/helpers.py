"""Shared helpers of the test suite: test problems and host-to-GPU conversion.

Tests build their problems with NumPy on the host and move them to the GPU as
``warp.array`` with :func:`to_wp`, the array type the solvers work in. CuPy,
PyTorch and other frameworks appear only in tests about accepting their arrays.
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp

from cupiqp import Status

DEVICE = "cuda"
DATA_DIR = Path(__file__).resolve().parent / "data"


# ----------------------------------------------------------------------
# Host -> GPU
# ----------------------------------------------------------------------

def _numpy_dtype(dtype):
    """NumPy dtype of a NumPy or Warp scalar type (None passes through)."""
    if dtype is None or isinstance(dtype, (str, np.dtype)):
        return dtype
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return dtype
    return wp.dtype_to_numpy(dtype)


def to_wp(a, dtype=None) -> wp.array:
    """Copy ``a`` (NumPy array, list or scalar) into a new contiguous
    ``warp.array`` on the GPU. ``dtype`` is a NumPy or Warp scalar type; by
    default the dtype of ``np.asarray(a)`` is kept (float64 for Python floats).
    ``None`` passes through, so optional problem fields can be mapped blindly."""
    if a is None:
        return None
    if isinstance(a, wp.array):
        return a if dtype is None else to_wp(a.numpy(), dtype)
    arr = np.ascontiguousarray(np.asarray(a, dtype=_numpy_dtype(dtype)))
    return wp.array(arr, dtype=wp.dtype_from_numpy(arr.dtype), device=DEVICE)


def to_wp_dict(qp: dict, dtype=None) -> dict:
    """:func:`to_wp` applied to every value of a problem dict."""
    return {k: to_wp(v, dtype) for k, v in qp.items()}


def broadcast(a: wp.array, batch_size: int) -> wp.array:
    """``(batch_size, *a.shape)`` zero-stride view of ``a``: every problem of
    the batch aliases the same data, like ``np.broadcast_to``. The view keeps
    ``a`` alive."""
    view = wp.array(ptr=a.ptr, dtype=a.dtype, shape=(batch_size, *a.shape),
                    strides=(0, *a.strides), device=a.device, copy=False)
    view._broadcast_source = a
    return view


def csr_triple(M, dtype=np.float64):
    """GPU CSR triple ``(indptr, indices, values)`` of a host matrix ``M``
    (dense NumPy array or SciPy sparse), with sorted column indices. The
    pattern is int32; explicit zeros of a dense ``M`` are not stored."""
    M = sp.csr_matrix(M)
    M.sort_indices()
    return (to_wp(M.indptr, np.int32), to_wp(M.indices, np.int32), to_wp(M.data, dtype))


# ----------------------------------------------------------------------
# Test problems (host NumPy)
# ----------------------------------------------------------------------

def random_qp(rng=0, n=8, p=3, m=5) -> dict:
    """Random strictly convex QP with equalities, two-sided inequalities and
    box bounds, as host arrays. ``rng`` is a seed or a NumPy Generator. The
    bounds are wide, so the problem is feasible with an interior solution."""
    rng = np.random.default_rng(rng)
    M = rng.standard_normal((n, n))
    return dict(
        P=M @ M.T + n * np.eye(n),
        c=rng.standard_normal(n),
        A=rng.standard_normal((p, n)),
        b=rng.standard_normal(p),
        G=rng.standard_normal((m, n)),
        h_u=np.abs(rng.standard_normal(m)) + 2.0,
        h_l=-np.abs(rng.standard_normal(m)) - 2.0,
        x_u=np.abs(rng.standard_normal(n)) + 3.0,
        x_l=-np.abs(rng.standard_normal(n)) - 3.0,
    )


def primal_infeasible(qp: dict) -> dict:
    """Copy of ``qp`` made primal infeasible: rows 0 and 1 of ``G`` are
    opposite, but both are required to lie in [1, 2]."""
    qp = {k: np.array(v, copy=True) for k, v in qp.items()}
    qp["G"][1] = -qp["G"][0]
    qp["h_l"][:2] = 1.0
    qp["h_u"][:2] = 2.0
    return qp


def stack_qps(qps: list) -> dict:
    """Stack a list of same-shaped host problems into one batched problem."""
    return {k: np.stack([q[k] for q in qps]) for k in qps[0]}


# ----------------------------------------------------------------------
# Solvers and results
# ----------------------------------------------------------------------

def quiet(solver):
    """Turn off the solver's console output; returns the solver."""
    solver.settings.verbose = False
    return solver


def statuses(solver) -> list:
    """Per-problem ``Status`` of the last solve (waits for the solve)."""
    return list(solver.result.info.to_host().status)


def all_solved(status_list) -> bool:
    """True if every entry of ``status_list`` is ``CUPIQP_SOLVED``."""
    return all(st == Status.CUPIQP_SOLVED for st in status_list)


def layer_statuses(layer) -> tuple:
    """Per-problem ``Status`` of the latest call of a PyTorch solver layer
    (an explicit device-to-host read)."""
    return tuple(Status(int(v)) for v in layer.result.status.cpu().tolist())
