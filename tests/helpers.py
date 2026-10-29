"""Shared helpers of the test suite: test problems and host-to-GPU conversion.

Tests build their problems with NumPy on the host and move them to the GPU as
``warp.array`` with :func:`to_wp`, the array type the solvers work in. CuPy,
PyTorch and other frameworks appear only in tests about accepting their arrays.
"""
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from cupiqp import Status

DEVICE = "cuda"
DATA_DIR = Path(__file__).resolve().parent / "data"

# ----------------------------------------------------------------------
# Precisions and comparisons
# ----------------------------------------------------------------------

# Precisions every numerical test runs in. A test rounds its host inputs to
# the test dtype (rounded()), so the GPU and the float64 NumPy reference see
# the same numbers.
DTYPES = [pytest.param(wp.float64, id="f64"), pytest.param(wp.float32, id="f32")]


def rounded(a, dtype) -> np.ndarray:
    """``a`` rounded to ``dtype`` and returned as float64 host data: the
    exact values a ``dtype`` GPU array of ``a`` holds."""
    return np.asarray(a).astype(_numpy_dtype(dtype)).astype(np.float64)


def host(a: wp.array) -> np.ndarray:
    """Host float64 copy of a GPU array (waits for it)."""
    return a.numpy().astype(np.float64)


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


def map_arrays(qp: dict, fn) -> dict:
    """Apply ``fn`` to every array of ``qp``, keeping ``(diag, offdiag)``
    pairs and ``None`` entries."""
    out = {}
    for name, value in qp.items():
        if value is None:
            out[name] = None
        elif isinstance(value, tuple):
            out[name] = tuple(None if v is None else fn(v) for v in value)
        else:
            out[name] = fn(value)
    return out


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

def random_spd(rng, n, batch_size=None) -> np.ndarray:
    """Random symmetric positive definite ``M M^T + n I`` (``(n, n)``, or
    ``(batch_size, n, n)``). ``rng`` is a seed or a NumPy Generator."""
    rng = np.random.default_rng(rng)
    lead = () if batch_size is None else (batch_size,)
    M = rng.standard_normal(lead + (n, n))
    P = M @ np.swapaxes(M, -1, -2) + n * np.eye(n)
    return 0.5 * (P + np.swapaxes(P, -1, -2))


def random_sparse_spd(rng, n, density=0.4) -> sp.csr_matrix:
    """Random sparse symmetric positive definite ``M M^T + n I`` with sorted
    indices, from a random sparse ``M`` of the given ``density``."""
    rng = np.random.default_rng(rng)
    M = sp.random(n, n, density=density, format="csr", random_state=rng)
    P = (M @ M.T + n * sp.eye(n)).tocsr()
    P = (0.5 * (P + P.T)).tocsr()
    P.sort_indices()
    return P


def _infinite_masks(rng, n, m, inf_fraction) -> dict:
    """Random masks of the ``inf_fraction`` of each bound that is infinite."""
    if not inf_fraction:
        return {}
    return {k: rng.random(size) < inf_fraction for k, size in (("h_u", m), ("h_l", m), ("x_u", n), ("x_l", n))}


def _vectors_around(rng, A, G, infinite, margin=2.0) -> dict:
    """``c``, ``b`` and the bounds of a QP with constraint matrices ``A`` and
    ``G``, built around a random point ``x_star``: ``b = A x_star`` and every
    finite bound is at least ``margin`` away from ``G x_star`` or ``x_star``.
    The problem is feasible with ``x_star`` strictly inside the inequalities."""
    n, m = G.shape[1], G.shape[0]
    x_star = rng.uniform(-1.0, 1.0, n)
    Gx = G @ x_star
    v = dict(
        c=rng.standard_normal(n),
        b=A @ x_star,
        h_u=Gx + np.abs(rng.standard_normal(m)) + margin,
        h_l=Gx - np.abs(rng.standard_normal(m)) - margin,
        x_u=x_star + np.abs(rng.standard_normal(n)) + margin,
        x_l=x_star - np.abs(rng.standard_normal(n)) - margin,
    )
    for k, mask in infinite.items():
        v[k][mask] = np.inf if k.endswith("u") else -np.inf
    return v


def random_dense_qp(rng=0, n=8, p=3, m=5, batch_size=None, inf_fraction=0.0) -> dict:
    """Random strictly convex, feasible QP with equalities, two-sided
    inequalities and box bounds, as host arrays. ``rng`` is a seed or a NumPy
    Generator. The problem is built around a point that satisfies the
    equalities and lies strictly inside every finite bound.

    ``batch_size`` gives every field a leading batch axis (independent
    problems). ``inf_fraction`` of the bounds are infinite, at the same
    entries for every problem of the batch, as the solvers require.
    """
    rng = np.random.default_rng(rng)
    infinite = _infinite_masks(rng, n, m, inf_fraction)

    def one():
        qp = dict(P=random_spd(rng, n), A=rng.standard_normal((p, n)), G=rng.standard_normal((m, n)))
        return {**qp, **_vectors_around(rng, qp["A"], qp["G"], infinite)}

    return one() if batch_size is None else stack_qps([one() for _ in range(batch_size)])


def random_sparse_qp(rng=0, n=8, p=3, m=5, density=0.5, inf_fraction=0.0) -> dict:
    """:func:`random_dense_qp` with SciPy CSR matrices (sorted indices) of the given
    ``density``: a random sparse SPD ``P`` and sparse ``A`` and ``G`` with a
    nonzero in every row. Strictly convex and feasible like :func:`random_dense_qp`."""
    rng = np.random.default_rng(rng)
    infinite = _infinite_masks(rng, n, m, inf_fraction)
    qp = dict(P=random_sparse_spd(rng, n, density))
    for k, rows in (("A", p), ("G", m)):
        keep = rng.random((rows, n)) < density
        keep[np.arange(rows), np.arange(rows) % n] = True
        M = sp.csr_matrix(np.where(keep, rng.standard_normal((rows, n)), 0.0))
        M.sort_indices()
        qp[k] = M
    return {**qp, **_vectors_around(rng, qp["A"], qp["G"], infinite)}


def random_feasible_qp(rng=0, n=5, p=1, m=3, batch_size=None, infinite=None) -> dict:
    """Random strictly convex, feasible QP like :func:`random_dense_qp`, but with
    bounds close to the interior point (margin 0.2), so some are typically
    active at the solution.

    ``batch_size`` gives the vectors (``c``, ``b`` and the bounds) a leading
    batch axis, one interior point per problem; ``P``, ``A`` and ``G`` are
    shared. ``infinite`` maps any of ``h_l``, ``h_u``, ``x_l``, ``x_u`` to a
    boolean mask of the entries set to +/-inf.
    """
    rng = np.random.default_rng(rng)
    qp = dict(P=random_spd(rng, n), A=rng.standard_normal((p, n)), G=rng.standard_normal((m, n)))
    infinite = {k: np.asarray(mask, bool) for k, mask in (infinite or {}).items()}
    vecs = [_vectors_around(rng, qp["A"], qp["G"], infinite, margin=0.2) for _ in range(batch_size or 1)]
    return {**qp, **(vecs[0] if batch_size is None else stack_qps(vecs))}


def random_multistage_P(rng, N, d, batch_size=None, has_P_E=True):
    """Blocks ``(P_D, P_E)`` of a random block-tridiagonal positive definite
    P: ``N`` diagonal blocks of size ``d`` and, if ``has_P_E``, ``N - 1``
    lower off-diagonal blocks (else ``P_E`` is None). The smallest
    eigenvalue of the assembled P is at least ``3 d``. ``batch_size`` adds a
    leading batch axis."""
    lead = () if batch_size is None else (batch_size,)
    P_D = rng.standard_normal(lead + (N, d, d))
    P_D = P_D @ np.swapaxes(P_D, -1, -2) + 3 * d * np.eye(d)
    if not has_P_E:
        return P_D, None
    P_E = 0.3 * rng.standard_normal(lead + (N - 1, d, d))
    # Shift block k by ||P_E[k-1]|| + ||P_E[k]||: then x^T P x >= 3d |x|^2
    # by the block Gershgorin bound.
    norms = np.linalg.norm(P_E, ord=2, axis=(-2, -1))
    pad = np.zeros(lead + (1,))
    margin = np.concatenate([pad, norms], axis=-1) + np.concatenate([norms, pad], axis=-1)
    return P_D + margin[..., None, None] * np.eye(d), P_E


def random_multistage_qp(rng=0, N=4, d=3, rA=2, rG=2, batch_size=None, has_P_E=True, has_A_E=True,
                         has_G_E=False) -> dict:
    """A random strictly convex, feasible multistage QP as host arrays, in the
    solver's argument layout: matrices as (diag, offdiag) pairs, vectors in
    the block layout. ``N`` stages of size ``d``, ``rA`` / ``rG`` rows per
    block of A / G; ``has_*_E`` selects the off-diagonal blocks.
    ``batch_size`` adds a leading batch axis to every array."""
    rng = np.random.default_rng(rng)
    lead = () if batch_size is None else (batch_size,)
    P_D, P_E = random_multistage_P(rng, N, d, batch_size, has_P_E)
    A_D = rng.standard_normal(lead + (N, rA, d))
    A_E = rng.standard_normal(lead + (N, rA, d)) if has_A_E else None
    G_D = rng.standard_normal(lead + (N, rG, d))
    G_E = rng.standard_normal(lead + (N, rG, d)) if has_G_E else None
    # b = A x0 and G x0 in the interior, so that the problem is feasible.
    x0 = rng.uniform(-0.3, 0.3, lead + (N, d))

    def bidiag_apply(D, E):
        out = np.einsum("...krd,...kd->...kr", D, x0)
        if E is None:
            return out
        shifted = np.einsum("...krd,...kd->...kr", E, x0)
        zero = np.zeros(out.shape[:-2] + (1, out.shape[-1]))
        return np.concatenate([out, zero], axis=-2) + np.concatenate([zero, shifted], axis=-2)

    Gx = bidiag_apply(G_D, G_E)
    return dict(
        P=(P_D, P_E), c=rng.standard_normal(lead + (N, d)),
        A=(A_D, A_E), b=bidiag_apply(A_D, A_E),
        G=(G_D, G_E), h_l=Gx - 1.0, h_u=Gx + 0.5,
        x_l=-np.ones(lead + (N, d)), x_u=np.ones(lead + (N, d)),
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


def assemble_multistage(N, d, P_D, P_E, A_D=None, A_E=None, G_D=None, G_E=None):
    """Dense ``(P, A, G)`` of one multistage problem from its blocks.

    P is block tridiagonal with lower off-diagonal blocks ``P_E`` (block
    diagonal when ``P_E`` is None). A and G are block bidiagonal: ``N + 1``
    block rows with lower off-diagonal blocks, ``N`` without. An absent A or
    G (``A_D`` / ``G_D`` None) gives a ``(0, N * d)`` matrix.
    """
    n = N * d
    P = np.zeros((n, n))
    for k in range(N):
        P[k * d:(k + 1) * d, k * d:(k + 1) * d] = P_D[k]
    if P_E is not None:
        for k in range(N - 1):
            P[(k + 1) * d:(k + 2) * d, k * d:(k + 1) * d] = P_E[k]
            P[k * d:(k + 1) * d, (k + 1) * d:(k + 2) * d] = P_E[k].T

    def bidiag(D, E):
        if D is None:
            return np.zeros((0, n))
        r = D.shape[1]
        M = np.zeros(((N + (E is not None)) * r, n))
        for k in range(N):
            M[k * r:(k + 1) * r, k * d:(k + 1) * d] = D[k]
        if E is not None:
            for k in range(N):
                M[(k + 1) * r:(k + 2) * r, k * d:(k + 1) * d] = E[k]
        return M

    return P, bidiag(A_D, A_E), bidiag(G_D, G_E)


# ----------------------------------------------------------------------
# Solvers and results
# ----------------------------------------------------------------------

def quiet(solver, **settings):
    """Turn off the console output of ``solver`` (a solver or solver layer)
    and set the given ``settings`` fields; returns ``solver``."""
    solver.settings.verbose = False
    for name, value in settings.items():
        setattr(solver.settings, name, value)
    return solver


def tight(solver, **settings):
    """:func:`quiet` with stopping tolerances tight enough for
    finite-difference gradient checks. float64 solves only: float32 cannot
    reach a 1e-11 residual."""
    return quiet(solver, **{"eps_abs": 1e-11, "eps_rel": 1e-11, **settings})


def statuses(solver) -> list:
    """Per-problem ``Status`` of the last solve (waits for the solve)."""
    return list(solver.result.info.to_host().status)


def all_solved(status_list) -> bool:
    """True if every entry of ``status_list`` is ``CUPIQP_SOLVED``."""
    return all(st == Status.CUPIQP_SOLVED for st in status_list)


def assert_all_solved(solver) -> None:
    """Every problem of the last solve of ``solver`` ended ``CUPIQP_SOLVED``."""
    st = statuses(solver)
    assert all_solved(st), st
