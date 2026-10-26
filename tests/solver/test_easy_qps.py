"""Regression tests on small, well-conditioned QPs that must always solve.

The problem family is

    min  1/2 x'Px + c'x   s.t.  G x <= h_u      (no h_l, no A, no box)

with n = 4, m = 8, P positive definite and h > 0, so x = 0 is strictly
feasible. Every problem of a large random batch must be SOLVED on every
backend, with and without the Ruiz preconditioner. Before the step-length cap
(see test_step_length.py) a few problems per 100k stalled at
CUPIQP_MAX_ITER_REACHED.

The multistage test covers blocks with more constraint rows r than columns d:
the multistage KKT solve used to ignore rows ``d, ..., r-1`` of every block.
"""
import numpy as np
import pytest

from cupiqp import DenseSolver, SparseSolver, MultistageSolver, Status
from tests.helpers import to_wp


SETTINGS = [
    pytest.param({}, id="default"),
    pytest.param({"preconditioner_iter": 0}, id="no_preconditioner"),
]


def _sample_family(B, n=4, m=8, seed=0):
    """``B`` random problems of the easy family, as batched NumPy arrays."""
    rng = np.random.default_rng(seed)
    L = np.tril(rng.normal(0.0, 0.3, (B, n, n)))
    P = L @ L.transpose(0, 2, 1) + 0.1 * np.eye(n)
    c = rng.normal(0.0, 0.1, (B, n))
    G = rng.normal(0.0, 0.3, (B, m, n))
    h = np.abs(rng.normal(0.0, 0.3, (B, m)))
    return P, c, G, h


def _dense_csr(M):
    """CSR triple of a fully dense pattern; ``M`` is ``(B, rows, cols)``."""
    B, rows, cols = M.shape
    indptr = np.arange(0, rows * cols + 1, cols, dtype=np.int32)
    indices = np.tile(np.arange(cols, dtype=np.int32), rows)
    return to_wp(indptr), to_wp(indices), to_wp(M.reshape(B, rows * cols))


def _solve(backend, P, c, G, h, settings):
    """Solve the batch with one backend; return ``(statuses, x, z_u)``.

    For the multistage backend the problem is one stage (N = 1): block row 0
    holds ``G`` and the trailing block row is all zero with ``h_u = 1``, so
    the first ``m`` entries of ``z_u`` are the multipliers of ``G x <= h``.
    """
    B, n = c.shape
    m = G.shape[1]
    solver = {"dense": DenseSolver, "sparse": SparseSolver, "multistage": MultistageSolver}[backend]()
    for key, value in settings.items():
        setattr(solver.settings, key, value)
    if backend == "dense":
        solver.setup(P=to_wp(P), c=to_wp(c), G=to_wp(G), h_u=to_wp(h))
    elif backend == "sparse":
        solver.setup(P=_dense_csr(P), c=to_wp(c), G=_dense_csr(G), h_u=to_wp(h))
    else:
        h_blocks = np.stack([h, np.ones_like(h)], axis=1)  # (B, N+1, m)
        solver.setup(P=(to_wp(P[:, None]), None), c=to_wp(c[:, None]),
                     G=(to_wp(G[:, None]), to_wp(np.zeros_like(G)[:, None])), h_u=to_wp(h_blocks))
    solver.solve()
    statuses = np.array([s.value for s in solver.result.info.to_host().status])
    x = solver.result.x.numpy().reshape(B, n)
    z = solver.result.z_u.numpy().reshape(B, -1)[:, :m]
    return statuses, x, z


def _assert_kkt(P, c, G, h, x, z, tol=1e-6):
    """Check the KKT conditions of every problem of the batch."""
    slack = h - np.einsum("bij,bj->bi", G, x)
    stationarity = np.einsum("bij,bj->bi", P, x) + c + np.einsum("bji,bj->bi", G, z)
    assert np.abs(stationarity).max() < tol
    assert slack.min() > -tol
    assert z.min() > -tol
    assert np.abs(slack * z).max() < tol


@pytest.mark.parametrize("backend", ["dense", "sparse", "multistage"])
@pytest.mark.parametrize("settings", SETTINGS)
def test_random_easy_family_all_solved(backend, settings):
    """Every problem of a large batch of the easy family is SOLVED and
    satisfies the KKT conditions."""
    P, c, G, h = _sample_family(100_000, seed=0)
    statuses, x, z = _solve(backend, P, c, G, h, settings)
    failed = np.flatnonzero(statuses != Status.CUPIQP_SOLVED.value)
    assert failed.size == 0, f"{failed.size} problems not solved, e.g. {failed[:10]}"
    _assert_kkt(P, c, G, h, x, z)


@pytest.mark.parametrize("m", [5, 8, 12])
def test_multistage_more_constraint_rows_than_block_size(m):
    """Multistage blocks with r = m constraint rows and d = 4 columns, r > d."""
    P, c, G, h = _sample_family(1000, n=4, m=m, seed=1)
    statuses, x, z = _solve("multistage", P, c, G, h, {})
    assert np.all(statuses == Status.CUPIQP_SOLVED.value)
    _assert_kkt(P, c, G, h, x, z)
