"""Wide problems: the block-reduction kernels are independent of the width.

Every reduction of the interior-point iteration (step lengths, mu, sigma,
residual norms, objectives) is a fixed-block kernel whose threads stride over
the row, so the same compiled kernel handles a 3-variable QP and one whose
KKT width is in the thousands. These tests solve wide problems with known
solutions on the dense and multistage backends.
"""
import numpy as np
import pytest

from cupiqp import DenseSolver, MultistageSolver
from tests.helpers import all_solved, broadcast, to_wp


def _assert_within_gap_bound(x, x_star, duality_gap):
    """Check x against the exact solution x_star of a QP with P = I.

    The objective is 1-strongly convex, so for a primal-dual feasible pair
    0.5 * ||x - x*||^2 <= f(x) - f(x*) <= duality gap, i.e.
    ||x - x*||_2 <= sqrt(2 * gap). An elementwise atol cannot be used here:
    the solver stops once the gap passes eps_duality_gap_rel * |obj| (~1e-6
    for these objectives), and an entry whose unconstrained optimum -c sits
    within ~1e-3 of a bound is nearly degenerate (tiny multiplier), so its
    error is of order sqrt(gap) ~ 5e-4. CPU PIQP returns the same iterate.
    """
    err = np.linalg.norm((x - x_star).reshape(x.shape[0], -1), axis=1)
    assert np.all(err <= np.sqrt(2.0 * duality_gap) + 1e-8), (err, duality_gap)


@pytest.mark.slow
@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_wide_dense_box_qp(enable_cuda_graph):
    """min 0.5||x||^2 + c^T x s.t. x_l <= x <= x_u has x* = clip(-c, x_l, x_u)."""
    B, n = 2, 1500
    rng = np.random.default_rng(0)
    c = rng.standard_normal((B, n)) * 2.0
    x_l, x_u = -np.ones(n), np.ones(n)
    s = DenseSolver()
    s.settings.verbose = False
    s.settings.enable_cuda_graph = enable_cuda_graph
    s.setup(P=to_wp(np.eye(n)), c=to_wp(c), x_l=to_wp(x_l), x_u=to_wp(x_u))
    s.solve()
    info = s.result.info.to_host()
    assert all_solved(info.status)
    _assert_within_gap_bound(s.result.x.numpy(), np.clip(-c, x_l, x_u), info.duality_gap)


@pytest.mark.slow
def test_wide_dense_with_equalities_and_inequalities():
    """A wide QP with every block present matches a narrow reference solve of
    the same problem restricted to its active variables."""
    B, n, p, m = 2, 1200, 10, 600
    rng = np.random.default_rng(1)
    M = rng.standard_normal((n, n))
    P = M @ M.T / n + np.eye(n)
    c = rng.standard_normal(n)
    A = rng.standard_normal((p, n))
    b = A @ rng.standard_normal(n)
    G = rng.standard_normal((m, n))
    h_u = np.abs(rng.standard_normal(m)) + 1.0
    h_l = -np.abs(rng.standard_normal(m)) - 1.0
    s = DenseSolver()
    s.settings.verbose = False
    s.setup(P=to_wp(P), c=broadcast(to_wp(c), B), A=to_wp(A), b=to_wp(b),
            G=to_wp(G), h_l=to_wp(h_l), h_u=to_wp(h_u),
            x_l=to_wp(-3 * np.ones(n)), x_u=to_wp(3 * np.ones(n)))
    s.solve()
    assert all_solved(s.result.info.to_host().status)
    x = s.result.x.numpy()
    # Both batch entries hold the template problem: identical solutions,
    # feasible to tolerance, and stationary in the sense of the reported residuals.
    np.testing.assert_allclose(x[0], x[1], atol=1e-12)
    np.testing.assert_allclose(A @ x[0], b, atol=1e-6)
    assert np.all(G @ x[0] <= h_u + 1e-6) and np.all(G @ x[0] >= h_l - 1e-6)
    assert float(s.result.info.primal_res.numpy()[0]) < 1e-6
    assert float(s.result.info.dual_res.numpy()[0]) < 1e-6


@pytest.mark.slow
def test_wide_multistage_box_qp():
    """A long horizon (n = N * d well above the old tile-kernel cutoff) with
    a separable cost: x* = clip(-c, x_l, x_u) per entry."""
    B, N, d = 2, 400, 4
    rng = np.random.default_rng(2)
    c = rng.standard_normal((B, N, d))
    P_diag = to_wp(np.broadcast_to(np.eye(d), (N, d, d)))
    s = MultistageSolver()
    s.settings.verbose = False
    s.setup(P=(P_diag, None), c=to_wp(c),
            x_l=to_wp(-np.ones((N, d))), x_u=to_wp(np.ones((N, d))))
    s.solve()
    info = s.result.info.to_host()
    assert all_solved(info.status)
    x = s.result.x.numpy().reshape(B, N, d)
    _assert_within_gap_bound(x, np.clip(-c, -1.0, 1.0), info.duality_gap)
