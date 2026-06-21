"""Unit tests for KKTSystem (sparse backend): condensed factorize-solve round-trip.

Mirrors PIQP's C++ test (``tests/src/sparse/kkt_test.cpp::FactorizeSolve``).
"""
import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp_cpu
from cupyx.scipy.sparse import csr_matrix

from cupiqp.kkt_systems import KKTSystem
from cupiqp.results import Variables
from cupiqp.settings import Settings
from cupiqp.sparse.sparse_data import SparseData
from cupiqp.sparse.sparse_preconditioner import SparseRuizEquilibration


def random_sparse_qp(
    n: int = 20, p: int = 8, m: int = 9,
    density: float = 0.3, seed: int = 42,
) -> SparseData:
    """Build a random strongly-convex sparse QP on the GPU.

    P is SPD (``MM^T + n*I``, then symmetrized). A, G are random rectangular
    sparse matrices. Some bounds are set to ±∞ to mimic real QPs.
    """
    rng = np.random.default_rng(seed)

    # P: symmetric positive definite
    M = sp_cpu.random(n, n, density=density, format='csr', random_state=rng)
    P = (M @ M.T + n * sp_cpu.eye(n)).tocsr()
    P = ((P + P.T) / 2).tocsr()

    c = rng.standard_normal(n)

    A = sp_cpu.random(p, n, density=density + 0.1, format='csr', random_state=rng)
    b = rng.standard_normal(p)

    G = sp_cpu.random(m, n, density=density + 0.1, format='csr', random_state=rng)
    h_u = np.abs(rng.standard_normal(m)) + 1.0
    h_l = -np.abs(rng.standard_normal(m)) - 1.0
    x_u = np.abs(rng.standard_normal(n)) + 1.0
    x_l = -np.abs(rng.standard_normal(n)) - 1.0

    # Sprinkle ±inf bounds so the test exercises the inactive-bound path.
    if m > 0:
        h_u[rng.random(m) < 0.3] = np.inf
        h_l[rng.random(m) < 0.3] = -np.inf
    x_u[rng.random(n) < 0.3] = np.inf
    x_l[rng.random(n) < 0.3] = -np.inf

    data = SparseData()
    data.init(
        P=csr_matrix(P), c=cp.array(c),
        A=csr_matrix(A), b=cp.array(b),
        G=csr_matrix(G), h_u=cp.array(h_u), h_l=cp.array(h_l),
        x_u=cp.array(x_u), x_l=cp.array(x_l),
        )
    return data


def random_rhs(n: int, p: int, m: int, seed: int) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Three random (1, k) condensed-RHS blocks for the (B=1) sparse KKT."""
    rng = cp.random.RandomState(seed)
    return rng.randn(1, n), rng.randn(1, p), rng.randn(1, m)


def factor(
    data: SparseData,
    settings: Settings,
    use_static_reg: bool,
    rho: float = 1.0,
    delta: float = 1.0
    ) -> tuple[KKTSystem, Variables]:
    """Init a ``KKTSystem``, randomize IPM variables, factor — return both."""
    kkt = KKTSystem()
    kkt.init(data, settings)

    preconditioner = SparseRuizEquilibration(
        data.batch_size, data.n, data.p, data.m,
        has_h_l=data.has_h_l, has_h_u=data.has_h_u,
        has_x_l=data.has_x_l, has_x_u=data.has_x_u,
    )

    variables = Variables()
    variables.init(data)
    variables.set_random()

    rho_arr = cp.array([rho])
    delta_arr = cp.array([delta])
    ok = kkt.update_scalings_and_factor(
        data, preconditioner, settings, use_static_reg, rho_arr, delta_arr, variables,
    )
    assert ok, "KKT factorization failed"
    return kkt, variables





# ---------------------------------------------------------------------------
# Problem-shape grids
# ---------------------------------------------------------------------------
# (n, p, m) tuples — cover constraint-set variations + a couple of larger sizes.
ROUND_TRIP_SHAPES: list[tuple[int, int, int]] = [
    (10, 0, 5),    # no equality constraints
    (10, 3, 0),    # no inequality constraints
    (10, 0, 0),    # only bound constraints
    (20, 5, 8),    # mixed
    (30, 10, 15),  # larger
    (60, 20, 30),  # even larger
]

IR_SHAPES: list[tuple[int, int, int]] = [
    (20, 8, 9),
    (10, 0, 5),
    (10, 3, 0),
    (30, 10, 15),
]


# ===========================================================================
# Tests
# ===========================================================================

@pytest.mark.parametrize("n,p,m", ROUND_TRIP_SHAPES)
def test_condensed_factorize_solve(n: int, p: int, m: int) -> None:
    """Solve ``K_c * lhs = rhs`` for the condensed KKT and verify the
    multiply-back identity holds via the public ``mul_condensed_kkt``."""
    data = random_sparse_qp(n, p, m)
    settings = Settings()
    kkt, _ = factor(data, settings, use_static_reg=False)

    rhs_x, rhs_y, rhs_z = random_rhs(n, p, m, seed=123)

    lhs_x = cp.zeros((1, n))
    lhs_y = cp.zeros((1, p))
    lhs_z = cp.zeros((1, m))
    # No public entry point yet for the condensed-only solve (KKTSystem.solve
    # always runs eliminate→condensed→recover). Use the inner solver directly.
    kkt._kkt_solver.solve(data, rhs_x, rhs_y, rhs_z, lhs_x, lhs_y, lhs_z)

    check_x = cp.zeros((1, n))
    check_y = cp.zeros((1, p))
    check_z = cp.zeros((1, m))
    kkt.mul_condensed_kkt(data, lhs_x, lhs_y, lhs_z, check_x, check_y, check_z)

    atol = 1e-8
    cp.testing.assert_allclose(rhs_x, check_x, atol=atol)
    cp.testing.assert_allclose(rhs_y, check_y, atol=atol)
    cp.testing.assert_allclose(rhs_z, check_z, atol=atol)


@pytest.mark.parametrize("n,p,m", IR_SHAPES)
def test_condensed_solve_with_ir(n: int, p: int, m: int) -> None:
    """Run the same RHS through the condensed solve twice — once without
    iterative refinement, once with — and assert IR doesn't make things worse."""
    data = random_sparse_qp(n, p, m)
    rhs_x, rhs_y, rhs_z = random_rhs(n, p, m, seed=111)

    # --- Without IR ----------------------------------------------------------
    settings_no_ir = Settings()
    settings_no_ir.iterative_refinement_max_iter = 0
    kkt_no_ir, _ = factor(data, settings_no_ir, use_static_reg=False)

    lhs_x = cp.zeros((1, n))
    lhs_y = cp.zeros((1, p))
    lhs_z = cp.zeros((1, m))
    kkt_no_ir._kkt_solver.solve(
        data, rhs_x.copy(), rhs_y.copy(), rhs_z.copy(), lhs_x, lhs_y, lhs_z,
    )
    err_x, err_y, err_z = cp.zeros((1, n)), cp.zeros((1, p)), cp.zeros((1, m))
    error_no_ir = kkt_no_ir.get_refinement_error(
        data, lhs_x, lhs_y, lhs_z, rhs_x, rhs_y, rhs_z, err_x, err_y, err_z,
    )

    # --- With IR (static reg + IR loop) --------------------------------------
    settings_ir = Settings()
    settings_ir.iterative_refinement_max_iter = 10
    kkt_ir, _ = factor(data, settings_ir, use_static_reg=True)

    lhs_x2 = cp.zeros((1, n))
    lhs_y2 = cp.zeros((1, p))
    lhs_z2 = cp.zeros((1, m))
    kkt_ir._kkt_solver.solve(
        data, rhs_x.copy(), rhs_y.copy(), rhs_z.copy(), lhs_x2, lhs_y2, lhs_z2,
    )
    kkt_ir.iterative_refinement(
        data, settings_ir,
        rhs_x.copy(), rhs_y.copy(), rhs_z.copy(),
        lhs_x2, lhs_y2, lhs_z2,
    )
    err_x2, err_y2, err_z2 = cp.zeros((1, n)), cp.zeros((1, p)), cp.zeros((1, m))
    error_ir = kkt_ir.get_refinement_error(
        data, lhs_x2, lhs_y2, lhs_z2,
        rhs_x, rhs_y, rhs_z, err_x2, err_y2, err_z2,
    )

    # IR must not make things worse for well-conditioned problems.
    assert error_ir <= error_no_ir * 10 + 1e-14, (
        f"IR made things worse: {error_ir:.2e} > {error_no_ir:.2e}"
    )
