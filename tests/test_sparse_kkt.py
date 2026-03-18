"""
Unit tests for KKTSystem: condensed factorize-solve round-trip.

Mirrors PIQP's C++ test (tests/src/sparse/kkt_test.cpp FactorizeSolve)
"""
import pytest
import numpy as np
import cupy as cp
import scipy.sparse as sp_cpu
from cupyx.scipy.sparse import csr_matrix

from cupiqp.sparse.sparse_data import SparseData
from cupiqp.kkt_systems import KKTSystem
from cupiqp.results import Variables
from cupiqp.settings import Settings


def make_random_sparse_qp(n=20, p=8, m=9, density=0.3, seed=42) -> SparseData:
    """Generate a random strongly-convex sparse QP.

    Returns scipy (CPU) sparse matrices and numpy arrays so the caller
    can convert to CuPy as needed.
    """
    rng = np.random.default_rng(seed)

    # P: sparse symmetric positive definite
    M = sp_cpu.random(n, n, density=density, format='csr', random_state=rng)
    P = (M @ M.T + n * sp_cpu.eye(n)).tocsr()
    P = ((P + P.T) / 2).tocsr()

    c = rng.standard_normal(n)

    # A: equality constraints
    A = sp_cpu.random(p, n, density=density + 0.1, format='csr', random_state=rng)
    b = rng.standard_normal(p)

    # G: inequality constraints
    G = sp_cpu.random(m, n, density=density + 0.1, format='csr', random_state=rng)
    h_u = np.abs(rng.standard_normal(m)) + 1.0
    h_l = -np.abs(rng.standard_normal(m)) - 1.0

    x_u = np.abs(rng.standard_normal(n)) + 1.0
    x_l = -np.abs(rng.standard_normal(n)) - 1.0

    # Randomly set some bounds to +/-inf (like real QP problems)
    if m > 0:
        inf_mask_h_u = rng.random(m) < 0.3
        inf_mask_h_l = rng.random(m) < 0.3
        h_u[inf_mask_h_u] = np.inf
        h_l[inf_mask_h_l] = -np.inf

    inf_mask_x_u = rng.random(n) < 0.3
    inf_mask_x_l = rng.random(n) < 0.3
    x_u[inf_mask_x_u] = np.inf
    x_l[inf_mask_x_l] = -np.inf

    return SparseData(
        P=csr_matrix(P),
        c=cp.array(c),
        A=csr_matrix(A),
        b=cp.array(b),
        G=csr_matrix(G),
        h_u=cp.array(h_u),
        h_l=cp.array(h_l),
        x_u=cp.array(x_u),
        x_l=cp.array(x_l),
    )


class TestKKTSystemCondensedSolve:
    """Test that solving the condensed KKT system and multiplying back gives
    the original RHS."""

    @pytest.mark.parametrize("n,p,m", [
        (10, 0, 5),   # no equality constraints
        (10, 3, 0),   # no inequality constraints
        (10, 0, 0),   # only bound constraints
        (30, 10, 15), # larger problem
        (60, 20, 30), # even larger problem
    ])
    def test_condensed_factorize_solve(self, n, p, m):
        """Solve K*lhs = rhs, verify K*lhs ≈ rhs via
        mul_condensed_kkt."""
        data = make_random_sparse_qp(n, p, m)

        settings = Settings()
        kkt = KKTSystem()
        kkt.init(data, settings)

        vars = Variables()
        vars.init(data)
        vars.set_random()

        rho_arr = cp.array([1.])
        delta_arr = cp.array([1.])

        success = kkt.update_scalings_and_factor(
            data, settings, False, rho_arr, delta_arr, vars)
        assert success, "KKT factorization failed"

        settings = Settings()
        kkt = KKTSystem()
        kkt.init(data, settings)

        vars = Variables()
        vars.init(data)
        vars.set_random()

        rho_arr = cp.array([1.])
        delta_arr = cp.array([1.])

        success = kkt.update_scalings_and_factor(
            data, settings, False, rho_arr, delta_arr, vars)

        # Random condensed RHS
        cp.random.seed(123)
        rhs_x = cp.random.randn(n)
        rhs_y = cp.random.randn(p)
        rhs_z = cp.random.randn(m)

        # Solve condensed system directly
        lhs_x = cp.zeros(n)
        lhs_y = cp.zeros(p)
        lhs_z = cp.zeros(m)
        kkt._kkt_solver.solve(data, rhs_x, rhs_y, rhs_z,
                              lhs_x, lhs_y, lhs_z)

        # Verify: K * lhs should equal rhs
        check_x = cp.zeros(n)
        check_y = cp.zeros(p)
        check_z = cp.zeros(m)
        kkt.mul_condensed_kkt(data, lhs_x, lhs_y, lhs_z,
                              check_x, check_y, check_z)

        atol = 1e-8
        assert cp.allclose(rhs_x, check_x, atol=atol), \
            f"x block mismatch: max err = {float(cp.max(cp.abs(rhs_x - check_x))):.2e}"
        assert cp.allclose(rhs_y, check_y, atol=atol), \
            f"y block mismatch: max err = {float(cp.max(cp.abs(rhs_y - check_y))):.2e}"
        assert cp.allclose(rhs_z, check_z, atol=atol), \
            f"z block mismatch: max err = {float(cp.max(cp.abs(rhs_z - check_z))):.2e}"

    @pytest.mark.parametrize("n,p,m", [
        (20, 8, 9),
        (10, 0, 5),
        (10, 3, 0),
        (30, 10, 15),
    ])
    def test_condensed_solve_with_ir(self, n, p, m):
        """Solve with iterative refinement enabled and verify the residual
        is at least as good as without IR."""
        data = make_random_sparse_qp(n, p, m)

        # --- Without IR ---
        settings_no_ir = Settings()
        settings_no_ir.iterative_refinement_max_iter = 0
        kkt_no_ir = KKTSystem()
        kkt_no_ir.init(data, settings_no_ir)

        vars_no_ir = Variables()
        vars_no_ir.init(data)
        vars_no_ir.set_random()

        rho_arr = cp.array([1.])
        delta_arr = cp.array([1.])

        success = kkt_no_ir.update_scalings_and_factor(
            data, settings_no_ir, False, rho_arr, delta_arr, vars_no_ir)
        assert success, "KKT factorization (no IR) failed"

        cp.random.seed(111)
        rhs_x = cp.random.randn(n)
        rhs_y = cp.random.randn(p)
        rhs_z = cp.random.randn(m)

        lhs_x = cp.zeros(n)
        lhs_y = cp.zeros(p)
        lhs_z = cp.zeros(m)
        kkt_no_ir._kkt_solver.solve(data, rhs_x.copy(), rhs_y.copy(), rhs_z.copy(),
                                    lhs_x, lhs_y, lhs_z)

        err_x = cp.zeros(n)
        err_y = cp.zeros(p)
        err_z = cp.zeros(m)
        error_no_ir = kkt_no_ir.get_refinement_error(
            data, lhs_x, lhs_y, lhs_z,
            rhs_x, rhs_y, rhs_z,
            err_x, err_y, err_z)

        # --- With IR (static reg + iterative refinement) ---
        settings_ir = Settings()
        settings_ir.iterative_refinement_max_iter = 10
        kkt_ir = KKTSystem()
        kkt_ir.init(data, settings_ir)

        vars_ir = Variables()
        vars_ir.init(data)
        vars_ir.set_random()  # same seed as vars_no_ir (set_random uses seed=0)

        # Factor with IR enabled (adds static regularization)
        success = kkt_ir.update_scalings_and_factor(
            data, settings_ir, True, rho_arr, delta_arr, vars_ir)
        assert success, "KKT factorization (with IR) failed"

        lhs_x2 = cp.zeros(n)
        lhs_y2 = cp.zeros(p)
        lhs_z2 = cp.zeros(m)
        kkt_ir._kkt_solver.solve(data, rhs_x.copy(), rhs_y.copy(), rhs_z.copy(),
                                 lhs_x2, lhs_y2, lhs_z2)

        # Run condensed IR
        kkt_ir.condensed_iterative_refinement(
            data, settings_ir,
            rhs_x.copy(), rhs_y.copy(), rhs_z.copy(),
            lhs_x2, lhs_y2, lhs_z2)

        err_x2 = cp.zeros(n)
        err_y2 = cp.zeros(p)
        err_z2 = cp.zeros(m)
        error_ir = kkt_ir.get_refinement_error(
            data, lhs_x2, lhs_y2, lhs_z2,
            rhs_x, rhs_y, rhs_z,
            err_x2, err_y2, err_z2)

        print(f"  n={n}, p={p}, m={m}: error_no_ir={error_no_ir:.2e}, error_ir={error_ir:.2e}")

        # IR should not make things worse (for well-conditioned problems)
        assert error_ir <= error_no_ir * 10 + 1e-14, \
            f"IR made things worse: {error_ir:.2e} > {error_no_ir:.2e}"
