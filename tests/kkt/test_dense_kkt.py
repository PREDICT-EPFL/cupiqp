"""Tests for the dense KKT path: low-level ``DenseKKTSolver`` and the
higher-level ``KKTSystem`` interface.

Every test is parameterized over batch size so the single-problem path
(``B=1``) and the genuinely batched path (``B>1``) share coverage -
``DenseData`` is internally batched and ``B=1`` is just the smallest case -
and over the precision (float64 / float32). Problems and right-hand sides
are built with NumPy on the host and moved to the GPU as Warp arrays, the
only array type the KKT internals take; the references are computed in
float64 from the values the GPU holds.
"""
import pytest
import numpy as np
import warp as wp

from cupiqp.dense.dense_data import DenseData
from cupiqp.dense.dense_kkt_solver import DenseKKTSolver
from cupiqp.settings import Settings
from tests.helpers import DTYPES, host, random_dense_qp, rounded, to_wp, to_wp_dict
from tests.kkt.kkt_helpers import (assert_inactive_rows_decoupled, condensed_residual, condensed_solve, empty,
                                   factored_kkt_system, numpy_condensed_kkt_solve, perturb_lhs, zeros)
from tests.tolerances import KKT_RESIDUAL_TOL, KKT_SOLVE_TOL, MATVEC_TOL


def random_dense_data(B=1, n=20, p=8, m=9, seed=42, dtype=wp.float64) -> DenseData:
    """A batched random strictly convex QP (:func:`random_dense_qp`, some bounds
    infinite) as DenseData; absent constraint fields are not passed."""
    qp = random_dense_qp(seed, n, p, m, batch_size=B, inf_fraction=0.3)
    data = DenseData(dtype=dtype)
    data.init(**to_wp_dict({k: v for k, v in qp.items() if v.size}, dtype))
    return data


# NOTE: keep the number of distinct shapes small. The dense KKT kernels are
# specialized and compiled per (n, p, m), constraint layout and dtype, so
# every new shape costs a fresh compile of the kernel set, which dominates
# the run time of this file on a cold cache. The batch size is not part of
# the specialization, so B is free to vary. Hence:
#   - SIZES covers every constraint layout on (n, p, m) = (10, 3, 5), with p
#     and / or m set to zero for a missing block, plus (20, 8, 9). n = 10
#     runs the Warp matrix-product kernels, n = 20 the cuBLAS ones (see
#     WARP_MATMUL_MAX_COLS / _ROWS in cupiqp/dense/dense_matmul.py);
#   - only the tests of a backend path that SIZES misses add a shape:
#     TALL_G_SIZES for the cuBLAS products of a G with many rows, and
#     CHOLESKY_SIZES for both sides of the tile Cholesky limits;
#   - each shape runs at B = 1 and B = 8 only, because B = 1 and B > 1 take
#     different matrix-product and factorization paths.
# The same shapes are used in test_kkt_system.py and test_sparse_kkt.py, so a
# run of tests/kkt compiles them once.
SIZES = [
    pytest.param(1, 10, 3, 5, id="B1-n10-p3-m5"),
    pytest.param(8, 10, 3, 5, id="B8-n10-p3-m5"),
    pytest.param(1, 10, 0, 5, id="B1-n10-p0-m5"),  # no equality
    pytest.param(8, 10, 0, 5, id="B8-n10-p0-m5"),
    pytest.param(1, 10, 3, 0, id="B1-n10-p3-m0"),  # no inequality
    pytest.param(8, 10, 3, 0, id="B8-n10-p3-m0"),
    pytest.param(1, 10, 0, 0, id="B1-n10-p0-m0"),  # box bounds only
    pytest.param(8, 10, 0, 0, id="B8-n10-p0-m0"),
    pytest.param(1, 20, 8, 9, id="B1-n20-p8-m9"),  # cuBLAS products
    pytest.param(8, 20, 8, 9, id="B8-n20-p8-m9"),
]

# More G rows than WARP_MATMUL_MAX_ROWS: the products of G run on cuBLAS,
# those of A on Warp.
TALL_G_SIZES = [
    pytest.param(1, 10, 3, 40, id="B1-n10-p3-m40"),
    pytest.param(8, 10, 3, 40, id="B8-n10-p3-m40"),
]

# The tile Cholesky runs up to n = 48 in float64 and n = 64 in float32,
# cuSOLVER above.
CHOLESKY_SIZES = [
    pytest.param(1, 48, 16, 24, id="B1-n48-p16-m24"),  # largest float64 tile
    pytest.param(8, 48, 16, 24, id="B8-n48-p16-m24"),
    pytest.param(1, 64, 21, 32, id="B1-n64-p21-m32"),  # largest float32 tile, float64 cuSOLVER
    pytest.param(8, 64, 21, 32, id="B8-n64-p21-m32"),
    pytest.param(1, 72, 24, 36, id="B1-n72-p24-m36"),  # cuSOLVER in both
    pytest.param(8, 72, 24, 36, id="B8-n72-p24-m36"),
]


# ===========================================================================
# Low-level DenseKKTSolver primitives
# ===========================================================================
@pytest.mark.parametrize("dtype", DTYPES)
class TestDenseKKTSolverMatvec:
    """Batched matrix-vector primitives on the solver."""

    @pytest.mark.parametrize("B,n,p,m", SIZES + TALL_G_SIZES)
    def test_eval_P_x(self, dtype, B, n, p, m):
        data = random_dense_data(B, n, p=p, m=m, dtype=dtype)
        solver = DenseKKTSolver(data)

        x = rounded(np.random.default_rng(B * 100 + n).standard_normal((B, n)), dtype)
        z = zeros(dtype, B, n)
        solver.eval_P_x(data, 2.0, to_wp(x, dtype), z)

        P, z = host(data.P), host(z)
        for i in range(B):
            expected = 2.0 * P[i] @ x[i]
            np.testing.assert_allclose(z[i], expected, **MATVEC_TOL[dtype])

    @pytest.mark.parametrize("B,n,p,m", SIZES + TALL_G_SIZES)
    def test_eval_A_xn_and_AT_xt(self, dtype, B, n, p, m):
        if p == 0:
            pytest.skip("no equality constraints - A has zero rows")
        data = random_dense_data(B, n, p=p, m=m, dtype=dtype)
        solver = DenseKKTSolver(data)

        x = rounded(np.random.default_rng(B * 100 + n + 1).standard_normal((B, n)), dtype)
        y = zeros(dtype, B, p)
        solver.eval_A_xn(data, 1.0, to_wp(x, dtype), y)

        xt = rounded(np.random.default_rng(B * 100 + p + 2).standard_normal((B, p)), dtype)
        zt = zeros(dtype, B, n)
        solver.eval_AT_xt(data, 1.0, to_wp(xt, dtype), zt)

        A, y, zt = host(data.A), host(y), host(zt)
        for i in range(B):
            np.testing.assert_allclose(y[i], A[i] @ x[i], **MATVEC_TOL[dtype])
            np.testing.assert_allclose(zt[i], A[i].T @ xt[i], **MATVEC_TOL[dtype])

    @pytest.mark.parametrize("B,n,p,m", SIZES + TALL_G_SIZES)
    def test_eval_G_xn_and_GT_xt(self, dtype, B, n, p, m):
        if m == 0:
            pytest.skip("no inequality constraints - G has zero rows")
        data = random_dense_data(B, n, p=p, m=m, dtype=dtype)
        solver = DenseKKTSolver(data)

        x = rounded(np.random.default_rng(B * 100 + n + 3).standard_normal((B, n)), dtype)
        y = zeros(dtype, B, m)
        solver.eval_G_xn(data, 1.0, to_wp(x, dtype), y)

        xt = rounded(np.random.default_rng(B * 100 + m + 4).standard_normal((B, m)), dtype)
        zt = zeros(dtype, B, n)
        solver.eval_GT_xt(data, 1.0, to_wp(xt, dtype), zt)

        G, y, zt = host(data.G), host(y), host(zt)
        for i in range(B):
            np.testing.assert_allclose(y[i], G[i] @ x[i], **MATVEC_TOL[dtype])
            np.testing.assert_allclose(zt[i], G[i].T @ xt[i], **MATVEC_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
class TestDenseKKTSolverAssembly:
    """The factored KKT matrix matches the per-problem reference assembly."""

    @pytest.mark.parametrize("B,n,p,m", SIZES + TALL_G_SIZES + CHOLESKY_SIZES)
    def test_kkt_matrix_assembly(self, dtype, B, n, p, m):
        data = random_dense_data(B, n, p=p, m=m, dtype=dtype)
        solver = DenseKKTSolver(data)

        rng = np.random.default_rng(99 + B)
        x_reg = rounded(np.abs(rng.standard_normal((B, n))) + 0.1, dtype)
        # z_reg_inv = condensed row weight (w_l + w_u); z_reg = 1/weight is the
        # explicit augmented diagonal magnitude. Both are passed in.
        z_reg_inv = rounded(np.abs(rng.standard_normal((B, m))) + 0.1, dtype)
        z_reg = 1.0 / z_reg_inv
        delta = rounded(np.abs(rng.standard_normal(B)) + 0.1, dtype)

        solver.update_and_factor_kkt(data, to_wp(delta, dtype), to_wp(x_reg, dtype), to_wp(z_reg, dtype),
                                     to_wp(z_reg_inv, dtype))
        assert not solver.factor_status.numpy().any()

        # With rhs_y = rhs_z = 0 the condensed solve returns K^{-1} rhs_x, so
        # K_ref @ delta_x must reproduce rhs_x.
        rhs_x = rounded(rng.standard_normal((B, n)), dtype)
        delta_x, delta_y, delta_z = empty(dtype, B, n), empty(dtype, B, p), empty(dtype, B, m)
        solver.solve(data, to_wp(rhs_x, dtype), zeros(dtype, B, p), zeros(dtype, B, m),
                     delta_x, delta_y, delta_z)

        P, A, G = host(data.P), host(data.A), host(data.G)
        delta_x = host(delta_x)
        for i in range(B):
            ref_kkt = P[i] + np.diag(x_reg[i]) + (1.0 / float(delta[i])) * (A[i].T @ A[i])
            ref_kkt += G[i].T @ (z_reg_inv[i][:, None] * G[i])
            np.testing.assert_allclose(ref_kkt @ delta_x[i], rhs_x[i], err_msg=f"batch={i}",
                                       **KKT_SOLVE_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
class TestDenseKKTSolverSolve:
    """Full DenseKKTSolver pipeline: assemble, factor, solve vs NumPy."""

    @staticmethod
    def _run_solve(B, n, p, m, dtype, seed=42):
        data = random_dense_data(B, n, p=p, m=m, seed=seed, dtype=dtype)
        solver = DenseKKTSolver(data)

        rng = np.random.default_rng(seed + 100)
        x_reg = rounded(np.abs(rng.standard_normal((B, n))) + 1.0, dtype)
        # z_reg_inv = condensed row weight; z_reg = 1/weight (augmented diagonal).
        z_reg_inv = rounded(np.abs(rng.standard_normal((B, m))) + 1.0, dtype)
        z_reg = 1.0 / z_reg_inv
        delta = rounded(np.abs(rng.standard_normal(B)) + 1.0, dtype)

        rhs_x = rounded(rng.standard_normal((B, n)), dtype)
        rhs_y = rounded(rng.standard_normal((B, p)), dtype)
        rhs_z = rounded(rng.standard_normal((B, m)), dtype)

        solver.update_and_factor_kkt(data, to_wp(delta, dtype), to_wp(x_reg, dtype), to_wp(z_reg, dtype),
                                     to_wp(z_reg_inv, dtype))
        assert not solver.factor_status.numpy().any()

        delta_x, delta_y, delta_z = empty(dtype, B, n), empty(dtype, B, p), empty(dtype, B, m)
        solver.solve(data, to_wp(rhs_x, dtype), to_wp(rhs_y, dtype), to_wp(rhs_z, dtype),
                     delta_x, delta_y, delta_z)

        P, A, G = host(data.P), host(data.A), host(data.G)
        delta_x, delta_y, delta_z = host(delta_x), host(delta_y), host(delta_z)
        for i in range(B):
            ref_dx, ref_dy, ref_dz = numpy_condensed_kkt_solve(
                P[i], A[i], G[i], x_reg[i], z_reg_inv[i], float(delta[i]), rhs_x[i], rhs_y[i], rhs_z[i])
            np.testing.assert_allclose(delta_x[i], ref_dx, err_msg=f"delta_x mismatch at batch {i}",
                                       **KKT_SOLVE_TOL[dtype])
            if p > 0:
                np.testing.assert_allclose(delta_y[i], ref_dy, err_msg=f"delta_y mismatch at batch {i}",
                                           **KKT_SOLVE_TOL[dtype])
            if m > 0:
                np.testing.assert_allclose(delta_z[i], ref_dz, err_msg=f"delta_z mismatch at batch {i}",
                                           **KKT_SOLVE_TOL[dtype])

    @pytest.mark.parametrize("B,n,p,m", SIZES + TALL_G_SIZES + CHOLESKY_SIZES)
    def test_full_pipeline(self, dtype, B, n, p, m):
        self._run_solve(B=B, n=n, p=p, m=m, dtype=dtype, seed=B * 100 + n + p + m)


def _random_rhs(B, n, p, m, seed, dtype):
    """Random host condensed right-hand sides holding ``dtype`` values."""
    rng = np.random.RandomState(seed)
    return rounded(rng.randn(B, n), dtype), rounded(rng.randn(B, p), dtype), rounded(rng.randn(B, m), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
class TestDenseKKTSystemCondensedSolve:
    """``KKTSystem`` round-trip: K * solve(rhs) ~= rhs via mul_condensed_kkt."""

    @pytest.mark.parametrize("B,n,p,m", SIZES)
    def test_condensed_factorize_solve(self, dtype, B, n, p, m):
        """The round trip holds on the active rows; inactive rows are covered
        by ``test_inactive_rows_are_decoupled``."""
        data = random_dense_data(B=B, n=n, p=p, m=m, dtype=dtype)
        kkt, _, _ = factored_kkt_system(data)
        active = host(data.active_G_row)

        rhs_x, rhs_y, rhs_z = _random_rhs(B, n, p, m, seed=123, dtype=dtype)
        rhs_z = rhs_z * active
        lhs = condensed_solve(kkt, data, rhs_x, rhs_y, rhs_z)

        check_x, check_y, check_z = zeros(dtype, B, n), zeros(dtype, B, p), zeros(dtype, B, m)
        kkt.mul_condensed_kkt(data, *lhs, check_x, check_y, check_z)

        t = KKT_RESIDUAL_TOL[dtype]
        np.testing.assert_allclose(host(check_x), rhs_x, err_msg="x block", **t)
        np.testing.assert_allclose(host(check_y), rhs_y, err_msg="y block", **t)
        np.testing.assert_allclose(host(check_z) * active, rhs_z, err_msg="z block", **t)

    def test_inactive_rows_are_decoupled(self, dtype):
        """A G row with both bounds infinite is decoupled from the condensed
        system: its right-hand side changes no other component of the solution."""
        B, n, p, m = 2, 10, 3, 5
        qp = random_dense_qp(5, n, p, m, batch_size=B)
        qp["h_l"][:, [1, 4]] = -np.inf
        qp["h_u"][:, [1, 4]] = np.inf
        data = DenseData(dtype=dtype)
        data.init(**to_wp_dict(qp, dtype))
        kkt, _, _ = factored_kkt_system(data)
        assert_inactive_rows_decoupled(kkt, data, *_random_rhs(B, n, p, m, seed=6, dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B,n,p,m", SIZES)
class TestDenseKKTSystemIR:
    """Iterative refinement at the KKTSystem level, against the residual of
    the same operator before and after refinement."""

    @staticmethod
    def _setup(B, n, p, m, dtype):
        data = random_dense_data(B=B, n=n, p=p, m=m, dtype=dtype)
        settings = Settings.for_dtype(dtype)
        settings.iterative_refinement_max_iter = 10
        kkt, _, _ = factored_kkt_system(data, settings)
        rhs_x, rhs_y, rhs_z = _random_rhs(B, n, p, m, seed=111, dtype=dtype)
        return data, kkt, (rhs_x, rhs_y, rhs_z * host(data.active_G_row))

    def test_does_not_increase_residual(self, dtype, B, n, p, m):
        """Refining a direct solve never increases its residual: refinement
        accepts a correction only if it lowers the residual."""
        data, kkt, rhs = self._setup(B, n, p, m, dtype)
        lhs = condensed_solve(kkt, data, *rhs)
        before = condensed_residual(kkt, data, lhs, *rhs)
        kkt.iterative_refinement(data, *(to_wp(r, dtype) for r in rhs), *lhs)
        after = condensed_residual(kkt, data, lhs, *rhs)
        assert after <= before, f"refinement increased the residual: {after:.2e} > {before:.2e}"

    def test_corrects_a_perturbed_solution(self, dtype, B, n, p, m):
        """Refinement started from a perturbed solution (residual of order
        one) reduces the residual to the solve accuracy of the dtype."""
        data, kkt, rhs = self._setup(B, n, p, m, dtype)
        lhs = condensed_solve(kkt, data, *rhs)
        perturb_lhs(lhs, data, seed=7)
        before = condensed_residual(kkt, data, lhs, *rhs)
        kkt.iterative_refinement(data, *(to_wp(r, dtype) for r in rhs), *lhs)
        after = condensed_residual(kkt, data, lhs, *rhs)
        assert after <= {wp.float64: 1e-8, wp.float32: 1e-3}[dtype] * before, \
            f"refinement did not correct the solution: {before:.2e} -> {after:.2e}"
