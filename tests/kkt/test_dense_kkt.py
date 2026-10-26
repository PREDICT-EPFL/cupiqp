"""Tests for the dense KKT path: low-level ``DenseKKTSolver`` and the
higher-level ``KKTSystem`` interface.

Every test is parameterized over batch size so the single-problem path
(``B=1``) and the genuinely batched path (``B>1``) share coverage -
``DenseData`` is internally batched and ``B=1`` is just the smallest case.
Problems and right-hand sides are built with NumPy on the host and moved to
the GPU as Warp arrays, the only array type the KKT internals take.
"""
import pytest
import numpy as np
import warp as wp

from cupiqp.dense.dense_data import DenseData
from cupiqp.dense.dense_kkt_solver import DenseKKTSolver
from cupiqp.dense.dense_preconditioner import DenseRuizEquilibration
from cupiqp.kkt_systems import KKTSystem
from cupiqp.results import Variables
from cupiqp.settings import Settings
from tests.helpers import DEVICE, to_wp, to_wp_dict, device_settings


def _zeros(*shape):
    """Zero-filled float64 GPU array."""
    return to_wp(np.zeros(shape))


def _empty(*shape):
    """Uninitialized float64 GPU array."""
    return wp.empty(shape, dtype=wp.float64, device=DEVICE)


def make_preconditioner(data):
    return DenseRuizEquilibration(
        data.batch_size, data.n, data.p, data.m,
        has_h_l=data.has_h_l, has_h_u=data.has_h_u,
        has_x_l=data.has_x_l, has_x_u=data.has_x_u,
    )


def random_dense_qp(B=1, n=20, p=8, m=9, seed=42) -> DenseData:
    """Generate a batched strongly-convex random dense QP.

    Bound sparsity structure (which entries are +/-inf) is shared across
    the batch so every sample has identical ``idx_xl/xu`` and ``idx_hl/hu``
    - this matches how ``DenseData`` and the solver expect batched inputs.
    """
    rng = np.random.default_rng(seed)

    # Shared bound structure
    h_l_mask = rng.random(m) > 0.3 if m > 0 else np.zeros(0, dtype=bool)
    h_u_mask = rng.random(m) > 0.3 if m > 0 else np.zeros(0, dtype=bool)
    x_l_mask = rng.random(n) > 0.3
    x_u_mask = rng.random(n) > 0.3

    Ps, cs, As, bs, Gs = [], [], [], [], []
    h_us, h_ls, x_us, x_ls = [], [], [], []

    for _ in range(B):
        M = rng.standard_normal((n, n))
        P = M @ M.T + n * np.eye(n)
        P = (P + P.T) / 2
        Ps.append(P)
        cs.append(rng.standard_normal(n))

        As.append(rng.standard_normal((p, n)) if p > 0 else np.zeros((0, n)))
        bs.append(rng.standard_normal(p))

        Gs.append(rng.standard_normal((m, n)) if m > 0 else np.zeros((0, n)))
        if m > 0:
            h_us.append(np.where(h_u_mask, np.abs(rng.standard_normal(m)) + 1.0, np.inf))
            h_ls.append(np.where(h_l_mask, -np.abs(rng.standard_normal(m)) - 1.0, -np.inf))
        else:
            h_us.append(np.zeros(0))
            h_ls.append(np.zeros(0))
        x_us.append(np.where(x_u_mask, np.abs(rng.standard_normal(n)) + 1.0, np.inf))
        x_ls.append(np.where(x_l_mask, -np.abs(rng.standard_normal(n)) - 1.0, -np.inf))

    kw = dict(
        P=np.stack(Ps),
        c=np.stack(cs),
        x_u=np.stack(x_us),
        x_l=np.stack(x_ls),
    )
    if p > 0:
        kw["A"] = np.stack(As)
        kw["b"] = np.stack(bs)
    if m > 0:
        kw["G"] = np.stack(Gs)
        kw["h_u"] = np.stack(h_us)
        kw["h_l"] = np.stack(h_ls)

    data = DenseData()
    data.init(**to_wp_dict(kw))
    return data


def _numpy_condensed_kkt_solve(P, A, G, x_reg, z_reg_inv, delta, rhs_x, rhs_y, rhs_z):
    """Reference: assemble and solve the condensed KKT system with NumPy.

    ``z_reg_inv`` is the condensed inequality row weight (w_l + w_u): the
    condensed Schur term is ``G^T diag(z_reg_inv) G`` and the eliminated dual
    recovers as ``dz = z_reg_inv * (G dx - rhs_z)``.
    """
    p = A.shape[0]
    m = G.shape[0]

    delta_inv = 1.0 / delta

    kkt = P.copy() + np.diag(x_reg)
    if p > 0:
        kkt += delta_inv * (A.T @ A)
    if m > 0:
        z_sqrt = np.sqrt(z_reg_inv)
        G_scaled = z_sqrt[:, None] * G
        kkt += G_scaled.T @ G_scaled

    rhs = rhs_x.copy()
    if p > 0:
        rhs += delta_inv * (A.T @ rhs_y)
    if m > 0:
        rhs += G.T @ (z_reg_inv * rhs_z)

    dx = np.linalg.solve(kkt, rhs)
    dy = (A @ dx - rhs_y) / delta if p > 0 else np.empty(0)
    dz = z_reg_inv * (G @ dx - rhs_z) if m > 0 else np.empty(0)
    return dx, dy, dz


# Batch sizes exercised everywhere. B=1 is the original "single problem"
# path; the others cover the genuine batched code paths.
BATCH_SIZES = [1, 3, 8]

# (B, n, p, m) grid for the matvec primitives. Mixes B, dim n, and the
# constraint counts p/m so the eval_* tests exercise rectangular A and G
# of varying aspect ratios, not just a fixed reference shape.
MATVEC_SIZES = [
    pytest.param( 1,   4,  1,  2, id="B1-n4-p1-m2"),
    pytest.param( 1,  16,  0,  5, id="B1-n16-p0-m5"),    # no equality
    pytest.param( 1,  16,  6,  0, id="B1-n16-p6-m0"),    # no inequality
    pytest.param( 3,   5,  2,  3, id="B3-n5-p2-m3"),
    pytest.param( 4,  10,  4,  6, id="B4-n10-p4-m6"),
    pytest.param( 4,  20,  8, 15, id="B4-n20-p8-m15"),   # p, m close to n
    pytest.param( 8,  32, 12, 16, id="B8-n32-p12-m16"),
]


# ===========================================================================
# Low-level DenseKKTSolver primitives
# ===========================================================================
class TestDenseKKTSolverMatvec:
    """Batched matrix-vector primitives on the solver."""

    @pytest.mark.parametrize("B,n,p,m", MATVEC_SIZES)
    def test_eval_P_x(self, B, n, p, m):
        data = random_dense_qp(B, n, p=p, m=m)
        solver = DenseKKTSolver(data)

        x = np.random.default_rng(B * 100 + n).standard_normal((B, n))
        z = _zeros(B, n)
        solver.eval_P_x(data, 2.0, to_wp(x), z)

        P, z = data.P.numpy(), z.numpy()
        for i in range(B):
            expected = 2.0 * P[i] @ x[i]
            np.testing.assert_allclose(z[i], expected, atol=1e-12)

    @pytest.mark.parametrize("B,n,p,m", MATVEC_SIZES)
    def test_eval_A_xn_and_AT_xt(self, B, n, p, m):
        if p == 0:
            pytest.skip("no equality constraints - A has zero rows")
        data = random_dense_qp(B, n, p=p, m=m)
        solver = DenseKKTSolver(data)

        x = np.random.default_rng(B * 100 + n + 1).standard_normal((B, n))
        y = _zeros(B, p)
        solver.eval_A_xn(data, 1.0, to_wp(x), y)

        xt = np.random.default_rng(B * 100 + p + 2).standard_normal((B, p))
        zt = _zeros(B, n)
        solver.eval_AT_xt(data, 1.0, to_wp(xt), zt)

        A, y, zt = data.A.numpy(), y.numpy(), zt.numpy()
        for i in range(B):
            np.testing.assert_allclose(
                y[i],
                A[i] @ x[i], atol=1e-12)
            np.testing.assert_allclose(
                zt[i],
                A[i].T @ xt[i], atol=1e-12)

    @pytest.mark.parametrize("B,n,p,m", MATVEC_SIZES)
    def test_eval_G_xn_and_GT_xt(self, B, n, p, m):
        if m == 0:
            pytest.skip("no inequality constraints - G has zero rows")
        data = random_dense_qp(B, n, p=p, m=m)
        solver = DenseKKTSolver(data)

        x = np.random.default_rng(B * 100 + n + 3).standard_normal((B, n))
        y = _zeros(B, m)
        solver.eval_G_xn(data, 1.0, to_wp(x), y)

        xt = np.random.default_rng(B * 100 + m + 4).standard_normal((B, m))
        zt = _zeros(B, n)
        solver.eval_GT_xt(data, 1.0, to_wp(xt), zt)

        G, y, zt = data.G.numpy(), y.numpy(), zt.numpy()
        for i in range(B):
            np.testing.assert_allclose(
                y[i],
                G[i] @ x[i], atol=1e-12)
            np.testing.assert_allclose(
                zt[i],
                G[i].T @ xt[i], atol=1e-12)


class TestDenseKKTSolverAssembly:
    """KKT matrix assembly matches per-problem reference."""

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_kkt_matrix_assembly(self, B):
        n, p, m = 5, 2, 4
        data = random_dense_qp(B, n, p=p, m=m)
        solver = DenseKKTSolver(data)

        rng = np.random.default_rng(99 + B)
        x_reg = np.abs(rng.standard_normal((B, n))) + 0.1
        # z_reg_inv = condensed row weight (w_l + w_u); z_reg = 1/weight is the
        # explicit augmented diagonal magnitude. Both are passed to update_kkt.
        z_reg_inv = np.abs(rng.standard_normal((B, m))) + 0.1
        z_reg = 1.0 / z_reg_inv
        delta = np.abs(rng.standard_normal(B)) + 0.1

        solver.update_kkt(data, to_wp(delta), to_wp(x_reg), to_wp(z_reg), to_wp(z_reg_inv))

        P, A, G = data.P.numpy(), data.A.numpy(), data.G.numpy()
        kkt_mat = solver._kkt_mat.numpy()
        for i in range(B):
            Pi = P[i]
            Ai = A[i]
            Gi = G[i]
            xr = x_reg[i]
            zr = z_reg_inv[i]  # weight used in the reference assembly
            d = float(delta[i])

            ref_kkt = Pi + np.diag(xr) + (1.0 / d) * (Ai.T @ Ai)
            G_sc = np.sqrt(zr)[:, None] * Gi
            ref_kkt += G_sc.T @ G_sc

            actual_kkt = kkt_mat[i]

            # The Cholesky solvers read the row-major upper triangle
            # (cuSOLVER FILL_MODE_LOWER in its column-major view); the syrk
            # paths populate only that triangle. Compare the triangle the
            # solver actually uses.
            for r in range(n):
                for c in range(r, n):
                    np.testing.assert_allclose(
                        actual_kkt[r, c], ref_kkt[r, c], atol=1e-10,
                        err_msg=f"batch={i}, ({r},{c})")


class TestDenseKKTSolverSolve:
    """Full DenseKKTSolver pipeline: assemble, factor, solve vs NumPy."""

    @staticmethod
    def _run_solve(B, n, p, m, seed=42):
        data = random_dense_qp(B, n, p=p, m=m, seed=seed)
        solver = DenseKKTSolver(data)

        rng = np.random.default_rng(seed + 100)
        x_reg = np.abs(rng.standard_normal((B, n))) + 1.0
        # z_reg_inv = condensed row weight; z_reg = 1/weight (augmented diagonal).
        z_reg_inv = (np.abs(rng.standard_normal((B, m))) + 1.0
                     if m > 0 else np.empty((B, 0)))
        z_reg = (1.0 / z_reg_inv if m > 0 else np.empty((B, 0)))
        delta = np.abs(rng.standard_normal(B)) + 1.0

        rhs_x = rng.standard_normal((B, n))
        rhs_y = rng.standard_normal((B, p)) if p > 0 else np.empty((B, 0))
        rhs_z = rng.standard_normal((B, m)) if m > 0 else np.empty((B, 0))

        solver.update_kkt(data, to_wp(delta), to_wp(x_reg), to_wp(z_reg), to_wp(z_reg_inv))
        solver.factor()
        assert not solver.factor_status.numpy().any()

        delta_x = _empty(B, n)
        delta_y = _empty(B, p) if p > 0 else _empty(B, 0)
        delta_z = _empty(B, m) if m > 0 else _empty(B, 0)

        solver.solve(data, to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z), delta_x, delta_y, delta_z)

        P, A, G = data.P.numpy(), data.A.numpy(), data.G.numpy()
        delta_x, delta_y, delta_z = delta_x.numpy(), delta_y.numpy(), delta_z.numpy()
        for i in range(B):
            ref_dx, ref_dy, ref_dz = _numpy_condensed_kkt_solve(
                P[i],
                A[i] if p > 0 else np.zeros((0, n)),
                G[i] if m > 0 else np.zeros((0, n)),
                x_reg[i],
                z_reg_inv[i] if m > 0 else np.empty(0),
                float(delta[i]),
                rhs_x[i],
                rhs_y[i] if p > 0 else np.empty(0),
                rhs_z[i] if m > 0 else np.empty(0),
            )
            np.testing.assert_allclose(delta_x[i], ref_dx, atol=1e-9,
                                       err_msg=f"delta_x mismatch at batch {i}")
            if p > 0:
                np.testing.assert_allclose(delta_y[i], ref_dy, atol=1e-9,
                                           err_msg=f"delta_y mismatch at batch {i}")
            if m > 0:
                np.testing.assert_allclose(delta_z[i], ref_dz, atol=1e-9,
                                           err_msg=f"delta_z mismatch at batch {i}")

    # Fused matrix: BATCH_SIZES x constraint shapes (full / no-eq / no-ineq /
    # only-bounds), plus a large-batch case and a sweep over n at fixed B=4.
    # Replaces the former separate test_full_pipeline / test_large_batch /
    # test_various_sizes - all called the same _run_solve helper.
    @pytest.mark.parametrize("B,n,p,m", [
        # BATCH_SIZES x constraint shape
        *[(B, n, p, m) for B in BATCH_SIZES for (n, p, m) in (
            (6, 3, 4), (5, 0, 3), (5, 2, 0), (4, 0, 0),
        )],
        # Large batch
        (64, 8, 3, 5),
        # n sweep at fixed B=4 with p=max(1, n//3), m=max(1, n//2)
        *[(4, n, max(1, n // 3), max(1, n // 2)) for n in (2, 4, 8, 12, 16)],
    ])
    def test_full_pipeline(self, B, n, p, m):
        self._run_solve(B=B, n=n, p=p, m=m, seed=B * 100 + n + p + m)

class TestDenseKKTSystemCondensedSolve:
    """``KKTSystem`` round-trip: K * solve(rhs) ~= rhs via mul_condensed_kkt."""

    @pytest.mark.parametrize("B", BATCH_SIZES)
    @pytest.mark.parametrize("n,p,m", [
        (10, 0, 5),
        (10, 3, 0),
        (10, 0, 0),
        (30, 10, 15),
        (60, 20, 30),
    ])
    def test_condensed_factorize_solve(self, B, n, p, m):
        data = random_dense_qp(B=B, n=n, p=p, m=m)

        settings = Settings()
        kkt = KKTSystem()
        kkt.init(data, settings, device_settings(settings, data))
        preconditioner = make_preconditioner(data)

        vars_ = Variables()
        vars_.init(data)
        vars_.set_random()

        rho_arr = to_wp(np.full(B, 1.0))
        delta_arr = to_wp(np.full(B, 1.0))

        kkt.update_scalings_and_factor(
            data, preconditioner, settings, False, rho_arr, delta_arr, vars_)

        assert not kkt.factor_status.numpy().any(), "KKT factorization failed"

        rng = np.random.RandomState(123)
        rhs_x = rng.randn(B, n)
        rhs_y = rng.randn(B, p)
        rhs_z = rng.randn(B, m)
        if m > 0:
            rhs_z *= data.active_G_row.numpy()

        lhs_x = _zeros(B, n)
        lhs_y = _zeros(B, p)
        lhs_z = _zeros(B, m)
        kkt._kkt_solver.solve(data, to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z), lhs_x, lhs_y, lhs_z)

        check_x = _zeros(B, n)
        check_y = _zeros(B, p)
        check_z = _zeros(B, m)
        kkt.mul_condensed_kkt(data, lhs_x, lhs_y, lhs_z,
                              check_x, check_y, check_z)
        check_x, check_y, check_z = check_x.numpy(), check_y.numpy(), check_z.numpy()

        atol = 1e-8
        assert np.allclose(rhs_x, check_x, atol=atol), \
            f"x block mismatch: max err = {float(np.max(np.abs(rhs_x - check_x))):.2e}"
        assert np.allclose(rhs_y, check_y, atol=atol), \
            f"y block mismatch: max err = {float(np.max(np.abs(rhs_y - check_y))):.2e}"
        if m > 0:
            check_z *= data.active_G_row.numpy()
        assert np.allclose(rhs_z, check_z, atol=atol), \
            f"z block mismatch: max err = {float(np.max(np.abs(rhs_z - check_z))):.2e}"


class TestDenseKKTSystemIR:
    """Iterative refinement at the KKTSystem level."""

    @pytest.mark.parametrize("B", BATCH_SIZES)
    @pytest.mark.parametrize("n,p,m", [
        (20, 8, 9),
        (10, 0, 5),
        (10, 3, 0),
        (30, 10, 15),
    ])
    def test_condensed_solve_with_ir(self, B, n, p, m):
        """IR should never make the residual meaningfully worse on a
        well-conditioned problem."""
        data = random_dense_qp(B=B, n=n, p=p, m=m)

        rho_arr = to_wp(np.full(B, 1.0))
        delta_arr = to_wp(np.full(B, 1.0))

        # --- Without IR ---
        settings_no_ir = Settings()
        settings_no_ir.iterative_refinement_max_iter = 0
        kkt_no_ir = KKTSystem()
        kkt_no_ir.init(data, settings_no_ir, device_settings(settings_no_ir, data))
        preconditioner = make_preconditioner(data)

        vars_no_ir = Variables()
        vars_no_ir.init(data)
        vars_no_ir.set_random()

        kkt_no_ir.update_scalings_and_factor(
            data, preconditioner, settings_no_ir, False, rho_arr, delta_arr, vars_no_ir)

        assert not kkt_no_ir.factor_status.numpy().any()

        rng = np.random.RandomState(111)
        rhs_x = rng.randn(B, n)
        rhs_y = rng.randn(B, p)
        rhs_z = rng.randn(B, m)
        if m > 0:
            rhs_z *= data.active_G_row.numpy()

        lhs_x = _zeros(B, n)
        lhs_y = _zeros(B, p)
        lhs_z = _zeros(B, m)
        kkt_no_ir._kkt_solver.solve(
            data, to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z), lhs_x, lhs_y, lhs_z)

        err_x = _zeros(B, n)
        err_y = _zeros(B, p)
        err_z = _zeros(B, m)
        error_no_ir = kkt_no_ir.get_refinement_error(
            data, lhs_x, lhs_y, lhs_z,
            to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z),
            err_x, err_y, err_z)

        # --- With IR (static reg + iterative refinement) ---
        settings_ir = Settings()
        settings_ir.iterative_refinement_max_iter = 10
        kkt_ir = KKTSystem()
        kkt_ir.init(data, settings_ir, device_settings(settings_ir, data))
        preconditioner = make_preconditioner(data)

        vars_ir = Variables()
        vars_ir.init(data)
        vars_ir.set_random()

        kkt_ir.update_scalings_and_factor(
            data, preconditioner, settings_ir, True, rho_arr, delta_arr, vars_ir)

        assert not kkt_ir.factor_status.numpy().any()

        lhs_x2 = _zeros(B, n)
        lhs_y2 = _zeros(B, p)
        lhs_z2 = _zeros(B, m)
        kkt_ir._kkt_solver.solve(
            data, to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z), lhs_x2, lhs_y2, lhs_z2)

        kkt_ir.iterative_refinement(
            data,
            to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z),
            lhs_x2, lhs_y2, lhs_z2)

        err_x2 = _zeros(B, n)
        err_y2 = _zeros(B, p)
        err_z2 = _zeros(B, m)
        error_ir = kkt_ir.get_refinement_error(
            data, lhs_x2, lhs_y2, lhs_z2,
            to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z),
            err_x2, err_y2, err_z2)

        assert error_ir <= error_no_ir * 10 + 1e-14, \
            f"IR made things worse: {error_ir:.2e} > {error_no_ir:.2e}"
