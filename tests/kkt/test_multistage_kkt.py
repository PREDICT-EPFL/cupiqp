"""Tests for the multistage KKT path: low-level ``MultistageKKTSolver`` and the
higher-level ``KKTSystem`` interface on multistage data.

The block-structured problem of every test is also assembled as a dense
problem on the host (``assemble_multistage``), which gives the NumPy
references: dense matrix products for the matvecs, a dense solve of the
condensed system for ``update_and_factor_kkt`` + ``solve``, and the dense
backend's ``KKTSystem`` for the full pipeline.

Every test covers each combination of present / absent off-diagonal blocks
of P, A and G (absent blocks are compiled out of the kernels), single and
batched problems, and one stage (``N = 1``, no coupling at all) as well as
several.
"""
import itertools

import numpy as np
import pytest

from cupiqp.dense.dense_data import DenseData
from cupiqp.dense.dense_preconditioner import DenseRuizEquilibration
from cupiqp.kkt_systems import KKTSystem
from cupiqp.multistage.multistage_data import MultistageData
from cupiqp.multistage.multistage_kkt_solver import MultistageKKTSolver
from cupiqp.multistage.multistage_preconditioner import MultistageRuizEquilibration
from cupiqp.results import Variables
from cupiqp.settings import Settings
from tests.helpers import assemble_multistage, device_settings, to_wp, to_wp_dict
from tests.kkt.test_dense_kkt import _numpy_condensed_kkt_solve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nan(*shape):
    """NaN-filled float64 GPU array: an output a kernel must overwrite entirely."""
    return to_wp(np.full(shape, np.nan))


def _zeros(*shape):
    """Zero-filled float64 GPU array."""
    return to_wp(np.zeros(shape))


def random_multistage_qp(B, N, d, rA, rG, has_P_E=True, has_A_E=True, has_G_E=True, seed=0):
    """A batched strongly convex multistage QP.

    Returns ``(data, dense)``: the ``MultistageData`` and, per problem, the
    host ``(P, A, G)`` it assembles to. ``rA`` / ``rG`` are the rows per block
    of A / G (0 for no equality / no inequality). The bound structure (which
    entries are +/-inf) is shared across the batch, as the solver requires.
    """
    rng = np.random.default_rng(seed)
    P_D = rng.standard_normal((B, N, d, d))
    P_D = P_D @ P_D.transpose(0, 1, 3, 2) + 3 * d * np.eye(d)
    P_E = 0.3 * rng.standard_normal((B, N - 1, d, d)) if has_P_E else None
    A_D = rng.standard_normal((B, N, rA, d)) if rA else None
    A_E = rng.standard_normal((B, N, rA, d)) if rA and has_A_E else None
    G_D = rng.standard_normal((B, N, rG, d)) if rG else None
    G_E = rng.standard_normal((B, N, rG, d)) if rG and has_G_E else None

    dense = [assemble_multistage(N, d, *(None if M is None else M[i] for M in (P_D, P_E, A_D, A_E, G_D, G_E)))
             for i in range(B)]
    n, p, m = N * d, dense[0][1].shape[0], dense[0][2].shape[0]

    x_l_mask, x_u_mask = rng.random(n) > 0.3, rng.random(n) > 0.3
    kw = dict(
        P=(P_D, P_E), c=rng.standard_normal((B, n)),
        x_l=np.where(x_l_mask, -np.abs(rng.standard_normal((B, n))) - 1.0, -np.inf),
        x_u=np.where(x_u_mask, np.abs(rng.standard_normal((B, n))) + 1.0, np.inf),
    )
    if rA:
        kw.update(A=(A_D, A_E), b=rng.standard_normal((B, p)))
    if rG:
        h_l_mask, h_u_mask = rng.random(m) > 0.3, rng.random(m) > 0.3
        kw.update(G=(G_D, G_E),
                  h_l=np.where(h_l_mask, -np.abs(rng.standard_normal((B, m))) - 1.0, -np.inf),
                  h_u=np.where(h_u_mask, np.abs(rng.standard_normal((B, m))) + 1.0, np.inf))

    data = MultistageData()
    data.init(**{k: (to_wp(v[0]), to_wp(v[1])) if isinstance(v, tuple) else to_wp(v) for k, v in kw.items()})
    return data, dense


def _dense_twin(data, dense):
    """``DenseData`` of the same problems as the multistage ``data``."""
    kw = dict(P=np.stack([P for P, _, _ in dense]), c=data.c.numpy(),
              x_l=data.x_l.numpy(), x_u=data.x_u.numpy())
    if data.p > 0:
        kw.update(A=np.stack([A for _, A, _ in dense]), b=data.b.numpy())
    if data.m > 0:
        kw.update(G=np.stack([G for _, _, G in dense]), h_l=data.h_l.numpy(), h_u=data.h_u.numpy())
    twin = DenseData()
    twin.init(**to_wp_dict(kw))
    return twin


OFFDIAG_COMBOS = [pytest.param(*c, id="PE{}-AE{}-GE{}".format(*map(int, c)))
                  for c in itertools.product([True, False], repeat=3)]
BATCH_SIZES = [1, 3]

# (N, d, rA, rG): several stages, one stage, no equality, no inequality,
# and more constraint rows than variables per block.
SHAPES = [
    pytest.param(5, 3, 2, 4, id="N5-d3-rA2-rG4"),
    pytest.param(1, 4, 2, 3, id="N1-d4-rA2-rG3"),
    pytest.param(4, 3, 0, 2, id="N4-d3-noA"),
    pytest.param(4, 3, 2, 0, id="N4-d3-noG"),
    pytest.param(3, 2, 3, 5, id="N3-d2-rA3-rG5"),
]


# ===========================================================================
# Low-level MultistageKKTSolver primitives
# ===========================================================================
class TestMultistageKKTSolverMatvec:
    """Block matrix-vector products against the assembled dense matrices.
    Outputs start as NaN, so every entry must be written."""

    @pytest.mark.parametrize("has_P_E,has_A_E,has_G_E", OFFDIAG_COMBOS)
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_eval_P_x(self, B, has_P_E, has_A_E, has_G_E):
        data, dense = random_multistage_qp(B, 5, 3, 2, 4, has_P_E, has_A_E, has_G_E, seed=B)
        solver = MultistageKKTSolver(data)

        x = np.random.default_rng(1).standard_normal((B, data.n))
        z = _nan(B, data.n)
        solver.eval_P_x(data, 2.0, to_wp(x), z)

        z = z.numpy()
        for i in range(B):
            np.testing.assert_allclose(z[i], 2.0 * dense[i][0] @ x[i], atol=1e-12)

    @pytest.mark.parametrize("has_P_E,has_A_E,has_G_E", OFFDIAG_COMBOS)
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_eval_A_xn_and_AT_xt(self, B, has_P_E, has_A_E, has_G_E):
        data, dense = random_multistage_qp(B, 5, 3, 2, 4, has_P_E, has_A_E, has_G_E, seed=B + 10)
        solver = MultistageKKTSolver(data)
        rng = np.random.default_rng(2)

        x = rng.standard_normal((B, data.n))
        y = _nan(B, data.p)
        solver.eval_A_xn(data, 1.5, to_wp(x), y)

        xt = rng.standard_normal((B, data.p))
        zt = _nan(B, data.n)
        solver.eval_AT_xt(data, -0.5, to_wp(xt), zt)

        y, zt = y.numpy(), zt.numpy()
        for i in range(B):
            A = dense[i][1]
            np.testing.assert_allclose(y[i], 1.5 * A @ x[i], atol=1e-12)
            np.testing.assert_allclose(zt[i], -0.5 * A.T @ xt[i], atol=1e-12)

    @pytest.mark.parametrize("has_P_E,has_A_E,has_G_E", OFFDIAG_COMBOS)
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_eval_G_xn_and_GT_xt(self, B, has_P_E, has_A_E, has_G_E):
        data, dense = random_multistage_qp(B, 5, 3, 2, 4, has_P_E, has_A_E, has_G_E, seed=B + 20)
        solver = MultistageKKTSolver(data)
        rng = np.random.default_rng(3)

        x = rng.standard_normal((B, data.n))
        y = _nan(B, data.m)
        solver.eval_G_xn(data, 1.0, to_wp(x), y)

        xt = rng.standard_normal((B, data.m))
        zt = _nan(B, data.n)
        solver.eval_GT_xt(data, 1.0, to_wp(xt), zt)

        y, zt = y.numpy(), zt.numpy()
        for i in range(B):
            G = dense[i][2]
            np.testing.assert_allclose(y[i], G @ x[i], atol=1e-12)
            np.testing.assert_allclose(zt[i], G.T @ xt[i], atol=1e-12)


class TestMultistageKKTSolverSolve:
    """``update_and_factor_kkt`` + ``solve`` against a NumPy solve of the
    assembled condensed system."""

    @staticmethod
    def _run_solve(B, N, d, rA, rG, has_P_E, has_A_E, has_G_E, seed):
        data, dense = random_multistage_qp(B, N, d, rA, rG, has_P_E, has_A_E, has_G_E, seed=seed)
        n, p, m = data.n, data.p, data.m
        solver = MultistageKKTSolver(data)

        rng = np.random.default_rng(seed + 100)
        x_reg = np.abs(rng.standard_normal((B, n))) + 1.0
        # z_reg_inv = condensed row weight; z_reg = 1/weight (augmented diagonal).
        z_reg_inv = np.abs(rng.standard_normal((B, m))) + 1.0
        z_reg = 1.0 / z_reg_inv
        delta = np.abs(rng.standard_normal(B)) + 1.0

        solver.update_and_factor_kkt(data, to_wp(delta), to_wp(x_reg), to_wp(z_reg), to_wp(z_reg_inv))
        assert not solver.factor_status.numpy().any()

        rhs_x = rng.standard_normal((B, n))
        rhs_y = rng.standard_normal((B, p))
        rhs_z = rng.standard_normal((B, m))
        delta_x, delta_y, delta_z = _nan(B, n), _nan(B, p), _nan(B, m)
        solver.solve(data, to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z), delta_x, delta_y, delta_z)
        assert not solver.solve_status.numpy().any()

        delta_x, delta_y, delta_z = delta_x.numpy(), delta_y.numpy(), delta_z.numpy()
        for i in range(B):
            P, A, G = dense[i]
            ref_dx, ref_dy, ref_dz = _numpy_condensed_kkt_solve(
                P, A, G, x_reg[i], z_reg_inv[i], float(delta[i]), rhs_x[i], rhs_y[i], rhs_z[i])
            np.testing.assert_allclose(delta_x[i], ref_dx, atol=1e-9, err_msg=f"delta_x, batch {i}")
            np.testing.assert_allclose(delta_y[i], ref_dy, atol=1e-9, err_msg=f"delta_y, batch {i}")
            np.testing.assert_allclose(delta_z[i], ref_dz, atol=1e-9, err_msg=f"delta_z, batch {i}")

    @pytest.mark.parametrize("has_P_E,has_A_E,has_G_E", OFFDIAG_COMBOS)
    @pytest.mark.parametrize("N,d,rA,rG", SHAPES)
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_full_pipeline(self, B, N, d, rA, rG, has_P_E, has_A_E, has_G_E):
        self._run_solve(B, N, d, rA, rG, has_P_E, has_A_E, has_G_E,
                        seed=B * 100 + N * 10 + 4 * has_P_E + 2 * has_A_E + has_G_E)

    def test_large_batch(self):
        self._run_solve(64, 6, 4, 3, 5, True, True, True, seed=7)

    def test_refactor_with_new_regularization(self):
        """A second ``update_and_factor_kkt`` replaces the first factor."""
        B, N, d = 3, 4, 3
        data, dense = random_multistage_qp(B, N, d, 2, 3, seed=11)
        n, p, m = data.n, data.p, data.m
        solver = MultistageKKTSolver(data)
        rng = np.random.default_rng(12)
        for _ in range(2):
            x_reg = np.abs(rng.standard_normal((B, n))) + 0.5
            z_reg_inv = np.abs(rng.standard_normal((B, m))) + 0.5
            delta = np.abs(rng.standard_normal(B)) + 0.5
            solver.update_and_factor_kkt(data, to_wp(delta), to_wp(x_reg), to_wp(1.0 / z_reg_inv),
                                         to_wp(z_reg_inv))
            rhs_x = rng.standard_normal((B, n))
            delta_x, delta_y, delta_z = _nan(B, n), _nan(B, p), _nan(B, m)
            solver.solve(data, to_wp(rhs_x), _zeros(B, p), _zeros(B, m), delta_x, delta_y, delta_z)
            delta_x = delta_x.numpy()
            for i in range(B):
                P, A, G = dense[i]
                ref_dx, _, _ = _numpy_condensed_kkt_solve(
                    P, A, G, x_reg[i], z_reg_inv[i], float(delta[i]), rhs_x[i], np.zeros(p), np.zeros(m))
                np.testing.assert_allclose(delta_x[i], ref_dx, atol=1e-9, err_msg=f"batch {i}")

    def test_factor_status_flags_only_the_failed_problem(self):
        """A NaN in one problem's data is reported for that problem alone."""
        B, N, d = 4, 3, 2
        data, _ = random_multistage_qp(B, N, d, 1, 2, seed=5)
        solver = MultistageKKTSolver(data)
        x_reg = np.ones((B, data.n))
        x_reg[2, 3] = np.nan
        z_reg_inv = np.ones((B, data.m))
        solver.update_and_factor_kkt(data, to_wp(np.ones(B)), to_wp(x_reg), to_wp(z_reg_inv),
                                     to_wp(z_reg_inv))
        np.testing.assert_array_equal(solver.factor_status.numpy() != 0, [False, False, True, False])


# ===========================================================================
# KKTSystem on multistage data
# ===========================================================================
def _positive_vars(data, seed):
    """Variables with s, z > 0 (required for the IPM scalings)."""
    rng = np.random.default_rng(seed)
    B = data.batch_size
    v = Variables()
    v.init(data)
    v.x = to_wp(rng.standard_normal((B, data.n)))
    if data.p > 0:
        v.y = to_wp(rng.standard_normal((B, data.p)))
    if data.num_ineq > 0:
        v.s_all = to_wp(np.abs(rng.standard_normal((B, data.num_ineq))) + 0.5)
        v.z_all = to_wp(np.abs(rng.standard_normal((B, data.num_ineq))) + 0.5)
    return v


def _random_rhs(data, seed):
    rng = np.random.default_rng(seed)
    B = data.batch_size
    rhs = Variables()
    rhs.init(data)
    rhs.x = to_wp(rng.standard_normal((B, data.n)))
    if data.p > 0:
        rhs.y = to_wp(rng.standard_normal((B, data.p)))
    if data.num_ineq > 0:
        rhs.s_all = to_wp(rng.standard_normal((B, data.num_ineq)))
        rhs.z_all = to_wp(rng.standard_normal((B, data.num_ineq)))
    return rhs


def _kkt_system_solve(data, preconditioner, vars_, rho, delta, rhs):
    """``update_scalings_and_factor`` + ``solve`` of a fresh KKTSystem."""
    settings = Settings()
    kkt = KKTSystem()
    kkt.init(data, settings, device_settings(settings, data))
    kkt.update_scalings_and_factor(data, preconditioner, settings, False, to_wp(rho), to_wp(delta), vars_)
    assert not kkt.factor_status.numpy().any()
    lhs = Variables()
    lhs.init(data)
    kkt.solve(data, preconditioner, settings, rhs, lhs)
    return lhs


class TestMultistageKKTSystem:
    """The full KKTSystem pipeline (regularization, slack elimination,
    condensed solve, recovery) on multistage data matches the dense backend
    on the assembled problem, given the same variables and right-hand side."""

    @pytest.mark.parametrize("has_P_E,has_A_E,has_G_E", OFFDIAG_COMBOS)
    @pytest.mark.parametrize("N,d,rA,rG", SHAPES)
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_matches_dense_backend(self, B, N, d, rA, rG, has_P_E, has_A_E, has_G_E):
        seed = B * 100 + N * 10 + 4 * has_P_E + 2 * has_A_E + has_G_E
        data, dense = random_multistage_qp(B, N, d, rA, rG, has_P_E, has_A_E, has_G_E, seed=seed)
        twin = _dense_twin(data, dense)
        rng = np.random.default_rng(seed + 1)
        rho = np.abs(rng.standard_normal(B)) + 0.1
        delta = np.abs(rng.standard_normal(B)) + 0.1

        flags = dict(has_h_l=data.has_h_l, has_h_u=data.has_h_u, has_x_l=data.has_x_l, has_x_u=data.has_x_u)
        ms_pc = MultistageRuizEquilibration(B, data.n, data.p, data.m, **flags, data=data)
        dense_pc = DenseRuizEquilibration(B, data.n, data.p, data.m, **flags)

        ms_lhs = _kkt_system_solve(data, ms_pc, _positive_vars(data, seed + 2), rho, delta,
                                   _random_rhs(data, seed + 3))
        dense_lhs = _kkt_system_solve(twin, dense_pc, _positive_vars(twin, seed + 2), rho, delta,
                                      _random_rhs(twin, seed + 3))

        for field in ("x", "y", "z_all", "s_all"):
            np.testing.assert_allclose(getattr(ms_lhs, field).numpy(), getattr(dense_lhs, field).numpy(),
                                       atol=1e-9, err_msg=field)
