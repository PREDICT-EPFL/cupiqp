"""Tests for the multistage KKT path: low-level ``MultistageKKTSolver`` and the
higher-level ``KKTSystem`` interface on multistage data.

The block-structured problem of every test is also assembled as a dense
problem on the host (``assemble_multistage``), which gives the NumPy
references: dense matrix products for the matvecs, a dense solve of the
condensed system for ``update_and_factor_kkt`` + ``solve``, and the dense
backend's ``KKTSystem`` for the full pipeline.

Every test covers each combination of present / absent off-diagonal blocks
of P, A and G (absent blocks are compiled out of the kernels), single and
batched problems, one stage (``N = 1``, no coupling at all) as well as
several, and both precisions (float64 / float32).
"""
import itertools

import numpy as np
import pytest
import warp as wp

from cupiqp.dense.dense_data import DenseData
from cupiqp.multistage.multistage_data import MultistageData
from cupiqp.multistage.multistage_kkt_solver import MultistageKKTSolver
from cupiqp.results import Variables
from cupiqp.settings import Settings
from tests.helpers import (DTYPES, assemble_multistage, host, random_multistage_P, rounded,
                           to_wp, to_wp_dict)
from tests.kkt.kkt_helpers import (factored_kkt_system, nan, numpy_condensed_kkt_solve,
                                   positive_variables, random_variables, zeros)
from tests.tolerances import KKT_SOLVE_TOL, MATVEC_TOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_multistage_data(B, N, d, rA, rG, has_P_E=True, has_A_E=True, has_G_E=True, seed=0,
                         dtype=wp.float64):
    """A batched strongly convex multistage QP.

    Returns ``(data, dense)``: the ``MultistageData`` and, per problem, the
    host ``(P, A, G)`` it assembles to. ``rA`` / ``rG`` are the rows per block
    of A / G (0 for no equality / no inequality). The bound structure (which
    entries are +/-inf) is shared across the batch, as the solver requires.
    The blocks are rounded to ``dtype`` before the assembly, so ``dense`` is
    exactly the problem the GPU holds.
    """
    rng = np.random.default_rng(seed)
    P_D, P_E = random_multistage_P(rng, N, d, B, has_P_E)
    A_D = rng.standard_normal((B, N, rA, d)) if rA else None
    A_E = rng.standard_normal((B, N, rA, d)) if rA and has_A_E else None
    G_D = rng.standard_normal((B, N, rG, d)) if rG else None
    G_E = rng.standard_normal((B, N, rG, d)) if rG and has_G_E else None
    P_D, P_E, A_D, A_E, G_D, G_E = (None if M is None else rounded(M, dtype)
                                    for M in (P_D, P_E, A_D, A_E, G_D, G_E))

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

    data = MultistageData(dtype=dtype)
    data.init(**{k: (to_wp(v[0], dtype), to_wp(v[1], dtype)) if isinstance(v, tuple) else to_wp(v, dtype)
                 for k, v in kw.items()})
    return data, dense


def _dense_twin(data, dense):
    """``DenseData`` of the same problems as the multistage ``data``."""
    kw = dict(P=np.stack([P for P, _, _ in dense]), c=data.c.numpy(),
              x_l=data.x_l.numpy(), x_u=data.x_u.numpy())
    if data.p > 0:
        kw.update(A=np.stack([A for _, A, _ in dense]), b=data.b.numpy())
    if data.m > 0:
        kw.update(G=np.stack([G for _, _, G in dense]), h_l=data.h_l.numpy(), h_u=data.h_u.numpy())
    twin = DenseData(dtype=data.dtype)
    twin.init(**to_wp_dict(kw, data.dtype))
    return twin


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


def _pipeline_cases():
    """(B, N, d, rA, rG, has_P_E, has_A_E, has_G_E) cases of the solve
    tests: every off-diagonal structure on the first shape, and every shape
    and batch size with all and with no off-diagonal blocks."""
    cases = {(3, 5, 3, 2, 4, *flags) for flags in itertools.product([True, False], repeat=3)}
    cases |= {(B, *shape.values, *flags) for B in BATCH_SIZES for shape in SHAPES
              for flags in ((True,) * 3, (False,) * 3)}
    return [pytest.param(*c, id="B{}-N{}-d{}-rA{}-rG{}-PE{:d}-AE{:d}-GE{:d}".format(*c)) for c in sorted(cases)]


PIPELINE_CASES = _pipeline_cases()


# ===========================================================================
# Low-level MultistageKKTSolver primitives
# ===========================================================================
@pytest.mark.parametrize("dtype", DTYPES)
class TestMultistageKKTSolverMatvec:
    """Block matrix-vector products against the assembled dense matrices.
    Outputs start as NaN, so every entry must be written."""

    @pytest.mark.parametrize("has_P_E", [True, False], ids=["P_E1", "P_E0"])
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_eval_P_x(self, dtype, B, has_P_E):
        data, dense = random_multistage_data(B, 5, 3, 2, 4, has_P_E, True, True, seed=B, dtype=dtype)
        solver = MultistageKKTSolver(data)

        x = rounded(np.random.default_rng(1).standard_normal((B, data.n)), dtype)
        z = nan(dtype, B, data.n)
        solver.eval_P_x(data, 2.0, to_wp(x, dtype), z)

        z = host(z)
        for i in range(B):
            np.testing.assert_allclose(z[i], 2.0 * dense[i][0] @ x[i], **MATVEC_TOL[dtype])

    @pytest.mark.parametrize("has_A_E", [True, False], ids=["A_E1", "A_E0"])
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_eval_A_xn_and_AT_xt(self, dtype, B, has_A_E):
        data, dense = random_multistage_data(B, 5, 3, 2, 4, True, has_A_E, True, seed=B + 10, dtype=dtype)
        solver = MultistageKKTSolver(data)
        rng = np.random.default_rng(2)

        x = rounded(rng.standard_normal((B, data.n)), dtype)
        y = nan(dtype, B, data.p)
        solver.eval_A_xn(data, 1.5, to_wp(x, dtype), y)

        xt = rounded(rng.standard_normal((B, data.p)), dtype)
        zt = nan(dtype, B, data.n)
        solver.eval_AT_xt(data, -0.5, to_wp(xt, dtype), zt)

        y, zt = host(y), host(zt)
        t = MATVEC_TOL[dtype]
        for i in range(B):
            A = dense[i][1]
            np.testing.assert_allclose(y[i], 1.5 * A @ x[i], **t)
            np.testing.assert_allclose(zt[i], -0.5 * A.T @ xt[i], **t)

    @pytest.mark.parametrize("has_G_E", [True, False], ids=["G_E1", "G_E0"])
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_eval_G_xn_and_GT_xt(self, dtype, B, has_G_E):
        data, dense = random_multistage_data(B, 5, 3, 2, 4, True, True, has_G_E, seed=B + 20, dtype=dtype)
        solver = MultistageKKTSolver(data)
        rng = np.random.default_rng(3)

        x = rounded(rng.standard_normal((B, data.n)), dtype)
        y = nan(dtype, B, data.m)
        solver.eval_G_xn(data, 1.0, to_wp(x, dtype), y)

        xt = rounded(rng.standard_normal((B, data.m)), dtype)
        zt = nan(dtype, B, data.n)
        solver.eval_GT_xt(data, 1.0, to_wp(xt, dtype), zt)

        y, zt = host(y), host(zt)
        t = MATVEC_TOL[dtype]
        for i in range(B):
            G = dense[i][2]
            np.testing.assert_allclose(y[i], G @ x[i], **t)
            np.testing.assert_allclose(zt[i], G.T @ xt[i], **t)


@pytest.mark.parametrize("dtype", DTYPES)
class TestMultistageKKTSolverSolve:
    """``update_and_factor_kkt`` + ``solve`` against a NumPy solve of the
    assembled condensed system."""

    @staticmethod
    def _run_solve(B, N, d, rA, rG, has_P_E, has_A_E, has_G_E, seed, dtype):
        data, dense = random_multistage_data(B, N, d, rA, rG, has_P_E, has_A_E, has_G_E, seed=seed, dtype=dtype)
        n, p, m = data.n, data.p, data.m
        solver = MultistageKKTSolver(data)

        rng = np.random.default_rng(seed + 100)
        x_reg = rounded(np.abs(rng.standard_normal((B, n))) + 1.0, dtype)
        # z_reg_inv = condensed row weight; z_reg = 1/weight (augmented diagonal).
        z_reg_inv = rounded(np.abs(rng.standard_normal((B, m))) + 1.0, dtype)
        z_reg = 1.0 / z_reg_inv
        delta = rounded(np.abs(rng.standard_normal(B)) + 1.0, dtype)

        solver.update_and_factor_kkt(data, to_wp(delta, dtype), to_wp(x_reg, dtype), to_wp(z_reg, dtype),
                                     to_wp(z_reg_inv, dtype))
        assert not solver.factor_status.numpy().any()

        rhs_x = rounded(rng.standard_normal((B, n)), dtype)
        rhs_y = rounded(rng.standard_normal((B, p)), dtype)
        rhs_z = rounded(rng.standard_normal((B, m)), dtype)
        delta_x, delta_y, delta_z = nan(dtype, B, n), nan(dtype, B, p), nan(dtype, B, m)
        solver.solve(data, to_wp(rhs_x, dtype), to_wp(rhs_y, dtype), to_wp(rhs_z, dtype),
                     delta_x, delta_y, delta_z)
        assert not solver.solve_status.numpy().any()

        delta_x, delta_y, delta_z = host(delta_x), host(delta_y), host(delta_z)
        t = KKT_SOLVE_TOL[dtype]
        for i in range(B):
            P, A, G = dense[i]
            ref_dx, ref_dy, ref_dz = numpy_condensed_kkt_solve(
                P, A, G, x_reg[i], z_reg_inv[i], float(delta[i]), rhs_x[i], rhs_y[i], rhs_z[i])
            np.testing.assert_allclose(delta_x[i], ref_dx, err_msg=f"delta_x, batch {i}", **t)
            np.testing.assert_allclose(delta_y[i], ref_dy, err_msg=f"delta_y, batch {i}", **t)
            np.testing.assert_allclose(delta_z[i], ref_dz, err_msg=f"delta_z, batch {i}", **t)

    @pytest.mark.parametrize("B,N,d,rA,rG,has_P_E,has_A_E,has_G_E", PIPELINE_CASES)
    def test_full_pipeline(self, dtype, B, N, d, rA, rG, has_P_E, has_A_E, has_G_E):
        self._run_solve(B, N, d, rA, rG, has_P_E, has_A_E, has_G_E,
                        seed=B * 100 + N * 10 + 4 * has_P_E + 2 * has_A_E + has_G_E, dtype=dtype)

    def test_large_batch(self, dtype):
        self._run_solve(64, 6, 4, 3, 5, True, True, True, seed=7, dtype=dtype)

    def test_refactor_with_new_regularization(self, dtype):
        """A second ``update_and_factor_kkt`` replaces the first factor."""
        B, N, d = 3, 4, 3
        data, dense = random_multistage_data(B, N, d, 2, 3, seed=11, dtype=dtype)
        n, p, m = data.n, data.p, data.m
        solver = MultistageKKTSolver(data)
        rng = np.random.default_rng(12)
        for _ in range(2):
            x_reg = rounded(np.abs(rng.standard_normal((B, n))) + 0.5, dtype)
            z_reg_inv = rounded(np.abs(rng.standard_normal((B, m))) + 0.5, dtype)
            delta = rounded(np.abs(rng.standard_normal(B)) + 0.5, dtype)
            solver.update_and_factor_kkt(data, to_wp(delta, dtype), to_wp(x_reg, dtype),
                                         to_wp(1.0 / z_reg_inv, dtype), to_wp(z_reg_inv, dtype))
            rhs_x = rounded(rng.standard_normal((B, n)), dtype)
            delta_x, delta_y, delta_z = nan(dtype, B, n), nan(dtype, B, p), nan(dtype, B, m)
            solver.solve(data, to_wp(rhs_x, dtype), zeros(dtype, B, p), zeros(dtype, B, m),
                         delta_x, delta_y, delta_z)
            delta_x = host(delta_x)
            for i in range(B):
                P, A, G = dense[i]
                ref_dx, _, _ = numpy_condensed_kkt_solve(
                    P, A, G, x_reg[i], z_reg_inv[i], float(delta[i]), rhs_x[i], np.zeros(p), np.zeros(m))
                np.testing.assert_allclose(delta_x[i], ref_dx, err_msg=f"batch {i}",
                                           **KKT_SOLVE_TOL[dtype])

    def test_factor_status_flags_only_the_failed_problem(self, dtype):
        """A NaN in one problem's data is reported for that problem alone."""
        B, N, d = 4, 3, 2
        data, _ = random_multistage_data(B, N, d, 1, 2, seed=5, dtype=dtype)
        solver = MultistageKKTSolver(data)
        x_reg = np.ones((B, data.n))
        x_reg[2, 3] = np.nan
        z_reg_inv = to_wp(np.ones((B, data.m)), dtype)
        solver.update_and_factor_kkt(data, to_wp(np.ones(B), dtype), to_wp(x_reg, dtype), z_reg_inv, z_reg_inv)
        np.testing.assert_array_equal(solver.factor_status.numpy() != 0, [False, False, True, False])


# ===========================================================================
# KKTSystem on multistage data
# ===========================================================================
def _kkt_system_solve(data, rho, delta, seed):
    """``update_scalings_and_factor`` + ``solve`` of a fresh KKTSystem, at
    and with random variables drawn from ``seed``."""
    settings = Settings.for_dtype(data.dtype)
    kkt, preconditioner, _ = factored_kkt_system(data, settings, rho, delta, positive_variables(data, seed))
    lhs = Variables()
    lhs.init(data)
    kkt.solve(data, preconditioner, settings, random_variables(data, seed + 1), lhs)
    return lhs


@pytest.mark.parametrize("dtype", DTYPES)
class TestMultistageKKTSystem:
    """The full KKTSystem pipeline (regularization, slack elimination,
    condensed solve, recovery) on multistage data matches the dense backend
    on the assembled problem, given the same variables and right-hand side."""

    @pytest.mark.parametrize("B,N,d,rA,rG,has_P_E,has_A_E,has_G_E", PIPELINE_CASES)
    def test_matches_dense_backend(self, dtype, B, N, d, rA, rG, has_P_E, has_A_E, has_G_E):
        seed = B * 100 + N * 10 + 4 * has_P_E + 2 * has_A_E + has_G_E
        data, dense = random_multistage_data(B, N, d, rA, rG, has_P_E, has_A_E, has_G_E, seed=seed, dtype=dtype)
        twin = _dense_twin(data, dense)
        rng = np.random.default_rng(seed + 1)
        rho = np.abs(rng.standard_normal(B)) + 0.1
        delta = np.abs(rng.standard_normal(B)) + 0.1

        ms_lhs = _kkt_system_solve(data, rho, delta, seed + 2)
        dense_lhs = _kkt_system_solve(twin, rho, delta, seed + 2)

        t = KKT_SOLVE_TOL[dtype]
        for field in ("x", "y", "z_all", "s_all"):
            np.testing.assert_allclose(host(getattr(ms_lhs, field)), host(getattr(dense_lhs, field)),
                                       err_msg=field, **t)
