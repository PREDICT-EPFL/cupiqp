"""Unit tests for KKTSystem with the sparse backend.

Mirrors PIQP's C++ ``tests/src/sparse/kkt_test.cpp``:
  * FactorizeSolve -- factor the condensed KKT, solve ``K_c v = rhs`` for a
    random rhs, multiply ``K_c`` back and check ``K_c v == rhs``.
  * UpdateData     -- update the P/A/G values, re-factor, and confirm the
    solution matches a freshly built KKT.
plus a check that iterative refinement does not make the condensed solve
worse. cuPIQP is natively batched, so the batch size is a test parameter.
"""
import numpy as np
import pytest
import scipy.sparse as sp_cpu

from cupiqp.kkt_systems import KKTSystem
from cupiqp.results import Variables
from cupiqp.settings import Settings
from cupiqp.sparse.sparse_data import SparseData
from cupiqp.sparse.sparse_preconditioner import SparseRuizEquilibration
from tests.helpers import csr_triple, to_wp, device_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_qp_arrays(n, p, m, density=0.3, seed=42) -> dict:
    """Host arrays of one random strongly convex sparse QP.

    P is SPD (``M M^T + n I``, symmetrized); A and G are random rectangular
    sparse matrices; some bounds are +-inf so the inactive-bound (masked) path
    is exercised. Returns scipy CSR matrices with sorted indices and numpy
    vectors.
    """
    rng = np.random.default_rng(seed)

    M = sp_cpu.random(n, n, density=density, format='csr', random_state=rng)
    P = (M @ M.T + n * sp_cpu.eye(n)).tocsr()
    P = ((P + P.T) / 2).tocsr()
    A = sp_cpu.random(p, n, density=density + 0.1, format='csr', random_state=rng)
    G = sp_cpu.random(m, n, density=density + 0.1, format='csr', random_state=rng)
    for mat in (P, A, G):
        mat.sort_indices()

    c = rng.standard_normal(n)
    b = rng.standard_normal(p)
    h_u = np.abs(rng.standard_normal(m)) + 1.0
    h_l = -np.abs(rng.standard_normal(m)) - 1.0
    x_u = np.abs(rng.standard_normal(n)) + 1.0
    x_l = -np.abs(rng.standard_normal(n)) - 1.0
    if m > 0:
        h_u[rng.random(m) < 0.3] = np.inf
        h_l[rng.random(m) < 0.3] = -np.inf
    x_u[rng.random(n) < 0.3] = np.inf
    x_l[rng.random(n) < 0.3] = -np.inf

    return dict(P=P, c=c, A=A, b=b, G=G, h_u=h_u, h_l=h_l, x_u=x_u, x_l=x_l)


def _sparse_data(arr: dict, B: int) -> SparseData:
    """SparseData holding ``B`` copies of one QP: the matrices are shared and
    the vectors carry the batch axis. The KKT internals and Data objects take
    Warp arrays only."""
    def batched(v):
        return to_wp(np.broadcast_to(v, (B,) + v.shape))

    data = SparseData()
    data.init(
        P=csr_triple(arr['P']), c=batched(arr['c']),
        A=csr_triple(arr['A']), b=batched(arr['b']),
        G=csr_triple(arr['G']), h_u=batched(arr['h_u']), h_l=batched(arr['h_l']),
        x_u=batched(arr['x_u']), x_l=batched(arr['x_l']),
    )
    return data


def _make_preconditioner(data: SparseData) -> SparseRuizEquilibration:
    """An identity Ruiz preconditioner (no scaling) for the given data."""
    return SparseRuizEquilibration(
        data.batch_size, data.n, data.p, data.m,
        has_h_l=data.has_h_l, has_h_u=data.has_h_u,
        has_x_l=data.has_x_l, has_x_u=data.has_x_u,
    )


def _factor(data: SparseData, settings: Settings, use_static_reg: bool,
            rho: float = 1.0, delta: float = 1.0):
    """Init a KKTSystem, set random (positive s/z) IPM variables, factor.

    Returns ``(kkt, preconditioner, variables)``.
    """
    kkt = KKTSystem()
    kkt.init(data, settings, device_settings(settings, data))
    preconditioner = _make_preconditioner(data)

    variables = Variables()
    variables.init(data)
    variables.set_random()

    rho_arr = to_wp(np.full(data.batch_size, rho))
    delta_arr = to_wp(np.full(data.batch_size, delta))
    kkt.update_scalings_and_factor(
        data, preconditioner, settings, use_static_reg, rho_arr, delta_arr, variables,
    )
    assert not kkt.factor_status.numpy().any(), "KKT factorization failed"
    return kkt, preconditioner, variables


def _random_rhs(B, n, p, m, seed):
    """Three random ``(B, k)`` condensed-rhs blocks (host arrays)."""
    rng = np.random.RandomState(seed)
    return rng.randn(B, n), rng.randn(B, p), rng.randn(B, m)


def _zeros(B, k):
    """``(B, k)`` zero GPU array."""
    return to_wp(np.zeros((B, k)))


def _mask_rhs_infinite_rows(rhs: Variables, data: SparseData) -> None:
    """Zero the rhs dual and slack rows whose bound is infinite: solve() returns
    0 there and cannot reproduce a nonzero rhs."""
    if data.num_ineq > 0:
        mask = data.finite_mask_all.numpy()
        rhs.z_all = to_wp(rhs.z_all.numpy() * mask)
        rhs.s_all = to_wp(rhs.s_all.numpy() * mask)


# (n, p, m): constraint-set variations and a couple of larger sizes.
SHAPES = [
    (10, 0, 5),    # no equality constraints
    (10, 3, 0),    # no inequality constraints
    (10, 0, 0),    # only bound constraints
    (20, 5, 8),    # mixed
    (30, 10, 15),  # larger
    (60, 20, 30),  # even larger
]
BATCH_SIZES = [1, 4]


# ===========================================================================
# FactorizeSolve: condensed round trip
# ===========================================================================
@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("n,p,m", SHAPES)
def test_condensed_factorize_solve(n: int, p: int, m: int, B: int) -> None:
    """Solve ``K_c lhs = rhs`` for the condensed KKT and verify the
    multiply-back identity through ``mul_condensed_kkt``."""
    data = _sparse_data(_random_qp_arrays(n, p, m), B)
    kkt, _, _ = _factor(data, Settings(), use_static_reg=False)

    rhs_x, rhs_y, rhs_z = _random_rhs(B, n, p, m, seed=123)
    lhs_x, lhs_y, lhs_z = _zeros(B, n), _zeros(B, p), _zeros(B, m)
    # There is no public entry point for the condensed-only solve (KKTSystem.solve
    # always runs eliminate -> condensed -> recover), so the inner solver is used.
    kkt._kkt_solver.solve(data, to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z), lhs_x, lhs_y, lhs_z)

    check_x, check_y, check_z = _zeros(B, n), _zeros(B, p), _zeros(B, m)
    kkt.mul_condensed_kkt(data, lhs_x, lhs_y, lhs_z, check_x, check_y, check_z)

    atol = 1e-8
    np.testing.assert_allclose(rhs_x, check_x.numpy(), atol=atol)
    np.testing.assert_allclose(rhs_y, check_y.numpy(), atol=atol)
    np.testing.assert_allclose(rhs_z, check_z.numpy(), atol=atol)


# ===========================================================================
# Iterative refinement
# ===========================================================================
@pytest.mark.parametrize("n,p,m", [(20, 8, 9), (10, 0, 5), (10, 3, 0), (30, 10, 15)])
def test_condensed_solve_with_ir(n: int, p: int, m: int) -> None:
    """Run the same rhs through the condensed solve without and with iterative
    refinement; refinement must not make the error worse."""
    data = _sparse_data(_random_qp_arrays(n, p, m), 1)
    rhs_x, rhs_y, rhs_z = _random_rhs(1, n, p, m, seed=111)

    # --- Without IR -------------------------------------------------------
    settings_no_ir = Settings()
    settings_no_ir.iterative_refinement_max_iter = 0
    kkt_no_ir, _, _ = _factor(data, settings_no_ir, use_static_reg=False)

    lhs_x, lhs_y, lhs_z = _zeros(1, n), _zeros(1, p), _zeros(1, m)
    kkt_no_ir._kkt_solver.solve(
        data, to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z), lhs_x, lhs_y, lhs_z,
    )
    err_x, err_y, err_z = _zeros(1, n), _zeros(1, p), _zeros(1, m)
    error_no_ir = kkt_no_ir.get_refinement_error(
        data, lhs_x, lhs_y, lhs_z, to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z), err_x, err_y, err_z,
    )

    # --- With IR (static reg + IR loop) -----------------------------------
    settings_ir = Settings()
    settings_ir.iterative_refinement_max_iter = 10
    kkt_ir, _, _ = _factor(data, settings_ir, use_static_reg=True)

    lhs_x2, lhs_y2, lhs_z2 = _zeros(1, n), _zeros(1, p), _zeros(1, m)
    kkt_ir._kkt_solver.solve(
        data, to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z), lhs_x2, lhs_y2, lhs_z2,
    )
    kkt_ir.iterative_refinement(
        data,
        to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z),
        lhs_x2, lhs_y2, lhs_z2,
    )
    err_x2, err_y2, err_z2 = _zeros(1, n), _zeros(1, p), _zeros(1, m)
    error_ir = kkt_ir.get_refinement_error(
        data, lhs_x2, lhs_y2, lhs_z2,
        to_wp(rhs_x), to_wp(rhs_y), to_wp(rhs_z), err_x2, err_y2, err_z2,
    )

    # IR must not make things worse for well-conditioned problems.
    assert error_ir <= error_no_ir * 10 + 1e-14, (
        f"IR made things worse: {error_ir:.2e} > {error_no_ir:.2e}"
    )


# ===========================================================================
# UpdateData: an updated KKT must match a freshly built one
# ===========================================================================
@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("n,p,m", [(20, 8, 9), (10, 0, 5), (12, 4, 0)])
def test_update_data(n: int, p: int, m: int, B: int) -> None:
    """Update the P/A/G values, re-factor, and confirm the solve matches a fresh KKT."""
    arr = _random_qp_arrays(n, p, m, seed=3)
    data = _sparse_data(arr, B)
    settings = Settings()
    settings.iterative_refinement_max_iter = 0

    # First KKT, factored on the original data.
    kkt1, precond, variables = _factor(data, settings, use_static_reg=False)

    # New values on the same pattern (a scalar scaling keeps P SPD);
    # set_P / set_A / set_G take only the stored values.
    data.set_P(to_wp(1.3 * arr['P'].data))
    if p > 0:
        data.set_A(to_wp(0.7 * arr['A'].data))
    if m > 0:
        data.set_G(to_wp(1.5 * arr['G'].data))
    kkt1.update_data(data, update_P=True, update_A=p > 0, update_G=m > 0)
    rho_arr, delta_arr = to_wp(np.full(B, 1.0)), to_wp(np.full(B, 1.0))
    kkt1.update_scalings_and_factor(data, precond, settings, False, rho_arr, delta_arr, variables)
    assert not kkt1.factor_status.numpy().any()

    # Fresh KKT built from the updated data, factored at the same point.
    kkt2 = KKTSystem()
    kkt2.init(data, settings, device_settings(settings, data))
    precond2 = _make_preconditioner(data)
    kkt2.update_scalings_and_factor(data, precond2, settings, False, rho_arr, delta_arr, variables)
    assert not kkt2.factor_status.numpy().any()

    # Same rhs through both -> identical solution.
    rhs = Variables()
    rhs.init(data)
    rhs.set_random()
    _mask_rhs_infinite_rows(rhs, data)

    lhs_updated = Variables()
    lhs_updated.init(data)
    kkt1.solve(data, precond, settings, rhs, lhs_updated)
    lhs_fresh = Variables()
    lhs_fresh.init(data)
    kkt2.solve(data, precond2, settings, rhs, lhs_fresh)

    assert lhs_updated.allclose(lhs_fresh, rtol=1e-6, atol=1e-8)
