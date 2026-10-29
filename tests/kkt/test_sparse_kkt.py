"""Unit tests for KKTSystem with the sparse backend.

Mirrors PIQP's C++ ``tests/src/sparse/kkt_test.cpp``:
  * FactorizeSolve -- factor the condensed KKT, solve ``K_c v = rhs`` for a
    random rhs, multiply ``K_c`` back and check ``K_c v == rhs``.
  * UpdateData     -- update the P/A/G values, re-factor, and confirm the
    solution matches a freshly built KKT.
plus checks that inactive inequality rows are decoupled and that iterative
refinement corrects an inexact solution without ever making one worse. cuPIQP is natively batched, so the batch size is a test parameter.
"""
import numpy as np
import pytest
import warp as wp

from cupiqp.results import Variables
from cupiqp.settings import Settings
from cupiqp.sparse.sparse_data import SparseData
from tests.helpers import DTYPES, csr_triple, host, random_sparse_qp, rounded, to_wp
from tests.kkt.kkt_helpers import (assert_inactive_rows_decoupled, condensed_residual,
                                   condensed_solve, factored_kkt_system, perturb_lhs, random_variables, zeros)
from tests.tolerances import KKT_RESIDUAL_TOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sparse_data(arr: dict, B: int, dtype) -> SparseData:
    """SparseData holding ``B`` copies of one QP: the matrices are shared and
    the vectors carry the batch axis. The KKT internals and Data objects take
    Warp arrays only."""
    def batched(v):
        return to_wp(np.broadcast_to(v, (B,) + v.shape), dtype)

    def csr(M):
        return csr_triple(M, wp.dtype_to_numpy(dtype))

    data = SparseData(dtype=dtype)
    data.init(
        P=csr(arr['P']), c=batched(arr['c']),
        A=csr(arr['A']), b=batched(arr['b']),
        G=csr(arr['G']), h_u=batched(arr['h_u']), h_l=batched(arr['h_l']),
        x_u=batched(arr['x_u']), x_l=batched(arr['x_l']),
    )
    return data


def _random_rhs(B, n, p, m, seed, dtype):
    """Three random ``(B, k)`` condensed-rhs blocks (host arrays holding
    ``dtype`` values)."""
    rng = np.random.RandomState(seed)
    return rounded(rng.randn(B, n), dtype), rounded(rng.randn(B, p), dtype), rounded(rng.randn(B, m), dtype)


def _mask_rhs_infinite_rows(rhs: Variables, data: SparseData) -> None:
    """Zero the rhs dual and slack rows whose bound is infinite: solve() returns
    0 there and cannot reproduce a nonzero rhs."""
    if data.num_ineq > 0:
        mask = data.finite_mask_all.numpy()
        rhs.z_all = to_wp(rhs.z_all.numpy() * mask, data.dtype)
        rhs.s_all = to_wp(rhs.s_all.numpy() * mask, data.dtype)


# NOTE: keep the number of distinct shapes small. The KKT kernels are
# specialized and compiled per (n, p, m), constraint layout and dtype, so
# every new shape costs a fresh compile of the kernel set, which dominates
# the run time of this file on a cold cache. The batch size is not part of
# the specialization, so B is free to vary. Hence all tests share the
# shapes below: every constraint layout on (n, p, m) = (10, 3, 5), with p
# and / or m set to zero for a missing block, plus (20, 8, 9). The sparse
# backend has no size-dependent code path, so no larger shape is needed.
# The same shapes are used in test_kkt_system.py and test_dense_kkt.py, so a
# run of tests/kkt compiles them once.
SHAPES = [
    pytest.param(10, 3, 5, id="n10-p3-m5"),
    pytest.param(10, 0, 5, id="n10-p0-m5"),  # no equality
    pytest.param(10, 3, 0, id="n10-p3-m0"),  # no inequality
    pytest.param(10, 0, 0, id="n10-p0-m0"),  # box bounds only
    pytest.param(20, 8, 9, id="n20-p8-m9"),
]
BATCH_SIZES = [1, 8]


# ===========================================================================
# FactorizeSolve: condensed round trip
# ===========================================================================
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("n,p,m", SHAPES)
def test_condensed_factorize_solve(n: int, p: int, m: int, B: int, dtype) -> None:
    """Solve ``K_c lhs = rhs`` for the condensed KKT and verify the
    multiply-back identity through ``mul_condensed_kkt`` on the active rows
    (inactive rows are covered by :func:`test_inactive_rows_are_decoupled`)."""
    data = _sparse_data(random_sparse_qp(42, n, p, m, density=0.3, inf_fraction=0.3), B, dtype)
    kkt, _, _ = factored_kkt_system(data)
    active = host(data.active_G_row)

    rhs_x, rhs_y, rhs_z = _random_rhs(B, n, p, m, seed=123, dtype=dtype)
    rhs_z = rhs_z * active
    lhs = condensed_solve(kkt, data, rhs_x, rhs_y, rhs_z)

    check_x, check_y, check_z = zeros(dtype, B, n), zeros(dtype, B, p), zeros(dtype, B, m)
    kkt.mul_condensed_kkt(data, *lhs, check_x, check_y, check_z)

    t = KKT_RESIDUAL_TOL[dtype]
    np.testing.assert_allclose(host(check_x), rhs_x, **t)
    np.testing.assert_allclose(host(check_y), rhs_y, **t)
    np.testing.assert_allclose(host(check_z) * active, rhs_z, **t)


@pytest.mark.parametrize("dtype", DTYPES)
def test_inactive_rows_are_decoupled(dtype) -> None:
    """A G row with both bounds infinite is decoupled from the condensed
    system: its right-hand side changes no other component of the solution."""
    n, p, m = 10, 3, 5
    qp = random_sparse_qp(5, n, p, m, density=0.3)
    qp["h_l"][[1, 4]] = -np.inf
    qp["h_u"][[1, 4]] = np.inf
    data = _sparse_data(qp, 2, dtype)
    kkt, _, _ = factored_kkt_system(data)
    assert_inactive_rows_decoupled(kkt, data, *_random_rhs(2, n, p, m, seed=6, dtype=dtype))


# ===========================================================================
# Iterative refinement
# ===========================================================================
def _ir_kkt(n, p, m, dtype):
    """A factored sparse KKTSystem with refinement enabled, and a rhs."""
    data = _sparse_data(random_sparse_qp(42, n, p, m, density=0.3, inf_fraction=0.3), 1, dtype)
    settings = Settings.for_dtype(dtype)
    settings.iterative_refinement_max_iter = 10
    kkt, _, _ = factored_kkt_system(data, settings)
    rhs_x, rhs_y, rhs_z = _random_rhs(1, n, p, m, seed=111, dtype=dtype)
    return data, kkt, (rhs_x, rhs_y, rhs_z * host(data.active_G_row))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n,p,m", SHAPES)
def test_ir_does_not_increase_residual(n: int, p: int, m: int, dtype) -> None:
    """Refining a direct solve never increases its residual: refinement
    accepts a correction only if it lowers the residual of the same
    operator."""
    data, kkt, rhs = _ir_kkt(n, p, m, dtype)
    lhs = condensed_solve(kkt, data, *rhs)
    before = condensed_residual(kkt, data, lhs, *rhs)
    kkt.iterative_refinement(data, *(to_wp(r, dtype) for r in rhs), *lhs)
    after = condensed_residual(kkt, data, lhs, *rhs)
    assert after <= before, f"refinement increased the residual: {after:.2e} > {before:.2e}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n,p,m", SHAPES)
def test_ir_corrects_a_perturbed_solution(n: int, p: int, m: int, dtype) -> None:
    """Refinement started from a perturbed solution (residual of order one)
    reduces the residual to the solve accuracy of the dtype."""
    data, kkt, rhs = _ir_kkt(n, p, m, dtype)
    lhs = condensed_solve(kkt, data, *rhs)
    perturb_lhs(lhs, data, seed=7)
    before = condensed_residual(kkt, data, lhs, *rhs)
    kkt.iterative_refinement(data, *(to_wp(r, dtype) for r in rhs), *lhs)
    after = condensed_residual(kkt, data, lhs, *rhs)
    assert after <= {wp.float64: 1e-8, wp.float32: 1e-3}[dtype] * before, (
        f"refinement did not correct the solution: {before:.2e} -> {after:.2e}")


# ===========================================================================
# UpdateData: an updated KKT must match a freshly built one
# ===========================================================================
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("n,p,m", SHAPES)
def test_update_data(n: int, p: int, m: int, B: int, dtype) -> None:
    """Update the P/A/G values, re-factor, and confirm the solve matches a fresh KKT."""
    arr = random_sparse_qp(3, n, p, m, density=0.3, inf_fraction=0.3)
    data = _sparse_data(arr, B, dtype)
    settings = Settings.for_dtype(dtype)
    settings.iterative_refinement_max_iter = 0

    # First KKT, factored on the original data.
    kkt1, precond, variables = factored_kkt_system(data, settings)

    # New values on the same pattern (a scalar scaling keeps P SPD);
    # set_P / set_A / set_G take only the stored values.
    data.set_P(to_wp(1.3 * arr['P'].data, dtype))
    if p > 0:
        data.set_A(to_wp(0.7 * arr['A'].data, dtype))
    if m > 0:
        data.set_G(to_wp(1.5 * arr['G'].data, dtype))
    kkt1.update_data(data, update_P=True, update_A=p > 0, update_G=m > 0)
    rho_arr, delta_arr = to_wp(np.full(B, 1.0), dtype), to_wp(np.full(B, 1.0), dtype)
    kkt1.update_scalings_and_factor(data, precond, settings, False, rho_arr, delta_arr, variables)
    assert not kkt1.factor_status.numpy().any()

    # Fresh KKT built from the updated data, factored at the same point.
    kkt2, precond2, _ = factored_kkt_system(data, settings, variables=variables)

    # Same rhs through both -> identical solution.
    rhs = random_variables(data, seed=4)
    _mask_rhs_infinite_rows(rhs, data)

    lhs_updated = Variables()
    lhs_updated.init(data)
    kkt1.solve(data, precond, settings, rhs, lhs_updated)
    lhs_fresh = Variables()
    lhs_fresh.init(data)
    kkt2.solve(data, precond2, settings, rhs, lhs_fresh)

    # Two factorizations of the same matrix, assembled in a different order.
    atol, rtol = (1e-8, 1e-6) if dtype == wp.float64 else (1e-4, 1e-4)
    assert lhs_updated.allclose(lhs_fresh, atol=atol, rtol=rtol)
