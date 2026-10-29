"""Tests for the unified KKTSystem class - both single (B=1) and batched modes.

Verifies the full pipeline: regularization -> eliminate -> condensed solve
-> recover, against a per-problem NumPy reference.

The tests cover both precisions (float64 / float32), single-problem mode
(B = 1) and batched mode (B > 1), and every constraint layout: with / without
general inequality, equality and box bounds.

Problems and right-hand sides are built with NumPy on the host and moved to
the GPU as Warp arrays, the only array type the KKT internals take.
"""
import pytest
import numpy as np
import warp as wp

from cupiqp.dense.dense_data import DenseData
from cupiqp.results import Variables
from cupiqp.settings import Settings
from tests.helpers import DTYPES, host, random_dense_qp, rounded, to_wp, to_wp_dict
from tests.kkt.kkt_helpers import (condensed_solve, factored_kkt_system,
                                   kkt_system, positive_variables, random_variables, zeros)
from tests.tolerances import KKT_RESIDUAL_TOL, KKT_SOLVE_TOL, MATVEC_TOL


# ======================================================================
# NumPy reference implementation
# ======================================================================

def _ref_compute_scalings(P, A, G, s_u, s_l, s_bu, s_bl, z_u, z_l, z_bu, z_bl,
                          x_b_scaling, idx_hu, idx_hl, idx_xu, idx_xl,
                          rho, delta):
    """NumPy reference: compute x_reg, z_reg, and intermediate scalings."""
    n, m = P.shape[0], G.shape[0]

    m_z_u_inv = 1.0 / z_u if z_u.size else z_u
    m_z_l_inv = 1.0 / z_l if z_l.size else z_l
    m_z_bu_inv = 1.0 / z_bu if z_bu.size else z_bu
    m_z_bl_inv = 1.0 / z_bl if z_bl.size else z_bl

    w_u  = 1.0 / (s_u * m_z_u_inv + delta) if s_u.size else s_u
    w_l  = 1.0 / (s_l * m_z_l_inv + delta) if s_l.size else s_l
    w_bu = 1.0 / (s_bu * m_z_bu_inv + delta) if s_bu.size else s_bu
    w_bl = 1.0 / (s_bl * m_z_bl_inv + delta) if s_bl.size else s_bl

    x_reg = np.full(n, rho)
    if idx_xu.size:
        xbs = x_b_scaling[idx_xu]
        x_reg[idx_xu] += xbs * xbs * w_bu
    if idx_xl.size:
        xbs = x_b_scaling[idx_xl]
        x_reg[idx_xl] += xbs * xbs * w_bl

    z_reg = np.zeros(m)
    if idx_hu.size:
        z_reg[idx_hu] += w_u
    if idx_hl.size:
        z_reg[idx_hl] += w_l
    if m > 0:
        z_reg = 1.0 / z_reg

    return (x_reg, z_reg,
            s_u, s_l, s_bu, s_bl,
            m_z_u_inv, m_z_l_inv, m_z_bu_inv, m_z_bl_inv,
            w_u, w_l, w_bu, w_bl)


def _ref_full_solve(P, A, G, x_b_scaling,
                    idx_hu, idx_hl, idx_xu, idx_xl,
                    scalings, delta,
                    rhs_x, rhs_y,
                    rhs_z_u, rhs_z_l, rhs_z_bu, rhs_z_bl,
                    rhs_s_u, rhs_s_l, rhs_s_bu, rhs_s_bl):
    """NumPy reference: eliminate -> condensed solve -> recover."""
    (x_reg, z_reg,
     m_s_u, m_s_l, m_s_bu, m_s_bl,
     m_z_u_inv, m_z_l_inv, m_z_bu_inv, m_z_bl_inv,
     w_u, w_l, w_bu, w_bl) = scalings

    n, p, m = P.shape[0], A.shape[0], G.shape[0]

    # Step 1: eliminate slacks
    urhs_z_u  = rhs_z_u  - m_z_u_inv  * rhs_s_u  if rhs_z_u.size  else rhs_z_u
    urhs_z_l  = rhs_z_l  - m_z_l_inv  * rhs_s_l  if rhs_z_l.size  else rhs_z_l
    urhs_z_bu = rhs_z_bu - m_z_bu_inv * rhs_s_bu if rhs_z_bu.size else rhs_z_bu
    urhs_z_bl = rhs_z_bl - m_z_bl_inv * rhs_s_bl if rhs_z_bl.size else rhs_z_bl

    # Step 2: eliminate duals -> rhs_x_bar, rhs_z_bar
    rhs_x_bar = rhs_x.copy()
    if idx_xu.size:
        rhs_x_bar[idx_xu] += x_b_scaling[idx_xu] * w_bu * urhs_z_bu
    if idx_xl.size:
        rhs_x_bar[idx_xl] -= x_b_scaling[idx_xl] * w_bl * urhs_z_bl

    rhs_z_bar = np.zeros(m)
    if idx_hu.size:
        rhs_z_bar[idx_hu] += w_u * urhs_z_u
    if idx_hl.size:
        rhs_z_bar[idx_hl] -= w_l * urhs_z_l
    rhs_z_bar *= z_reg

    # Step 3: assemble & solve condensed system
    z_reg_inv = 1.0 / z_reg if m > 0 else np.empty(0)
    kkt = P + np.diag(x_reg)
    if p > 0:
        kkt += (1.0 / delta) * A.T @ A
    if m > 0:
        G_sc = np.sqrt(z_reg_inv)[:, None] * G
        kkt += G_sc.T @ G_sc

    rhs_cond = rhs_x_bar.copy()
    if p > 0:
        rhs_cond += (1.0 / delta) * A.T @ rhs_y
    if m > 0:
        rhs_cond += G.T @ (z_reg_inv * rhs_z_bar)

    dx = np.linalg.solve(kkt, rhs_cond)
    dy = (A @ dx - rhs_y) / delta if p > 0 else np.empty(0)

    # Step 4: recover inequality duals
    G_dx = G @ dx if m > 0 else np.empty(0)
    dz_u  = w_u * (G_dx[idx_hu] - urhs_z_u) if idx_hu.size else np.empty(0)
    dz_l  = w_l * (-G_dx[idx_hl] - urhs_z_l) if idx_hl.size else np.empty(0)

    dz_bu, dz_bl = np.empty(0), np.empty(0)
    if idx_xu.size:
        dz_bu = w_bu * (x_b_scaling[idx_xu] * dx[idx_xu] - rhs_z_bu + m_z_bu_inv * rhs_s_bu)
    if idx_xl.size:
        dz_bl = -w_bl * (x_b_scaling[idx_xl] * dx[idx_xl] + rhs_z_bl - m_z_bl_inv * rhs_s_bl)

    # Step 5: recover slacks
    ds_u  = m_z_u_inv  * (rhs_s_u  - m_s_u  * dz_u)  if idx_hu.size else np.empty(0)
    ds_l  = m_z_l_inv  * (rhs_s_l  - m_s_l  * dz_l)  if idx_hl.size else np.empty(0)
    ds_bu = m_z_bu_inv * (rhs_s_bu - m_s_bu * dz_bu) if idx_xu.size else np.empty(0)
    ds_bl = m_z_bl_inv * (rhs_s_bl - m_s_bl * dz_bl) if idx_xl.size else np.empty(0)

    return dx, dy, dz_u, dz_l, dz_bu, dz_bl, ds_u, ds_l, ds_bu, ds_bl


# ======================================================================
# Helpers
# ======================================================================

def _make_data(B, n, p, m, dtype, seed=42, with_box=True):
    """A batched DenseData of the requested size.

    A and b are supplied only when ``p > 0``; G / h_u / h_l only when
    ``m > 0``; x_u / x_l only when ``with_box=True``. This matches how a
    real caller constructs the data and exercises DenseData's "this field
    is absent" code paths, instead of always passing size-0 arrays.
    """
    qp = random_dense_qp(seed, n, p, m, batch_size=B)
    if not with_box:
        del qp["x_l"], qp["x_u"]
    data = DenseData(dtype=dtype)
    data.init(**to_wp_dict({k: v for k, v in qp.items() if v.size}, dtype))
    return data


def _run_full_solve(B, n, p, m, dtype, seed=42, with_box=True):
    """Run the KKTSystem pipeline and verify against a per-problem NumPy reference."""
    data = _make_data(B, n, p, m, dtype, seed=seed, with_box=with_box)
    settings = Settings.for_dtype(dtype)
    rng = np.random.default_rng(seed + 200)
    rho = rounded(np.abs(rng.standard_normal(B)) + 0.1, dtype)
    delta = rounded(np.abs(rng.standard_normal(B)) + 0.1, dtype)
    kkt_sys, pc, vars = factored_kkt_system(data, settings, rho, delta, positive_variables(data, seed + 100))
    rhs = random_variables(data, seed + 300)

    lhs = Variables()
    lhs.init(data)
    kkt_sys.solve(data, pc, settings, rhs, lhs)

    # --- Per-problem NumPy reference ---
    # Host copies of the data, the variables and the solution.
    P, A, G, x_b_scaling = host(data.P), host(data.A), host(data.G), host(pc.x_b_scaling)
    h_vars = {k: host(getattr(vars, k)) for k in ("s_u", "s_l", "s_bu", "s_bl", "z_u", "z_l", "z_bu", "z_bl")}
    fields = ("x", "y", "z_u", "z_l", "z_bu", "z_bl", "s_u", "s_l", "s_bu", "s_bl")
    h_rhs = {k: host(getattr(rhs, k)) for k in fields}
    h_lhs = {k: host(getattr(lhs, k)) for k in fields}
    for i in range(B):
        Pi  = P[i]
        Ai  = A[i] if data.p > 0 else np.zeros((0, data.n))
        Gi  = G[i] if data.m > 0 else np.zeros((0, data.n))
        xbs = x_b_scaling[i]
        # Full-length dual layout: the index maps are the identity.
        idx_hu = np.arange(data.m)
        idx_hl = np.arange(data.m)
        idx_xu = np.arange(data.n)
        idx_xl = np.arange(data.n)

        scalings = _ref_compute_scalings(
            Pi, Ai, Gi,
            h_vars["s_u"][i], h_vars["s_l"][i],
            h_vars["s_bu"][i], h_vars["s_bl"][i],
            h_vars["z_u"][i], h_vars["z_l"][i],
            h_vars["z_bu"][i], h_vars["z_bl"][i],
            xbs, idx_hu, idx_hl, idx_xu, idx_xl,
            float(rho[i]), float(delta[i]),
        )

        ref = _ref_full_solve(
            Pi, Ai, Gi, xbs,
            idx_hu, idx_hl, idx_xu, idx_xl,
            scalings, float(delta[i]),
            h_rhs["x"][i], h_rhs["y"][i],
            h_rhs["z_u"][i], h_rhs["z_l"][i],
            h_rhs["z_bu"][i], h_rhs["z_bl"][i],
            h_rhs["s_u"][i], h_rhs["s_l"][i],
            h_rhs["s_bu"][i], h_rhs["s_bl"][i],
        )
        ref_dx, ref_dy, ref_dz_u, ref_dz_l, ref_dz_bu, ref_dz_bl, \
            ref_ds_u, ref_ds_l, ref_ds_bu, ref_ds_bl = ref

        t = KKT_SOLVE_TOL[dtype]
        np.testing.assert_allclose(h_lhs["x"][i], ref_dx, **t,
                                   err_msg=f"dx problem {i}")
        if p > 0:
            np.testing.assert_allclose(h_lhs["y"][i], ref_dy, **t,
                                       err_msg=f"dy problem {i}")
        if data.num_hu > 0:
            np.testing.assert_allclose(h_lhs["z_u"][i], ref_dz_u, **t,
                                       err_msg=f"dz_u problem {i}")
            np.testing.assert_allclose(h_lhs["s_u"][i], ref_ds_u, **t,
                                       err_msg=f"ds_u problem {i}")
        if data.num_hl > 0:
            np.testing.assert_allclose(h_lhs["z_l"][i], ref_dz_l, **t,
                                       err_msg=f"dz_l problem {i}")
            np.testing.assert_allclose(h_lhs["s_l"][i], ref_ds_l, **t,
                                       err_msg=f"ds_l problem {i}")
        if data.num_xu > 0:
            np.testing.assert_allclose(h_lhs["z_bu"][i], ref_dz_bu, **t,
                                       err_msg=f"dz_bu problem {i}")
            np.testing.assert_allclose(h_lhs["s_bu"][i], ref_ds_bu, **t,
                                       err_msg=f"ds_bu problem {i}")
        if data.num_xl > 0:
            np.testing.assert_allclose(h_lhs["z_bl"][i], ref_dz_bl, **t,
                                       err_msg=f"dz_bl problem {i}")
            np.testing.assert_allclose(h_lhs["s_bl"][i], ref_ds_bl, **t,
                                       err_msg=f"ds_bl problem {i}")


# ======================================================================
# Size matrices for the parametrized tests
# ======================================================================

# NOTE: keep the number of distinct shapes small. The KKT kernels are
# specialized and compiled per (n, p, m), constraint layout and dtype, so
# every new shape costs a fresh compile of the whole kernel set, which
# dominates the run time of this file on a cold cache. The batch size is not
# part of the specialization, so B is free to vary. Hence:
#   - all tests share the shapes (n, p, m) = (10, 3, 5) and (20, 8, 9);
#     n = 10 runs the Warp matrix-product kernels, n = 20 the cuBLAS ones
#     (see WARP_MATMUL_MAX_COLS in cupiqp/dense/dense_matmul.py);
#   - a missing block sets p and / or m of (10, 3, 5) to zero instead of
#     picking a new shape;
#   - each shape runs at B = 1 and B = 8 only, because B = 1 and B > 1 take
#     different matrix-product paths.
# (20, 8, 9), (10, 0, 5) and (10, 3, 0) are also in IR_SHAPES of
# test_dense_kkt.py, so a run of both files compiles them once.

# All four constraint types present: equality A, inequality G, box bounds.
FULL_SIZES = [
    pytest.param(1, 10, 3, 5, id="B1-n10-p3-m5"),
    pytest.param(8, 10, 3, 5, id="B8-n10-p3-m5"),
    pytest.param(1, 20, 8, 9, id="B1-n20-p8-m9"),
    pytest.param(8, 20, 8, 9, id="B8-n20-p8-m9"),
]

# No general inequality (m = 0); equality + box bounds.
NO_INEQ_SIZES = [
    pytest.param(1, 10, 3, 0, id="B1-n10-p3"),
    pytest.param(8, 10, 3, 0, id="B8-n10-p3"),
]

# No equality (p = 0); inequality + box bounds.
NO_EQ_SIZES = [
    pytest.param(1, 10, 0, 5, id="B1-n10-m5"),
    pytest.param(8, 10, 0, 5, id="B8-n10-m5"),
]

# Box bounds only (p = 0, m = 0).
BOX_ONLY_SIZES = [
    pytest.param(1, 10, 0, 0, id="B1-n10"),
    pytest.param(8, 10, 0, 0, id="B8-n10"),
]

# Equality only, no inequality of any kind (no G, no box bounds).
# This is a *different* DenseData code path from BOX_ONLY: ``num_ineq``
# is zero, so no slacks/duals get allocated.
EQ_ONLY_SIZES = [
    pytest.param(1, 10, 3, id="B1-n10-p3"),
    pytest.param(8, 10, 3, id="B8-n10-p3"),
]


# ======================================================================
# Full-solve correctness tests (parametrized over size matrix)
# ======================================================================

@pytest.mark.parametrize("dtype", DTYPES)
class TestKKTSystemFullSolve:
    """Verify the full KKTSystem pipeline against the NumPy reference, across
    a matrix of problem sizes covering both single and batched modes."""

    @pytest.mark.parametrize("B,n,p,m", FULL_SIZES)
    def test_all_constraints(self, dtype, B, n, p, m):
        _run_full_solve(B=B, n=n, p=p, m=m, dtype=dtype, seed=B * 100 + n * 10 + p + m)

    @pytest.mark.parametrize("B,n,p,m", NO_EQ_SIZES)
    def test_no_equality(self, dtype, B, n, p, m):
        _run_full_solve(B=B, n=n, p=p, m=m, dtype=dtype, seed=B * 100 + n * 10 + m + 1)

    @pytest.mark.parametrize("B,n,p,m", NO_INEQ_SIZES)
    def test_no_inequality_with_box(self, dtype, B, n, p, m):
        """``m = 0`` -- no general inequality, but ``x_u`` / ``x_l`` are present."""
        _run_full_solve(B=B, n=n, p=p, m=m, dtype=dtype, seed=B * 100 + n * 10 + p + 2)

    @pytest.mark.parametrize("B,n,p,m", BOX_ONLY_SIZES)
    def test_box_bounds_only(self, dtype, B, n, p, m):
        _run_full_solve(B=B, n=n, p=p, m=m, dtype=dtype, seed=B * 100 + n * 10 + 3)


@pytest.mark.parametrize("dtype", DTYPES)
class TestKKTSystemEqualityOnly:
    """Exercise the path where the data has no inequality of any kind
    (``num_ineq == 0``). Box bounds are deliberately suppressed; this is a
    different code path from ``test_box_bounds_only`` where ``num_ineq > 0``
    via x_u/x_l."""

    @pytest.mark.parametrize("B,n,p", EQ_ONLY_SIZES)
    def test_equality_only(self, dtype, B, n, p):
        data = _make_data(B, n, p, 0, dtype, seed=B * 100 + n * 10 + p + 4, with_box=False)
        rng = np.random.default_rng(B * 100 + n * 10 + p + 5)
        settings = Settings.for_dtype(dtype)
        # rho = delta = 1e-2 as GPU numbers, so the reference uses the same values.
        # Smaller values make K = P + rho I + A^T A / delta ill-conditioned.
        reg = float(rounded(1e-2, dtype))
        kkt_sys, pc, _ = factored_kkt_system(data, settings, reg, reg)

        rhs = Variables(); rhs.init(data)
        rhs_x = rounded(rng.standard_normal((B, n)), dtype)
        rhs_y = rounded(rng.standard_normal((B, p)), dtype)
        rhs.x = to_wp(rhs_x, dtype)
        rhs.y = to_wp(rhs_y, dtype)

        lhs = Variables(); lhs.init(data)
        kkt_sys.solve(data, pc, settings, rhs, lhs)

        P, A, lhs_x = host(data.P), host(data.A), host(lhs.x)
        for i in range(B):
            Pi = P[i]
            Ai = A[i]
            kkt = Pi + reg * np.eye(n) + (1.0 / reg) * Ai.T @ Ai
            rhs_cond = (rhs_x[i]
                        + (1.0 / reg) * Ai.T @ rhs_y[i])
            ref_dx = np.linalg.solve(kkt, rhs_cond)
            # The small regularization (1 / reg = 100 on A^T A) costs about
            # two digits of the KKT_SOLVE_TOL accuracy.
            atol, rtol = (1e-8, 1e-8) if dtype == wp.float64 else (1e-4, 1e-4)
            np.testing.assert_allclose(lhs_x[i], ref_dx, err_msg=f"problem {i}", atol=atol, rtol=rtol)


# ======================================================================
# Condensed K * lhs ~= rhs sanity check
# ======================================================================

@pytest.mark.parametrize("dtype", DTYPES)
class TestKKTSystemCondensedMatvec:
    """After ``solve``, multiplying ``lhs`` by the condensed KKT operator
    should reproduce the original ``rhs`` (within solver tolerance)."""

    @pytest.mark.parametrize("B,n,p,m", FULL_SIZES)
    def test_K_times_lhs_recovers_rhs(self, dtype, B, n, p, m):
        data = _make_data(B, n, p, m, dtype)
        kkt_sys, _, _ = factored_kkt_system(data)

        rng = np.random.default_rng(B * 100 + n * 10 + p + m)
        rhs_x = rounded(rng.standard_normal((B, n)), dtype)
        rhs_y = rounded(rng.standard_normal((B, p)), dtype)
        active = host(data.active_G_row)
        rhs_z = rounded(rng.standard_normal((B, m)), dtype) * active
        lhs = condensed_solve(kkt_sys, data, rhs_x, rhs_y, rhs_z)

        check_x, check_y, check_z = zeros(dtype, B, n), zeros(dtype, B, p), zeros(dtype, B, m)
        kkt_sys.mul_condensed_kkt(data, *lhs, check_x, check_y, check_z)

        t = KKT_RESIDUAL_TOL[dtype]
        np.testing.assert_allclose(host(check_x), rhs_x, err_msg="x block", **t)
        np.testing.assert_allclose(host(check_y), rhs_y, err_msg="y block", **t)
        np.testing.assert_allclose(host(check_z) * active, rhs_z, err_msg="z block", **t)


# ======================================================================
# Forwarded matvec ops
# ======================================================================

@pytest.mark.parametrize("dtype", DTYPES)
class TestKKTSystemMatvec:
    """Sanity check for forwarded matvec operations."""

    @pytest.mark.parametrize("B,n,p,m", FULL_SIZES)
    def test_eval_P_x(self, dtype, B, n, p, m):
        data = _make_data(B=B, n=n, p=p, m=m, dtype=dtype)
        kkt_sys = kkt_system(data, Settings.for_dtype(dtype))

        x = rounded(np.random.default_rng(0).standard_normal((B, n)), dtype)
        z = zeros(dtype, B, n)
        kkt_sys.eval_P_x(data, 1.0, to_wp(x, dtype), z)

        P, z = host(data.P), host(z)
        for i in range(B):
            expected = P[i] @ x[i]
            np.testing.assert_allclose(z[i], expected, **MATVEC_TOL[dtype])

    @pytest.mark.parametrize("B,n,p,m", FULL_SIZES)
    def test_eval_A_xn_and_AT_xt(self, dtype, B, n, p, m):
        data = _make_data(B=B, n=n, p=p, m=m, dtype=dtype)
        kkt_sys = kkt_system(data, Settings.for_dtype(dtype))

        x = rounded(np.random.default_rng(1).standard_normal((B, n)), dtype)
        y = rounded(np.random.default_rng(2).standard_normal((B, p)), dtype)
        out_Ax  = zeros(dtype, B, p)
        out_ATy = zeros(dtype, B, n)

        kkt_sys.eval_A_xn(data, 1.0, to_wp(x, dtype), out_Ax)
        kkt_sys.eval_AT_xt(data, 1.0, to_wp(y, dtype), out_ATy)

        A, out_Ax, out_ATy = host(data.A), host(out_Ax), host(out_ATy)
        t = MATVEC_TOL[dtype]
        for i in range(B):
            Ai = A[i]
            np.testing.assert_allclose(out_Ax[i],  Ai @ x[i], **t)
            np.testing.assert_allclose(out_ATy[i], Ai.T @ y[i], **t)
