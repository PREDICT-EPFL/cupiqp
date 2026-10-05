"""Tests for the unified KKTSystem class - both single (B=1) and batched modes.

Verifies the full pipeline: regularization -> eliminate -> condensed solve
-> recover, against a per-problem NumPy reference.

The full-solve tests are parametrized over a matrix of ``(B, n, p, m)``
sizes that covers:

  * single-problem mode (B = 1) and batched mode (B > 1),
  * with / without general inequality (``m > 0`` / ``m == 0``),
  * with / without equality (``p > 0`` / ``p == 0``),
  * with / without box bounds.
"""
import pytest
import numpy as np
import cupy as cp
import warp as wp

from cupiqp.dense.dense_data import DenseData
from cupiqp.dense.dense_preconditioner import DenseRuizEquilibration
from cupiqp.kkt_systems import KKTSystem
from cupiqp.results import Variables
from cupiqp.settings import Settings



def _dense_data(**kw):
    d = DenseData()
    d.init(**{k: _w(v) for k, v in kw.items()})
    return d

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

def _make_data(B, n, p, m, seed=42, with_box=True):
    """Build a batched DenseData of the requested size.

    A and b are supplied only when ``p > 0``; G / h_u / h_l only when
    ``m > 0``; x_u / x_l only when ``with_box=True``. This matches how a
    real caller constructs the data and exercises DenseData's "this field
    is absent" code paths, instead of always passing size-0 arrays.
    """
    rng = np.random.default_rng(seed)
    Ps = []
    for _ in range(B):
        M = rng.standard_normal((n, n))
        Pi = M @ M.T + n * np.eye(n)
        Ps.append((Pi + Pi.T) / 2)
    kw = dict(
        P=cp.array(np.stack(Ps)),
        c=cp.array(rng.standard_normal((B, n))),
    )
    if p > 0:
        kw['A'] = cp.array(rng.standard_normal((B, p, n)))
        kw['b'] = cp.array(rng.standard_normal((B, p)))
    if m > 0:
        kw['G']   = cp.array(rng.standard_normal((B, m, n)))
        kw['h_u'] = cp.ones((B, m)) * 5.0
        kw['h_l'] = -cp.ones((B, m)) * 5.0
    if with_box:
        kw['x_u'] = cp.ones((B, n)) * 5.0
        kw['x_l'] = -cp.ones((B, n)) * 5.0
    return _dense_data(**kw)


def _w(a):
    """Zero-copy Warp view: the KKT solver takes Warp arrays only."""
    return wp.array(a, copy=False)


def _make_positive_vars(data, seed=99):
    """Create Variables with s, z > 0 (required for IPM scalings)."""
    rng = np.random.default_rng(seed)
    B = data.batch_size
    v = Variables()
    v.init(data)
    v.x = cp.array(rng.standard_normal((B, data.n)))
    if data.p > 0:
        v.y = cp.array(rng.standard_normal((B, data.p)))
    if data.num_ineq > 0:
        v.s_all = cp.array(np.abs(rng.standard_normal((B, data.num_ineq))) + 0.5)
        v.z_all = cp.array(np.abs(rng.standard_normal((B, data.num_ineq))) + 0.5)
    return v


def _run_full_solve(B, n, p, m, seed=42, with_box=True):
    """Run the KKTSystem pipeline and verify against a per-problem NumPy reference."""
    data = _make_data(B, n, p, m, seed=seed, with_box=with_box)
    settings = Settings()
    kkt_sys = KKTSystem()
    kkt_sys.init(data, settings)
    pc = DenseRuizEquilibration(B, data.n, data.p, data.m, has_h_l=data.has_h_l, has_h_u=data.has_h_u, has_x_l=data.has_x_l, has_x_u=data.has_x_u)

    rng = np.random.default_rng(seed + 200)
    vars = _make_positive_vars(data, seed=seed + 100)
    rho = cp.array(np.abs(rng.standard_normal(B)) + 0.1)
    delta = cp.array(np.abs(rng.standard_normal(B)) + 0.1)

    kkt_sys.update_scalings_and_factor(data, pc, settings, False, rho, delta, vars)

    assert not kkt_sys.factor_status.numpy().any(), "Cholesky factorization failed"

    rhs = Variables()
    rhs.init(data)
    rhs.x = cp.array(rng.standard_normal((B, n)))
    if p > 0:
        rhs.y = cp.array(rng.standard_normal((B, p)))
    if data.num_ineq > 0:
        rhs.s_all = cp.array(rng.standard_normal((B, data.num_ineq)))
        rhs.z_all = cp.array(rng.standard_normal((B, data.num_ineq)))

    lhs = Variables()
    lhs.init(data)
    kkt_sys.solve(data, pc, settings, rhs, lhs)

    # --- Per-problem NumPy reference ---
    for i in range(B):
        Pi  = cp.asnumpy(cp.asarray(data.P)[i])
        Ai  = cp.asnumpy(cp.asarray(data.A)[i]) if data.p > 0 else np.zeros((0, data.n))
        Gi  = cp.asnumpy(cp.asarray(data.G)[i]) if data.m > 0 else np.zeros((0, data.n))
        xbs = cp.asnumpy(pc.x_b_scaling[i])
        # Full-length dual layout: the index maps are the identity.
        idx_hu = np.arange(data.m)
        idx_hl = np.arange(data.m)
        idx_xu = np.arange(data.n)
        idx_xl = np.arange(data.n)

        scalings = _ref_compute_scalings(
            Pi, Ai, Gi,
            cp.asnumpy(cp.asarray(vars.s_u)[i]), cp.asnumpy(cp.asarray(vars.s_l)[i]),
            cp.asnumpy(cp.asarray(vars.s_bu)[i]), cp.asnumpy(cp.asarray(vars.s_bl)[i]),
            cp.asnumpy(cp.asarray(vars.z_u)[i]), cp.asnumpy(cp.asarray(vars.z_l)[i]),
            cp.asnumpy(cp.asarray(vars.z_bu)[i]), cp.asnumpy(cp.asarray(vars.z_bl)[i]),
            xbs, idx_hu, idx_hl, idx_xu, idx_xl,
            float(rho[i]), float(delta[i]),
        )

        ref = _ref_full_solve(
            Pi, Ai, Gi, xbs,
            idx_hu, idx_hl, idx_xu, idx_xl,
            scalings, float(delta[i]),
            cp.asnumpy(cp.asarray(rhs.x)[i]), cp.asnumpy(cp.asarray(rhs.y)[i]),
            cp.asnumpy(cp.asarray(rhs.z_u)[i]), cp.asnumpy(cp.asarray(rhs.z_l)[i]),
            cp.asnumpy(cp.asarray(rhs.z_bu)[i]), cp.asnumpy(cp.asarray(rhs.z_bl)[i]),
            cp.asnumpy(cp.asarray(rhs.s_u)[i]), cp.asnumpy(cp.asarray(rhs.s_l)[i]),
            cp.asnumpy(cp.asarray(rhs.s_bu)[i]), cp.asnumpy(cp.asarray(rhs.s_bl)[i]),
        )
        ref_dx, ref_dy, ref_dz_u, ref_dz_l, ref_dz_bu, ref_dz_bl, \
            ref_ds_u, ref_ds_l, ref_ds_bu, ref_ds_bl = ref

        atol = 1e-8
        np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.x)[i]), ref_dx, atol=atol,
                                   err_msg=f"dx problem {i}")
        if p > 0:
            np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.y)[i]), ref_dy, atol=atol,
                                       err_msg=f"dy problem {i}")
        if data.num_hu > 0:
            np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.z_u)[i]), ref_dz_u, atol=atol,
                                       err_msg=f"dz_u problem {i}")
            np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.s_u)[i]), ref_ds_u, atol=atol,
                                       err_msg=f"ds_u problem {i}")
        if data.num_hl > 0:
            np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.z_l)[i]), ref_dz_l, atol=atol,
                                       err_msg=f"dz_l problem {i}")
            np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.s_l)[i]), ref_ds_l, atol=atol,
                                       err_msg=f"ds_l problem {i}")
        if data.num_xu > 0:
            np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.z_bu)[i]), ref_dz_bu, atol=atol,
                                       err_msg=f"dz_bu problem {i}")
            np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.s_bu)[i]), ref_ds_bu, atol=atol,
                                       err_msg=f"ds_bu problem {i}")
        if data.num_xl > 0:
            np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.z_bl)[i]), ref_dz_bl, atol=atol,
                                       err_msg=f"dz_bl problem {i}")
            np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.s_bl)[i]), ref_ds_bl, atol=atol,
                                       err_msg=f"ds_bl problem {i}")


# ======================================================================
# Size matrices for the parametrized tests
# ======================================================================

# All four constraint types present: equality A, inequality G, box bounds.
# Mix of single (B=1) and batched (B>1) problems.
FULL_SIZES = [
    pytest.param( 1,  3, 1,  2, id="B1-n3-p1-m2"),
    pytest.param( 1,  6, 3,  4, id="B1-n6-p3-m4"),
    pytest.param( 1, 10, 3,  5, id="B1-n10-p3-m5"),
    pytest.param( 1, 16, 5,  8, id="B1-n16-p5-m8"),
    pytest.param( 1, 24, 8, 12, id="B1-n24-p8-m12"),
    pytest.param( 2,  5, 2,  3, id="B2-n5-p2-m3"),
    pytest.param( 4,  6, 3,  4, id="B4-n6-p3-m4"),
    pytest.param( 8, 10, 3,  5, id="B8-n10-p3-m5"),
    pytest.param(16,  8, 3,  5, id="B16-n8-p3-m5"),
    pytest.param(32, 12, 4,  6, id="B32-n12-p4-m6"),
    pytest.param(64, 16, 5,  8, id="B64-n16-p5-m8"),
]

# No general inequality (m = 0); equality + box bounds.
NO_INEQ_SIZES = [
    pytest.param( 1,  5, 3, 0, id="B1-n5-p3"),
    pytest.param( 1, 10, 5, 0, id="B1-n10-p5"),
    pytest.param( 3,  5, 2, 0, id="B3-n5-p2"),
    pytest.param( 8,  8, 3, 0, id="B8-n8-p3"),
    pytest.param(32, 12, 4, 0, id="B32-n12-p4"),
]

# No equality (p = 0); inequality + box bounds.
NO_EQ_SIZES = [
    pytest.param( 1,  6, 0, 4, id="B1-n6-m4"),
    pytest.param( 1, 10, 0, 6, id="B1-n10-m6"),
    pytest.param( 3,  6, 0, 4, id="B3-n6-m4"),
    pytest.param(16,  8, 0, 5, id="B16-n8-m5"),
    pytest.param(32, 12, 0, 8, id="B32-n12-m8"),
]

# Box bounds only (p = 0, m = 0).
BOX_ONLY_SIZES = [
    pytest.param( 1,  8, 0, 0, id="B1-n8"),
    pytest.param( 1, 16, 0, 0, id="B1-n16"),
    pytest.param( 4,  8, 0, 0, id="B4-n8"),
    pytest.param(16, 10, 0, 0, id="B16-n10"),
]

# Equality only, no inequality of any kind (no G, no box bounds).
# This is a *different* DenseData code path from BOX_ONLY: ``num_ineq``
# is zero, so no slacks/duals get allocated.
EQ_ONLY_SIZES = [
    pytest.param( 1,  5, 3, id="B1-n5-p3"),
    pytest.param( 1, 10, 5, id="B1-n10-p5"),
    pytest.param( 3,  5, 2, id="B3-n5-p2"),
    pytest.param( 8,  8, 3, id="B8-n8-p3"),
]


# ======================================================================
# Full-solve correctness tests (parametrized over size matrix)
# ======================================================================

class TestKKTSystemFullSolve:
    """Verify the full KKTSystem pipeline against the NumPy reference, across
    a matrix of problem sizes covering both single and batched modes."""

    @pytest.mark.parametrize("B,n,p,m", FULL_SIZES)
    def test_all_constraints(self, B, n, p, m):
        _run_full_solve(B=B, n=n, p=p, m=m, seed=B * 100 + n * 10 + p + m)

    @pytest.mark.parametrize("B,n,p,m", NO_EQ_SIZES)
    def test_no_equality(self, B, n, p, m):
        _run_full_solve(B=B, n=n, p=p, m=m, seed=B * 100 + n * 10 + m + 1)

    @pytest.mark.parametrize("B,n,p,m", NO_INEQ_SIZES)
    def test_no_inequality_with_box(self, B, n, p, m):
        """``m = 0`` -- no general inequality, but ``x_u`` / ``x_l`` are present."""
        _run_full_solve(B=B, n=n, p=p, m=m, seed=B * 100 + n * 10 + p + 2)

    @pytest.mark.parametrize("B,n,p,m", BOX_ONLY_SIZES)
    def test_box_bounds_only(self, B, n, p, m):
        _run_full_solve(B=B, n=n, p=p, m=m, seed=B * 100 + n * 10 + 3)


class TestKKTSystemEqualityOnly:
    """Exercise the path where the data has no inequality of any kind
    (``num_ineq == 0``). Box bounds are deliberately suppressed; this is a
    different code path from ``test_box_bounds_only`` where ``num_ineq > 0``
    via x_u/x_l."""

    @pytest.mark.parametrize("B,n,p", EQ_ONLY_SIZES)
    def test_equality_only(self, B, n, p):
        rng = np.random.default_rng(B * 100 + n * 10 + p + 4)
        Ps = []
        for _ in range(B):
            M = rng.standard_normal((n, n))
            Pi = M @ M.T + n * np.eye(n)
            Ps.append((Pi + Pi.T) / 2)
        data = _dense_data(
            P=cp.array(np.stack(Ps)),
            c=cp.zeros((B, n)),
            A=cp.array(rng.standard_normal((B, p, n))),
            b=cp.zeros((B, p)),
        )
        settings = Settings()
        kkt_sys = KKTSystem()
        kkt_sys.init(data, settings)
        pc = DenseRuizEquilibration(B, data.n, data.p, data.m, has_h_l=data.has_h_l, has_h_u=data.has_h_u, has_x_l=data.has_x_l, has_x_u=data.has_x_u)

        vars = Variables(); vars.init(data)
        vars.x = 0.0
        vars.y = 0.0

        rho = cp.full(B, 1e-4)
        delta = cp.full(B, 1e-4)
        kkt_sys.update_scalings_and_factor(data, pc, settings, False, rho, delta, vars)
        assert not kkt_sys.factor_status.numpy().any()

        rhs = Variables(); rhs.init(data)
        rhs.x = cp.array(rng.standard_normal((B, n)))
        rhs.y = cp.array(rng.standard_normal((B, p)))

        lhs = Variables(); lhs.init(data)
        kkt_sys.solve(data, pc, settings, rhs, lhs)

        for i in range(B):
            Pi = cp.asnumpy(cp.asarray(data.P)[i])
            Ai = cp.asnumpy(cp.asarray(data.A)[i])
            d_v, r_v = 1e-4, 1e-4
            kkt = Pi + r_v * np.eye(n) + (1.0 / d_v) * Ai.T @ Ai
            rhs_cond = (cp.asnumpy(cp.asarray(rhs.x)[i])
                        + (1.0 / d_v) * Ai.T @ cp.asnumpy(cp.asarray(rhs.y)[i]))
            ref_dx = np.linalg.solve(kkt, rhs_cond)
            np.testing.assert_allclose(cp.asnumpy(cp.asarray(lhs.x)[i]), ref_dx, atol=1e-8,
                                       err_msg=f"problem {i}")


# ======================================================================
# Condensed K * lhs ~= rhs sanity check
# ======================================================================

class TestKKTSystemCondensedMatvec:
    """After ``solve``, multiplying ``lhs`` by the condensed KKT operator
    should reproduce the original ``rhs`` (within solver tolerance)."""

    CONDENSED_SIZES = [
        pytest.param( 1, 10, 3, 5, id="B1-n10-p3-m5"),
        pytest.param( 1, 20, 5, 8, id="B1-n20-p5-m8"),
        pytest.param( 4,  8, 2, 4, id="B4-n8-p2-m4"),
        pytest.param(16, 12, 3, 6, id="B16-n12-p3-m6"),
    ]

    @pytest.mark.parametrize("B,n,p,m", CONDENSED_SIZES)
    def test_K_times_lhs_recovers_rhs(self, B, n, p, m):
        data = _make_data(B, n, p, m)
        settings = Settings()
        kkt_sys = KKTSystem()
        kkt_sys.init(data, settings)
        pc = DenseRuizEquilibration(B, data.n, data.p, data.m, has_h_l=data.has_h_l, has_h_u=data.has_h_u, has_x_l=data.has_x_l, has_x_u=data.has_x_u)

        vars = _make_positive_vars(data)
        rho = cp.full(B, 1.0)
        delta = cp.full(B, 1.0)
        kkt_sys.update_scalings_and_factor(data, pc, settings, False, rho, delta, vars)

        rhs_x = cp.random.randn(B, n)
        rhs_y = cp.random.randn(B, p)
        rhs_z = cp.random.randn(B, m)
        if m > 0:
            rhs_z *= data.active_G_row
        lhs_x = cp.zeros((B, n))
        lhs_y = cp.zeros((B, p))
        lhs_z = cp.zeros((B, m))
        kkt_sys._kkt_solver.solve(data, _w(rhs_x), _w(rhs_y), _w(rhs_z), _w(lhs_x), _w(lhs_y), _w(lhs_z))

        check_x = cp.zeros((B, n))
        check_y = cp.zeros((B, p))
        check_z = cp.zeros((B, m))
        kkt_sys.mul_condensed_kkt(data, _w(lhs_x), _w(lhs_y), _w(lhs_z),
                                  _w(check_x), _w(check_y), _w(check_z))

        assert cp.allclose(rhs_x, check_x, atol=1e-8), \
            f"x mismatch: {float(cp.max(cp.abs(rhs_x - check_x))):.2e}"
        assert cp.allclose(rhs_y, check_y, atol=1e-8), \
            f"y mismatch: {float(cp.max(cp.abs(rhs_y - check_y))):.2e}"
        assert cp.allclose(rhs_z, check_z, atol=1e-8), \
            f"z mismatch: {float(cp.max(cp.abs(rhs_z - check_z))):.2e}"


# ======================================================================
# Forwarded matvec ops
# ======================================================================

class TestKKTSystemMatvec:
    """Sanity check for forwarded matvec operations."""

    MATVEC_SIZES = [
        pytest.param( 1,  4, 1, 2, id="B1-n4-p1-m2"),
        pytest.param( 4,  5, 3, 2, id="B4-n5-p3-m2"),
        pytest.param(16, 10, 4, 6, id="B16-n10-p4-m6"),
    ]

    @pytest.mark.parametrize("B,n,p,m", MATVEC_SIZES)
    def test_eval_P_x(self, B, n, p, m):
        data = _make_data(B=B, n=n, p=p, m=m)
        settings = Settings()
        kkt_sys = KKTSystem()
        kkt_sys.init(data, settings)

        x = cp.array(np.random.default_rng(0).standard_normal((B, n)))
        z = cp.zeros((B, n), dtype=cp.float64)
        kkt_sys.eval_P_x(data, 1.0, _w(x), _w(z))

        for i in range(B):
            expected = cp.asnumpy(data.P[i]) @ cp.asnumpy(x[i])
            np.testing.assert_allclose(cp.asnumpy(z[i]), expected, atol=1e-12)

    @pytest.mark.parametrize("B,n,p,m", MATVEC_SIZES)
    def test_eval_A_xn_and_AT_xt(self, B, n, p, m):
        data = _make_data(B=B, n=n, p=p, m=m)
        settings = Settings()
        kkt_sys = KKTSystem()
        kkt_sys.init(data, settings)

        x = cp.array(np.random.default_rng(1).standard_normal((B, n)))
        y = cp.array(np.random.default_rng(2).standard_normal((B, p)))
        out_Ax  = cp.zeros((B, p), dtype=cp.float64)
        out_ATy = cp.zeros((B, n), dtype=cp.float64)

        kkt_sys.eval_A_xn(data, 1.0, _w(x), _w(out_Ax))
        kkt_sys.eval_AT_xt(data, 1.0, _w(y), _w(out_ATy))

        for i in range(B):
            Ai = cp.asnumpy(data.A[i])
            np.testing.assert_allclose(cp.asnumpy(out_Ax[i]),  Ai @ cp.asnumpy(x[i]), atol=1e-12)
            np.testing.assert_allclose(cp.asnumpy(out_ATy[i]), Ai.T @ cp.asnumpy(y[i]), atol=1e-12)
