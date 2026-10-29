"""Shared helpers of the KKT tests: GPU buffers, an identity preconditioner,
a factored ``KKTSystem``, IPM variables and the NumPy reference solve of the
condensed KKT system.

These tests drive the internal KKT layer directly, so they build the pieces
the solver normally owns (device settings, preconditioner, variables).
"""
import numpy as np
import warp as wp

from cupiqp.dense.dense_data import DenseData
from cupiqp.dense.dense_preconditioner import DenseRuizEquilibration
from cupiqp.kkt_systems import KKTSystem
from cupiqp.multistage.multistage_data import MultistageData
from cupiqp.multistage.multistage_preconditioner import MultistageRuizEquilibration
from cupiqp.results import Variables
from cupiqp.settings import DeviceSettings, Settings
from cupiqp.sparse.sparse_data import SparseData
from cupiqp.sparse.sparse_preconditioner import SparseRuizEquilibration
from tests.helpers import DEVICE, to_wp


# ----------------------------------------------------------------------
# GPU buffers
# ----------------------------------------------------------------------

def zeros(dtype, *shape) -> wp.array:
    """Zero-filled GPU array."""
    return wp.zeros(shape, dtype=dtype, device=DEVICE)


def empty(dtype, *shape) -> wp.array:
    """Uninitialized GPU array."""
    return wp.empty(shape, dtype=dtype, device=DEVICE)


def nan(dtype, *shape) -> wp.array:
    """NaN-filled GPU array: an output a kernel must overwrite entirely."""
    return wp.full(shape, np.nan, dtype=dtype, device=DEVICE)


# ----------------------------------------------------------------------
# KKTSystem and its inputs
# ----------------------------------------------------------------------

def _identity_preconditioner(data):
    """A Ruiz preconditioner of the backend of ``data`` that has not run, so
    it scales nothing."""
    args = (data.batch_size, data.n, data.p, data.m)
    kw = dict(has_h_l=data.has_h_l, has_h_u=data.has_h_u, has_x_l=data.has_x_l, has_x_u=data.has_x_u,
              dtype=data.dtype)
    if isinstance(data, DenseData):
        return DenseRuizEquilibration(*args, **kw)
    if isinstance(data, SparseData):
        return SparseRuizEquilibration(*args, **kw)
    assert isinstance(data, MultistageData)
    return MultistageRuizEquilibration(*args, **kw, data=data)


def positive_variables(data, seed=0) -> Variables:
    """IPM variables with random x, y and s, z > 0, as the scalings need."""
    rng = np.random.default_rng(seed)
    B = data.batch_size
    v = Variables()
    v.init(data)
    v.x = to_wp(rng.standard_normal((B, data.n)), data.dtype)
    if data.p > 0:
        v.y = to_wp(rng.standard_normal((B, data.p)), data.dtype)
    if data.num_ineq > 0:
        v.s_all = to_wp(np.abs(rng.standard_normal((B, data.num_ineq))) + 0.5, data.dtype)
        v.z_all = to_wp(np.abs(rng.standard_normal((B, data.num_ineq))) + 0.5, data.dtype)
    return v


def random_variables(data, seed=0) -> Variables:
    """IPM variables (a right-hand side of the full KKT system) with random
    entries of either sign."""
    rng = np.random.default_rng(seed)
    B = data.batch_size
    v = Variables()
    v.init(data)
    v.x = to_wp(rng.standard_normal((B, data.n)), data.dtype)
    if data.p > 0:
        v.y = to_wp(rng.standard_normal((B, data.p)), data.dtype)
    if data.num_ineq > 0:
        v.s_all = to_wp(rng.standard_normal((B, data.num_ineq)), data.dtype)
        v.z_all = to_wp(rng.standard_normal((B, data.num_ineq)), data.dtype)
    return v


def kkt_system(data, settings: Settings) -> KKTSystem:
    """A KKTSystem initialized for ``data``."""
    # The solver owns and uploads the device settings otherwise.
    device_settings = DeviceSettings(data.dtype, data.device)
    device_settings.upload(settings)
    kkt = KKTSystem()
    kkt.init(data, settings, device_settings)
    return kkt


def factored_kkt_system(data, settings: Settings = None, rho=1.0, delta=1.0, variables=None,
                        use_static_reg=False):
    """A KKTSystem of ``data`` factored at ``variables`` (default:
    :func:`positive_variables`) with an identity preconditioner.

    ``rho`` and ``delta`` are scalars or per-problem host arrays; ``settings``
    defaults to the defaults of the data dtype. Returns ``(kkt,
    preconditioner, variables)``.
    """
    settings = Settings.for_dtype(data.dtype) if settings is None else settings
    variables = positive_variables(data) if variables is None else variables
    B = data.batch_size
    kkt = kkt_system(data, settings)
    preconditioner = _identity_preconditioner(data)
    kkt.update_scalings_and_factor(data, preconditioner, settings, use_static_reg,
                                   to_wp(np.broadcast_to(rho, (B,)), data.dtype),
                                   to_wp(np.broadcast_to(delta, (B,)), data.dtype), variables)
    assert not kkt.factor_status.numpy().any(), "KKT factorization failed"
    return kkt, preconditioner, variables


# ----------------------------------------------------------------------
# Condensed KKT system
# ----------------------------------------------------------------------

def condensed_solve(kkt: KKTSystem, data, rhs_x, rhs_y, rhs_z):
    """Solve the condensed KKT system for host right-hand sides; returns the
    ``(lhs_x, lhs_y, lhs_z)`` GPU arrays.

    There is no public entry point for the condensed-only solve
    (``KKTSystem.solve`` always runs eliminate -> condensed -> recover), so
    the inner solver is used.
    """
    dtype, B = data.dtype, data.batch_size
    lhs = zeros(dtype, B, data.n), zeros(dtype, B, data.p), zeros(dtype, B, data.m)
    kkt._kkt_solver.solve(data, to_wp(rhs_x, dtype), to_wp(rhs_y, dtype), to_wp(rhs_z, dtype), *lhs)
    return lhs


def condensed_residual(kkt: KKTSystem, data, lhs, rhs_x, rhs_y, rhs_z) -> float:
    """``||rhs - K_c lhs||_inf`` over the batch, for GPU ``lhs`` and host
    right-hand sides."""
    dtype, B = data.dtype, data.batch_size
    err = zeros(dtype, B, data.n), zeros(dtype, B, data.p), zeros(dtype, B, data.m)
    return kkt.get_refinement_error(data, *lhs, to_wp(rhs_x, dtype), to_wp(rhs_y, dtype), to_wp(rhs_z, dtype),
                                    *err)


def perturb_lhs(lhs, data, seed=0) -> None:
    """Add standard normal noise to a condensed solution ``lhs`` in place,
    leaving the inactive rows of ``lhs_z`` (both bounds infinite) as they are."""
    rng = np.random.default_rng(seed)
    masks = (1.0, 1.0, data.active_G_row.numpy())
    for a, mask in zip(lhs, masks):
        a.assign(to_wp(a.numpy() + mask * rng.standard_normal(a.shape), data.dtype))


def assert_inactive_rows_decoupled(kkt: KKTSystem, data, rhs_x, rhs_y, rhs_z) -> None:
    """The right-hand side of an inactive inequality row (both bounds
    infinite) leaves ``lhs_x``, ``lhs_y`` and the active rows of ``lhs_z``
    unchanged."""
    active = data.active_G_row.numpy()
    assert not active.all(), "the data has no inactive G row"
    first = condensed_solve(kkt, data, rhs_x, rhs_y, rhs_z)
    second = condensed_solve(kkt, data, rhs_x, rhs_y, rhs_z + 10.0 * (1.0 - active))
    for name, a, b in zip(("x", "y", "z"), first, second):
        a, b = a.numpy(), b.numpy()
        if name == "z":
            a, b = a * active, b * active
        # Same factor, right-hand sides equal on every coupled row: the
        # solutions agree up to the rounding of the solve.
        atol, rtol = (1e-12, 1e-12) if data.dtype == wp.float64 else (1e-5, 1e-5)
        np.testing.assert_allclose(b, a, err_msg=name, atol=atol, rtol=rtol)


# ----------------------------------------------------------------------
# NumPy reference
# ----------------------------------------------------------------------

def numpy_condensed_kkt_solve(P, A, G, x_reg, z_reg_inv, delta, rhs_x, rhs_y, rhs_z):
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
