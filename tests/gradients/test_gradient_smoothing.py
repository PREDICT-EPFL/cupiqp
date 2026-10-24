"""Tests for qpax-style gradient smoothing (relaxed-KKT differentiation).

The smoothed backward returns a surrogate VJP of the *relaxed* solution map
(``s * z = kappa`` instead of ``0``), while the forward result stays the true
optimum. The reference is the closed-form relaxed solution of the 1-D problem

    min 0.5 x^2 + c x   s.t.   x >= 0

whose relaxed solution and derivative are

    x_kappa(c)      = 0.5 (-c + sqrt(c^2 + 4 kappa))
    dx_kappa/dc     = 0.5 (-1 +   c / sqrt(c^2 + 4 kappa))

A loss ``l = x`` has ``dl/dc = dx_kappa/dc`` (the gradient cuPIQP returns in
``grad_data.c``). At ``c = 0`` the relaxed primal bias is ``x_kappa(0) = sqrt(kappa)``.
"""
import numpy as np
import pytest

from cupiqp import DenseSolver, SparseSolver, Status
from tests.helpers import csr_triple, to_wp


def _scalar_qp(c):
    """Device data of min 0.5 x^2 + c x  s.t.  x >= 0."""
    return dict(P=to_wp(np.array([[1.0]])), c=to_wp(np.array([float(c)])),
                x_l=to_wp(np.array([0.0])))


def _ones(shape):
    return to_wp(np.ones(shape))


KAPPA = 1e-3


def _expected_grad_c(c, kappa=KAPPA):
    return 0.5 * (-1.0 + c / np.sqrt(c * c + 4.0 * kappa))


def _dense_grad_c(c, *, smoothing, kappa=KAPPA, prec_iter=0, scale_cost=False):
    s = DenseSolver()
    s.settings.enable_grad = True
    s.settings.gradient_smoothing = smoothing
    s.settings.gradient_smoothing_mu = kappa
    s.settings.preconditioner_iter = prec_iter
    s.settings.preconditioner_scale_cost = scale_cost
    s.setup(**_scalar_qp(c))
    s.solve()
    status = s.result.info.to_host().status
    assert status[0] == Status.CUPIQP_SOLVED
    g = s.backward(grad_x=_ones((1, 1)))
    return float(s.result.x.numpy()[0, 0]), float(g.c.numpy()[0, 0])


@pytest.mark.parametrize("c", [-0.5, -0.1, 0.0, 0.1, 0.5, 1.0])
def test_smoothed_vjp_matches_analytic_relaxed_derivative(c):
    x_sol, grad_c = _dense_grad_c(c, smoothing=True)
    # forward solution is the TRUE optimum, not the relaxed point
    np.testing.assert_allclose(x_sol, max(-c, 0.0), atol=1e-6)
    np.testing.assert_allclose(grad_c, _expected_grad_c(c), atol=1e-5)


def test_smoothing_is_continuous_across_active_set_change():
    # Unsmoothed grad jumps 0 -> -1 across c=0; smoothed grad varies smoothly.
    grads = [_dense_grad_c(c, smoothing=True)[1] for c in (-0.05, -0.02, 0.0, 0.02, 0.05)]
    diffs = np.diff(grads)
    assert np.all(diffs > 0)                      # monotonic (grad increases in c)
    assert np.max(np.abs(diffs)) < 0.4            # no jump (a jump would be ~1)
    np.testing.assert_allclose(grads[2], -0.5, atol=1e-6)  # symmetric midpoint at c=0


def test_relaxed_primal_bias_is_sqrt_kappa_at_transition():
    # At c=0 the relaxed primal is sqrt(kappa); we observe it through the VJP,
    # which equals -0.5 there exactly regardless of kappa.
    for kappa in (1e-2, 1e-3, 1e-4):
        _, grad_c = _dense_grad_c(0.0, smoothing=True, kappa=kappa)
        np.testing.assert_allclose(grad_c, -0.5, atol=1e-8)


# c=0 is the exact kink (degenerate); the unsmoothed derivative is undefined there.
@pytest.mark.parametrize("c", [-0.1, 0.2, 0.5])
def test_smoothing_off_is_unsmoothed_and_kappa_independent(c):
    # Standard mode ignores kappa and gives the (nonsmooth) true-optimum VJP.
    _, g1 = _dense_grad_c(c, smoothing=False, kappa=1e-3)
    _, g2 = _dense_grad_c(c, smoothing=False, kappa=1e-1)
    np.testing.assert_allclose(g1, g2, atol=1e-9)
    expected = 0.0 if c > 0 else -1.0      # bound active (c>0 -> x=0) vs inactive
    np.testing.assert_allclose(g1, expected, atol=1e-6)


def test_kappa_to_zero_approaches_true_derivative_away_from_boundary():
    c = 0.4   # well inside: true x=0 (bound active), dx/dc -> 0
    errs = [abs(_dense_grad_c(c, smoothing=True, kappa=k)[1] - 0.0)
            for k in (1e-2, 1e-3, 1e-4)]
    assert errs[0] > errs[1] > errs[2]     # bias shrinks with kappa


@pytest.mark.parametrize("scale_cost", [False, True])
def test_preconditioner_and_cost_scaling_match_analytic(scale_cost):
    for c in (-0.1, 0.0, 0.3):
        _, grad_c = _dense_grad_c(c, smoothing=True, prec_iter=10, scale_cost=scale_cost)
        np.testing.assert_allclose(grad_c, _expected_grad_c(c), atol=1e-5)


def test_sparse_backend_matches_analytic():
    for c in (-0.1, 0.0, 0.3):
        s = SparseSolver()
        s.settings.enable_grad = True
        s.settings.gradient_smoothing = True
        s.settings.gradient_smoothing_mu = KAPPA
        s.settings.preconditioner_iter = 10
        s.setup(P=csr_triple(np.array([[1.0]])), c=to_wp(np.array([float(c)])),
                x_l=to_wp(np.array([0.0])))
        s.solve()
        assert s.result.info.to_host().status[0] == Status.CUPIQP_SOLVED
        g = s.backward(grad_x=_ones((1, 1)))
        np.testing.assert_allclose(float(g.c.numpy()[0, 0]), _expected_grad_c(c), atol=1e-5)


def test_multistage_matches_dense_box_constrained():
    from cupiqp import MultistageSolver

    # P=[[2,0.25],[0.25,3]], c=[1,-0.5], x>=0
    P = (to_wp(np.array([[[2.0]], [[3.0]]])), to_wp(np.array([[[0.25]]])))   # (diag, offdiag)
    c = to_wp(np.array([1.0, -0.5]))
    xl = to_wp(np.array([0.0, 0.0]))
    ms = MultistageSolver()
    ms.settings.enable_cuda_graph = False
    ms.settings.enable_grad = True
    ms.settings.gradient_smoothing = True
    ms.settings.gradient_smoothing_mu = KAPPA
    ms.setup(P=P, c=c, x_l=xl)
    ms.solve()
    assert ms.result.info.to_host().status[0] == Status.CUPIQP_SOLVED
    g_ms = ms.backward(grad_x=_ones((1, 2))).c.numpy().ravel()

    d = DenseSolver()
    d.settings.enable_grad = True
    d.settings.gradient_smoothing = True
    d.settings.gradient_smoothing_mu = KAPPA
    d.setup(P=to_wp(np.array([[2.0, 0.25], [0.25, 3.0]])), c=to_wp(np.array([1.0, -0.5])),
            x_l=to_wp(np.array([0.0, 0.0])))
    d.solve()
    g_d = d.backward(grad_x=_ones((1, 2))).c.numpy().ravel()

    np.testing.assert_allclose(g_ms, g_d, atol=1e-5)


def test_smoothing_mode_frozen_at_setup():
    # enabling after setup (set up with smoothing OFF explicitly, independent
    # of the default)
    s = DenseSolver()
    s.settings.enable_grad = True
    s.settings.gradient_smoothing = False
    s.setup(**_scalar_qp(-0.3))
    s.solve()
    s.settings.gradient_smoothing = True
    with pytest.raises(RuntimeError):
        s.backward(grad_x=_ones((1, 1)))
    # disabling after setup
    s2 = DenseSolver()
    s2.settings.enable_grad = True
    s2.settings.gradient_smoothing = True
    s2.setup(**_scalar_qp(-0.3))
    s2.solve()
    s2.settings.gradient_smoothing = False
    with pytest.raises(RuntimeError):
        s2.backward(grad_x=_ones((1, 1)))


def _solved_smoothing_solver(c=0.5, **smoothing):
    s = DenseSolver()
    s.settings.enable_grad = True
    s.settings.gradient_smoothing = True
    s.settings.preconditioner_iter = 0
    for k, v in smoothing.items():
        setattr(s.settings, k, v)
    s.setup(**_scalar_qp(c))
    s.solve()
    assert s.result.info.to_host().status[0] == Status.CUPIQP_SOLVED
    return s


@pytest.mark.parametrize("field, bad", [
    ("gradient_smoothing_mu", -1.0),
    ("gradient_smoothing_mu", 0.0),
    ("gradient_smoothing_mu", float("nan")),
    ("gradient_smoothing_mu", float("inf")),
    ("gradient_smoothing_tol", -1e-5),
    ("gradient_smoothing_tol", float("nan")),
    ("gradient_smoothing_max_iter", 0),
])
def test_invalid_smoothing_settings_rejected_at_backward(field, bad):
    # Settings are mutable between backward() calls, so they are revalidated each
    # smoothed backward -- an invalid value must raise, not produce a NaN gradient.
    s = _solved_smoothing_solver()
    setattr(s.settings, field, bad)
    with pytest.raises(ValueError):
        s.backward(grad_x=_ones((1, 1)))


def test_relaxation_nonconvergence_raises():
    # If the relaxed Newton loop cannot reach the tolerance within the iteration
    # budget, backward() raises rather than returning a stale-factor / under-
    # relaxed (silently biased) gradient.
    s = _solved_smoothing_solver(c=0.5, gradient_smoothing_tol=1e-16,
                                 gradient_smoothing_max_iter=1)
    with pytest.raises(RuntimeError):
        s.backward(grad_x=_ones((1, 1)))


def test_unconverged_solve_follows_gradient_on_failure_under_smoothing():
    # An unconverged problem does not stop backward(); the gradient policy is
    # settings.gradient_on_failure, decided on the device.
    s = DenseSolver()
    s.settings.enable_grad = True
    s.settings.gradient_smoothing = True
    s.settings.max_iter = 1            # force MAX_ITER (not solved)
    s.setup(**_scalar_qp(-0.3))
    s.solve()
    status = s.result.info.to_host().status
    assert status[0] != Status.CUPIQP_SOLVED
    g = s.backward(grad_x=_ones((1, 1)))              # "differentiate": proceeds
    assert np.isfinite(g.c.numpy()).all()
    s.settings.gradient_on_failure = "zero"
    g = s.backward(grad_x=_ones((1, 1)))
    assert float(g.c.numpy()[0, 0]) == 0.0


def test_update_without_resolve_rejected_then_ok_after_solve():
    s = DenseSolver()
    s.settings.enable_grad = True
    s.settings.gradient_smoothing = True
    s.setup(**_scalar_qp(-0.3))
    s.solve()
    s.update(c=to_wp(np.array([0.4])))              # the cached factor belongs to the old data
    with pytest.raises(RuntimeError, match="solve"):
        s.backward(grad_x=_ones((1, 1)))
    s.solve()
    g = s.backward(grad_x=_ones((1, 1)))   # ok after re-solve
    np.testing.assert_allclose(float(g.c.numpy()[0, 0]), _expected_grad_c(0.4), atol=1e-5)


@pytest.mark.parametrize("c", [-0.5, -0.1, 0.0, 0.1, 0.5])
def test_dual_and_slack_cotangents_match_relaxed_map(c):
    # The relaxed point is smooth in all of (x, y, s, z), so cotangents on the
    # dual/slack outputs give the relaxed-map VJP. For min 0.5x^2+cx s.t. x>=0:
    #   z_bl_kappa = 0.5(c + sqrt(c^2+4k)),  s_bl_kappa = 0.5(-c + sqrt(c^2+4k))
    r = np.sqrt(c * c + 4.0 * KAPPA)
    dz_dc = 0.5 * (1.0 + c / r)
    ds_dc = 0.5 * (-1.0 + c / r)
    s = DenseSolver()
    s.settings.enable_grad = True
    s.settings.gradient_smoothing = True
    s.settings.gradient_smoothing_mu = KAPPA
    s.setup(**_scalar_qp(c))
    s.solve()
    gz = float(s.backward(grad_z_bl=_ones((1, 1))).c.numpy()[0, 0])
    gs = float(s.backward(grad_s_bl=_ones((1, 1))).c.numpy()[0, 0])
    np.testing.assert_allclose(gz, dz_dc, atol=1e-5)
    np.testing.assert_allclose(gs, ds_dc, atol=1e-5)
