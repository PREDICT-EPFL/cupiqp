"""Tests for backward_from_state(): the caller saves the final state of a
solve (the ten solution fields, rho, delta, status) and the solver later
refactorizes the KKT system there. The gradients agree with the immediate
backward for every backend, with and without preconditioning and gradient
smoothing; the caller's arrays are never modified; bad arguments are
rejected by name before any work."""

import numpy as np
import pytest
import warp as wp

import cupiqp
from cupiqp import DenseSolver, MultistageSolver, SparseSolver, Status
from tests.helpers import csr_triple, to_wp, to_wp_dict


STATE_FIELDS = ("x", "y", "z_u", "z_l", "z_bu", "z_bl", "s_u", "s_l", "s_bu", "s_bl")


# ---------------------------------------------------------------------------
# Problems and helpers
# ---------------------------------------------------------------------------
def _dense_qp(n=5, p=1, m=3, seed=0, batch_size=1):
    """Strictly convex QP with equalities, two-sided inequalities and box
    bounds; a batch of perturbed copies when batch_size > 1."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)
    A = rng.standard_normal((p, n))
    G = rng.standard_normal((m, n))
    qp = dict(P=P, A=A, G=G)
    vecs = []
    for b in range(batch_size):
        # Feasible by construction: a point x* satisfies the equalities and
        # lies strictly inside the inequalities and the box.
        x_star = rng.standard_normal(n)
        c = rng.standard_normal(n)
        bb = A @ x_star
        mid = G @ x_star
        rad = 0.2 + rng.random(m)
        xr = 0.5 + rng.random(n)
        vecs.append(dict(c=c, b=bb, h_l=mid - rad, h_u=mid + rad, x_l=x_star - xr, x_u=x_star + xr))
    for key in vecs[0]:
        qp[key] = np.stack([v[key] for v in vecs]) if batch_size > 1 else vecs[0][key]
    return qp


def _settings(solver, preconditioner_iter, dtype):
    s = solver.settings
    s.enable_grad = True
    s.preconditioner_iter = preconditioner_iter
    s.verbose = False
    s.enable_cuda_graph = False
    # Tight tolerances: the immediate backward linearizes at the last Newton
    # iterate, backward_from_state() at the final point; they agree to the
    # solve tolerance.
    s.eps_abs = 1e-12 if dtype == np.float64 else 1e-6
    s.eps_rel = 1e-12 if dtype == np.float64 else 1e-6


def _dense_solver(qp, preconditioner_iter=0, dtype=np.float64, **extra):
    solver = DenseSolver(dtype=cupiqp.utils.to_warp_dtype(dtype))
    _settings(solver, preconditioner_iter, dtype)
    for name, value in extra.items():
        setattr(solver.settings, name, value)
    solver.setup(**to_wp_dict(qp, dtype))
    return solver


def _sparse_solver(qp, preconditioner_iter=0):
    solver = SparseSolver()
    _settings(solver, preconditioner_iter, np.float64)
    d = to_wp_dict(qp)
    tri = {name: csr_triple(qp[name]) for name in ("P", "A", "G")}
    solver.setup(P=tri["P"], c=d["c"], A=tri["A"], b=d["b"], G=tri["G"],
                 h_u=d["h_u"], h_l=d["h_l"], x_u=d["x_u"], x_l=d["x_l"])
    return solver


def _multistage_solver(offdiag=0.25, preconditioner_iter=0):
    solver = MultistageSolver()
    _settings(solver, preconditioner_iter, np.float64)
    P_diag = to_wp(np.array([[[2.0]], [[3.0]]]))
    P_offdiag = to_wp(np.array([[[offdiag]]]))
    c = to_wp(np.array([[1.0], [-0.5]]))
    solver.setup(P=(P_diag, P_offdiag), c=c)
    return solver


def _saved_state(solver):
    """Independent Warp copies of the latest solve's state: what a caller
    keeps for a later backward_from_state()."""
    result = solver.result
    with wp.ScopedStream(solver.stream):
        state = {name: wp.clone(getattr(result, name)) for name in STATE_FIELDS}
        state["rho"] = wp.clone(result.info.rho)
        state["delta"] = wp.clone(result.info.delta)
        state["status"] = wp.clone(result.info.status)
    wp.synchronize_stream(solver.stream)
    return state


def _with_entry(a, index, value):
    """Copy of the device array ``a`` with ``a[index] = value``."""
    host = a.numpy()
    host[index] = value
    return to_wp(host)


def _grads_to_numpy(grads, names):
    out = {}
    for name in names:
        value = getattr(grads, name)
        arr = value.data if hasattr(value, "indptr") else value
        out[name] = arr.numpy().copy()
    return out


def _assert_solved(solver):
    statuses = solver.result.info.to_host().status
    assert all(s == Status.CUPIQP_SOLVED for s in statuses), statuses


DENSE_GRADS = ("P", "c", "A", "b", "G", "h_u", "h_l", "x_u", "x_l")


# ---------------------------------------------------------------------------
# Agreement with the immediate backward
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("preconditioner_iter", [0, 10])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_dense_matches_immediate(preconditioner_iter, dtype):
    qp = _dense_qp(batch_size=3)
    solver = _dense_solver(qp, preconditioner_iter, dtype)
    solver.solve()
    _assert_solved(solver)
    B, n = 3, 5
    grad_x = to_wp(np.random.default_rng(1).standard_normal((B, n)), dtype)
    grad_zu = to_wp(np.ones((B, 3)), dtype)

    state = _saved_state(solver)
    immediate = _grads_to_numpy(solver.backward(grad_x=grad_x, grad_z_u=grad_zu), DENSE_GRADS)
    from_state = _grads_to_numpy(
        solver.backward_from_state(**state, grad_x=grad_x, grad_z_u=grad_zu), DENSE_GRADS)
    # The immediate backward linearizes at the last Newton iterate,
    # backward_from_state() at the final point. On problems with a nearly
    # degenerate active set they differ at the 1e-3 level.
    for name in DENSE_GRADS:
        assert np.isfinite(from_state[name]).all(), name
    if dtype == np.float64:
        for name in DENSE_GRADS:
            np.testing.assert_allclose(from_state[name], immediate[name], atol=2e-3, rtol=2e-3, err_msg=name)
    else:
        np.testing.assert_allclose(from_state["c"], immediate["c"], atol=1e-1, rtol=1e-1, err_msg="c")


@pytest.mark.parametrize("preconditioner_iter", [0, 10])
def test_sparse_matches_immediate(preconditioner_iter):
    qp = _dense_qp(batch_size=2, seed=3)
    solver = _sparse_solver(qp, preconditioner_iter)
    solver.solve()
    _assert_solved(solver)
    grad_x = to_wp(np.random.default_rng(2).standard_normal((2, 5)))
    state = _saved_state(solver)
    immediate = _grads_to_numpy(solver.backward(grad_x=grad_x), DENSE_GRADS)
    from_state = _grads_to_numpy(solver.backward_from_state(**state, grad_x=grad_x), DENSE_GRADS)
    for name in DENSE_GRADS:
        np.testing.assert_allclose(from_state[name], immediate[name], atol=1e-6, rtol=1e-6, err_msg=name)


@pytest.mark.parametrize("preconditioner_iter", [0, 10])
def test_multistage_matches_immediate(preconditioner_iter):
    solver = _multistage_solver(preconditioner_iter=preconditioner_iter)
    solver.solve()
    _assert_solved(solver)
    grad_x = to_wp(np.ones((1, 2)))
    state = _saved_state(solver)
    names = ("P_diag", "P_offdiag", "c")
    immediate = _grads_to_numpy(solver.backward(grad_x=grad_x), names)
    from_state = _grads_to_numpy(solver.backward_from_state(**state, grad_x=grad_x), names)
    for name in names:
        np.testing.assert_allclose(from_state[name], immediate[name], atol=1e-8, rtol=1e-6, err_msg=name)


@pytest.mark.parametrize("preconditioner_iter", [0, 10])
def test_smoothing_matches_immediate(preconditioner_iter):
    """With gradient smoothing the relaxed Newton loop starts from the given
    point; the result must match the immediate smoothed backward."""
    qp = _dense_qp(batch_size=2, seed=8)
    solver = _dense_solver(qp, preconditioner_iter, gradient_smoothing=True)
    solver.solve()
    _assert_solved(solver)
    grad_x = to_wp(np.random.default_rng(3).standard_normal((2, 5)))
    state = _saved_state(solver)
    immediate = _grads_to_numpy(solver.backward(grad_x=grad_x), DENSE_GRADS)
    from_state = _grads_to_numpy(solver.backward_from_state(**state, grad_x=grad_x), DENSE_GRADS)
    for name in DENSE_GRADS:
        np.testing.assert_allclose(from_state[name], immediate[name], atol=1e-8, rtol=1e-6, err_msg=name)


def test_zero_width_fields_are_accepted():
    """Without inequalities and bounds, eight of the ten fields are (B, 0)
    arrays; they are passed like the others."""
    rng = np.random.default_rng(21)
    n, B = 4, 2
    M = rng.standard_normal((n, n))
    qp = dict(P=M @ M.T + n * np.eye(n), c=rng.standard_normal((B, n)),
              A=rng.standard_normal((1, n)), b=rng.standard_normal((B, 1)))
    solver = _dense_solver(qp)
    solver.solve()
    _assert_solved(solver)
    state = _saved_state(solver)
    widths = {name: state[name].shape[1] for name in STATE_FIELDS}
    assert widths == dict(x=n, y=1, z_u=0, z_l=0, z_bu=0, z_bl=0, s_u=0, s_l=0, s_bu=0, s_bl=0)
    # Exact gradients of sum(x) for an equality-constrained QP, from the
    # adjoint of [[P, A'], [A, 0]]: d/dc = -lam_x, d/db = lam_y.
    K = np.block([[qp["P"], qp["A"].T], [qp["A"], np.zeros((1, 1))]])
    lam = np.linalg.solve(K.T, np.concatenate([np.ones(n), [0.0]]))
    grads = _grads_to_numpy(solver.backward_from_state(**state, grad_x=to_wp(np.ones((B, n)))), ("c", "b"))
    # Off by the final regularization (delta = 1e-6) only.
    np.testing.assert_allclose(grads["c"], np.tile(-lam[:n], (B, 1)), atol=1e-5, rtol=0)
    np.testing.assert_allclose(grads["b"], np.tile(lam[n:], (B, 1)), atol=1e-5, rtol=0)


def test_accepts_warp_and_torch_arrays():
    torch = pytest.importorskip("torch")
    qp = _dense_qp(batch_size=2, seed=4)
    solver = _dense_solver(qp)
    solver.solve()
    state = _saved_state(solver)
    grad_x = to_wp(np.ones((2, 5)))
    reference = _grads_to_numpy(solver.backward_from_state(**state, grad_x=grad_x), ("c",))["c"]
    as_warp = {k: wp.clone(v) for k, v in state.items()}
    got = _grads_to_numpy(solver.backward_from_state(**as_warp, grad_x=grad_x), ("c",))["c"]
    np.testing.assert_array_equal(got, reference)
    as_torch = {k: torch.as_tensor(v.numpy(), device="cuda") for k, v in state.items()}
    got = _grads_to_numpy(solver.backward_from_state(**as_torch, grad_x=grad_x), ("c",))["c"]
    np.testing.assert_array_equal(got, reference)


# ---------------------------------------------------------------------------
# Rollout: several solves, backward later in any order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("preconditioner_iter", [0, 10])
@pytest.mark.parametrize("order", ["AB", "BA"])
def test_two_solves_backward_later_in_any_order(preconditioner_iter, order):
    qp_a = _dense_qp(seed=10)
    qp_b = dict(qp_a, c=qp_a["c"] + 0.7, h_u=qp_a["h_u"] + 0.3, x_l=qp_a["x_l"] - 0.2)
    grad_x = to_wp(np.random.default_rng(4).standard_normal((1, 5)))

    def reference(qp):
        s = _dense_solver(qp, preconditioner_iter)
        s.solve()
        _assert_solved(s)
        return _grads_to_numpy(s.backward(grad_x=grad_x), DENSE_GRADS)

    ref_a, ref_b = reference(qp_a), reference(qp_b)

    def load(qp):
        solver.update(c=to_wp(qp["c"]), h_u=to_wp(qp["h_u"]), x_l=to_wp(qp["x_l"]))

    solver = _dense_solver(qp_a, preconditioner_iter)
    solver.solve()
    state_a = _saved_state(solver)
    load(qp_b)
    solver.solve()
    state_b = _saved_state(solver)

    def backward(qp, state, ref, label):
        load(qp)
        got = _grads_to_numpy(solver.backward_from_state(**state, grad_x=grad_x), DENSE_GRADS)
        for name in DENSE_GRADS:
            np.testing.assert_allclose(got[name], ref[name], atol=1e-5, rtol=1e-5, err_msg=label + name)

    if order == "AB":
        backward(qp_a, state_a, ref_a, "A "); backward(qp_b, state_b, ref_b, "B ")
    else:
        backward(qp_b, state_b, ref_b, "B "); backward(qp_a, state_a, ref_a, "A ")


def test_state_saved_inside_a_user_capture():
    """Saving a state is plain device copies: a user can record them in a CUDA
    graph with update() and solve(), and differentiate the saved state later."""
    qp = _dense_qp(batch_size=2, seed=14)
    costs = [qp["c"] + 0.3 * k for k in range(2)]
    solver = DenseSolver()
    _settings(solver, 0, np.float64)
    c_buf = wp.array(qp["c"], dtype=wp.float64)
    d = to_wp_dict(qp)
    d["c"] = c_buf
    solver.setup(**d)
    solver.solve()                           # warm-up outside the capture
    result = solver.result
    saved = {name: wp.empty_like(getattr(result, name)) for name in STATE_FIELDS}
    saved["rho"] = wp.empty_like(result.info.rho)
    saved["delta"] = wp.empty_like(result.info.delta)
    saved["status"] = wp.empty_like(result.info.status)
    with wp.ScopedStream(solver.stream), wp.ScopedCapture() as capture:
        solver.update(c=c_buf)
        solver.solve()
        for name in STATE_FIELDS:
            if saved[name].size > 0:
                wp.copy(saved[name], getattr(result, name))
        wp.copy(saved["rho"], result.info.rho)
        wp.copy(saved["delta"], result.info.delta)
        wp.copy(saved["status"], result.info.status)

    grad_x = to_wp(np.ones((2, 5)))
    for c in costs:
        c_buf.assign(c)
        wp.capture_launch(capture.graph, stream=solver.stream)
        ref = _dense_solver(dict(qp, c=c))
        ref.solve()
        expected = _grads_to_numpy(ref.backward(grad_x=grad_x), DENSE_GRADS)
        got = _grads_to_numpy(solver.backward_from_state(**saved, grad_x=grad_x), DENSE_GRADS)
        for name in DENSE_GRADS:
            np.testing.assert_allclose(got[name], expected[name], atol=2e-3, rtol=2e-3, err_msg=name)


# ---------------------------------------------------------------------------
# The caller's arrays, validation, lifecycle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smoothing", [False, True])
def test_caller_state_is_not_modified(smoothing):
    """Scaling, refactorization and smoothing work on solver-owned copies."""
    qp = _dense_qp(batch_size=2, seed=5)
    solver = _dense_solver(qp, preconditioner_iter=10, gradient_smoothing=smoothing)
    solver.solve()
    _assert_solved(solver)
    state = _saved_state(solver)
    before = {k: v.numpy() for k, v in state.items()}
    solver.backward_from_state(**state, grad_x=to_wp(np.ones((2, 5))))
    solver.backward_from_state(**state, grad_x=to_wp(np.ones((2, 5))))
    for k, v in state.items():
        np.testing.assert_array_equal(v.numpy(), before[k], err_msg=k)


def _expect_rejected(solver, state, exc, match, **changes):
    bad = dict(state, **changes)
    with pytest.raises(exc, match=match):
        solver.backward_from_state(**bad, grad_x=to_wp(np.ones((2, 5))))


def test_validation_names_the_bad_argument():
    qp = _dense_qp(batch_size=2, seed=6)
    solver = _dense_solver(qp)
    solver.solve()
    state = _saved_state(solver)
    _expect_rejected(solver, state, ValueError, r"'x' must have shape \(2, 5\); got \(2, 4\)",
                     x=state["x"][:, :4])
    _expect_rejected(solver, state, ValueError, r"'z_u' must have shape \(2, 3\); got \(1, 3\)",
                     z_u=state["z_u"][:1])
    _expect_rejected(solver, state, ValueError, r"'rho' must have shape \(2,\)",
                     rho=state["rho"].reshape((2, 1)))
    _expect_rejected(solver, state, TypeError, "'s_bl' has dtype float32",
                     s_bl=to_wp(state["s_bl"].numpy(), np.float32))
    _expect_rejected(solver, state, TypeError, "'delta' has dtype float32",
                     delta=to_wp(state["delta"].numpy(), np.float32))
    _expect_rejected(solver, state, TypeError, "'status' has dtype float64",
                     status=to_wp(state["status"].numpy(), np.float64))
    _expect_rejected(solver, state, TypeError, "'y' must be a GPU array",
                     y=state["y"].numpy())
    _expect_rejected(solver, state, TypeError, "'s_u' is required", s_u=None)
    with pytest.raises(TypeError):
        solver.backward_from_state(state["x"], grad_x=to_wp(np.ones((2, 5))))     # keyword-only
    missing = {k: v for k, v in state.items() if k != "z_bl"}
    with pytest.raises(TypeError, match="z_bl"):
        solver.backward_from_state(**missing)


def test_zero_width_fields_are_validated():
    rng = np.random.default_rng(22)
    n = 3
    M = rng.standard_normal((n, n))
    solver = _dense_solver(dict(P=M @ M.T + n * np.eye(n), c=rng.standard_normal((2, n))))
    solver.solve()
    state = _saved_state(solver)
    assert state["z_u"].shape == (2, 0)
    with pytest.raises(ValueError, match=r"'z_u' must have shape \(2, 0\); got \(1, 0\)"):
        solver.backward_from_state(**dict(state, z_u=to_wp(np.zeros((1, 0)))),
                                   grad_x=to_wp(np.ones((2, n))))
    with pytest.raises(TypeError, match="'s_l' has dtype float32"):
        solver.backward_from_state(**dict(state, s_l=to_wp(np.zeros((2, 0)), np.float32)),
                                   grad_x=to_wp(np.ones((2, n))))


@pytest.mark.skipif(wp.get_cuda_device_count() < 2, reason="needs two GPUs")
def test_state_on_another_device_is_rejected():
    qp = _dense_qp(batch_size=2, seed=6)
    solver = _dense_solver(qp)
    solver.solve()
    state = _saved_state(solver)
    elsewhere = wp.array(state["x"].numpy(), device="cuda:1")
    _expect_rejected(solver, state, ValueError, "'x' must be on", x=elsewhere)


def test_rejected_call_leaves_the_cached_factor():
    """Validation runs before any work: after a rejected call, the immediate
    backward of the latest solve is still available."""
    qp = _dense_qp(batch_size=2, seed=7)
    solver = _dense_solver(qp)
    solver.solve()
    state = _saved_state(solver)
    expected = _grads_to_numpy(solver.backward(grad_x=to_wp(np.ones((2, 5)))), ("c",))["c"]
    _expect_rejected(solver, state, TypeError, "'status' has dtype",
                     status=to_wp(state["status"].numpy(), np.float64))
    got = _grads_to_numpy(solver.backward(grad_x=to_wp(np.ones((2, 5)))), ("c",))["c"]
    np.testing.assert_array_equal(got, expected)


def test_backward_from_state_needs_no_solve_of_its_own():
    """A solver that never solved differentiates a state another solver of
    the same structure produced: backward_from_state() depends on the data
    and the state only (the JAX interface relies on this)."""
    qp = _dense_qp(batch_size=2, seed=9)
    producer = _dense_solver(qp)
    producer.solve()
    _assert_solved(producer)
    state = _saved_state(producer)
    grad_x = to_wp(np.ones((2, 5)))
    expected = _grads_to_numpy(producer.backward_from_state(**state, grad_x=grad_x), ("P", "c", "b"))

    fresh = _dense_solver(qp)
    got = _grads_to_numpy(fresh.backward_from_state(**state, grad_x=grad_x), ("P", "c", "b"))
    for name in expected:
        np.testing.assert_allclose(got[name], expected[name], rtol=1e-10, atol=1e-12, err_msg=name)
    with pytest.raises(RuntimeError, match="prior solve"):
        fresh.backward(grad_x=grad_x)          # backward() still needs the cached factor


def test_lifecycle_and_cached_factor():
    qp = _dense_qp(batch_size=2, seed=9)
    solver = _dense_solver(qp)
    solver.solve()
    state = _saved_state(solver)
    # An update() does not invalidate the saved state; the caller restores data.
    solver.update(c=to_wp(qp["c"] + 1.0))
    with pytest.raises(RuntimeError, match="solve"):
        solver.backward(grad_x=to_wp(np.ones((2, 5))))
    solver.update(c=to_wp(qp["c"]))
    solver.backward_from_state(**state, grad_x=to_wp(np.ones((2, 5))))
    # The cached factor now belongs to that state, not to the latest solve.
    with pytest.raises(RuntimeError, match="backward_from_state"):
        solver.backward(grad_x=to_wp(np.ones((2, 5))))
    solver.solve()
    solver.backward(grad_x=to_wp(np.ones((2, 5))))

    no_grad = DenseSolver()
    no_grad.settings.verbose = False
    no_grad.setup(**to_wp_dict(qp))
    no_grad.solve()
    with pytest.raises(RuntimeError, match="enable_grad"):
        no_grad.backward_from_state(**state)


def test_checkpoint_api_is_gone():
    assert not hasattr(cupiqp, "SolutionCheckpoint")
    for name in ("new_checkpoint", "save_checkpoint", "backward_from_checkpoint"):
        assert not hasattr(DenseSolver, name), name


# ---------------------------------------------------------------------------
# Failure policy and invalid problems
# ---------------------------------------------------------------------------
def test_follows_gradient_on_failure():
    P = np.eye(2)
    c = np.array([1.0, 1.0])
    A = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 1.0]])            # problem 1 infeasible
    solver = DenseSolver()
    _settings(solver, 0, np.float64)
    solver.settings.gradient_on_failure = "zero"
    solver.setup(to_wp(P), to_wp(c), A=to_wp(A), b=to_wp(b))
    solver.solve()
    status = solver.result.info.to_host().status
    assert status[0] == Status.CUPIQP_SOLVED and status[1] != Status.CUPIQP_SOLVED
    state = _saved_state(solver)
    grads = solver.backward_from_state(**state, grad_x=to_wp(np.ones((2, 2))))
    grad_c = grads.c.numpy()
    assert np.all(grad_c[1] == 0.0)
    np.testing.assert_allclose(grad_c[0], [0.0, -1.0], atol=1e-4)
    # The given status decides, not the solver's latest one.
    forged = dict(state, status=to_wp(np.full(2, Status.CUPIQP_SOLVED.value, dtype=np.int32)))
    solver.backward_from_state(**forged, grad_x=to_wp(np.ones((2, 2))))
    assert list(solver.grad_valid.numpy()) == [1, 1]


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_grad_valid_mask_for_a_failed_factorization(backend):
    """A state whose regularization is NaN makes that problem's KKT
    factorization or adjoint solve fail: its gradients are zero under every
    policy, the others are untouched, and grad_valid reports it. (A NaN slack
    would not do: the regularization kernels floor the slacks with a max,
    which swallows NaN.)"""
    qp = _dense_qp(batch_size=3, seed=12)
    solver = _dense_solver(qp) if backend == "dense" else _sparse_solver(qp)
    solver.solve()
    _assert_solved(solver)
    grad_x = to_wp(np.ones((3, 5)))
    reference = _grads_to_numpy(solver.backward(grad_x=grad_x), DENSE_GRADS)
    assert (solver.grad_valid.numpy() == 1).all()

    state = _saved_state(solver)
    state["rho"] = _with_entry(state["rho"], 1, np.nan)     # poison problem 1 only
    for policy in ("differentiate", "zero"):
        solver.settings.gradient_on_failure = policy
        grads = _grads_to_numpy(solver.backward_from_state(**state, grad_x=grad_x), DENSE_GRADS)
        assert list(solver.grad_valid.numpy()) == [1, 0, 1]
        for name in DENSE_GRADS:
            if grads[name].shape[0] == 3:
                assert np.all(grads[name][1] == 0.0), name
                assert np.isfinite(grads[name]).all(), name
                np.testing.assert_allclose(grads[name][[0, 2]], reference[name][[0, 2]],
                                           atol=2e-3, rtol=2e-3, err_msg=name)
    solver.solve()
    solver.backward(grad_x=grad_x)
    assert (solver.grad_valid.numpy() == 1).all()


def test_dense_indefinite_regularization_is_a_factor_failure():
    """The dense Cholesky rejects an indefinite matrix: a state with a large
    negative rho fails to factorize and its gradients are zeroed."""
    qp = _dense_qp(batch_size=2, seed=13)
    solver = _dense_solver(qp)
    solver.solve()
    state = _saved_state(solver)
    state["rho"] = _with_entry(state["rho"], 0, -1e3)
    grads = _grads_to_numpy(solver.backward_from_state(**state, grad_x=to_wp(np.ones((2, 5)))), ("c", "P"))
    assert list(solver.grad_valid.numpy()) == [0, 1]
    assert np.all(grads["c"][0] == 0.0) and np.all(grads["P"][0] == 0.0)
    assert np.isfinite(grads["c"]).all()


def test_structural_settings_are_frozen_at_setup():
    qp = _dense_qp()
    solver = _dense_solver(qp, preconditioner_iter=10)
    solver.solve()
    solver.settings.max_iter = 50                            # numerical: fine
    solver.solve()
    state = _saved_state(solver)
    solver.settings.preconditioner_iter = 0
    with pytest.raises(RuntimeError, match="preconditioner_iter"):
        solver.solve()
    with pytest.raises(RuntimeError, match="preconditioner_iter"):
        solver.update(c=to_wp(qp["c"]))
    with pytest.raises(RuntimeError, match="preconditioner_iter"):
        solver.backward(grad_x=to_wp(np.ones((1, 5))))
    with pytest.raises(RuntimeError, match="preconditioner_iter"):
        solver.backward_from_state(**state, grad_x=to_wp(np.ones((1, 5))))
