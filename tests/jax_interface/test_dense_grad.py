"""Gradients through cupiqp.jax.DenseSolver.solve: finite-difference checks
for every input and every solution field, shared inputs summed over the
batch, jit/eager equality, purity across call histories, the failure policy,
non-differentiable diagnostics, lax.scan, grad(vmap) and vmap(grad)."""

import os

import numpy as np
import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import warp as wp

import cupiqp
from cupiqp import Status
from cupiqp.jax import DenseSolver, QPSolution

from tests.helpers import primal_infeasible, random_qp

pytestmark = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="a CUDA JAX device is required"
)

SOLUTION_FIELDS = ("x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu")
DATA_FIELDS = ("P", "c", "A", "b", "G", "h_u", "h_l", "x_u", "x_l")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tight_solver(preconditioner_iter=10) -> DenseSolver:
    """A solver converged tightly enough for 1e-6 finite differences."""
    solver = DenseSolver()
    solver.settings.eps_abs = 1e-11
    solver.settings.eps_rel = 1e-11
    solver.settings.preconditioner_iter = preconditioner_iter
    return solver


def active_qp(seed=0, n=6, p=2, m=4) -> dict:
    """A random QP whose solution has strictly active inequalities and box
    bounds on both sides, with strict complementarity: the bounds cut off
    the equality-constrained optimum by a margin, the other bounds stay
    far. Every multiplier and slack then has a well-defined gradient."""
    qp = random_qp(seed, n=n, p=p, m=m)
    # The equality-constrained optimum, from the KKT system.
    P, c, A = qp["P"], qp["c"], qp["A"]
    kkt = np.block([[P, A.T], [A, np.zeros((p, p))]])
    x_star = np.linalg.solve(kkt, np.concatenate([-c, qp["b"]]))[:n]
    Gx = qp["G"] @ x_star
    qp["h_u"][0] = Gx[0] - 0.2                       # strictly active upper row
    qp["h_l"][1] = Gx[1] + 0.2                       # strictly active lower row
    qp["x_u"][0] = x_star[0] - 0.2                   # strictly active upper bound
    qp["x_l"][1] = x_star[1] + 0.2                   # strictly active lower bound
    return qp


def to_jax(qp: dict) -> dict:
    return {k: jnp.asarray(v) for k, v in qp.items()}


def directional_fd(f, args: dict, name: str, direction: np.ndarray, eps=1e-6) -> float:
    """Central finite difference of ``f(args)`` along ``direction`` in ``args[name]``."""
    plus = dict(args, **{name: args[name] + eps * direction})
    minus = dict(args, **{name: args[name] - eps * direction})
    return (float(f(**plus)) - float(f(**minus))) / (2 * eps)


def symmetric(R: np.ndarray) -> np.ndarray:
    return 0.5 * (R + R.T)


# ---------------------------------------------------------------------------
# Finite-difference checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preconditioner_iter", [0, 10])
@pytest.mark.parametrize("name", DATA_FIELDS)
def test_gradient_wrt_each_input(name, preconditioner_iter):
    """d loss / d <input> against a directional finite difference, with a
    loss on all solution fields. P is perturbed symmetrically."""
    solver = tight_solver(preconditioner_iter)
    qp = active_qp(1)
    args = to_jax(qp)
    rng = np.random.default_rng(7)
    fields = SOLUTION_FIELDS
    weights = {f: rng.standard_normal(1) for f in fields}

    def loss(**kw):
        sol = solver.solve(**kw)
        assert int(sol.status) == Status.CUPIQP_SOLVED
        return sum(float(weights[f][0]) * jnp.sum(getattr(sol, f) ** 2) for f in fields)

    direction = rng.standard_normal(np.shape(qp[name]))
    if name == "P":
        direction = symmetric(direction)
    grad = jax.grad(lambda v: loss(**dict(args, **{name: v})))(args[name])
    analytic = float(jnp.sum(grad * direction))
    numeric = directional_fd(loss, args, name, direction)
    # An inactive bound has a zero gradient; the finite difference of the
    # loss is then dominated by the solve tolerance, so compare absolutely
    # on the scale of the loss.
    scale = abs(float(loss(**args)))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6 * scale)


@pytest.mark.parametrize("field", SOLUTION_FIELDS)
def test_cotangent_on_each_solution_field(field):
    """A loss on one solution field at a time, differentiated w.r.t. c."""
    solver = tight_solver()
    qp = active_qp(2)
    args = to_jax(qp)
    rng = np.random.default_rng(3)
    width = np.asarray(getattr(solver.solve(**args), field)).shape
    weights = jnp.asarray(rng.standard_normal(width))

    def loss(**kw):
        return jnp.sum(getattr(solver.solve(**kw), field) * weights)

    direction = np.random.default_rng(4).standard_normal(qp["c"].shape)
    grad = jax.grad(lambda c: loss(**dict(args, c=c)))(args["c"])
    numeric = directional_fd(loss, args, "c", direction)
    np.testing.assert_allclose(float(jnp.sum(grad * direction)), numeric, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "keep",
    [
        ("A", "b"),                                   # equality only
        ("x_l", "x_u"),                               # box only
        ("G", "h_l", "h_u"),                          # inequality only
        ("G", "h_u"),                                 # one-sided inequality
        ("G", "h_l", "x_u"),                          # one-sided inequality and box
        (),                                           # unconstrained
    ],
)
def test_gradient_for_block_combinations(keep):
    """Every given input against a directional finite difference, for
    problems with only some constraint blocks. Absent blocks give
    zero-width solution fields that a loss may still touch."""
    solver = tight_solver()
    qp = {k: v for k, v in active_qp(20).items() if k in ("P", "c") + keep}
    args = to_jax(qp)
    rng = np.random.default_rng(21)
    fields = SOLUTION_FIELDS
    weights = {f: float(rng.standard_normal()) for f in fields}

    def loss(**kw):
        sol = solver.solve(**kw)
        assert int(sol.status) == Status.CUPIQP_SOLVED
        return sum(w * jnp.sum(getattr(sol, f) ** 2) for f, w in weights.items())

    scale = abs(float(loss(**args)))
    for name in qp:
        direction = rng.standard_normal(np.shape(qp[name]))
        if name == "P":
            direction = symmetric(direction)
        grad = jax.grad(lambda v: loss(**dict(args, **{name: v})))(args[name])
        assert grad.shape == args[name].shape
        numeric = directional_fd(loss, args, name, direction)
        np.testing.assert_allclose(float(jnp.sum(grad * direction)), numeric,
                                   rtol=1e-5, atol=1e-6 * scale, err_msg=name)


def test_shared_input_gradient_is_the_batch_sum():
    solver = tight_solver()
    qps = [active_qp(seed) for seed in (5, 6, 7)]
    base = to_jax(qps[0])
    cs = jnp.stack([jnp.asarray(q["c"]) for q in qps])

    def loss(P, cs):
        return jnp.sum(solver.solve(P, cs, **{k: base[k] for k in base if k not in ("P", "c")}).x ** 2)

    g_P, g_cs = jax.grad(loss, argnums=(0, 1))(base["P"], cs)
    assert g_P.shape == base["P"].shape and g_cs.shape == cs.shape
    per_problem = [
        jax.grad(lambda P, c: loss(P, c[None]), argnums=(0, 1))(base["P"], cs[i]) for i in range(3)
    ]
    # The batched and the single solves reach the optimum within the solve
    # tolerance, not bitwise: compare the gradients to that tolerance.
    np.testing.assert_allclose(np.asarray(g_P), sum(np.asarray(g[0]) for g in per_problem), rtol=1e-6, atol=1e-9)
    for i in range(3):
        np.testing.assert_allclose(np.asarray(g_cs[i]), np.asarray(per_problem[i][1]), rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def test_jit_grad_matches_eager_bitwise_and_is_pure():
    solver = tight_solver()
    args = to_jax(active_qp(8))
    other = to_jax(active_qp(9))
    infeasible = to_jax(primal_infeasible(random_qp(10, n=6, p=2, m=4)))

    def loss(c):
        return jnp.sum(solver.solve(**dict(args, c=c)).x ** 2)

    eager = jax.grad(loss)(args["c"])
    jitted = jax.jit(jax.grad(loss))(args["c"])
    assert np.array_equal(np.asarray(eager), np.asarray(jitted))
    for history in (other, infeasible):
        jax.grad(lambda c: jnp.sum(solver.solve(**dict(history, c=c)).x ** 2))(history["c"])
    assert np.array_equal(np.asarray(jax.grad(loss)(args["c"])), np.asarray(eager))


def test_grad_of_vmap_and_vmap_of_grad():
    solver = tight_solver()
    qps = [active_qp(seed) for seed in (11, 12, 13, 14)]
    base = to_jax(qps[0])
    cs = jnp.stack([jnp.asarray(q["c"]) for q in qps])

    def loss(c):
        return jnp.sum(solver.solve(**dict(base, c=c)).x ** 2)

    expected = np.stack([np.asarray(jax.grad(loss)(cs[i])) for i in range(4)])
    g_vmap_grad = jax.jit(jax.vmap(jax.grad(loss)))(cs)
    g_grad_vmap = jax.jit(jax.grad(lambda cs: jnp.sum(jax.vmap(loss)(cs))))(cs)
    # The batched and the single solves differ by the solve tolerance,
    # which the gradient at active constraints amplifies.
    np.testing.assert_allclose(np.asarray(g_vmap_grad), expected, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(g_grad_vmap), expected, rtol=1e-5, atol=1e-9)
    for f in (jax.jit(jax.vmap(jax.grad(loss))), jax.jit(jax.grad(lambda cs: jnp.sum(jax.vmap(loss)(cs))))):
        assert f.lower(cs).as_text().count("custom_call") == 2        # one forward, one backward


def test_grad_through_scan_rollout():
    solver = tight_solver()
    args = to_jax(active_qp(15))

    def rollout(c):
        def step(carry, _):
            x = solver.solve(**dict(args, c=carry)).x
            return carry + 0.1 * x, jnp.sum(x ** 2)
        _, losses = jax.lax.scan(step, c, None, length=3)
        return jnp.sum(losses)

    grad = jax.jit(jax.grad(rollout))(args["c"])
    direction = np.random.default_rng(16).standard_normal(args["c"].shape)
    numeric = directional_fd(lambda c: rollout(c), {"c": args["c"]}, "c", direction)
    np.testing.assert_allclose(float(jnp.sum(grad * direction)), numeric, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def test_diagnostics_have_no_gradient():
    solver = tight_solver()
    args = to_jax(active_qp(17))
    for field in ("primal_res", "dual_res"):
        grad = jax.grad(lambda c: jnp.sum(getattr(solver.solve(**dict(args, c=c)), field)))(args["c"])
        assert np.all(np.asarray(grad) == 0.0)
    grad = jax.grad(lambda c: jnp.sum(solver.solve(**dict(args, c=c)).iterations.astype(jnp.float64)))(args["c"])
    assert np.all(np.asarray(grad) == 0.0)


def test_gradient_on_failure_policy():
    """A batch with one infeasible problem: 'zero' zeroes its gradient and
    leaves the others untouched, whatever was solved before."""
    good = random_qp(18, n=6, p=2, m=4)
    bad = primal_infeasible(good)
    base = to_jax(good)
    stacked = {k: jnp.stack([jnp.asarray(good[k]), jnp.asarray(bad[k])]) for k in ("G", "h_l", "h_u")}
    rest = {k: base[k] for k in base if k not in stacked}

    def loss(G, h_l, h_u):
        sol = solver.solve(G=G, h_l=h_l, h_u=h_u, **rest)
        return jnp.sum(sol.x ** 2)

    solver = tight_solver()
    statuses = np.asarray(solver.solve(**stacked, **rest).status)
    assert statuses[0] == Status.CUPIQP_SOLVED and statuses[1] != Status.CUPIQP_SOLVED
    single = np.asarray(jax.grad(lambda h_u: jnp.sum(solver.solve(**dict(base, h_u=h_u)).x ** 2))(base["h_u"]))
    solver.settings.gradient_on_failure = "zero"
    grad_zero = np.asarray(jax.grad(loss, argnums=2)(stacked["G"], stacked["h_l"], stacked["h_u"]))
    assert np.all(grad_zero[1] == 0.0)
    np.testing.assert_allclose(grad_zero[0], single, rtol=1e-6, atol=1e-9)
    # The same gradient after the cached solvers handled other problems,
    # feasible and infeasible, forward and backward.
    other = to_jax(random_qp(22, n=6, p=2, m=4))
    jax.grad(lambda h_u: jnp.sum(solver.solve(**dict(other, h_u=h_u)).x ** 2))(other["h_u"])
    swapped = {k: v[::-1] for k, v in stacked.items()}
    jax.grad(loss, argnums=2)(swapped["G"], swapped["h_l"], swapped["h_u"])
    again = np.asarray(jax.grad(loss, argnums=2)(stacked["G"], stacked["h_l"], stacked["h_u"]))
    assert np.array_equal(again, grad_zero)
    solver.settings.gradient_on_failure = "differentiate"
    grad_diff = np.asarray(jax.grad(loss, argnums=2)(stacked["G"], stacked["h_l"], stacked["h_u"]))
    np.testing.assert_allclose(grad_diff[0], single, rtol=1e-6, atol=1e-9)
    assert np.all(np.isfinite(grad_diff[1]))


def test_backward_settings_are_part_of_the_signature():
    """Changing a backward setting compiles a new solve (forward and
    backward), like any other setting; the old one stays valid."""
    from cupiqp.interfaces.jax.base import cache_info   # test instrumentation, not public API
    solver = tight_solver()
    args = to_jax(active_qp(19, n=7, p=2, m=3))     # a shape no other test uses
    loss = lambda c: jnp.sum(solver.solve(**dict(args, c=c)).x ** 2)
    first = np.asarray(jax.grad(loss)(args["c"]))
    before = cache_info()
    solver.settings.gradient_on_failure = "zero"
    np.testing.assert_array_equal(np.asarray(jax.grad(loss)(args["c"])), first)   # all solved: same gradient
    after = cache_info()
    assert after["signatures"] == before["signatures"] + 1
    assert after["live_solvers"] == before["live_solvers"] + 2


def test_gradient_equals_the_core_backward_from_state():
    """The JAX gradient is the core's backward_from_state() of the same
    solution with the same cotangents, for every input: the interface adds
    no error of its own. (The core's own accuracy, including the effect of
    its regularized adjoint, is a property of the core.)"""
    from tests.helpers import to_wp_dict
    qp = active_qp(23)
    args = to_jax(qp)
    solver = tight_solver()
    sol, vjp = jax.vjp(lambda kw: solver.solve(**kw), args)
    rng = np.random.default_rng(24)
    weights = {f: rng.standard_normal(np.asarray(getattr(sol, f)).shape) for f in SOLUTION_FIELDS}
    (jax_grads,) = vjp(sol._replace(**{f: jnp.asarray(w) for f, w in weights.items()}))

    # The core along the interface's path: setup, update, solve.
    core = cupiqp.DenseSolver()
    core.settings = tight_solver().settings
    core.settings.enable_grad = True
    data = to_wp_dict(qp)
    core.setup(**data)
    core.update(**data)
    core.solve()
    result = core.result
    for f in SOLUTION_FIELDS:                     # same solution, bitwise
        assert np.array_equal(getattr(result, f).numpy()[0], np.asarray(getattr(sol, f))), f
    core_grads = core.backward_from_state(
        **{f: getattr(result, f) for f in SOLUTION_FIELDS},
        rho=result.info.rho, delta=result.info.delta, status=result.info.status,
        **{f"grad_{f}": wp.array(w[None], dtype=wp.float64) for f, w in weights.items()},
    )
    for name in DATA_FIELDS:
        np.testing.assert_array_equal(np.asarray(jax_grads[name]), getattr(core_grads, name).numpy()[0],
                                      err_msg=name)
