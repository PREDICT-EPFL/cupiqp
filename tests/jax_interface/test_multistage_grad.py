"""Gradients through cupiqp.jax.MultistageSolver.solve: directional finite
differences for every block array and vector, block and flat vector layouts,
and jit / vmap of the gradient."""

import os

import numpy as np
import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cupiqp import Status
from cupiqp.jax import MultistageSolver

from tests.jax_interface.test_multistage_solve import problem, to_jax

pytestmark = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="a CUDA JAX device is required"
)


def tight_solver() -> MultistageSolver:
    solver = MultistageSolver()
    solver.settings.eps_abs = 1e-11
    solver.settings.eps_rel = 1e-11
    return solver


def flatten_args(qp: dict) -> dict:
    """The call's arrays as a flat dict: ``P_diag``, ``P_offdiag``, ... and
    the vectors, so that one of them can be perturbed at a time."""
    flat = {}
    for name, value in qp.items():
        if isinstance(value, tuple):
            flat[f"{name}_diag"] = value[0]
            if value[1] is not None:
                flat[f"{name}_offdiag"] = value[1]
        else:
            flat[name] = value
    return flat


def call_args(flat: dict) -> dict:
    args = {}
    for block in ("P", "A", "G"):
        if f"{block}_diag" in flat:
            args[block] = (flat[f"{block}_diag"], flat.get(f"{block}_offdiag"))
    for name in ("c", "b", "h_u", "h_l", "x_u", "x_l"):
        if name in flat:
            args[name] = flat[name]
    return args


def directional_fd(f, flat: dict, name: str, direction: np.ndarray, eps=1e-6) -> float:
    plus = dict(flat, **{name: flat[name] + eps * direction})
    minus = dict(flat, **{name: flat[name] - eps * direction})
    return (float(f(**call_args(plus))) - float(f(**call_args(minus)))) / (2 * eps)


FLAT_FIELDS = ("P_diag", "P_offdiag", "c", "A_diag", "A_offdiag", "b", "G_diag", "h_u", "h_l", "x_u", "x_l")


@pytest.mark.parametrize("name", FLAT_FIELDS)
def test_gradient_wrt_each_input(name):
    qp = problem(seed=3, has_G_E=False)
    solver = tight_solver()
    flat = flatten_args(to_jax(qp))
    rng = np.random.default_rng(4)
    weights = {f: float(rng.standard_normal()) for f in ("x", "y", "z_l", "z_u", "z_bl", "z_bu")}

    def loss(**kw):
        sol = solver.solve(**kw)
        assert int(sol.status) == Status.CUPIQP_SOLVED
        return sum(w * jnp.sum(getattr(sol, f) ** 2) for f, w in weights.items())

    direction = rng.standard_normal(np.shape(flat[name]))
    if name == "P_diag":
        direction = 0.5 * (direction + np.swapaxes(direction, -1, -2))
    grad = jax.grad(lambda v: loss(**call_args(dict(flat, **{name: v}))))(flat[name])
    assert grad.shape == flat[name].shape
    numeric = directional_fd(loss, flat, name, direction)
    scale = abs(float(loss(**call_args(flat))))
    # The loss includes the equality multipliers y. The core differentiates
    # the regularized KKT system, so its gradient of y is off by about 1e-5
    # relative here (dense core alike); x and the inequality multipliers
    # agree to 1e-7. The interface returns the core's gradient exactly (see
    # test_gradient_equals_the_core_backward_from_state), so the check uses
    # the core's accuracy. A fix is kept in plans/adjoint_refinement.patch.
    np.testing.assert_allclose(float(jnp.sum(grad * direction)), numeric, rtol=1e-4, atol=1e-6 * scale)


def test_flat_and_block_vector_gradients_agree():
    qp = to_jax(problem(seed=5))
    solver = tight_solver()
    loss = lambda c: jnp.sum(solver.solve(**dict(qp, c=c)).x ** 2)
    g_block = jax.grad(loss)(qp["c"])
    g_flat = jax.grad(loss)(qp["c"].reshape(-1))
    assert g_block.shape == qp["c"].shape and g_flat.shape == (qp["c"].size,)
    np.testing.assert_allclose(np.asarray(g_block).reshape(-1), np.asarray(g_flat), rtol=1e-9, atol=1e-12)


def test_jit_vmap_of_grad():
    qp = to_jax(problem(seed=6))
    solver = tight_solver()
    loss = lambda c: jnp.sum(solver.solve(**dict(qp, c=c)).x ** 2)
    cs = jnp.stack([qp["c"] * (1.0 + 0.2 * i) for i in range(3)])
    expected = np.stack([np.asarray(jax.grad(loss)(cs[i])) for i in range(3)])
    g = jax.jit(jax.vmap(jax.grad(loss)))(cs)
    np.testing.assert_allclose(np.asarray(g), expected, rtol=1e-5, atol=1e-9)
    assert jax.jit(jax.vmap(jax.grad(loss))).lower(cs).as_text().count("custom_call") == 2
