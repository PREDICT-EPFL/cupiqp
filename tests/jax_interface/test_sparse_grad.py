"""Gradients through cupiqp.jax.SparseSolver.solve: directional finite
differences for the stored values and the vectors, shared values summed over
the batch, and jit / vmap of the gradient."""

import os

import numpy as np
import pytest
import scipy.sparse as sp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cupiqp import Status
from cupiqp.jax import SparseSolver

from tests.jax_interface.test_sparse_solve import patterns, sparse_qp, values

pytestmark = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="a CUDA JAX device is required"
)

DATA_FIELDS = ("P", "c", "A", "b", "G", "h_u", "h_l", "x_u", "x_l")


def tight_solver(qp) -> SparseSolver:
    solver = SparseSolver(**patterns(qp))
    solver.settings.eps_abs = 1e-11
    solver.settings.eps_rel = 1e-11
    return solver


def symmetric_values(M: sp.csr_matrix, direction: np.ndarray) -> np.ndarray:
    """``direction`` (values of M's pattern) symmetrized on the pattern."""
    D = sp.csr_matrix((direction, M.indices, M.indptr), shape=M.shape)
    S = 0.5 * (D + D.T)
    S.sort_indices()
    assert np.array_equal(S.indices, M.indices) and np.array_equal(S.indptr, M.indptr)
    return S.data


def directional_fd(f, args: dict, name: str, direction: np.ndarray, eps=1e-6) -> float:
    plus = dict(args, **{name: args[name] + eps * direction})
    minus = dict(args, **{name: args[name] - eps * direction})
    return (float(f(**plus)) - float(f(**minus))) / (2 * eps)


@pytest.mark.parametrize("name", DATA_FIELDS)
def test_gradient_wrt_each_input(name):
    qp = sparse_qp(0)
    solver = tight_solver(qp)
    args = values(qp)
    rng = np.random.default_rng(1)
    weights = {f: float(rng.standard_normal()) for f in ("x", "y", "z_l", "z_u", "z_bl", "z_bu")}

    def loss(**kw):
        sol = solver.solve(**kw)
        assert int(sol.status) == Status.CUPIQP_SOLVED
        return sum(w * jnp.sum(getattr(sol, f) ** 2) for f, w in weights.items())

    direction = rng.standard_normal(np.shape(args[name]))
    if name == "P":
        direction = symmetric_values(qp["P"], direction)
    grad = jax.grad(lambda v: loss(**dict(args, **{name: v})))(args[name])
    assert grad.shape == args[name].shape
    numeric = directional_fd(loss, args, name, direction)
    scale = abs(float(loss(**args)))
    np.testing.assert_allclose(float(jnp.sum(grad * direction)), numeric, rtol=1e-5, atol=1e-6 * scale)


def test_shared_values_gradient_is_the_batch_sum_and_vmap_of_grad():
    qp = sparse_qp(2)
    solver = tight_solver(qp)
    args = values(qp)
    cs = jnp.stack([args["c"] * (1.0 + 0.3 * i) for i in range(3)])
    rest = {k: args[k] for k in args if k not in ("P", "c")}

    def loss(P, c):
        return jnp.sum(solver.solve(P, c, **rest).x ** 2)

    g_P, g_cs = jax.grad(loss, argnums=(0, 1))(args["P"], cs)
    singles = [jax.grad(loss, argnums=(0, 1))(args["P"], cs[i]) for i in range(3)]
    np.testing.assert_allclose(np.asarray(g_P), sum(np.asarray(g[0]) for g in singles), rtol=1e-5, atol=1e-9)
    for i in range(3):
        np.testing.assert_allclose(np.asarray(g_cs[i]), np.asarray(singles[i][1]), rtol=1e-5, atol=1e-9)
    g_vmap = jax.jit(jax.vmap(lambda c: jax.grad(loss, argnums=1)(args["P"], c)))(cs)
    np.testing.assert_allclose(np.asarray(g_vmap), np.asarray(g_cs), rtol=1e-5, atol=1e-9)
