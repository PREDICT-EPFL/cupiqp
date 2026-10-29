"""Gradients through cupiqp.jax.SparseSolver.solve: directional finite
differences for the stored values and the vectors, shared values summed over
the batch, and jit / vmap of the gradient."""

import numpy as np
import pytest
import scipy.sparse as sp

# Configures JAX: must come before any import of JAX.
from tests.jax_interface.jax_helpers import DATA_FIELDS, directional_fd, jax, jnp, patterns, requires_gpu, values

from cupiqp import Status
from cupiqp.jax import SparseSolver
from tests.helpers import random_sparse_qp, tight

pytestmark = requires_gpu


def tight_solver(qp) -> SparseSolver:
    return tight(SparseSolver(**patterns(qp)))


def symmetric_values(M: sp.csr_matrix, direction: np.ndarray) -> np.ndarray:
    """``direction`` (values of M's pattern) symmetrized on the pattern."""
    D = sp.csr_matrix((direction, M.indices, M.indptr), shape=M.shape)
    S = 0.5 * (D + D.T)
    S.sort_indices()
    assert np.array_equal(S.indices, M.indices) and np.array_equal(S.indptr, M.indptr)
    return S.data


@pytest.mark.parametrize("name", DATA_FIELDS)
def test_gradient_wrt_each_input(name):
    qp = random_sparse_qp(0)
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
    qp = random_sparse_qp(2)
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
