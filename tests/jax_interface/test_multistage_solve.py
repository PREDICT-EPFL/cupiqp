"""Tests for cupiqp.jax.MultistageSolver: agreement with the core multistage
solver for every combination of off-diagonal blocks, block and flat vector
layouts, shared and batched arrays, jit and vmap, and input validation."""

import itertools
import os

import numpy as np
import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import cupiqp
from cupiqp import Status
from cupiqp.jax import MultistageSolver, QPSolution

from tests.helpers import to_wp

pytestmark = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="a CUDA JAX device is required"
)

FIELDS = QPSolution._fields
VARIABLE_FIELDS = ("x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu")
TOL = dict(rtol=1e-9, atol=1e-10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def problem(N=4, d=3, rA=2, rG=2, B=None, has_P_E=True, has_A_E=True, has_G_E=False, seed=0) -> dict:
    """A random strictly convex, feasible multistage QP as host arrays, in the
    core's argument layout: matrices as (diag, offdiag) pairs, vectors in the
    block layout. ``B`` adds a leading batch axis to every array."""
    rng = np.random.default_rng(seed)
    lead = () if B is None else (B,)
    P_D = rng.standard_normal(lead + (N, d, d))
    P_D = P_D @ np.swapaxes(P_D, -1, -2) + 3 * d * np.eye(d)
    P_E = 0.3 * rng.standard_normal(lead + (N - 1, d, d)) if has_P_E else None
    A_D = rng.standard_normal(lead + (N, rA, d))
    A_E = rng.standard_normal(lead + (N, rA, d)) if has_A_E else None
    G_D = rng.standard_normal(lead + (N, rG, d))
    G_E = rng.standard_normal(lead + (N, rG, d)) if has_G_E else None
    # b = A x0 and G x0 in the interior, so that the problem is feasible.
    x0 = rng.uniform(-0.3, 0.3, lead + (N, d))

    def bidiag_apply(D, E):
        out = np.einsum("...krd,...kd->...kr", D, x0)
        if E is None:
            return out
        shifted = np.einsum("...krd,...kd->...kr", E, x0)
        zero = np.zeros(out.shape[:-2] + (1, out.shape[-1]))
        return np.concatenate([out, zero], axis=-2) + np.concatenate([zero, shifted], axis=-2)

    Gx = bidiag_apply(G_D, G_E)
    return dict(
        P=(P_D, P_E), c=rng.standard_normal(lead + (N, d)),
        A=(A_D, A_E), b=bidiag_apply(A_D, A_E),
        G=(G_D, G_E), h_l=Gx - 1.0, h_u=Gx + 0.5,
        x_l=-np.ones(lead + (N, d)), x_u=np.ones(lead + (N, d)),
    )


def convert(qp: dict, to) -> dict:
    """Apply ``to`` to every array of ``qp``, keeping pairs and ``None``."""
    out = {}
    for name, value in qp.items():
        if value is None:
            out[name] = None
        elif isinstance(value, tuple):
            out[name] = tuple(None if v is None else to(v) for v in value)
        else:
            out[name] = to(value)
    return out


def to_jax(qp: dict) -> dict:
    return convert(qp, jnp.asarray)


def core_solve(qp: dict) -> dict:
    solver = cupiqp.MultistageSolver()
    solver.setup(**{k: v for k, v in convert(qp, to_wp).items() if v is not None})
    solver.solve()
    result = solver.result
    out = {name: getattr(result, name).numpy() for name in VARIABLE_FIELDS}
    out["status"] = result.info.status.numpy()
    out["iterations"] = result.info.iter.numpy()
    out["primal_res"] = result.info.primal_res.numpy()
    out["dual_res"] = result.info.dual_res.numpy()
    return out


def assert_close(sol: QPSolution, ref: dict, unbatched=False) -> None:
    for name in FIELDS:
        got = np.asarray(getattr(sol, name))
        expected = ref[name][0] if unbatched else ref[name]
        assert got.shape == expected.shape, name
        if name in ("status", "iterations"):
            np.testing.assert_array_equal(got, expected, err_msg=name)
        elif got.size:
            np.testing.assert_allclose(got, expected, err_msg=name, **TOL)


def assert_bitwise(a: QPSolution, b: QPSolution) -> None:
    for name in FIELDS:
        assert np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))), name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("has_P_E,has_A_E,has_G_E", list(itertools.product([True, False], repeat=3)))
def test_matches_core_for_every_offdiag_combination(has_P_E, has_A_E, has_G_E):
    qp = problem(has_P_E=has_P_E, has_A_E=has_A_E, has_G_E=has_G_E, seed=1)
    sol = MultistageSolver().solve(**to_jax(qp))
    assert int(sol.status) == Status.CUPIQP_SOLVED
    assert sol.x.shape == (12,)
    assert sol.y.shape == ((4 + has_A_E) * 2,) and sol.z_l.shape == ((4 + has_G_E) * 2,)
    assert_close(sol, core_solve(qp), unbatched=True)


def test_block_and_flat_vectors_agree():
    qp = to_jax(problem(seed=2))
    flat = dict(qp, **{k: qp[k].reshape(-1) for k in ("c", "b", "h_l", "h_u", "x_l", "x_u")})
    solver = MultistageSolver()
    assert_bitwise(solver.solve(**qp), solver.solve(**flat))


def test_jit_eager_and_one_custom_call():
    qp = to_jax(problem(seed=3))
    solver = MultistageSolver()
    eager = solver.solve(**qp)
    jitted = jax.jit(lambda **kw: solver.solve(**kw))(**qp)
    assert_bitwise(eager, jitted)
    hlo = jax.jit(lambda **kw: solver.solve(**kw).x).lower(**qp).as_text()
    assert hlo.count("custom_call") == 1


def test_batched_and_shared_arrays():
    """Batched P_diag and c (block layout) and x_l (flat layout) with the
    rest shared, b in the flat layout: agrees with the core batched solve."""
    shared = problem(seed=4)
    batched = problem(B=3, seed=4)
    qp = dict(shared, P=(batched["P"][0], shared["P"][1]), c=batched["c"],
              b=shared["b"].reshape(-1),
              x_l=np.ascontiguousarray(np.broadcast_to(shared["x_l"].reshape(-1), (3, 12))))
    sol = jax.jit(lambda **kw: MultistageSolver().solve(**kw))(**to_jax(qp))
    assert sol.x.shape == (3, 12) and sol.status.shape == (3,)
    assert np.all(np.asarray(sol.status) == Status.CUPIQP_SOLVED)
    assert_close(sol, core_solve(qp))


def test_vmap_is_one_batched_solve():
    qp = to_jax(problem(seed=5))
    cs = jnp.stack([qp["c"] * (i + 1) for i in range(4)])     # (4, N, d)
    solver = MultistageSolver()

    def solve_one(c):
        return solver.solve(**dict(qp, c=c))

    vmapped = jax.jit(jax.vmap(solve_one))
    sol = vmapped(cs)
    assert sol.x.shape == (4, 12)
    assert_bitwise(sol, solver.solve(**dict(qp, c=cs)))
    assert vmapped.lower(cs).as_text().count("custom_call") == 1


def test_minimal_problem_and_absent_blocks():
    qp = problem(seed=6)
    minimal = {"P": (qp["P"][0], None), "c": qp["c"]}
    sol = MultistageSolver().solve(**to_jax(minimal))
    assert int(sol.status) == Status.CUPIQP_SOLVED
    assert sol.y.shape == (0,) and sol.z_l.shape == (0,) and sol.z_bl.shape == (0,)
    assert_close(sol, core_solve(minimal), unbatched=True)


def test_result_does_not_depend_on_earlier_calls():
    solver = MultistageSolver()
    target = to_jax(problem(seed=8))
    first = solver.solve(**target)

    solver.solve(**to_jax(problem(seed=9)))

    assert_bitwise(solver.solve(**target), first)


def test_input_validation():
    qp = to_jax(problem(seed=7))
    solver = MultistageSolver()
    with pytest.raises(TypeError, match=r"\(diag, offdiag\) tuple"):
        solver.solve(**dict(qp, P=qp["P"][0]))
    with pytest.raises(ValueError, match="P diag must have shape"):
        solver.solve(**dict(qp, P=(qp["P"][0][..., :2], None)))
    with pytest.raises(ValueError, match="A diag must have shape"):
        solver.solve(**dict(qp, A=(qp["A"][0][:2], qp["A"][1])))
    with pytest.raises(ValueError, match="expected"):
        solver.solve(**dict(qp, P=(qp["P"][0], qp["P"][1][:1])))
    with pytest.raises(ValueError, match="expected"):
        solver.solve(**dict(qp, b=qp["b"][:-1]))          # needs N + 1 block rows
    with pytest.raises(ValueError, match="A and b"):
        solver.solve(**dict(qp, b=None))
    with pytest.raises(TypeError, match="dtype"):
        solver.solve(**dict(qp, c=qp["c"].astype(jnp.float32)))
