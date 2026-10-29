"""Tests for cupiqp.jax.MultistageSolver: agreement with the core multistage
solver for every combination of off-diagonal blocks, block and flat vector
layouts, shared and batched arrays, jit and vmap, and input validation."""

import itertools

import numpy as np
import pytest

# Configures JAX: must come before any import of JAX.
from tests.jax_interface.jax_helpers import (assert_bitwise, assert_close, core_result, jax, jnp,
                                             requires_gpu, to_jax)

import cupiqp
from cupiqp import Status
from cupiqp.jax import MultistageSolver
from tests.helpers import map_arrays, random_multistage_qp, to_wp

pytestmark = requires_gpu


def core_solve(qp: dict) -> dict:
    solver = cupiqp.MultistageSolver()
    solver.setup(**{k: v for k, v in map_arrays(qp, to_wp).items() if v is not None})
    solver.solve()
    return core_result(solver)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("has_P_E,has_A_E,has_G_E", list(itertools.product([True, False], repeat=3)))
def test_matches_core_for_every_offdiag_combination(has_P_E, has_A_E, has_G_E):
    qp = random_multistage_qp(1, has_P_E=has_P_E, has_A_E=has_A_E, has_G_E=has_G_E)
    sol = MultistageSolver().solve(**to_jax(qp))
    assert int(sol.status) == Status.CUPIQP_SOLVED
    assert sol.x.shape == (12,)
    assert sol.y.shape == ((4 + has_A_E) * 2,) and sol.z_l.shape == ((4 + has_G_E) * 2,)
    assert_close(sol, core_solve(qp), unbatched=True)


def test_block_and_flat_vectors_agree():
    qp = to_jax(random_multistage_qp(2))
    flat = dict(qp, **{k: qp[k].reshape(-1) for k in ("c", "b", "h_l", "h_u", "x_l", "x_u")})
    solver = MultistageSolver()
    assert_bitwise(solver.solve(**qp), solver.solve(**flat))


def test_jit_eager_and_one_custom_call():
    qp = to_jax(random_multistage_qp(3))
    solver = MultistageSolver()
    eager = solver.solve(**qp)
    jitted = jax.jit(lambda **kw: solver.solve(**kw))(**qp)
    assert_bitwise(eager, jitted)
    hlo = jax.jit(lambda **kw: solver.solve(**kw).x).lower(**qp).as_text()
    assert hlo.count("custom_call") == 1


def test_batched_and_shared_arrays():
    """Batched P_diag and c (block layout) and x_l (flat layout) with the
    rest shared, b in the flat layout: agrees with the core batched solve."""
    shared = random_multistage_qp(4)
    batched = random_multistage_qp(4, batch_size=3)
    qp = dict(shared, P=(batched["P"][0], shared["P"][1]), c=batched["c"],
              b=shared["b"].reshape(-1),
              x_l=np.ascontiguousarray(np.broadcast_to(shared["x_l"].reshape(-1), (3, 12))))
    sol = jax.jit(lambda **kw: MultistageSolver().solve(**kw))(**to_jax(qp))
    assert sol.x.shape == (3, 12) and sol.status.shape == (3,)
    assert np.all(np.asarray(sol.status) == Status.CUPIQP_SOLVED)
    assert_close(sol, core_solve(qp))


def test_vmap_is_one_batched_solve():
    qp = to_jax(random_multistage_qp(5))
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
    qp = random_multistage_qp(6)
    minimal = {"P": (qp["P"][0], None), "c": qp["c"]}
    sol = MultistageSolver().solve(**to_jax(minimal))
    assert int(sol.status) == Status.CUPIQP_SOLVED
    assert sol.y.shape == (0,) and sol.z_l.shape == (0,) and sol.z_bl.shape == (0,)
    assert_close(sol, core_solve(minimal), unbatched=True)


def test_result_does_not_depend_on_earlier_calls():
    solver = MultistageSolver()
    target = to_jax(random_multistage_qp(8))
    first = solver.solve(**target)

    solver.solve(**to_jax(random_multistage_qp(9)))

    assert_bitwise(solver.solve(**target), first)


def test_input_validation():
    qp = to_jax(random_multistage_qp(7))
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
