"""Tests for cupiqp.jax.SparseSolver: agreement with the core sparse solver,
value arrays shared or batched, jit and vmap, pattern validation and the
pattern's role in the solver cache. The reference is the core SparseSolver."""

import os

import numpy as np
import pytest
import scipy.sparse as sp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import warp as wp

import cupiqp
from cupiqp import Status
from cupiqp.jax import QPSolution, SparseSolver
from cupiqp.interfaces.jax.base import cache_info   # test instrumentation, not public API

from tests.helpers import csr_triple, random_qp, to_wp

pytestmark = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="a CUDA JAX device is required"
)

FIELDS = QPSolution._fields
VARIABLE_FIELDS = ("x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu")
TOL = dict(rtol=1e-9, atol=1e-10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sparse_qp(seed=0, n=8, p=3, m=5, density=0.5) -> dict:
    """A random strictly convex QP whose matrices are SciPy CSR matrices
    with sorted indices and a sparse pattern."""
    rng = np.random.default_rng(seed)
    qp = random_qp(rng, n=n, p=p, m=m)
    mask = rng.random((n, n)) < density
    mask = mask | mask.T | np.eye(n, dtype=bool)
    P = sp.csr_matrix(np.where(mask, qp["P"], 0.0) + n * np.eye(n))
    A = sp.csr_matrix(np.where(rng.random((p, n)) < density, qp["A"], 0.0) + 0.1)
    G = sp.csr_matrix(np.where(rng.random((m, n)) < density, qp["G"], 0.0) + 0.1)
    for M in (P, A, G):
        M.sort_indices()
    return dict(qp, P=P, A=A, G=G)


def patterns(qp: dict) -> dict:
    """The ``(indptr, indices)`` pattern of every sparse matrix of ``qp``."""
    return {name: (M.indptr, M.indices) for name, M in qp.items() if sp.issparse(M)}


def values(qp: dict) -> dict:
    """JAX arguments of ``qp``: the stored values of each matrix and the vectors."""
    out = {}
    for name, value in qp.items():
        data = value.data if sp.issparse(value) else np.asarray(value, dtype=np.float64)
        out[name] = jnp.asarray(data)
    return out


def core_solve(qp: dict) -> dict:
    solver = cupiqp.SparseSolver()
    args = {}
    for name, value in qp.items():
        if sp.issparse(value):
            args[name] = csr_triple(value)
        else:
            args[name] = to_wp(value)
    solver.setup(**args)
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


def assert_same(a: QPSolution, b: QPSolution) -> None:
    """Two solutions of the same problem agree. The sparse backend is not
    bitwise reproducible from run to run unless cuDSS deterministic mode is
    on (``Settings.use_deterministic_mode_for_cudss``), so compare to
    tolerance; statuses and iteration counts must agree exactly."""
    for name in FIELDS:
        got, expected = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if name in ("status", "iterations"):
            np.testing.assert_array_equal(got, expected, err_msg=name)
        elif got.size:
            np.testing.assert_allclose(got, expected, err_msg=name, **TOL)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_unbatched_matches_core():
    qp = sparse_qp(0)
    solver = SparseSolver(**patterns(qp))
    sol = solver.solve(**values(qp))
    assert sol.x.shape == (8,) and sol.status.shape == ()
    assert int(sol.status) == Status.CUPIQP_SOLVED
    assert_close(sol, core_solve(qp), unbatched=True)


def test_pattern_index_arrays():
    """Host patterns, int32 or int64, and concrete JAX arrays are accepted;
    other GPU arrays are rejected with a clear message."""
    qp = sparse_qp(1)
    mixed = SparseSolver(
        P=(qp["P"].indptr, qp["P"].indices),
        A=(jnp.asarray(qp["A"].indptr), jnp.asarray(qp["A"].indices)),
        G=(qp["G"].indptr.astype(np.int64), qp["G"].indices.astype(np.int64)),
    )
    for device_array in (to_wp(qp["P"].indptr, np.int32),):
        with pytest.raises(TypeError, match="host array"):
            SparseSolver(P=(device_array, qp["P"].indices))
    with pytest.raises(TypeError, match="traced"):
        jax.jit(lambda ip: SparseSolver(P=(ip, qp["P"].indices)).solve(
            jnp.asarray(qp["P"].data), jnp.asarray(qp["c"])).x)(jnp.asarray(qp["P"].indptr))
    host = SparseSolver(**patterns(qp))
    before = cache_info()["live_solvers"]
    assert_same(mixed.solve(**values(qp)), host.solve(**values(qp)))
    assert cache_info()["live_solvers"] == before + 1                    # equal patterns share one solver


def test_jit_eager_and_one_custom_call():
    qp = sparse_qp(2)
    solver = SparseSolver(**patterns(qp))
    args = values(qp)
    eager = solver.solve(**args)
    jitted = jax.jit(lambda **kw: solver.solve(**kw))(**args)
    assert_same(eager, jitted)
    hlo = jax.jit(lambda **kw: solver.solve(**kw).x).lower(**args).as_text()
    assert hlo.count("custom_call") == 1


def test_batched_values_and_vectors():
    """Batched P values and c with the other arguments shared: agrees with
    the core batched solve and with one solve per problem."""
    qps = [sparse_qp(seed) for seed in range(3)]
    base = qps[0]
    # Same pattern for every problem: reuse base's pattern with new values.
    P_values = np.stack([base["P"].data * (1.0 + 0.5 * i) for i in range(3)])
    cs = np.stack([q["c"] for q in qps])
    solver = SparseSolver(**patterns(base))
    args = values(base)
    args["P"] = jnp.asarray(P_values)
    args["c"] = jnp.asarray(cs)
    sol = jax.jit(lambda **kw: solver.solve(**kw))(**args)
    assert sol.x.shape == (3, 8) and sol.status.shape == (3,)
    assert np.all(np.asarray(sol.status) == Status.CUPIQP_SOLVED)

    solver_core = cupiqp.SparseSolver()
    solver_core.setup(
        P=(base["P"].indptr, base["P"].indices, to_wp(P_values)), c=to_wp(cs),
        A=csr_triple(base["A"]), b=to_wp(base["b"]), G=csr_triple(base["G"]),
        h_u=to_wp(base["h_u"]), h_l=to_wp(base["h_l"]), x_u=to_wp(base["x_u"]), x_l=to_wp(base["x_l"]),
    )
    solver_core.solve()
    np.testing.assert_allclose(np.asarray(sol.x), solver_core.result.x.numpy(), **TOL)
    for i in range(3):
        single = dict(base, c=cs[i])
        single["P"] = sp.csr_matrix((P_values[i], base["P"].indices, base["P"].indptr), shape=base["P"].shape)
        np.testing.assert_allclose(np.asarray(sol.x)[i], core_solve(single)["x"][0], **TOL)


def test_vmap_is_one_batched_solve():
    qp = sparse_qp(3)
    solver = SparseSolver(**patterns(qp))
    args = values(qp)
    cs = jnp.stack([args["c"] * (i + 1) for i in range(4)])

    def solve_one(c):
        return solver.solve(args["P"], c, A=args["A"], b=args["b"], G=args["G"],
                            h_u=args["h_u"], h_l=args["h_l"], x_u=args["x_u"], x_l=args["x_l"])

    vmapped = jax.jit(jax.vmap(solve_one))
    sol = vmapped(cs)
    direct = solver.solve(args["P"], cs, A=args["A"], b=args["b"], G=args["G"],
                          h_u=args["h_u"], h_l=args["h_l"], x_u=args["x_u"], x_l=args["x_l"])
    assert_same(sol, direct)
    assert vmapped.lower(cs).as_text().count("custom_call") == 1


def test_block_combinations():
    qp = sparse_qp(4)
    args = values(qp)
    only_P = SparseSolver(P=patterns(qp)["P"]).solve(args["P"], args["c"])
    assert only_P.y.shape == (0,) and only_P.z_l.shape == (0,) and only_P.z_bl.shape == (0,)
    assert_close(only_P, core_solve({"P": qp["P"], "c": qp["c"]}), unbatched=True)
    one_sided = SparseSolver(P=patterns(qp)["P"], G=patterns(qp)["G"]).solve(args["P"], args["c"], G=args["G"], h_u=args["h_u"])
    assert one_sided.z_u.shape == (5,) and one_sided.z_l.shape == (0,)
    assert_close(one_sided, core_solve({"P": qp["P"], "c": qp["c"], "G": qp["G"], "h_u": qp["h_u"]}),
                 unbatched=True)
    box = SparseSolver(P=patterns(qp)["P"]).solve(args["P"], args["c"], x_l=args["x_l"], x_u=args["x_u"])
    assert_close(box, core_solve({"P": qp["P"], "c": qp["c"], "x_l": qp["x_l"], "x_u": qp["x_u"]}),
                 unbatched=True)


def test_values_must_match_the_pattern():
    qp = sparse_qp(5)
    args = values(qp)
    solver = SparseSolver(P=patterns(qp)["P"], A=patterns(qp)["A"])
    with pytest.raises(ValueError, match="expected"):
        solver.solve(args["P"][:-1], args["c"], A=args["A"], b=args["b"])
    with pytest.raises(ValueError, match="must not be given"):
        solver.solve(args["P"], args["c"], A=args["A"], b=args["b"], G=args["G"], h_u=args["h_u"])
    with pytest.raises(ValueError, match="must be given"):
        solver.solve(args["P"], args["c"])
    with pytest.raises(ValueError, match="two batch axes"):
        solver.solve(args["P"][None, None], args["c"], A=args["A"], b=args["b"])
    with pytest.raises(ValueError, match="must be a vector"):
        solver.solve(args["P"][None, None, None], args["c"], A=args["A"], b=args["b"])


def test_pattern_validation():
    qp = sparse_qp(6)
    P = qp["P"]
    with pytest.raises(TypeError, match="CSR pattern"):
        SparseSolver(P=P)                                # the matrix itself, not its pattern
    with pytest.raises(ValueError, match="strictly increasing"):
        SparseSolver(P=(P.indptr, P.indices[::-1].copy()))
    with pytest.raises(ValueError, match="indptr"):
        SparseSolver(P=(P.indptr[:-1], P.indices))
    with pytest.raises(ValueError, match="column indices"):
        SparseSolver(P=(P.indptr, P.indices), A=(qp["A"].indptr, qp["A"].indices + 100))
    with pytest.raises(TypeError, match="int32 or int64"):
        SparseSolver(P=(P.indptr.astype(np.float64), P.indices))


def test_pattern_is_part_of_the_cache_key():
    qp = sparse_qp(7)
    args = values(qp)
    before = cache_info()["live_solvers"]
    SparseSolver(**patterns(qp)).solve(**args)
    SparseSolver(**patterns(qp)).solve(**args)           # equal pattern: same solver
    assert cache_info()["live_solvers"] == before + 1
    denser = sp.csr_matrix(qp["P"].toarray() + 1.0)      # another pattern
    denser.sort_indices()
    SparseSolver(P=(denser.indptr, denser.indices), A=patterns(qp)["A"], G=patterns(qp)["G"]).solve(
        jnp.asarray(denser.data), args["c"], A=args["A"], b=args["b"], G=args["G"],
        h_u=args["h_u"], h_l=args["h_l"], x_u=args["x_u"], x_l=args["x_l"])
    assert cache_info()["live_solvers"] == before + 2


def test_result_does_not_depend_on_earlier_calls():
    qp = sparse_qp(9)
    solver = SparseSolver(**patterns(qp))
    target = values(qp)
    first = solver.solve(**target)

    other = dict(target, c=-target["c"])
    solver.solve(**other)

    assert_same(solver.solve(**target), first)


def test_failures_are_reported_in_status():
    qp = sparse_qp(8)
    bad = dict(qp)
    h_l, h_u = np.array(qp["h_l"]), np.array(qp["h_u"])
    h_l[0], h_u[0] = 1.0, 0.5                            # empty interval on row 0
    bad["h_l"], bad["h_u"] = h_l, h_u
    solver = SparseSolver(**patterns(qp))
    sol = solver.solve(**values(bad))
    assert int(sol.status) != Status.CUPIQP_SOLVED
    assert np.isfinite(np.asarray(sol.x)).all()
