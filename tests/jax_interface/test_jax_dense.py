"""Tests for cupiqp.jax.DenseSolver: agreement with the core solver, the shared
and batched argument layouts, jit and vmap behaviour (one custom call, no
recompilation on bound flips), status reporting, input validation, stream
ordering, and the solver cache."""

import os

import numpy as np
import pytest

# JAX reserves most of the GPU memory for itself when its backend starts;
# the core solver then cannot create its cuSOLVER / cuDSS handles. Leave the
# memory to be allocated on demand (must be set before JAX initializes).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import warp as wp

import cupiqp
from cupiqp import Status
from cupiqp.jax import DenseSolver, QPSolution, cache_size, clear_cache

from tests.helpers import primal_infeasible, random_qp, to_wp_dict


pytestmark = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="a CUDA JAX device is required"
)

FIELDS = QPSolution._fields
VARIABLE_FIELDS = ("x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu")
WARP_DTYPE = {np.float64: wp.float64, np.float32: wp.float32}
TOL = {np.float64: dict(rtol=1e-9, atol=1e-10), np.float32: dict(rtol=1e-4, atol=1e-4)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_jax(qp: dict, dtype=np.float64) -> dict:
    """The problem dict as JAX arrays of ``dtype`` (``None`` values dropped)."""
    return {k: jnp.asarray(np.asarray(v, dtype=dtype)) for k, v in qp.items() if v is not None}


def core_solve(qp: dict, dtype=np.float64, settings=None) -> dict:
    """Solve ``qp`` with the core DenseSolver; every public field as a
    ``(B, ...)`` NumPy array."""
    solver = cupiqp.DenseSolver(dtype=WARP_DTYPE[dtype])
    if settings is not None:
        solver.settings = settings
    solver.setup(**to_wp_dict({k: v for k, v in qp.items() if v is not None}, dtype))
    solver.solve()
    result = solver.result
    out = {name: getattr(result, name).numpy() for name in VARIABLE_FIELDS}
    out["status"] = result.info.status.numpy()
    out["iterations"] = result.info.iter.numpy()
    out["primal_res"] = result.info.primal_res.numpy()
    out["dual_res"] = result.info.dual_res.numpy()
    return out


def as_numpy(sol: QPSolution) -> dict:
    return {name: np.asarray(getattr(sol, name)) for name in FIELDS}


def assert_close(sol: QPSolution, ref: dict, dtype=np.float64, unbatched=False) -> None:
    """``sol`` agrees with the core reference ``ref`` to the dtype tolerance;
    statuses and iteration counts agree exactly."""
    got = as_numpy(sol)
    for name in FIELDS:
        expected = ref[name][0] if unbatched else ref[name]
        assert got[name].shape == expected.shape, name
        if name in ("status", "iterations"):
            np.testing.assert_array_equal(got[name], expected, err_msg=name)
        elif got[name].size:
            np.testing.assert_allclose(got[name], expected, err_msg=name, **TOL[dtype])


def assert_bitwise(a: QPSolution, b: QPSolution) -> None:
    for name in FIELDS:
        assert np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))), name


def solved(sol: QPSolution) -> bool:
    return bool(np.all(np.asarray(sol.status) == Status.CUPIQP_SOLVED))


def batch_of(qps: list, batched: tuple, dtype=np.float64) -> dict:
    """JAX arguments where the fields in ``batched`` are stacked over the
    problems and the others are taken (shared) from the first problem."""
    out = {}
    for name in qps[0]:
        if name in batched:
            out[name] = jnp.asarray(np.stack([np.asarray(q[name], dtype=dtype) for q in qps]))
        else:
            out[name] = jnp.asarray(np.asarray(qps[0][name], dtype=dtype))
    return out


def stacked(qps: list) -> dict:
    return {name: np.stack([q[name] for q in qps]) for name in qps[0]}


# ---------------------------------------------------------------------------
# Agreement with the core solver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_unbatched_matches_core(dtype):
    qp = random_qp(0, n=6, p=2, m=4)
    sol = DenseSolver(dtype=dtype).solve(**to_jax(qp, dtype))
    assert isinstance(sol, QPSolution)
    assert sol.x.dtype == dtype and sol.status.dtype == np.int32
    assert sol.x.shape == (6,) and sol.status.shape == ()
    assert solved(sol)
    assert_close(sol, core_solve(qp, dtype), dtype, unbatched=True)


def test_jit_matches_eager_bitwise():
    qp = to_jax(random_qp(1, n=5, p=1, m=3))
    op = DenseSolver()
    eager = op.solve(**qp)
    jitted = jax.jit(lambda **kw: op.solve(**kw))(**qp)
    assert_bitwise(eager, jitted)
    assert_bitwise(eager, op.solve(**qp))        # and the call is repeatable


def test_jit_lowers_to_one_custom_call():
    qp = to_jax(random_qp(2, n=5, p=1, m=3))
    op = DenseSolver()
    hlo = jax.jit(lambda **kw: op.solve(**kw).x).lower(**qp).as_text()
    assert hlo.count("custom_call") == 1


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_batched_and_shared_arguments(dtype):
    """Batched c, b, h_u with shared P, A, G, h_l, x_l, x_u: agrees with the
    core batched solve and with one unbatched solve per problem."""
    qps = [random_qp(seed, n=6, p=2, m=4) for seed in range(4)]
    base = qps[0]
    problems = [dict(base, c=q["c"], b=q["b"], h_u=q["h_u"]) for q in qps]
    op = DenseSolver(dtype=dtype)
    sol = jax.jit(lambda **kw: op.solve(**kw))(**batch_of(problems, ("c", "b", "h_u"), dtype))
    assert sol.x.shape == (4, 6) and sol.status.shape == (4,)
    assert solved(sol)
    assert_close(sol, core_solve(stacked(problems), dtype), dtype)
    for i, problem in enumerate(problems):
        single = op.solve(**to_jax(problem, dtype))
        got = as_numpy(sol)
        for name in VARIABLE_FIELDS:
            np.testing.assert_allclose(got[name][i], np.asarray(getattr(single, name)), **TOL[dtype])


def test_leading_axis_of_size_one_is_a_batch():
    qp = random_qp(3, n=4, p=1, m=2)
    args = to_jax(qp)
    args["c"] = args["c"][None]
    sol = DenseSolver().solve(**args)
    assert sol.x.shape == (1, 4) and sol.status.shape == (1,)
    assert_close(sol, core_solve(qp))


@pytest.mark.parametrize(
    "drop",
    [
        ("A", "b", "G", "h_u", "h_l", "x_u", "x_l"),      # P and c only
        ("G", "h_u", "h_l", "x_u", "x_l"),                # equalities only
        ("A", "b", "h_l", "x_u", "x_l"),                  # one-sided G
        ("A", "b", "h_u", "x_u", "x_l"),
        ("A", "b", "G", "h_u", "h_l"),                    # box only
        ("A", "b", "G", "h_u", "h_l", "x_l"),             # upper box only
        (),                                               # everything
    ],
)
def test_block_combinations(drop):
    qp = {k: (None if k in drop else v) for k, v in random_qp(4, n=5, p=2, m=3).items()}
    sol = DenseSolver().solve(**to_jax(qp))
    assert solved(sol)
    assert_close(sol, core_solve(qp), unbatched=True)
    widths = {
        "y": 0 if qp["A"] is None else 2,
        "z_l": 0 if qp["h_l"] is None else 3, "s_l": 0 if qp["h_l"] is None else 3,
        "z_u": 0 if qp["h_u"] is None else 3, "s_u": 0 if qp["h_u"] is None else 3,
        "z_bl": 0 if qp["x_l"] is None else 5, "s_bl": 0 if qp["x_l"] is None else 5,
        "z_bu": 0 if qp["x_u"] is None else 5, "s_bu": 0 if qp["x_u"] is None else 5,
    }
    for name, width in widths.items():
        assert getattr(sol, name).shape == (width,), name
        assert getattr(sol, name).dtype == np.float64


def test_infinite_bounds_change_without_recompilation():
    """Entries of a bound may flip between finite and infinite from call to
    call: same compiled function, same cached solver, results that match
    the core solver for every pattern."""
    qp = random_qp(5, n=6, p=1, m=4)
    op = DenseSolver()
    traces = 0

    @jax.jit
    def solve(**kw):
        nonlocal traces
        traces += 1
        return op.solve(**kw)

    clear_cache()
    patterns = [
        dict(qp),
        dict(qp, x_l=np.where(np.arange(6) % 2 == 0, -np.inf, qp["x_l"]), h_u=np.full(4, np.inf)),
        dict(qp, x_u=np.full(6, np.inf), h_l=np.where(np.arange(4) < 2, -np.inf, qp["h_l"])),
        dict(qp),
    ]
    for problem in patterns:
        sol = solve(**to_jax(problem))
        assert solved(sol)
        assert_close(sol, core_solve(problem), unbatched=True)
    assert traces == 1
    assert cache_size() == 1


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------

def test_vmap_is_one_batched_solve():
    qps = [random_qp(seed, n=5, p=1, m=3) for seed in range(3)]
    base = to_jax(qps[0])
    cs = jnp.stack([jnp.asarray(q["c"]) for q in qps])
    op = DenseSolver()

    def solve_one(c):
        return op.solve(base["P"], c, A=base["A"], b=base["b"], G=base["G"],
                  h_u=base["h_u"], h_l=base["h_l"], x_u=base["x_u"], x_l=base["x_l"])

    vmapped = jax.jit(jax.vmap(solve_one))
    sol = vmapped(cs)
    assert sol.x.shape == (3, 5) and sol.status.shape == (3,)
    direct = op.solve(base["P"], cs, A=base["A"], b=base["b"], G=base["G"],
                h_u=base["h_u"], h_l=base["h_l"], x_u=base["x_u"], x_l=base["x_l"])
    assert_bitwise(sol, direct)
    assert vmapped.lower(cs).as_text().count("custom_call") == 1
    assert_bitwise(jax.vmap(solve_one)(cs), direct)   # eager vmap too


def test_vmap_in_axes_and_shared_arguments():
    qps = [random_qp(seed, n=5, p=1, m=3) for seed in range(3)]
    base = to_jax(qps[0])
    cs = jnp.stack([jnp.asarray(q["c"]) for q in qps])
    bs = jnp.stack([jnp.asarray(q["b"]) for q in qps])
    op = DenseSolver()

    def solve(P, c, A, b):
        return op.solve(P, c, A=A, b=b).x

    direct = op.solve(base["P"], cs, A=base["A"], b=bs).x
    got = jax.vmap(solve, in_axes=(None, 0, None, 0))(base["P"], cs, base["A"], bs)
    assert np.array_equal(np.asarray(got), np.asarray(direct))
    got_t = jax.vmap(solve, in_axes=(None, 1, None, 1))(base["P"], cs.T, base["A"], bs.T)
    assert np.array_equal(np.asarray(got_t), np.asarray(direct))
    expected = core_solve(stacked([dict(qps[0], c=q["c"], b=q["b"]) for q in qps]) | {
        "G": None, "h_u": None, "h_l": None, "x_u": None, "x_l": None})["x"]
    np.testing.assert_allclose(np.asarray(got), expected, **TOL[np.float64])


def test_nested_batching_is_rejected():
    base = to_jax(random_qp(6, n=4, p=1, m=2))
    cs = jnp.stack([base["c"], 2 * base["c"]])
    op = DenseSolver()
    with pytest.raises(NotImplementedError, match="nested"):
        jax.vmap(lambda c: op.solve(base["P"], c).x)(cs[None])
    with pytest.raises(ValueError, match="two batch axes"):
        op.solve(base["P"], cs[None])


def test_grad_is_not_supported_yet():
    base = to_jax(random_qp(7, n=4, p=1, m=2))
    with pytest.raises(Exception, match="differentiated"):
        jax.grad(lambda c: DenseSolver().solve(base["P"], c).x.sum())(base["c"])


# ---------------------------------------------------------------------------
# Statuses and settings
# ---------------------------------------------------------------------------

def test_failures_are_reported_in_status():
    qp = random_qp(8, n=6, p=2, m=4)
    bad = primal_infeasible(qp)
    op = DenseSolver()
    sol = op.solve(**to_jax(bad))
    assert int(sol.status) == Status.CUPIQP_PRIMAL_INFEASIBLE
    both = batch_of([qp, bad], tuple(qp))
    sol = jax.jit(lambda **kw: op.solve(**kw))(**both)
    np.testing.assert_array_equal(
        np.asarray(sol.status), [Status.CUPIQP_SOLVED, Status.CUPIQP_PRIMAL_INFEASIBLE])
    assert np.isfinite(np.asarray(sol.x)).all()


def test_settings_are_read_at_each_call():
    qp = to_jax(random_qp(9, n=6, p=2, m=4))
    capped = DenseSolver()
    capped.settings.max_iter = 1                       # mutable, like the core solver
    default = DenseSolver()
    assert capped.settings.max_iter == 1 and default.settings.max_iter != 1

    @jax.jit
    def both(**kw):
        return default.solve(**kw).status, capped.solve(**kw).status, capped.solve(**kw).iterations

    st_default, st_capped, it_capped = both(**qp)
    assert int(st_default) == Status.CUPIQP_SOLVED
    assert int(st_capped) == Status.CUPIQP_MAX_ITER_REACHED
    assert int(it_capped) == 1

    capped.settings.max_iter = 250                     # eager calls see the change ...
    assert int(capped.solve(**qp).status) == Status.CUPIQP_SOLVED
    assert int(both(**qp)[1]) == Status.CUPIQP_MAX_ITER_REACHED   # ... a compiled function does not

    settings = cupiqp.Settings.for_dtype(wp.float64)
    settings.max_iter = 2
    assigned = DenseSolver(settings=settings)
    assert assigned.settings is settings
    assert int(assigned.solve(**qp).iterations) == 2


def test_solver_object():
    assert DenseSolver().dtype == np.float64 and DenseSolver(dtype=jnp.float32).dtype == np.float32
    assert DenseSolver().settings.dtype is wp.float64
    assert DenseSolver(dtype=jnp.float32).settings.dtype is wp.float32
    with pytest.raises(ValueError, match="settings.dtype"):
        DenseSolver(dtype=jnp.float32, settings=cupiqp.Settings.for_dtype(wp.float64))
    with pytest.raises(TypeError, match="Settings"):
        DenseSolver().settings = {"max_iter": 1}
    with pytest.raises(TypeError):
        DenseSolver(dtype=jnp.int32)


def test_lazy_import():
    import cupiqp as pkg
    assert pkg.jax.DenseSolver is DenseSolver


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_input_validation():
    qp = to_jax(random_qp(10, n=4, p=1, m=2))
    op = DenseSolver()
    with pytest.raises(TypeError, match="dtype"):
        op.solve(qp["P"].astype(jnp.float32), qp["c"])
    with pytest.raises(TypeError, match="JAX array"):
        op.solve([[1.0, 0.0], [0.0, 1.0]], qp["c"])
    with pytest.raises(ValueError, match="A and b"):
        op.solve(qp["P"], qp["c"], A=qp["A"])
    with pytest.raises(ValueError, match="require G"):
        op.solve(qp["P"], qp["c"], h_u=qp["h_u"])
    with pytest.raises(ValueError, match="at least one of h_l and h_u"):
        op.solve(qp["P"], qp["c"], G=qp["G"])
    with pytest.raises(ValueError, match="expected"):
        op.solve(qp["P"], qp["c"][:3])
    with pytest.raises(ValueError, match="expected"):
        op.solve(qp["P"], qp["c"], A=qp["A"], b=qp["b"][None].repeat(2, 0)[:, :0])
    with pytest.raises(ValueError, match="disagree"):
        op.solve(qp["P"], jnp.stack([qp["c"]] * 2), A=qp["A"], b=jnp.stack([qp["b"]] * 3))
    with pytest.raises(ValueError, match="must be a vector"):
        op.solve(qp["P"], qp["c"][None, None, None])


# ---------------------------------------------------------------------------
# Streams, result ownership, cache
# ---------------------------------------------------------------------------

def test_results_are_owned_by_jax():
    """Earlier results do not change when the same solver solves again."""
    qp = to_jax(random_qp(11, n=6, p=2, m=4))
    op = DenseSolver()
    first = op.solve(**qp)
    first_copy = as_numpy(first)
    op.solve(**dict(qp, c=-3.0 * qp["c"]))
    for name in FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(first, name)), first_copy[name], err_msg=name)


def test_inputs_produced_on_the_xla_stream_are_seen():
    """The solve reads inputs produced immediately before it by a long XLA
    computation in the same jitted function."""
    qp = random_qp(12, n=16, p=4, m=8)
    args = to_jax(qp)
    op = DenseSolver()
    big = jnp.asarray(np.random.default_rng(0).standard_normal((2048, 2048)))

    @jax.jit
    def pipeline(c0, M):
        v = c0
        for _ in range(8):
            v = jnp.tanh(M @ jnp.tile(v, 128))[:16]
        return v, op.solve(args["P"], v, A=args["A"], b=args["b"], G=args["G"],
                     h_u=args["h_u"], h_l=args["h_l"], x_u=args["x_u"], x_l=args["x_l"])

    for _ in range(3):
        c_used, sol = pipeline(args["c"], big)
        assert solved(sol)
        expected = core_solve(dict(qp, c=np.asarray(c_used)))
        assert_close(sol, expected, unbatched=True)


def test_cache_counts_signatures_and_clears():
    clear_cache()
    assert cache_size() == 0
    qp = to_jax(random_qp(13, n=4, p=1, m=2))
    op = DenseSolver()
    op.solve(qp["P"], qp["c"])
    op.solve(qp["P"], qp["c"])                                  # same signature
    op.solve(qp["P"], jnp.stack([qp["c"]] * 2))                 # batched c: new signature
    op.solve(qp["P"], qp["c"], A=qp["A"], b=qp["b"])            # more blocks: new signature
    DenseSolver(dtype=jnp.float32).solve(qp["P"].astype(jnp.float32), qp["c"].astype(jnp.float32))
    assert cache_size() == 4
    solve = jax.jit(lambda P, c: op.solve(P, c).x)
    before = np.asarray(solve(qp["P"], qp["c"]))
    clear_cache()
    assert cache_size() == 0
    np.testing.assert_array_equal(np.asarray(solve(qp["P"], qp["c"])), before)   # still works
    assert cache_size() == 1                                                  # and is tracked again
    clear_cache()
    assert cache_size() == 0
