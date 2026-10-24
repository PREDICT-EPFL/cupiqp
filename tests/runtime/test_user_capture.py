"""solve() and update() recorded into a CUDA graph captured by the user.

The user captures the solver stream together with their own work; every graph
launch must then run a complete solve on the current data, with the same
result as plain calls. Also covered: the user's own conditional loop around
the solve (nested conditional nodes), and the clear errors for what cannot be
captured.
"""
import numpy as np
import pytest
import warp as wp

from cupiqp import DenseSolver, MultistageSolver, SparseSolver, Status
from tests.helpers import csr_triple, to_wp, to_wp_dict


B, N = 4, 5


def _dense_problem(rng):
    M = rng.standard_normal((B, N, N))
    P = M @ M.transpose(0, 2, 1) + N * np.eye(N)
    G = rng.standard_normal((B, 3, N))
    return to_wp_dict(dict(P=P, G=G,
                           h_l=-np.ones((B, 3)), h_u=np.ones((B, 3)),
                           x_l=-2 * np.ones((B, N)), x_u=2 * np.ones((B, N))))


def _costs(rng, steps):
    return [rng.standard_normal((B, N)) for _ in range(steps)]


def _reference(problem, costs):
    ref = DenseSolver()
    ref.setup(c=to_wp(costs[0]), **problem)
    xs = []
    for c in costs:
        ref.update(c=to_wp(c))
        ref.solve()
        assert (ref.result.info.status.numpy() == Status.CUPIQP_SOLVED).all()
        xs.append(ref.result.x.numpy())
    return xs


def test_captured_update_and_solve_match_plain_calls():
    rng = np.random.default_rng(0)
    problem = _dense_problem(rng)
    costs = _costs(rng, 4)
    expected = _reference(problem, costs)

    solver = DenseSolver()
    c_buf = wp.array(costs[0], dtype=wp.float64)
    solver.setup(c=c_buf, **problem)
    solver.solve()  # warm-up: compiles every kernel, puts the settings on the GPU
    x_out = wp.zeros((B, N), dtype=wp.float64)
    with wp.ScopedStream(solver.stream), wp.ScopedCapture() as capture:
        solver.update(c=c_buf)
        solver.solve()
        wp.copy(x_out, solver.result.x)

    for c, x_ref in zip(costs, expected):
        c_buf.assign(c)
        wp.capture_launch(capture.graph, stream=solver.stream)
        np.testing.assert_allclose(x_out.numpy(), x_ref, rtol=1e-10, atol=1e-12)
        status = solver.result.info.status.numpy()
        assert (status == Status.CUPIQP_SOLVED).all()
        h = solver.result.info.to_host()
        assert h.status == [Status.CUPIQP_SOLVED] * B
        assert h.iter_total == int(h.iter.max())


@wp.kernel
def _next_step(step: wp.array(dtype=wp.int32), 
               costs: wp.array3d(dtype=wp.float64),
               c: wp.array2d(dtype=wp.float64)):
    b, j = wp.tid()
    c[b, j] = costs[step[0], b, j]


@wp.kernel
def _record(step: wp.array(dtype=wp.int32), 
            x: wp.array2d(dtype=wp.float64),
            history: wp.array3d(dtype=wp.float64)):
    b, j = wp.tid()
    history[step[0], b, j] = x[b, j]


@wp.kernel
def _advance(step: wp.array(dtype=wp.int32), 
             steps: wp.int32, 
             keep_going: wp.array(dtype=wp.int32)):
    step[0] = step[0] + 1
    keep_going[0] = wp.where(step[0] < steps, 1, 0)


def test_solve_inside_user_capture_while():
    """A closed loop run entirely on the GPU: the user's while loop sets the
    data of each step, solves and records the solution."""
    rng = np.random.default_rng(1)
    problem = _dense_problem(rng)
    steps = 5
    costs = _costs(rng, steps)
    expected = _reference(problem, costs)

    solver = DenseSolver()
    c_buf = wp.array(costs[0], dtype=wp.float64)
    solver.setup(c=c_buf, **problem)
    solver.solve()
    all_costs = wp.array(np.stack(costs), dtype=wp.float64)
    history = wp.zeros((steps, B, N), dtype=wp.float64)
    step = wp.zeros(1, dtype=wp.int32)
    keep_going = wp.ones(1, dtype=wp.int32)

    def body():
        wp.launch(_next_step, dim=(B, N), inputs=[step, all_costs, c_buf])
        solver.update(c=c_buf)
        solver.solve()
        wp.launch(_record, dim=(B, N), inputs=[step, solver.result.x, history])
        wp.launch(_advance, dim=1, inputs=[step, steps, keep_going])

    with wp.ScopedStream(solver.stream), wp.ScopedCapture() as capture:
        wp.capture_while(keep_going, body)
    wp.capture_launch(capture.graph, stream=solver.stream)
    np.testing.assert_allclose(history.numpy(), np.stack(expected), rtol=1e-10, atol=1e-12)


def test_multistage_captured_solve():
    P_diag = to_wp(np.array([[[4.0, 0.5], [0.5, 3.0]], [[5.0, 0.0], [0.0, 2.0]]]))
    P_offdiag = to_wp(np.array([[[0.3, 0.1], [0.0, 0.2]]]))
    rng = np.random.default_rng(2)
    costs = [rng.standard_normal((3, 4)) for _ in range(3)]

    ref = MultistageSolver()
    ref.setup(P=(P_diag, P_offdiag), c=to_wp(costs[0]), x_l=to_wp(-np.ones(4)), x_u=to_wp(np.ones(4)))
    expected = []
    for c in costs:
        ref.update(c=to_wp(c))
        ref.solve()
        expected.append(ref.result.x.numpy())

    solver = MultistageSolver()
    c_buf = wp.array(costs[0], dtype=wp.float64)
    solver.setup(P=(P_diag, P_offdiag), c=c_buf, x_l=to_wp(-np.ones(4)), x_u=to_wp(np.ones(4)))
    solver.solve()
    with wp.ScopedStream(solver.stream), wp.ScopedCapture() as capture:
        solver.update(c=c_buf)
        solver.solve()
    for c, x_ref in zip(costs, expected):
        c_buf.assign(c)
        wp.capture_launch(capture.graph, stream=solver.stream)
        np.testing.assert_allclose(solver.result.x.numpy(), x_ref, rtol=1e-10, atol=1e-12)
        assert (solver.result.info.status.numpy() == Status.CUPIQP_SOLVED).all()


def _captured(solver, fn):
    with wp.ScopedStream(solver.stream), wp.ScopedCapture():
        fn()


def _warm_dense():
    rng = np.random.default_rng(3)
    solver = DenseSolver()
    solver.setup(c=to_wp(_costs(rng, 1)[0]), **_dense_problem(rng))
    solver.solve()
    return solver


def test_sparse_solve_cannot_be_captured():
    n = 4
    solver = SparseSolver()
    solver.setup(P=csr_triple(np.eye(n)), c=to_wp(np.ones(n)), x_l=to_wp(-np.ones(n)),
                 x_u=to_wp(np.ones(n)))
    solver.solve()
    with pytest.raises(RuntimeError, match="conditional graph node"):
        _captured(solver, solver.solve)


def test_verbose_solve_cannot_be_captured():
    solver = _warm_dense()
    solver.settings.verbose = True
    with pytest.raises(RuntimeError, match="verbose"):
        _captured(solver, solver.solve)


def test_settings_changed_during_capture_raise():
    solver = _warm_dense()
    solver.settings.eps_abs = 1e-6
    with pytest.raises(RuntimeError, match="settings changed"):
        _captured(solver, solver.solve)


def test_to_host_inside_capture_raises():
    solver = _warm_dense()
    with pytest.raises(RuntimeError, match="captured"):
        _captured(solver, solver.result.info.to_host)
