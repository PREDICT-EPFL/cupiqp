"""Each solver instance owns (or borrows) one ``warp.Stream`` and runs on it.

Every public entry point makes that stream Warp's current stream for its
duration and restores the previous one afterwards; ordering with other
streams is the caller's business (or the framework adapter's).
"""
import gc

import cupy as cp
import numpy as np
import pytest
import warp as wp

from cupiqp import DenseSolver, MultistageSolver, SparseSolver, Status
from cupyx.scipy.sparse import csr_matrix


def _box_qp():
    n = 3
    return dict(P=cp.eye(n), c=cp.array([-1.0, 2.0, 0.5]), x_l=-cp.ones(n), x_u=cp.ones(n))


def _reference_x():
    ref = DenseSolver()
    ref.settings.verbose = False
    ref.setup(**_box_qp())
    ref.solve()
    return ref.result.x.numpy()


def test_solver_owns_a_stream_by_default():
    a, b = DenseSolver(), DenseSolver()
    assert isinstance(a.stream, wp.Stream)
    assert a.stream.cuda_stream != 0                       # not the legacy default stream
    assert a.stream.cuda_stream != b.stream.cuda_stream    # one stream per instance


def test_borrowed_warp_stream_is_used_and_not_destroyed():
    user = wp.Stream("cuda:0")
    solver = DenseSolver(stream=user)
    assert solver.stream is user
    solver.settings.verbose = False
    solver.setup(**_box_qp())
    solver.solve()
    assert solver.result.info.to_host().status == [Status.CUPIQP_SOLVED]
    del solver
    gc.collect()
    # The borrowed stream outlives the solver and still works.
    with wp.ScopedStream(user):
        a = wp.zeros(4, dtype=wp.float64)
        a.fill_(1.0)
    wp.synchronize_stream(user)
    assert np.all(a.numpy() == 1.0)


def test_only_warp_streams_are_accepted():
    with pytest.raises(TypeError, match="warp.Stream"):
        DenseSolver(stream=cp.cuda.Stream(non_blocking=True))
    with pytest.raises(TypeError, match="warp.Stream"):
        DenseSolver(stream=object())


def test_current_stream_is_restored_after_return_and_exception():
    before = wp.get_stream("cuda")
    solver = DenseSolver()
    solver.settings.verbose = False
    solver.setup(**_box_qp())
    assert wp.get_stream("cuda") is before
    solver.solve()
    assert wp.get_stream("cuda") is before
    with pytest.raises(RuntimeError):
        solver.setup(**_box_qp())            # second setup raises inside the scope
    assert wp.get_stream("cuda") is before


def test_work_on_a_user_warp_stream_is_ordered_with_the_solver():
    """Inputs produced on a non-blocking user stream that is passed to the
    solver are read after they are complete, and results consumed on it
    afterwards are the finished ones; compared against an independent solve."""
    x_ref = _reference_x()
    user = wp.Stream("cuda:0")
    with wp.ScopedStream(user), cp.cuda.ExternalStream(user.cuda_stream):
        qp = _box_qp()
        big = cp.ones((2048, 2048))              # keep the user stream busy
        for _ in range(5):
            big = big @ big * 1e-3
        qp["c"] = qp["c"] + 0.0 * big[0, :3]     # c depends on the busy work
    s = DenseSolver(stream=user)
    s.settings.verbose = False
    s.setup(**qp)
    s.solve()
    assert s.result.info.to_host().status == [Status.CUPIQP_SOLVED]
    with wp.ScopedStream(user):
        x = wp.clone(s.result.x)
    wp.synchronize_stream(user)
    np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-10)


def test_results_agree_with_default_stream_solve():
    """Inputs produced by plain cupy code on a cupy stream the solver knows
    nothing about are consumed correctly once that stream has finished.

    Ordering guarantee checked: none across streams. The solver stream and a
    cupy blocking stream are each ordered with the legacy NULL stream but not
    with each other, so the caller must complete its own stream (here with
    ``stream.synchronize()``) before handing the inputs to ``setup``. The
    test checks that after this host-side synchronization the solver copies
    the finished inputs and reproduces an independent solve exactly."""
    x_ref = _reference_x()
    stream = cp.cuda.Stream(non_blocking=False)
    with stream:
        qp = _box_qp()                                    # inputs on another blocking stream
    stream.synchronize()                                  # caller completes its stream first
    s = DenseSolver()
    s.settings.verbose = False
    s.setup(**qp)
    s.solve()
    np.testing.assert_allclose(s.result.x.numpy(), x_ref, atol=1e-10)


def test_repeated_graph_replay_and_destruction():
    """Captured graphs are replayed on the solver stream across solves and the
    solver (graphs, handles, stream) can be destroyed afterwards."""
    s = DenseSolver()
    s.settings.verbose = False
    s.settings.enable_cuda_graph = True
    qp = _box_qp()
    s.setup(**{**qp, "c": cp.broadcast_to(qp["c"], (4, 3))})   # four equal problems
    x_ref = _reference_x()
    for _ in range(3):
        s.solve()
        assert all(st == Status.CUPIQP_SOLVED for st in s.result.info.to_host().status)
        np.testing.assert_allclose(s.result.x.numpy()[0], x_ref[0], atol=1e-10)
    del s
    gc.collect()


def test_all_backends_run_on_their_stream():
    n = 4
    sp_ = SparseSolver()
    sp_.settings.verbose = False
    P = csr_matrix(cp.eye(n))
    sp_.setup(P=(P.indptr, P.indices, P.data), c=cp.ones((2, n)), x_l=-cp.ones(n), x_u=cp.ones(n))
    sp_.solve()
    assert all(st == Status.CUPIQP_SOLVED for st in sp_.result.info.to_host().status)

    N, d = 3, 2
    ms = MultistageSolver()
    ms.settings.verbose = False
    ms.setup(P=(cp.broadcast_to(2.0 * cp.eye(d), (N, d, d)), None), c=cp.ones((2, N, d)),
             x_l=-cp.ones((N, d)), x_u=cp.ones((N, d)))
    ms.solve()
    assert all(st == Status.CUPIQP_SOLVED for st in ms.result.info.to_host().status)

    for s in (sp_, ms):
        assert s.stream.cuda_stream != 0
    before = wp.get_stream("cuda")
    ms.solve()
    assert wp.get_stream("cuda") is before


def test_multistage_on_user_warp_stream():
    N, d = 3, 2
    ms = MultistageSolver(stream=wp.Stream("cuda:0"))
    ms.settings.verbose = False
    ms.setup(P=(cp.broadcast_to(2.0 * cp.eye(d), (N, d, d)), None), c=cp.ones((2, N, d)),
             x_l=-cp.ones((N, d)), x_u=cp.ones((N, d)))
    ms.solve()
    assert all(st == Status.CUPIQP_SOLVED for st in ms.result.info.to_host().status)
