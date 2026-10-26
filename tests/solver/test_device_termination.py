"""Device-side termination state.

The termination decisions of every IPM iteration are made by a GPU kernel
from the device ``Info``; the host only reads the outcome once the solve
has finished. These tests run problems that end in every status and verify
that the public device arrays ``result.info.status`` /
``result.info.iter`` hold the same values as the host snapshot returned by
``result.info.to_host()``.
"""
import numpy as np
import pytest
import warp as wp

from cupiqp import DenseSolver, SparseSolver, MultistageSolver, Status
from tests.helpers import csr_triple, primal_infeasible, random_qp, stack_qps, to_wp, to_wp_dict


def _dual_infeasible(qp):
    """Linear cost with no curvature and no constraints: unbounded below."""
    n, p, m = qp["P"].shape[0], qp["A"].shape[0], qp["G"].shape[0]
    return dict(
        P=np.zeros((n, n)), c=np.ones(n),
        A=np.zeros((p, n)), b=np.zeros(p),
        G=np.zeros((m, n)), h_u=qp["h_u"], h_l=qp["h_l"],
        x_u=np.full(n, np.inf), x_l=np.full(n, -np.inf),
    )


def _mixed_batch(seed=0):
    rng = np.random.default_rng(seed)
    qps = [random_qp(rng) for _ in range(6)]
    return qps + [primal_infeasible(qps[0]), _dual_infeasible(qps[1])]


def _stack(qps, np_dtype):
    return to_wp_dict(stack_qps(qps), np_dtype)


def _assert_device_matches_host(solver):
    info = solver.result.info
    h = info.to_host()
    np.testing.assert_array_equal(info.status.numpy(), [s.value for s in h.status])
    np.testing.assert_array_equal(info.iter.numpy(), h.iter)
    assert h.iter_total == int(info.iter_total.numpy()[0]) == int(h.iter.max())


DTYPES = [(wp.float64, np.float64), (wp.float32, np.float32)]


@pytest.mark.parametrize("dtype,np_dtype", DTYPES)
@pytest.mark.parametrize("enable_cuda_graph", [True, False])
@pytest.mark.parametrize("preconditioner_iter", [10, 0])
def test_dense_mixed_batch(dtype, np_dtype, enable_cuda_graph, preconditioner_iter):
    qps = _mixed_batch()
    solver = DenseSolver(dtype=dtype)
    solver.settings.enable_cuda_graph = enable_cuda_graph
    solver.settings.preconditioner_iter = preconditioner_iter
    solver.setup(**_stack(qps, np_dtype))

    solver.solve()
    statuses = solver.result.info.to_host().status
    assert statuses[:6] == [Status.CUPIQP_SOLVED] * 6
    assert statuses[6] == Status.CUPIQP_PRIMAL_INFEASIBLE
    # The unbounded problem is detected in float32; float64 tolerances leave it
    # stagnating until max_iter (exercising the finetune decision on the way).
    assert statuses[7] in (Status.CUPIQP_DUAL_INFEASIBLE, Status.CUPIQP_MAX_ITER_REACHED)
    _assert_device_matches_host(solver)

    # A second solve resets and rebuilds the device state.
    solver.solve()
    assert solver.result.info.to_host().status == statuses
    _assert_device_matches_host(solver)


@pytest.mark.parametrize("dtype,np_dtype", DTYPES)
def test_dense_max_iter(dtype, np_dtype):
    max_iter = 6
    solver = DenseSolver(dtype=dtype)
    solver.settings.max_iter = max_iter
    solver.settings.preconditioner_iter = 10
    solver.setup(**_stack(_mixed_batch(), np_dtype))

    solver.solve()
    statuses = solver.result.info.to_host().status
    assert Status.CUPIQP_MAX_ITER_REACHED in statuses
    _assert_device_matches_host(solver)
    info = solver.result.info
    # The last iteration is checked, so a problem that hits the limit has run
    # exactly max_iter iterations.
    for st, it in zip(statuses, info.iter.numpy()):
        if st == Status.CUPIQP_MAX_ITER_REACHED:
            assert it == max_iter


def test_dense_single_problem():
    qp = random_qp(np.random.default_rng(1))
    solver = DenseSolver()
    solver.setup(**to_wp_dict(qp))
    solver.solve()
    assert solver.result.info.to_host().status == [Status.CUPIQP_SOLVED]
    _assert_device_matches_host(solver)


def test_dense_equality_only():
    """No inequalities: the full-Newton path and its rho/delta kernel."""
    rng = np.random.default_rng(2)
    qps = [random_qp(rng) for _ in range(3)]
    solver = DenseSolver()
    solver.setup(**{k: to_wp(np.stack([q[k] for q in qps])) for k in ("P", "c", "A", "b")})
    solver.solve()
    assert solver.result.info.to_host().status == [Status.CUPIQP_SOLVED] * 3
    _assert_device_matches_host(solver)


def test_update_resets_device_status():
    solver = DenseSolver()
    solver.setup(**_stack(_mixed_batch(), np.float64))
    solver.solve()
    solver.update(c=to_wp(np.stack([q["c"] for q in _mixed_batch(seed=5)])))
    assert solver.result.info.to_host().status == [Status.CUPIQP_UNSOLVED] * 8
    np.testing.assert_array_equal(
        solver.result.info.status.numpy(), Status.CUPIQP_UNSOLVED.value)
    solver.solve()
    _assert_device_matches_host(solver)


def test_sparse_single_problems():
    """SparseSolver shares one sparsity pattern across the batch, so the
    mixed problems are solved one at a time."""
    batch = _mixed_batch()
    for qp, expected in ((batch[0], Status.CUPIQP_SOLVED),
                         (batch[6], Status.CUPIQP_PRIMAL_INFEASIBLE)):
        args = {k: (csr_triple(v) if k in ("P", "A", "G") else to_wp(v)) for k, v in qp.items()}
        solver = SparseSolver()
        solver.setup(**args)
        solver.solve()
        assert solver.result.info.to_host().status == [expected]
        _assert_device_matches_host(solver)


def test_multistage_batch():
    P_diag = to_wp([[[4.0, 0.5], [0.5, 3.0]], [[5.0, 0.0], [0.0, 2.0]]])
    P_offdiag = to_wp([[[0.3, 0.1], [0.0, 0.2]]])
    c = to_wp([[1.0, -1.0, 0.5, -0.5], [2.0, 1.0, -0.5, 0.0], [-1.0, 0.0, 1.0, 3.0]])
    solver = MultistageSolver()
    solver.setup(P=(P_diag, P_offdiag), c=c, x_l=to_wp(-np.ones(4)), x_u=to_wp(np.ones(4)))
    solver.solve()
    assert solver.result.info.to_host().status == [Status.CUPIQP_SOLVED] * 3
    _assert_device_matches_host(solver)
