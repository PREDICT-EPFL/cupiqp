"""Solve with poisoned buffers to catch reads of uninitialized memory.

Fresh GPU memory is usually zero, so a kernel that reads a buffer before writing
it (e.g. ``out = alpha * acc + beta * out`` with ``beta == 0``) passes until the
memory it gets was previously used and holds ``+/-inf`` or NaN. Every test of
this module runs under the ``poison_uninitialized_memory`` fixture, which fills
each buffer from ``wp.empty`` with ``0xFF`` bytes (NaN floats), so such a read
turns into a failed solve immediately and deterministically.
"""
import numpy as np
import pytest

from cupiqp import DenseSolver, MultistageSolver, SparseSolver
from tests.helpers import all_solved, broadcast, csr_triple, quiet, to_wp

pytestmark = pytest.mark.usefixtures("poison_uninitialized_memory")


@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_dense_solves_under_poison(enable_cuda_graph):
    n = 3
    P = to_wp(np.eye(n))
    c = to_wp(np.array([-1.0, 2.0, 0.5]))
    s = quiet(DenseSolver())
    s.settings.enable_cuda_graph = enable_cuda_graph
    s.setup(P=P, c=broadcast(c, 2), A=to_wp(np.ones((1, n))), b=to_wp(np.zeros(1)),
            G=to_wp(np.eye(n)), h_l=to_wp(-np.ones(n)), h_u=to_wp(np.ones(n)),
            x_l=to_wp(-2 * np.ones(n)), x_u=to_wp(2 * np.ones(n)))
    s.solve()
    assert all_solved(s.result.info.to_host().status)
    s.update(c=to_wp(np.array([[0.3, -0.2, 0.1], [1.0, 1.0, -1.0]])))
    s.solve()
    assert all_solved(s.result.info.to_host().status)


def test_dense_backward_under_poison():
    n = 3
    s = quiet(DenseSolver())
    s.settings.enable_grad = True
    s.settings.enable_cuda_graph = False
    s.setup(P=to_wp(np.eye(n)), c=to_wp(np.array([-1.0, 2.0, 0.5])),
            x_l=to_wp(-np.ones(n)), x_u=to_wp(np.ones(n)))
    s.solve()
    assert all_solved(s.result.info.to_host().status)
    grads = s.backward(grad_x=to_wp(np.ones((1, n))))
    assert bool(np.isfinite(grads.c.numpy()).all())


def test_sparse_solves_under_poison():
    n = 4
    P = csr_triple(np.eye(n))
    G = csr_triple(np.eye(n))
    s = quiet(SparseSolver())
    s.settings.enable_cuda_graph = False
    s.setup(P=P, c=broadcast(to_wp(np.arange(n, dtype=np.float64) - 1.5), 2),
            G=G, h_l=to_wp(-np.ones(n)), h_u=to_wp(np.ones(n)))
    s.solve()
    assert all_solved(s.result.info.to_host().status)
    s.update(P=to_wp(2.0 * np.ones((2, P[2].shape[0]))), c=to_wp(np.ones((2, n))))
    s.solve()
    assert all_solved(s.result.info.to_host().status)


@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_multistage_solves_under_poison(enable_cuda_graph):
    N, d = 3, 2
    P_diag = to_wp(np.broadcast_to(2.0 * np.eye(d), (N, d, d)))
    P_off = to_wp(np.broadcast_to(-0.3 * np.eye(d), (N - 1, d, d)))
    A_diag = to_wp(np.broadcast_to(np.eye(d), (N, d, d)))
    A_off = to_wp(np.broadcast_to(-np.eye(d), (N, d, d)))
    s = quiet(MultistageSolver())
    s.settings.enable_cuda_graph = enable_cuda_graph
    s.setup(P=(P_diag, P_off), c=to_wp(np.ones((2, N, d))), A=(A_diag, A_off),
            b=to_wp(np.zeros((N + 1, d))), x_l=to_wp(-np.ones((N, d))), x_u=to_wp(np.ones((N, d))))
    s.solve()
    assert all_solved(s.result.info.to_host().status)
    s.update(P=(None, to_wp(np.broadcast_to(-0.1 * np.eye(d), (2, N - 1, d, d)))))
    s.solve()
    assert all_solved(s.result.info.to_host().status)
