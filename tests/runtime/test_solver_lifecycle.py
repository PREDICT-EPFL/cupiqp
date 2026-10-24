"""Solver lifetime: a dropped solver must be freed by reference counting.

A reference cycle that reaches a solver leaves its CUDA graphs, library handles
and buffers to Python's cyclic garbage collector. The collector can then run
while another solver is capturing a CUDA graph, and destroying CUDA resources
during a capture invalidates it (later: a crash when the broken graph is
destroyed). These tests use the public API only.
"""
import gc
import weakref

import numpy as np
import pytest
import warp as wp

from cupiqp import DenseSolver, MultistageSolver, SparseSolver, Status
from tests.helpers import csr_triple, to_wp, to_wp_dict


def _dense_qp(n, p, m, B, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    qp = dict(P=M @ M.T + n * np.eye(n), c=rng.standard_normal((B, n)),
              x_l=-np.ones(n), x_u=np.ones(n))
    if p:
        qp.update(A=rng.standard_normal((p, n)), b=np.zeros(p))
    if m:
        qp.update(G=rng.standard_normal((m, n)), h_u=np.ones(m), h_l=-np.ones(m))
    return qp


def _dense_solver(n, p, m, B):
    s = DenseSolver()
    s.setup(**to_wp_dict(_dense_qp(n, p, m, B)))
    return s


def _sparse_solver(n=8, p=2, m=3, B=4):
    qp = _dense_qp(n, p, m, B)
    s = SparseSolver()
    s.setup(**{k: csr_triple(v) if k in ("P", "A", "G") else to_wp(v) for k, v in qp.items()})
    return s


def _multistage_solver(N=3, d=2, r=2):
    rng = np.random.default_rng(1)
    s = MultistageSolver()
    s.setup(P=(to_wp(np.stack([np.eye(d) * 2.0] * N)), None),
            c=to_wp(0.1 * rng.standard_normal((N, d))),
            G=(to_wp(rng.standard_normal((N, r, d))), to_wp(rng.standard_normal((N, r, d)))),
            h_u=to_wp(np.full((N + 1, r), 5.0)))
    return s


def _all_solved(solver):
    return bool(np.all(solver.result.info.status.numpy() == Status.CUPIQP_SOLVED.value))


CASES = {
    "dense_small_batched": lambda: _dense_solver(6, 2, 4, 4),     # Warp matrix products
    "dense_large_batched": lambda: _dense_solver(40, 10, 40, 4),  # cuBLAS matrix products
    "dense_mixed": lambda: _dense_solver(10, 2, 40, 4),           # Warp for P, A; cuBLAS for G
    "dense_single": lambda: _dense_solver(6, 2, 4, 1),            # single-problem Cholesky
    "sparse": _sparse_solver,
    "multistage": _multistage_solver,
}


# socu's kernel factories define @wp.func inside function bodies. Warp's
# decorator holds its own frame, which keeps the calling frames (and the
# solver in them) alive until the cyclic garbage collector runs.
SOCU_CYCLE = pytest.mark.xfail(
    strict=True, reason="socu kernel factories define nested @wp.func; needs a fix in socu")


@pytest.mark.parametrize("case", [
    pytest.param(c, marks=SOCU_CYCLE) if c == "multistage" else c for c in sorted(CASES)
])
def test_dropped_solver_is_freed_by_reference_counting(case):
    """After setup() and solve(), dropping the last reference frees the solver
    immediately, without the cyclic garbage collector."""
    gc.collect()
    gc.disable()
    try:
        solver = CASES[case]()
        solver.solve()
        wp.synchronize()
        assert _all_solved(solver)
        ref = weakref.ref(solver)
        del solver
        assert ref() is None, "a reference cycle keeps the dropped solver alive"
    finally:
        gc.enable()


def test_frequent_garbage_collection_during_capture():
    """Solvers created, solved and dropped repeatedly while the collector runs
    on almost every allocation, including during CUDA graph capture."""
    threshold = gc.get_threshold()
    gc.collect()
    gc.set_threshold(1, 1, 1)
    try:
        for _ in range(3):
            for case in sorted(CASES):
                solver = CASES[case]()
                solver.solve()
                assert _all_solved(solver), case
                del solver
        wp.synchronize()
    finally:
        gc.set_threshold(*threshold)
