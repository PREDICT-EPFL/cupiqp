"""Tests of the active-problem mask of the dense solver.

A dense configuration whose iteration is made only of Warp kernels (small
``n``) skips the problems of a batch that have terminated. These tests check,
through the public API, that skipping does not change what a terminated
problem reports and that the configurations without the optimization (large
``n``, which uses cuBLAS and cuSOLVER) behave the same way.
"""
import numpy as np
import pytest
import warp as wp

from cupiqp import DenseSolver, Settings, Status
from tests.helpers import random_qp, stack_qps, to_wp, to_wp_dict


def _infeasible_qp(n, p, m, seed):
    """Contradictory equalities: the solver runs to ``max_iter`` or gives up."""
    qp = random_qp(seed, n, p, m)
    qp["A"][1] = qp["A"][0]
    qp["b"][0], qp["b"][1] = 0.0, 1.0
    return qp


def _solver(qps, dtype=None, grad=False, **settings):
    s = DenseSolver(dtype=dtype) if dtype is not None else DenseSolver()
    s.settings.enable_grad = grad
    for name, value in settings.items():
        setattr(s.settings, name, value)
    npdtype = np.float64 if dtype is None else wp.dtype_to_numpy(dtype)
    s.setup(**to_wp_dict(stack_qps(qps), npdtype))
    return s


def _snapshot(solver):
    """Public per-problem results of the last solve."""
    r = solver.result
    h = r.info.to_host()
    fields = {name: np.array(getattr(r, name).numpy()) for name in
              ("x", "y", "z_u", "z_l", "z_bu", "z_bl", "s_u", "s_l", "s_bu", "s_bl")}
    return fields, list(h.status), np.array(h.iter)


N, P_EQ, M_INEQ = 8, 3, 5


@pytest.mark.parametrize("preconditioner_iter", [0, 10])
def test_terminated_problem_does_not_depend_on_the_rest_of_the_batch(preconditioner_iter):
    """Problem 0 reports the same result whether its batch mate converges as fast
    as it does or runs on to ``max_iter``: a terminated problem stops changing."""
    easy = random_qp(1, N, P_EQ, M_INEQ)
    other_easy = random_qp(2, N, P_EQ, M_INEQ)
    hard = _infeasible_qp(N, P_EQ, M_INEQ, seed=3)

    results = []
    for mate in (other_easy, hard):
        s = _solver([easy, mate], preconditioner_iter=preconditioner_iter, max_iter=60)
        s.solve()
        results.append(_snapshot(s))

    (fast, st_fast, it_fast), (slow, st_slow, it_slow) = results
    assert st_fast == [Status.CUPIQP_SOLVED] * 2
    assert st_slow[0] == Status.CUPIQP_SOLVED and st_slow[1] != Status.CUPIQP_SOLVED
    assert it_fast[0] == it_slow[0]
    assert it_slow[1] > it_slow[0]
    for name in fast:
        np.testing.assert_array_equal(fast[name][0], slow[name][0], err_msg=name)


@pytest.mark.parametrize("n", [N, 40])
def test_mixed_batch_matches_solo_solves(n):
    """Each problem of a batch with different iteration counts matches its solo
    solve. ``n = 40`` selects cuBLAS/cuSOLVER, which ignore the mask."""
    qps = [random_qp(s, n, P_EQ, M_INEQ) for s in (11, 12, 13, 14, 15)]
    batch = _solver(qps, preconditioner_iter=0)
    batch.solve()
    fields, status, iters = _snapshot(batch)
    assert status == [Status.CUPIQP_SOLVED] * len(qps)
    if n == N:
        assert len(set(iters)) > 1, "the test needs problems that terminate at different iterations"

    for i, qp in enumerate(qps):
        solo = _solver([qp], preconditioner_iter=0)
        solo.solve()
        solo_fields, _, solo_iters = _snapshot(solo)
        assert iters[i] == solo_iters[0]
        np.testing.assert_allclose(fields["x"][i], solo_fields["x"][0], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(fields["y"][i], solo_fields["y"][0], rtol=1e-6, atol=1e-8)


def test_float32_mixed_batch():
    qps = [random_qp(s, N, P_EQ, M_INEQ) for s in (21, 22, 23, 24)]
    batch = _solver(qps, dtype=wp.float32, preconditioner_iter=0)
    batch.solve()
    fields, status, iters = _snapshot(batch)
    assert status == [Status.CUPIQP_SOLVED] * len(qps)
    for i, qp in enumerate(qps):
        solo = _solver([qp], dtype=wp.float32, preconditioner_iter=0)
        solo.solve()
        solo_fields, _, _ = _snapshot(solo)
        np.testing.assert_allclose(fields["x"][i], solo_fields["x"][0], rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("n", [N, 40])
def test_gradient_of_terminated_problem_does_not_depend_on_the_rest_of_the_batch(n):
    """backward() differentiates through the factorization at the final iterate,
    also for a problem that terminated long before the batch ended."""
    easy = random_qp(31, n, P_EQ, M_INEQ)
    mates = {"fast": random_qp(32, n, P_EQ, M_INEQ),
             "slow": _infeasible_qp(n, P_EQ, M_INEQ, seed=33)}
    grads = {}
    for name, mate in mates.items():
        s = _solver([easy, mate], grad=True, preconditioner_iter=0, max_iter=60)
        s.solve()
        g = s.backward(grad_x=to_wp(np.ones((2, n))))
        assert bool(s.grad_valid.numpy()[0])
        grads[name] = {k: getattr(g, k).numpy()[0].copy() for k in ("P", "c", "A", "b", "G")}
    # Masked path: the factor is rebuilt at the final iterate, so the batch
    # runtime does not matter. cuSOLVER path: the factor is refreshed every
    # iteration at the frozen point, which differs at the tolerance level.
    rtol, atol = (1e-8, 1e-10) if n == N else (1e-4, 1e-6)
    for k in grads["fast"]:
        np.testing.assert_allclose(grads["fast"][k], grads["slow"][k], rtol=rtol, atol=atol, err_msg=k)


def test_repeated_solves_with_masking():
    """The mask is rebuilt at the start of every solve: solving the same batch
    twice, and a modified batch after an update, gives the expected results."""
    qps = [random_qp(s, N, P_EQ, M_INEQ) for s in (41, 42, 43)]
    s = _solver(qps, preconditioner_iter=0)
    s.solve()
    first = _snapshot(s)
    s.solve()
    second = _snapshot(s)
    for name in first[0]:
        np.testing.assert_array_equal(first[0][name], second[0][name], err_msg=name)
    np.testing.assert_array_equal(first[2], second[2])
