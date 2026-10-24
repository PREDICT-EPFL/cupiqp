"""Tests for BatchedSolver -- end-to-end QP solving."""
import numpy as np
import pytest

from cupiqp import DenseSolver, Settings, Status
from tests.helpers import stack_qps, to_wp, to_wp_dict


def _setup_batched(solver, **kw):
    """Set up ``solver`` from batched ``(B, ...)`` host inputs."""
    solver.setup(**{k: to_wp(v) for k, v in kw.items() if v is not None})


def _make_random_qp(n=8, p=3, m=5, seed=42):
    """Generate a single random QP (NumPy arrays)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)
    P = (P + P.T) / 2
    c = rng.standard_normal(n)
    A = rng.standard_normal((p, n))
    b = rng.standard_normal(p)
    G = rng.standard_normal((m, n))
    h_u = np.abs(rng.standard_normal(m)) + 2.0
    h_l = -np.abs(rng.standard_normal(m)) - 2.0
    x_u = np.abs(rng.standard_normal(n)) + 3.0
    x_l = -np.abs(rng.standard_normal(n)) - 3.0
    return dict(P=P, c=c, A=A, b=b, G=G, h_u=h_u, h_l=h_l, x_u=x_u, x_l=x_l)


def _iter_settings(max_iter=None):
    """Deterministic config (no preconditioner / cuda graph) for iter tests."""
    settings = Settings()
    settings.preconditioner_iter = 0
    settings.enable_cuda_graph = False
    if max_iter is not None:
        settings.max_iter = max_iter
    return settings


def _solve_single(qp, settings=None):
    """Solve one QP with the existing SolverBase."""
    settings = settings or Settings()
    settings.preconditioner_iter = 0
    settings.enable_cuda_graph = False
    solver = DenseSolver()
    solver.settings = settings
    solver.setup(**to_wp_dict(qp))
    solver.solve()
    status = solver.result.info.to_host().status[0]  # status is always a list; B=1 -> one entry
    return status, solver.result.x.numpy()[0].copy()


def _solve_single_iter(qp, settings=None, solver_cls=DenseSolver):
    """Solve one QP (B=1) and return (status, iteration_count)."""
    solver = solver_cls()
    solver.settings = settings or _iter_settings()
    solver.setup(**to_wp_dict(qp))
    solver.solve()
    h = solver.result.info.to_host()
    return h.status[0], int(h.iter[0])


class TestBatchedSolverCorrectness:
    """Compare batched solutions against per-problem SolverBase."""

    def test_batch_matches_individual(self):
        """Main correctness test: batch of 4 QPs should match individual solves."""
        seeds = [42, 123, 77, 999]
        B = len(seeds)
        n, p, m = 8, 3, 5

        # Solve individually
        ref_statuses = []
        ref_x = []
        for s in seeds:
            qp = _make_random_qp(n, p, m, seed=s)
            st, x = _solve_single(qp)
            ref_statuses.append(st)
            ref_x.append(x)

        # Solve batched
        qps = [_make_random_qp(n, p, m, seed=s) for s in seeds]
        settings = Settings()
        settings.preconditioner_iter = 0
        solver = DenseSolver()
        solver.settings = settings
        _setup_batched(solver, **stack_qps(qps))
        solver.solve()
        batch_statuses = solver.result.info.to_host().status

        for i in range(B):
            assert batch_statuses[i] == ref_statuses[i], (
                f"Problem {i}: batch={batch_statuses[i]}, ref={ref_statuses[i]}")
            if ref_statuses[i] == Status.CUPIQP_SOLVED:
                np.testing.assert_allclose(
                    solver.result.x.numpy()[i], ref_x[i], atol=1e-5,
                    err_msg=f"x mismatch at problem {i}")

    def test_single_problem(self):
        """Batch of size 1 should match individual solve."""
        qp = _make_random_qp(n=6, p=2, m=4, seed=42)
        ref_st, ref_x = _solve_single(qp)

        settings = Settings()
        settings.preconditioner_iter = 0
        solver = DenseSolver()
        solver.settings = settings
        _setup_batched(solver, **{k: v[None, ...] for k, v in qp.items()})
        solver.solve()
        statuses = solver.result.info.to_host().status  # always a list, one Status per problem

        assert statuses[0] == ref_st
        if ref_st == Status.CUPIQP_SOLVED:
            np.testing.assert_allclose(solver.result.x.numpy()[0], ref_x, atol=1e-5)

    def test_no_inequality(self):
        """QP with only equality constraints."""
        B, n, p = 3, 5, 3
        rng = np.random.default_rng(42)
        Ps, cs, As, bs = [], [], [], []
        for _ in range(B):
            M = rng.standard_normal((n, n))
            P = M @ M.T + n * np.eye(n)
            Ps.append((P + P.T) / 2)
            cs.append(rng.standard_normal(n))
            As.append(rng.standard_normal((p, n)))
            bs.append(rng.standard_normal(p))

        settings = Settings()
        settings.preconditioner_iter = 0
        solver = DenseSolver()
        solver.settings = settings
        _setup_batched(solver, 
            P=np.stack(Ps),
            c=np.stack(cs),
            A=np.stack(As),
            b=np.stack(bs),
        )
        solver.solve()
        statuses = solver.result.info.to_host().status
        for st in statuses:
            assert st == Status.CUPIQP_SOLVED, f"Expected SOLVED, got {st}"


class TestBatchedSolverBasic:

    def test_all_solved(self):
        """Simple well-conditioned problems should all solve."""
        B, n = 8, 4
        rng = np.random.default_rng(0)
        Ps = []
        for _ in range(B):
            M = rng.standard_normal((n, n))
            P = M @ M.T + 10 * np.eye(n)
            Ps.append((P + P.T) / 2)
        settings = Settings()
        settings.preconditioner_iter = 0
        solver = DenseSolver()
        solver.settings = settings
        _setup_batched(solver, 
            P=np.stack(Ps),
            c=rng.standard_normal((B, n)),
        )
        solver.solve()
        statuses = solver.result.info.to_host().status
        assert all(st == Status.CUPIQP_SOLVED for st in statuses)


class TestPerProblemIterations:
    """``result.info.iter`` records each problem's own iteration count.

    Each batch element terminates independently; its iteration counter must
    freeze at the iteration it reached a terminal state, and match the count
    the same problem gets when solved alone.
    """

    def _make_dense_batch(self, qps, settings):
        solver = DenseSolver()
        solver.settings = settings
        _setup_batched(solver, **stack_qps(qps))
        return solver

    def test_iter_is_per_problem_and_matches_individual(self):
        """Batched per-problem iter == standalone iter, and differs across
        problems (before the fix every problem shared the global counter)."""
        seeds = [42, 123, 77, 999]
        B = len(seeds)
        n, p, m = 8, 3, 5
        qps = [_make_random_qp(n, p, m, seed=s) for s in seeds]

        ref = [_solve_single_iter(q) for q in qps]
        assert all(st == Status.CUPIQP_SOLVED for st, _ in ref)

        solver = self._make_dense_batch(qps, _iter_settings())
        solver.solve()
        statuses = solver.result.info.to_host().status
        iters = [int(v) for v in solver.result.info.to_host().iter]

        for i in range(B):
            assert statuses[i] == Status.CUPIQP_SOLVED
            assert iters[i] == ref[i][1], (
                f"problem {i}: batched iter {iters[i]} != standalone {ref[i][1]}")
            assert 0 < iters[i] < solver.settings.max_iter

        # The whole point of the fix: counts are NOT all the same global number.
        assert len(set(iters)) > 1, f"expected per-problem iters to differ: {iters}"

    def test_terminated_problem_iter_does_not_advance(self):
        """A problem that solves early freezes its iter while a slower batch
        member keeps iterating up to max_iter."""
        # Two 2-variable problems sharing the same finite-bound pattern; they
        # differ only in theta = lower bound on x1 (x_l[:, 0]). theta=2 > x1<=1
        # makes an empty box, which does not converge -> MAX_ITER_REACHED.
        base = dict(
            P=np.array([[6.0, 0.0], [0.0, 4.0]]),
            c=np.array([-1.0, -4.0]),
            A=np.array([[1.0, -2.0]]),
            b=np.array([1.0]),
            G=np.array([[1.0, -2.0]]),
            h_l=np.array([-10.0]),
            h_u=np.array([10.0]),
            x_u=np.array([1.0, np.inf]),
        )
        feasible = dict(base, x_l=np.array([-1.0, -np.inf]))
        infeasible = dict(base, x_l=np.array([2.0, -np.inf]))
        max_iter = 40

        ref_st, ref_it = _solve_single_iter(feasible, _iter_settings(max_iter=max_iter))
        assert ref_st == Status.CUPIQP_SOLVED

        solver = self._make_dense_batch([feasible, infeasible], _iter_settings(max_iter=max_iter))
        solver.solve()
        statuses = solver.result.info.to_host().status
        iters = [int(v) for v in solver.result.info.to_host().iter]

        assert statuses[0] == Status.CUPIQP_SOLVED
        assert statuses[1] == Status.CUPIQP_MAX_ITER_REACHED
        # Solved member froze at its own (small) count, not the global max.
        assert iters[0] == ref_it
        assert iters[0] < max_iter
        assert iters[0] < iters[1]
        # Never-terminating member runs the full loop: iter == max_iter.
        assert iters[1] == max_iter

    def test_three_problems_per_problem_iter(self):
        """Three different problems in one batch each freeze at their own
        iteration count (the ``iter < 5`` warm-start branch of the rho/delta
        update is exercised along the way)."""
        seeds = [42, 123, 77]
        n, p, m = 8, 3, 5
        qps = [_make_random_qp(n, p, m, seed=s) for s in seeds]

        ref = [_solve_single_iter(q, _iter_settings(), solver_cls=DenseSolver)
               for q in qps]
        assert all(st == Status.CUPIQP_SOLVED for st, _ in ref)

        solver = DenseSolver()
        solver.settings = _iter_settings()
        _setup_batched(solver, **stack_qps(qps))
        solver.solve()
        statuses = solver.result.info.to_host().status
        iters = [int(v) for v in solver.result.info.to_host().iter]

        for i in range(len(seeds)):
            assert statuses[i] == Status.CUPIQP_SOLVED
            assert iters[i] == ref[i][1], (
                f"problem {i}: batched iter {iters[i]} != standalone {ref[i][1]}")
        assert len(set(iters)) > 1, f"expected per-problem iters to differ: {iters}"


def test_update_toggles_active_box_bounds():
    """update(x_l) that flips box bounds finite<->infinite must match a fresh
    setup with the new bounds. Guards the box scaling derived from the
    activity mask at setup: a variable without a finite bound at setup must
    still get its bound enforced once update() makes it finite."""
    inf = np.inf
    P = np.eye(3)
    c = np.array([1.0, 2.0, 3.0])
    x_l_1 = np.array([0.0, -inf, 0.0])   # x = [0, -2, 0]
    x_l_2 = np.array([0.0, 0.0, -inf])   # x = [0, 0, -3]; entries 1 and 2 toggle

    for preconditioner_iter in (0, 10):
        settings = Settings()
        settings.preconditioner_iter = preconditioner_iter
        settings.verbose = False

        ref = DenseSolver()
        ref.settings = settings
        ref.setup(to_wp(P), to_wp(c), x_l=to_wp(x_l_2))
        ref.solve()
        x_ref = ref.result.x.numpy()[0]
        np.testing.assert_allclose(x_ref, [0.0, 0.0, -3.0], atol=1e-6)

        solver = DenseSolver()
        solver.settings = settings
        solver.setup(to_wp(P), to_wp(c), x_l=to_wp(x_l_1))
        solver.solve()
        np.testing.assert_allclose(solver.result.x.numpy()[0], [0.0, -2.0, 0.0], atol=1e-6)

        solver.update(x_l=to_wp(x_l_2))
        solver.solve()
        assert solver.result.info.to_host().status[0] == Status.CUPIQP_SOLVED
        np.testing.assert_allclose(solver.result.x.numpy()[0], x_ref, atol=1e-6)
        z_bl = solver.result.z_bl.numpy()[0]
        assert z_bl[2] == 0.0          # now unbounded: dual exactly zero
        assert abs(z_bl[1]) > 1e-3     # now bounded and binding


def test_backward_with_unsolved_problem_in_batch():
    """backward() must not raise when a problem of the batch did not converge.
    Problem 1 is primal infeasible (x0 = 0 and x0 = 1). With "differentiate" the
    solved problem's gradient is unaffected; with "zero" the failed problem's
    gradients are exactly zero."""
    P = np.eye(2)
    c = np.array([1.0, 1.0])
    A = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 1.0]])            # problem 1: infeasible
    grad_x = np.ones((2, 2))

    def make(mode, b_in):
        s = DenseSolver()
        s.settings.enable_grad = True
        s.settings.preconditioner_iter = 0
        s.settings.verbose = False
        s.settings.gradient_on_failure = mode
        s.setup(to_wp(P), to_wp(c), A=to_wp(A), b=to_wp(b_in))
        s.solve()
        return s

    reference = make("differentiate", b[0])
    assert reference.result.info.to_host().status == [Status.CUPIQP_SOLVED]
    grad_c_ref = reference.backward(grad_x=to_wp(grad_x[:1])).c.numpy()[0]

    for mode in ("differentiate", "zero"):
        s = make(mode, b)
        status = s.result.info.to_host().status
        assert status[0] == Status.CUPIQP_SOLVED and status[1] != Status.CUPIQP_SOLVED
        grads = s.backward(grad_x=to_wp(grad_x))
        grad_c = grads.c.numpy()
        grad_P = grads.P.numpy()
        # The batch runs more iterations than the single problem (it waits
        # for the infeasible one to give up), so the iterates differ at the
        # tolerance level.
        np.testing.assert_allclose(grad_c[0], grad_c_ref, atol=1e-4)
        if mode == "zero":
            assert np.all(grad_c[1] == 0.0) and np.all(grad_P[1] == 0.0)
            assert np.all(grads.b.numpy()[1] == 0.0)


def test_structural_settings_are_frozen_at_setup():
    """Settings that fix the data coordinates, buffers or KKT structure are
    frozen by setup(): a later change is rejected by update(), solve() and
    backward(); numerical settings may still change."""
    rng = np.random.default_rng(3)
    M = rng.standard_normal((3, 3))
    solver = DenseSolver()
    solver.settings.enable_grad = True
    solver.settings.preconditioner_iter = 10
    solver.settings.verbose = False
    c = to_wp(rng.standard_normal(3))
    solver.setup(to_wp(M @ M.T + 3 * np.eye(3)), c, x_l=to_wp(-np.ones(3)), x_u=to_wp(np.ones(3)))
    solver.solve()
    solver.settings.max_iter = 50                    # numerical: allowed
    solver.solve()

    solver.settings.preconditioner_iter = 0
    for call in (solver.solve, lambda: solver.update(c=c),
                 lambda: solver.backward(grad_x=to_wp(np.ones((1, 3))))):
        with pytest.raises(RuntimeError, match=r"\['preconditioner_iter'\] changed after setup"):
            call()
    solver.settings.preconditioner_iter = 10

    solver.settings.gradient_smoothing = True
    with pytest.raises(RuntimeError, match=r"\['gradient_smoothing'\] changed after setup"):
        solver.backward(grad_x=to_wp(np.ones((1, 3))))
    solver.settings.gradient_smoothing = False
    solver.solve()                                   # restored: fine again
    solver.backward(grad_x=to_wp(np.ones((1, 3))))
