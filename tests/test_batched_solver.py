"""Tests for BatchedSolver — end-to-end QP solving."""
import numpy as np
import cupy as cp

from cupiqp import DenseSolver, Settings, Status


def _setup_batched(solver, **kw):
    """Set up ``solver`` from batched ``(B, ...)`` inputs."""
    solver.setup(**{k: cp.asarray(v) for k, v in kw.items() if v is not None})


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
    settings.kkt_solver = 'dense_cholesky'
    settings.preconditioner_iter = 0
    settings.enable_cuda_graph = False
    if max_iter is not None:
        settings.max_iter = max_iter
    return settings


def _solve_single(qp, settings=None):
    """Solve one QP with the existing SolverBase."""
    settings = settings or Settings()
    settings.kkt_solver = 'dense_cholesky'
    settings.preconditioner_iter = 0
    settings.enable_cuda_graph = False
    solver = DenseSolver()
    solver.settings = settings
    solver.setup(**{k: cp.array(v) for k, v in qp.items()})
    status = solver.solve()[0]  # solve() always returns a list; B=1 -> one entry
    return status, cp.asnumpy(cp.asarray(solver.result.x)[0].copy())


def _solve_single_iter(qp, settings=None, solver_cls=DenseSolver):
    """Solve one QP (B=1) and return (status, iteration_count)."""
    solver = solver_cls()
    solver.settings = settings or _iter_settings()
    solver.setup(**{k: cp.array(v) for k, v in qp.items()})
    solver.solve()
    return solver.result.info.status[0], int(solver.result.info.iter[0])


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
        settings.kkt_solver = 'dense_cholesky'
        settings.preconditioner_iter = 0
        solver = DenseSolver()
        solver.settings = settings
        _setup_batched(solver, 
            P=cp.array(np.stack([q['P'] for q in qps])),
            c=cp.array(np.stack([q['c'] for q in qps])),
            A=cp.array(np.stack([q['A'] for q in qps])),
            b=cp.array(np.stack([q['b'] for q in qps])),
            G=cp.array(np.stack([q['G'] for q in qps])),
            h_u=cp.array(np.stack([q['h_u'] for q in qps])),
            h_l=cp.array(np.stack([q['h_l'] for q in qps])),
            x_u=cp.array(np.stack([q['x_u'] for q in qps])),
            x_l=cp.array(np.stack([q['x_l'] for q in qps])),
        )
        batch_statuses = solver.solve()

        for i in range(B):
            assert batch_statuses[i] == ref_statuses[i], (
                f"Problem {i}: batch={batch_statuses[i]}, ref={ref_statuses[i]}")
            if ref_statuses[i] == Status.CUPIQP_SOLVED:
                np.testing.assert_allclose(
                    cp.asnumpy(solver.result.x[i]), ref_x[i], atol=1e-5,
                    err_msg=f"x mismatch at problem {i}")

    def test_single_problem(self):
        """Batch of size 1 should match individual solve."""
        qp = _make_random_qp(n=6, p=2, m=4, seed=42)
        ref_st, ref_x = _solve_single(qp)

        settings = Settings()
        settings.kkt_solver = 'dense_cholesky'
        settings.preconditioner_iter = 0
        solver = DenseSolver()
        solver.settings = settings
        _setup_batched(solver, **{k: cp.array(v[None, ...]) for k, v in qp.items()})
        statuses = solver.solve()  # always a list, one Status per problem

        assert statuses[0] == ref_st
        if ref_st == Status.CUPIQP_SOLVED:
            np.testing.assert_allclose(cp.asnumpy(solver.result.x[0]), ref_x, atol=1e-5)

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
        settings.kkt_solver = 'dense_cholesky'
        settings.preconditioner_iter = 0
        solver = DenseSolver()
        solver.settings = settings
        _setup_batched(solver, 
            P=cp.array(np.stack(Ps)),
            c=cp.array(np.stack(cs)),
            A=cp.array(np.stack(As)),
            b=cp.array(np.stack(bs)),
        )
        statuses = solver.solve()
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
        settings.kkt_solver = 'dense_cholesky'
        settings.preconditioner_iter = 0
        solver = DenseSolver()
        solver.settings = settings
        _setup_batched(solver, 
            P=cp.array(np.stack(Ps)),
            c=cp.array(rng.standard_normal((B, n))),
        )
        statuses = solver.solve()
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
        _setup_batched(solver, 
            P=cp.array(np.stack([q['P'] for q in qps])),
            c=cp.array(np.stack([q['c'] for q in qps])),
            A=cp.array(np.stack([q['A'] for q in qps])),
            b=cp.array(np.stack([q['b'] for q in qps])),
            G=cp.array(np.stack([q['G'] for q in qps])),
            h_u=cp.array(np.stack([q['h_u'] for q in qps])),
            h_l=cp.array(np.stack([q['h_l'] for q in qps])),
            x_u=cp.array(np.stack([q['x_u'] for q in qps])),
            x_l=cp.array(np.stack([q['x_l'] for q in qps])),
        )
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
        statuses = solver.solve()
        iters = [int(v) for v in solver.result.info.iter]

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
        statuses = solver.solve()
        iters = [int(v) for v in solver.result.info.iter]

        assert statuses[0] == Status.CUPIQP_SOLVED
        assert statuses[1] == Status.CUPIQP_MAX_ITER_REACHED
        # Solved member froze at its own (small) count, not the global max.
        assert iters[0] == ref_it
        assert iters[0] < max_iter - 1
        assert iters[0] < iters[1]
        # Never-terminating member runs the full loop: iter == max_iter - 1 (0-based).
        assert iters[1] == max_iter - 1

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
        _setup_batched(solver, 
            P=cp.array(np.stack([q['P'] for q in qps])),
            c=cp.array(np.stack([q['c'] for q in qps])),
            A=cp.array(np.stack([q['A'] for q in qps])),
            b=cp.array(np.stack([q['b'] for q in qps])),
            G=cp.array(np.stack([q['G'] for q in qps])),
            h_u=cp.array(np.stack([q['h_u'] for q in qps])),
            h_l=cp.array(np.stack([q['h_l'] for q in qps])),
            x_u=cp.array(np.stack([q['x_u'] for q in qps])),
            x_l=cp.array(np.stack([q['x_l'] for q in qps])),
        )
        statuses = solver.solve()
        iters = [int(v) for v in solver.result.info.iter]

        for i in range(len(seeds)):
            assert statuses[i] == Status.CUPIQP_SOLVED
            assert iters[i] == ref[i][1], (
                f"problem {i}: batched iter {iters[i]} != standalone {ref[i][1]}")
        assert len(set(iters)) > 1, f"expected per-problem iters to differ: {iters}"

