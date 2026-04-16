"""
Benchmark: Batched constrained least squares.

    min  ||Fx - g||^2
    s.t. Ax = b,  x >= 0

Compares cuPIQP, qpax, and jaxopt (BoxOSQP) using the unified
BatchedQPSolver interface.
"""
import sys
from pathlib import Path
_repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)  # cupiqp repo root
_batched_dir = str(Path(__file__).resolve().parent.parent)              # benchmarks/batched/
sys.path.insert(0, _repo_root)
sys.path.insert(0, _batched_dir)

import numpy as np
from batched_solver_interface import (
    BatchedQPData, CupiqpBatchedSolver, QpaxBatchedSolver, QpthBatchedSolver, MoreauBatchedSolver,
)


def make_constr_least_square_data(B, n, m_F, p, seed=42):
    """Generate B constrained least-squares QPs."""
    rng = np.random.default_rng(seed)

    F = rng.standard_normal((m_F, n))
    A_eq = rng.standard_normal((p, n))

    Fs = np.tile(F[None], (B, 1, 1))
    gs = rng.standard_normal((B, m_F))
    As = np.tile(A_eq[None], (B, 1, 1))
    bs = (As @ np.ones((B, n, 1))).squeeze(-1) + 0.1 * rng.standard_normal((B, p))

    Ps = 2 * np.einsum('bji,bjk->bik', Fs, Fs)
    Ps = (Ps + Ps.transpose(0, 2, 1)) / 2
    cs = -2 * np.einsum('bji,bj->bi', Fs, gs)

    return BatchedQPData(P=Ps, c=cs, A=As, b=bs, x_l=np.zeros((B, n)))


def main():
    n = 20  # number of optimization variables, x is in R^n
    row_F = 40
    row_A = 5
    batch_sizes = [10, 100, 500, 1000]
    n_repeats = 5
    tol = 1e-6
    max_iter = 200

    assert row_F > row_A

    print(f"Constrained Least Squares: n={n}, row_F={row_F}, row_A={row_A}")
    print(f"  min ||Fx - g||^2  s.t.  Ax = b,  x >= 0")
    print(f"  tol={tol}, max_iter={max_iter}, {n_repeats} repeats, median time reported")
    print()

    solvers_header = "  ".join(f"{'--- ' + name + ' ---':>30s}" for name in ["cuPIQP", "qpax", "qpth", "moreau"])
    print(f"{'':>6s}  {solvers_header}  {'--- speedup (solve) ---':>32s}")
    col = f"{'setup':>10s}  {'solve':>10s}  {'fail':>9s}"
    header = f"{'B':>6s}  {col}  {col}  {col}  {col}  {'vs_qpax':>8s}  {'vs_qpth':>8s}  {'vs_moreau':>10s}"
    print(header)
    print("-" * len(header))

    for B in batch_sizes:
        data = make_constr_least_square_data(B, n, row_F, row_A)

        # cuPIQP
        cu = CupiqpBatchedSolver(tol_abs=tol, max_iter=max_iter)
        r_cu = cu.benchmark(data, n_repeats)

        # qpax
        qp = QpaxBatchedSolver(tol_abs=tol, max_iter=max_iter)
        r_qp = qp.benchmark(data, n_repeats)

        # qpth
        qt = QpthBatchedSolver(tol_abs=tol, max_iter=max_iter)
        r_qt = qt.benchmark(data, n_repeats)

        # moreau
        mr = MoreauBatchedSolver(tol_abs=tol, max_iter=max_iter)
        r_mr = mr.benchmark(data, n_repeats)

        sp_qpax = r_qp.solve_time_ms / r_cu.solve_time_ms if r_cu.solve_time_ms > 0 else float('inf')
        sp_qpth = r_qt.solve_time_ms / r_cu.solve_time_ms if r_cu.solve_time_ms > 0 else float('inf')
        sp_moreau = r_mr.solve_time_ms / r_cu.solve_time_ms if r_cu.solve_time_ms > 0 else float('inf')

        def fmt(r, B):
            fail = B - r.n_solved
            return f"{r.setup_time_ms:10.2f}  {r.solve_time_ms:10.2f}  {fail:4d}/{B:<4d}"

        print(f"{B:6d}  {fmt(r_cu, B)}  {fmt(r_qp, B)}  {fmt(r_qt, B)}  {fmt(r_mr, B)}"
              f"  {sp_qpax:7.1f}x  {sp_qpth:7.1f}x  {sp_moreau:9.1f}x")


if __name__ == "__main__":
    main()
