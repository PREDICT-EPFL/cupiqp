"""
Benchmark cupiqp solver on Maros-Meszaros QP problems.

Usage:
    python bench_maros_cupiqp.py <problem_name> [n_runs]

Example:
    python bench_maros_cupiqp.py CVXQP1_S
    python bench_maros_cupiqp.py CVXQP1_S 20
"""

import sys
import os
import numpy as np
import scipy.io
import scipy.sparse as sp

import cupy as cp
from cupyx.scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cupiqp import SolverBase


def load_piqp_mat(path):
    data = scipy.io.loadmat(path)
    P = sp.csc_matrix(data["P"], dtype=np.float64)
    c = np.array(data["c"].flatten(), dtype=np.float64)
    A = sp.csc_matrix(data["A"], dtype=np.float64)
    b = np.array(data["b"].flatten(), dtype=np.float64)
    G = sp.csc_matrix(data["G"], dtype=np.float64)
    h_l = np.array(data["h_l"].flatten(), dtype=np.float64)
    h_u = np.array(data["h_u"].flatten(), dtype=np.float64)
    x_l = np.array(data["x_l"].flatten(), dtype=np.float64)
    x_u = np.array(data["x_u"].flatten(), dtype=np.float64)
    return P, c, A, b, G, h_l, h_u, x_l, x_u


def main(problem_name="", n_runs=10):
    # Command line args override function defaults
    if len(sys.argv) > 1:
        problem_name = sys.argv[1]
    if len(sys.argv) > 2:
        n_runs = int(sys.argv[2])

    if not problem_name:
        print("Usage: python bench_maros_cupiqp.py <problem_name> [n_runs]")
        print("Example: python bench_maros_cupiqp.py CVXQP1_S 10")
        return

    name = problem_name

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mat_path = os.path.join(script_dir, "..", "tests", "data", "maros_meszaros", name + ".mat")

    if not os.path.isfile(mat_path):
        print(f"File not found: {mat_path}")
        return

    print(f"Loading {name} ...")
    P, c, A, b, G, h_l, h_u, x_l, x_u = load_piqp_mat(mat_path)
    n = P.shape[0]
    p = A.shape[0]
    m = G.shape[0]
    print(f"  n={n}, p={p}, m={m}")

    with cp.cuda.Device(0):
        solver = SolverBase()
        solver.settings.kkt_solver = "sparse_ldlt"
        solver.settings.verbose = True
        solver.settings.max_iter = 100

        solver.setup(
            P=csr_matrix(P),
            c=cp.array(c),
            A=csr_matrix(A) if p > 0 else None,
            b=cp.array(b) if p > 0 else None,
            G=csr_matrix(G) if m > 0 else None,
            h_u=cp.array(h_u) if m > 0 else None,
            h_l=cp.array(h_l) if m > 0 else None,
            x_u=cp.array(x_u),
            x_l=cp.array(x_l),
        )

        # Warmup / profiling solve (verbose)
        solver.solve()
        print("Warmup solve finished.")
        solver.settings.verbose = False

        if n_runs == 0:
            cp.cuda.Device(0).synchronize()
            cp.cuda.profiler.start()
            solver.solve()
            cp.cuda.Device(0).synchronize()
            cp.cuda.profiler.stop()

            status = solver.result.info.status
            obj = float(solver.result.info.primal_obj[0])
            iters = int(solver.result.info.iter)

            print(f"\n===== Results (profiling mode) =====")
            print(f"Problem:    {name}")
            print(f"Status:     {status}")
            print(f"Objective:  {obj}")
            print(f"Iterations: {iters}")
            return

        
        # Timed solves
        times_ms = []
        for i in range(n_runs):
            cp.cuda.Device(0).synchronize()
            evt_start = cp.cuda.Event()
            evt_end = cp.cuda.Event()
            evt_start.record()
            solver.solve()
            evt_end.record()
            evt_end.synchronize()
            times_ms.append(cp.cuda.get_elapsed_time(evt_start, evt_end))

        status = solver.result.info.status
        obj = float(solver.result.info.primal_obj[0])
        iters = int(solver.result.info.iter)

    times_ms = np.array(times_ms)

    print(f"\n===== Results =====")
    print(f"Problem:    {name}")
    print(f"Status:     {status}")
    print(f"Objective:  {obj}")
    print(f"Iterations: {iters}")
    print(f"Runs:       {n_runs}")
    print(f"Solve time (ms):")
    print(f"  Median:   {np.median(times_ms):.3f}")
    print(f"  Mean:     {np.mean(times_ms):.3f}")
    print(f"  Min:      {np.min(times_ms):.3f}")
    print(f"  Max:      {np.max(times_ms):.3f}")
    print(f"  Std:      {np.std(times_ms):.3f}")


if __name__ == "__main__":
    main()
