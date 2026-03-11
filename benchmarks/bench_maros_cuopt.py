"""
Benchmark cuOpt solver on Maros-Meszaros QP problems.

Usage:
    python bench_maros_cuopt.py <problem_name> [n_runs]

Example:
    python bench_maros_cuopt.py CVXQP1_S
    python bench_maros_cuopt.py CVXQP1_S 20
"""

import sys
import os
import re
import ctypes
import numpy as np
import scipy.io
import scipy.sparse as sp

from cuopt.linear_programming.problem import Problem, QuadraticExpression, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings


# ---------------------------------------------------------------------------
# CUDA profiler helpers (for nsys --capture-range=cudaProfilerApi)
# ---------------------------------------------------------------------------

_cudart = ctypes.CDLL("libcudart.so")


def cuda_profiler_start():
    _cudart.cudaProfilerStart()


def cuda_profiler_stop():
    _cudart.cudaProfilerStop()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# cuOpt problem construction
# ---------------------------------------------------------------------------

def build_cuopt_problem(name, P, c, A, b, G, h_l, h_u, x_l, x_u):
    """Convert PIQP-format QP data into a cuOpt Problem object."""
    n = P.shape[0]
    p = A.shape[0]
    m = G.shape[0]

    prob = Problem(name)

    # 1. Create variables with bounds
    variables = []
    for i in range(n):
        lb = float(x_l[i]) if np.isfinite(x_l[i]) else None
        ub = float(x_u[i]) if np.isfinite(x_u[i]) else None
        variables.append(prob.addVariable(lb=lb, ub=ub))

    # 2. Quadratic objective: min 0.5 x'Px + c'x
    #    .mat files store P as upper triangular. Symmetrise first.
    P_full = P + P.T - sp.diags(P.diagonal())
    P_coo = P_full.tocoo()
    qvars1 = [variables[i] for i in P_coo.row]
    qvars2 = [variables[j] for j in P_coo.col]
    qcoeffs = (0.5 * P_coo.data).tolist()

    quad_expr = QuadraticExpression(
        qvars1=qvars1, qvars2=qvars2, qcoefficients=qcoeffs,
        vars=variables, coefficients=c.tolist(),
    )
    prob.setObjective(quad_expr, sense=MINIMIZE)

    # 3. Equality constraints: Ax = b
    if p > 0:
        A_csr = sp.csr_matrix(A)
        for i in range(p):
            s, e = A_csr.indptr[i], A_csr.indptr[i + 1]
            cols = A_csr.indices[s:e]
            vals = A_csr.data[s:e]
            expr = sum(float(v) * variables[int(j)] for j, v in zip(cols, vals))
            prob.addConstraint(expr == float(b[i]))

    # 4. Inequality constraints: h_l <= Gx <= h_u
    if m > 0:
        G_csr = sp.csr_matrix(G)
        for i in range(m):
            s, e = G_csr.indptr[i], G_csr.indptr[i + 1]
            cols = G_csr.indices[s:e]
            vals = G_csr.data[s:e]
            expr = sum(float(v) * variables[int(j)] for j, v in zip(cols, vals))
            if np.isfinite(h_u[i]):
                prob.addConstraint(expr <= float(h_u[i]))
            if np.isfinite(h_l[i]):
                prob.addConstraint(expr >= float(h_l[i]))

    return prob


# ---------------------------------------------------------------------------
# Iteration count extraction (cuOpt logs to stdout from C library)
# ---------------------------------------------------------------------------

def solve_and_capture_iters(prob, settings_verbose, result_prefix):
    """Solve with verbose settings and capture iteration count from log."""
    log_file = result_prefix + "_cuopt_log.txt"
    stdout_fd = os.dup(1)
    try:
        with open(log_file, "w") as f:
            os.dup2(f.fileno(), 1)
            prob.solve(settings_verbose)
            os.dup2(stdout_fd, 1)
    finally:
        os.close(stdout_fd)

    iters = -1
    try:
        with open(log_file) as f:
            log = f.read()
        m = re.search(r"found in (\d+) iterations", log)
        if m:
            iters = int(m.group(1))
    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)

    return iters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(problem_name="", n_runs=10):
    # Command line args override function defaults
    if len(sys.argv) > 1:
        problem_name = sys.argv[1]
    if len(sys.argv) > 2:
        n_runs = int(sys.argv[2])

    if not problem_name:
        print("Usage: python bench_maros_cuopt.py <problem_name> [n_runs]")
        print("Example: python bench_maros_cuopt.py CVXQP1_S 10")
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

    print("Building cuOpt problem ...")
    prob = build_cuopt_problem(name, P, c, A, b, G, h_l, h_u, x_l, x_u)

    settings_quiet = SolverSettings()
    settings_quiet.set_optimality_tolerance(1e-8)
    settings_quiet.set_parameter("log_to_console", "0")

    settings_verbose = SolverSettings()
    settings_verbose.set_optimality_tolerance(1e-8)

    # Warmup solve (verbose)
    prob.solve(settings_verbose)
    print("Warmup solve finished.")

    if n_runs == 0:
        # Profiling mode: single solve bracketed by CUDA profiler markers
        cuda_profiler_start()
        prob.solve(settings_quiet)
        cuda_profiler_stop()

        status = prob.Status.name  # e.g. "Optimal", "NoTermination"
        obj = float(prob.ObjValue) if status == "Optimal" else float("nan")
        solve_time = prob.SolveTime * 1e3 if not np.isnan(prob.SolveTime) else float("nan")

        # Capture iteration count from a verbose solve
        iters = solve_and_capture_iters(prob, settings_verbose, os.path.join(script_dir, name))

        print(f"\n===== Results (profiling mode) =====")
        print(f"Problem:    {name}")
        print(f"Status:     {status}")
        print(f"Objective:  {obj}")
        print(f"Iterations: {iters}")
        if not np.isnan(solve_time):
            print(f"Solve time: {solve_time:.3f} ms")
        return

    # Timed solves
    times_ms = []
    for i in range(n_runs):
        prob.solve(settings_quiet)
        times_ms.append(prob.SolveTime * 1e3)  # seconds → ms

    times_ms = np.array(times_ms)

    status = prob.Status.name
    obj = float(prob.ObjValue) if status == "Optimal" else float("nan")

    # Capture iteration count from a verbose solve
    iters = solve_and_capture_iters(prob, settings_verbose, os.path.join(script_dir, name))

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
