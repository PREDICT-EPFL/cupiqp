"""cuOpt benchmark subprocess for the chain mass OCP problem.

Called from benchmark_cupiqp_vs_cuopt.py with problem data passed via .npz files.
Runs in a separate process to avoid CUDA context conflicts with CuPy/Warp.

Usage:
    python bench_cuopt.py <problem.npz> <result.npz>
"""

import sys
import os
import re
import numpy as np
import scipy.sparse as sp

from cuopt.linear_programming.problem import Problem, QuadraticExpression, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings


def build_problem(data):
    """Build cuOpt Problem from loaded .npz data."""
    P_indptr = data['P_indptr']
    P_indices = data['P_indices']
    P_data = data['P_data']
    P_shape = tuple(data['P_shape'])

    c = data['c']
    xlb = data['xlb']
    xub = data['xub']

    n = P_shape[0]
    prob = Problem("chain_mass_ocp")

    # Create variables with bounds
    variables = []
    for i in range(n):
        lb = float(xlb[i]) if np.isfinite(xlb[i]) else None
        ub = float(xub[i]) if np.isfinite(xub[i]) else None
        variables.append(prob.addVariable(lb=lb, ub=ub))

    # Quadratic objective: 0.5 x'Px + c'x
    P_csr = sp.csr_matrix((P_data, P_indices, P_indptr), shape=P_shape)
    P_coo = P_csr.tocoo()
    qvars1 = [variables[i] for i in P_coo.row]
    qvars2 = [variables[j] for j in P_coo.col]
    qcoeffs = (0.5 * P_coo.data).tolist()

    quad_expr = QuadraticExpression(
        qvars1=qvars1, qvars2=qvars2, qcoefficients=qcoeffs,
        vars=variables, coefficients=c.tolist(),
    )
    prob.setObjective(quad_expr, sense=MINIMIZE)

    # Equality constraints
    n_eq = int(data['n_eq'])
    if n_eq > 0:
        Aeq_csr = sp.csr_matrix((data['Aeq_data'], data['Aeq_indices'],
                                  data['Aeq_indptr']),
                                 shape=tuple(data['Aeq_shape']))
        beq = data['beq']
        for i in range(n_eq):
            s, e = Aeq_csr.indptr[i], Aeq_csr.indptr[i + 1]
            cols = Aeq_csr.indices[s:e]
            vals = Aeq_csr.data[s:e]
            expr = sum(float(v) * variables[int(j)] for j, v in zip(cols, vals))
            prob.addConstraint(expr == float(beq[i]))

    # Inequality constraints (cuopt does not support double-sided inequalities, so we add separate constraints for upper and lower bounds)
    n_ineq = int(data['n_ineq'])
    if n_ineq > 0:
        Aineq_csr = sp.csr_matrix((data['Aineq_data'], data['Aineq_indices'],
                                    data['Aineq_indptr']),
                                   shape=tuple(data['Aineq_shape']))
        bineq_lb = data['bineq_lb']
        bineq_ub = data['bineq_ub']
        for i in range(n_ineq):
            s, e = Aineq_csr.indptr[i], Aineq_csr.indptr[i + 1]
            cols = Aineq_csr.indices[s:e]
            vals = Aineq_csr.data[s:e]
            expr = sum(float(v) * variables[int(j)] for j, v in zip(cols, vals))
            if np.isfinite(bineq_ub[i]):
                prob.addConstraint(expr <= float(bineq_ub[i]))
            if np.isfinite(bineq_lb[i]):
                prob.addConstraint(expr >= float(bineq_lb[i]))

    return prob


def main():
    problem_file = sys.argv[1]
    result_file = sys.argv[2]

    data = np.load(problem_file, allow_pickle=True)
    n_runs = int(data['n_runs'])
    verbose = bool(data['verbose'])

    prob = build_problem(data)

    settings_quiet = SolverSettings()
    settings_quiet.set_optimality_tolerance(1e-8)
    settings_quiet.set_parameter("log_to_console", "0")

    settings_verbose = SolverSettings()
    settings_verbose.set_optimality_tolerance(1e-8)

    # Warmup solve (quiet)
    prob.solve(settings_quiet)

    # Timed runs (quiet)
    times = []
    for _ in range(n_runs):
        prob.solve(settings_quiet)
        times.append(prob.SolveTime * 1e3)  # seconds → ms

    # Final solve with verbose to capture iteration count
    # Redirect stdout at fd level to capture C library output
    log_file = result_file.replace('_result.npz', '_log.txt')
    stdout_fd = os.dup(1)
    with open(log_file, 'w') as f:
        os.dup2(f.fileno(), 1)
        prob.solve(settings_verbose)
        os.dup2(stdout_fd, 1)
    os.close(stdout_fd)

    iters = -1
    try:
        with open(log_file) as f:
            log = f.read()
        m = re.search(r'found in (\d+) iterations', log)
        if m:
            iters = int(m.group(1))
        if verbose:
            print(log, file=sys.stderr)
    finally:
        if os.path.exists(log_file):
            os.unlink(log_file)

    obj = float(prob.ObjValue) if prob.ObjValue is not None else float('nan')

    np.savez(result_file,
             times=np.array(times),
             obj=np.array([obj]),
             iters=np.array([iters]),
             status=np.array(['OPTIMAL' if prob.ObjValue is not None else 'FAILED']))

    if verbose:
        print(f"cuOpt done: obj={obj:.6e}, iters={iters}, times={times}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
