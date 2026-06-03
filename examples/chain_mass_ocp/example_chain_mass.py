import sys, os
import argparse
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import piqp

from examples.chain_mass_ocp.chain_mass_ocp_problem import ChainMassOCPProblem
from cupiqp import SparseSolver, SparseLargeProblemSolver


def main():
    parser = argparse.ArgumentParser(description='Chain Mass OCP Example')
    parser.add_argument('--num_masses', type=int, default=20, help='Number of masses')
    parser.add_argument('--horizon', type=int, default=200, help='Horizon length')
    args = parser.parse_args()

    num_masses = args.num_masses
    horizon = args.horizon

    print(f"Running with num_masses={num_masses}, horizon={horizon}")

    # ---- Build problem data (numpy/scipy) ----
    chain_mass_ocp = ChainMassOCPProblem(
        M=num_masses, N=horizon, randomize_x0=False,
        use_u_diff_cost=True, use_u_diff_constr=True,
    )

    # ---- CPU piqp ----
    solver_cpu = piqp.SparseSolver()
    solver_cpu.settings.verbose = True
    solver_cpu.settings.max_iter = 50
    solver_cpu.settings.iterative_refinement_always_enabled = False
    solver_cpu.settings.iterative_refinement_max_iter = 0
    solver_cpu.setup(
        P=chain_mass_ocp.P, c=chain_mass_ocp.c,
        A=chain_mass_ocp.Aeq, b=chain_mass_ocp.beq,
        G=chain_mass_ocp.Aineq,
        h_u=chain_mass_ocp.bineq_ub,
        h_l=chain_mass_ocp.bineq_lb,
        x_u=chain_mass_ocp.xub, x_l=chain_mass_ocp.xlb,
    )

    t0 = time.perf_counter()
    status_cpu = solver_cpu.solve()
    t_cpu_solve = (time.perf_counter() - t0) * 1e3

    print(f"CPU piqp status: {status_cpu}")
    print(f"CPU piqp solve:  {t_cpu_solve:.2f} ms")

    # ---- GPU cupiqp (sparse backend) ----
    solver = SparseLargeProblemSolver()
    solver.settings.verbose = True
    solver.settings.max_iter = 100

    with cp.cuda.Device(0):
        solver.setup(
            P=csr_matrix(chain_mass_ocp.P), c=cp.array(chain_mass_ocp.c),
            A=csr_matrix(chain_mass_ocp.Aeq), b=cp.array(chain_mass_ocp.beq),
            G=csr_matrix(chain_mass_ocp.Aineq),
            h_u=cp.array(chain_mass_ocp.bineq_ub),
            h_l=cp.array(chain_mass_ocp.bineq_lb),
            x_u=cp.array(chain_mass_ocp.xub), x_l=cp.array(chain_mass_ocp.xlb),
        )

        evt_start = cp.cuda.Event()
        evt_end = cp.cuda.Event()
        evt_start.record()
        solver.solve()
        evt_end.record()
        evt_end.synchronize()
        t_gpu_solve = cp.cuda.get_elapsed_time(evt_start, evt_end)

    print(f"GPU cupiqp status: {solver._result.info.status}")
    print(f"GPU cupiqp solve:  {t_gpu_solve:.2f} ms")

    print(f"\n{'='*40}")
    print(f"Speedup (solve): {t_cpu_solve / t_gpu_solve:.2f}x")


if __name__ == "__main__":
    main()
