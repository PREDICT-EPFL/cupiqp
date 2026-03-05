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
from cupiqp.solver import SolverBase


def main():
    parser = argparse.ArgumentParser(description='Chain Mass OCP Example')
    parser.add_argument('--num_masses', type=int, default=20, help='Number of masses')
    parser.add_argument('--horizon', type=int, default=500, help='Horizon length')
    parser.add_argument('--kkt_solver', type=str, default='multistage_block_cholesky',
                        choices=['multistage_block_cholesky', 'sparse_ldlt', 'dense_cholesky'],
                        help='KKT solver type')
    args = parser.parse_args()

    num_masses = args.num_masses
    horizon = args.horizon

    print(f"Running with num_masses={num_masses}, horizon={horizon}, kkt_solver={args.kkt_solver}")

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

    # ---- GPU cupiqp ----
    solver = SolverBase()
    solver.settings.kkt_solver = args.kkt_solver
    solver.settings.verbose = True
    solver.settings.max_iter = 100

    with cp.cuda.Device(0):
        if args.kkt_solver == 'multistage_block_cholesky':
            solver.settings.multistage_block_size = chain_mass_ocp.ms_block_size
            solver.setup(
                P=chain_mass_ocp.ms_P, c=chain_mass_ocp.ms_c,
                A=chain_mass_ocp.ms_A, b=chain_mass_ocp.ms_b,
                G=chain_mass_ocp.ms_G, h_u=chain_mass_ocp.ms_h_u,
                h_l=chain_mass_ocp.ms_h_l, x_u=chain_mass_ocp.ms_x_u,
                x_l=chain_mass_ocp.ms_x_l,
            )
        elif args.kkt_solver == 'dense_cholesky':
            solver.setup(
                P=cp.array(chain_mass_ocp.P.toarray()), c=cp.array(chain_mass_ocp.c),
                A=cp.array(chain_mass_ocp.Aeq.toarray()), b=cp.array(chain_mass_ocp.beq),
                G=cp.array(chain_mass_ocp.Aineq.toarray()),
                h_u=cp.array(chain_mass_ocp.bineq_ub),
                h_l=cp.array(chain_mass_ocp.bineq_lb),
                x_u=cp.array(chain_mass_ocp.xub), x_l=cp.array(chain_mass_ocp.xlb),
            )
        else:  # sparse_ldlt
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
