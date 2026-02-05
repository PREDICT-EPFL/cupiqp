import sys, os
sys.path.append('./')
sys.path.append('../')

import cupy as cp
from cupyx.scipy.sparse import csr_matrix, bmat

from examples.chain_mass_ocp.chain_mass_ocp_problem import ChainMassOCPProblem
from cupiqp.solver import SolverBase

def main():
    num_masses = 2
    horizon = 200
    chain_mass_ocp = ChainMassOCPProblem(M=num_masses, N=horizon, randomize_x0=True, use_u_diff_cost=True, use_u_diff_constr=True)
    block_size = 3*num_masses - 1
    padding_size = (horizon + 1) * block_size - chain_mass_ocp.P.shape[0]

    # prepare data:
    P = bmat([
        [chain_mass_ocp.P, csr_matrix((chain_mass_ocp.P.shape[0], padding_size))],
        [csr_matrix((padding_size, chain_mass_ocp.P.shape[1])), csr_matrix((padding_size, padding_size))]], format='csr')
    c = cp.concatenate([cp.array(chain_mass_ocp.c), cp.zeros(padding_size)])
    A = bmat([[chain_mass_ocp.Aeq, csr_matrix((chain_mass_ocp.Aeq.shape[0], padding_size))]], format='csr')
    G = bmat([[chain_mass_ocp.Aineq, csr_matrix((chain_mass_ocp.Aineq.shape[0], padding_size))]], format='csr')
    xlb = cp.concatenate([cp.array(chain_mass_ocp.xlb), -cp.inf * cp.ones(padding_size)])
    xub = cp.concatenate([cp.array(chain_mass_ocp.xub), cp.inf * cp.ones(padding_size)])

    # solve QP
    solver = SolverBase()
    solver.settings.kkt_solver = 'multistage_block_cholesky'
    # solver.settings.kkt_solver = 'sparse_ldlt'
    # solver.settings.kkt_solver = 'dense_cholesky'
    # solver.settings.debug = True
    solver.settings.verbose = True
    solver.settings.max_iter = 50
    solver.settings.multistage_block_size = block_size
    with cp.cuda.Device(0):
        if solver.settings.kkt_solver == 'dense_cholesky':
            solver.setup(
                P=P.toarray(),
                c=c,
                A=A.toarray(),
                b=cp.array(chain_mass_ocp.beq),
                G=G.toarray(),
                h_u=cp.array(chain_mass_ocp.bineq_ub),
                h_l=cp.array(chain_mass_ocp.bineq_lb),
                x_u=xub,
                x_l=xlb
            )

        else:
            solver.setup(
                P=csr_matrix(P),
                c=c,
                A=csr_matrix(A),
                b=cp.array(chain_mass_ocp.beq),
                G=csr_matrix(G),
                h_u=cp.array(chain_mass_ocp.bineq_ub),
                h_l=cp.array(chain_mass_ocp.bineq_lb),
                x_u=xub,
                x_l=xlb
            )

        solver.solve()

    print("status: ", solver._result.info.status)


    


if __name__ == "__main__":
    main()