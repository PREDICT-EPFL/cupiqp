import pytest
import os
import glob
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cupiqp import SolverBase, Status
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix

import scipy.sparse as sp
import scipy.io


def get_problem_files():
    """Get all problem files from the maros_meszaros data folder"""
    data_folder = os.path.join(os.path.dirname(__file__), 'data/maros_meszaros')
    if not os.path.exists(data_folder):
        return []
    problem_files = glob.glob(os.path.join(data_folder, "*.mat"))
    problem_files.sort()
    return problem_files


@pytest.mark.parametrize("problem_file", get_problem_files(), ids=lambda f: os.path.basename(f))
def test_maros_meszaros_problem(problem_file):
    """Test that the solver can solve a single problem from the Maros-Meszaros set"""

    SPARSE = True
    # SPARSE = False
    MAX_PROBLEM_SIZE = 10_000  # skip problems larger than this size

    data = scipy.io.loadmat(problem_file)

    if not SPARSE:
        # P = data['P'].todense() if sp.issparse(data['P']) else data['P']
        P = data['P'] if sp.issparse(data['P']) else data['P']
        # skip large problems
        if P.shape[0] > MAX_PROBLEM_SIZE:
            pytest.skip(f"Skipping large problem with size {P.shape[0]}")
        P = P.todense()
        P = np.array(P)  # convert to np.ndarray
        c = np.array(data['c'].flatten(), dtype=np.float64)
        A = data['A'].todense() if 'A' in data else None
        A = np.array(A) if A is not None else None
        b = np.array(data['b'].flatten(), dtype=np.float64) if 'b' in data else None
        G = data['G'].todense() if 'G' in data else None
        G = np.array(G) if G is not None else None
        h_l = np.array(data['h_l'].flatten(), dtype=np.float64) if 'h_l' in data else None
        h_u = np.array(data['h_u'].flatten(), dtype=np.float64) if 'h_u' in data else None
        x_l = np.array(data['x_l'].flatten(), dtype=np.float64) if 'x_l' in data else None
        x_u = np.array(data['x_u'].flatten(), dtype=np.float64) if 'x_u' in data else None

    else:
        # -------- sparse data --------
        P = sp.csc_matrix(data['P'], dtype=np.float64)
        # Skip large problems
        if P.shape[0] > MAX_PROBLEM_SIZE:
            pytest.skip(f"Skipping large problem with size {P.shape[0]}")
        c = np.array(data['c'].flatten(), dtype=np.float64)
        A = sp.csc_matrix(data['A']) if 'A' in data else None
        b = np.array(data['b'].flatten(), dtype=np.float64) if 'b' in data else None
        G = sp.csc_matrix(data['G']) if 'G' in data else None
        h_l = np.array(data['h_l'].flatten(), dtype=np.float64) if 'h_l' in data else None
        h_u = np.array(data['h_u'].flatten(), dtype=np.float64) if 'h_u' in data else None
        x_l = np.array(data['x_l'].flatten(), dtype=np.float64) if 'x_l' in data else None
        x_u = np.array(data['x_u'].flatten(), dtype=np.float64) if 'x_u' in data else None

    with cp.cuda.Device(0):
        solver = SolverBase()
        solver.settings.kkt_solver = 'sparse_ldlt' if SPARSE else 'dense_cholesky'
        solver.settings.max_iter = 150
        # solver.settings.verbose = True

        with cp.cuda.Device(0):
            solver.setup(
                P=csr_matrix(P) if SPARSE else cp.array(P),
                c=cp.array(c),
                A=csr_matrix(A) if SPARSE else cp.array(A),
                b=cp.array(b),
                G=csr_matrix(G) if SPARSE else cp.array(G),
                h_u=cp.array(h_u),
                h_l=cp.array(h_l),
                x_u=cp.array(x_u),
                x_l=cp.array(x_l)
            )
            status = solver.solve()
    
    # Check that solver didn't fail
    assert status == Status.PIQP_SOLVED, f"Solver failed with status {status}"

if __name__ == "__main__":
    pytest.main([__file__])