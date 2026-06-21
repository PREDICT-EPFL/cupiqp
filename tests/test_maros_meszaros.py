"""Solve every Maros-Meszaros .mat problem with the sparse backend."""
import glob
import os

import cupy as cp
import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp
from cupyx.scipy.sparse import csr_matrix

from cupiqp import SparseLargeProblemSolver, Status


def _problem_files():
    data_folder = os.path.join(os.path.dirname(__file__), 'data/maros_meszaros')
    if not os.path.exists(data_folder):
        return []
    return sorted(glob.glob(os.path.join(data_folder, "*.mat")))


@pytest.mark.parametrize(
    "problem_file", _problem_files(), ids=lambda f: os.path.basename(f),
)
def test_maros_meszaros_problem(problem_file):
    data = scipy.io.loadmat(problem_file)
    P = sp.csc_matrix(data['P'], dtype=np.float64)
    c = np.array(data['c'].flatten(), dtype=np.float64)
    A = sp.csc_matrix(data['A']) if 'A' in data else None
    b = np.array(data['b'].flatten(), dtype=np.float64) if 'b' in data else None
    G = sp.csc_matrix(data['G']) if 'G' in data else None
    h_l = np.array(data['h_l'].flatten(), dtype=np.float64) if 'h_l' in data else None
    h_u = np.array(data['h_u'].flatten(), dtype=np.float64) if 'h_u' in data else None
    x_l = np.array(data['x_l'].flatten(), dtype=np.float64) if 'x_l' in data else None
    x_u = np.array(data['x_u'].flatten(), dtype=np.float64) if 'x_u' in data else None

    solver = SparseLargeProblemSolver()
    solver.settings.max_iter = 250
    solver.settings.eps_abs = 1e-6
    solver.settings.iterative_refinement_always_enabled = True
    solver.setup(
        P=csr_matrix(P), c=cp.array(c),
        A=csr_matrix(A) if A is not None else None,
        b=cp.array(b) if b is not None else None,
        G=csr_matrix(G) if G is not None else None,
        h_u=cp.array(h_u) if h_u is not None else None,
        h_l=cp.array(h_l) if h_l is not None else None,
        x_u=cp.array(x_u) if x_u is not None else None,
        x_l=cp.array(x_l) if x_l is not None else None,
    )
    status = solver.solve()
    assert status[0] == Status.CUPIQP_SOLVED, f"Solver failed with status {status}"
