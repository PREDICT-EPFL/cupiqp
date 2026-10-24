"""Solve every Maros-Meszaros .mat problem with the sparse backend."""
import os

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from cupiqp import SparseSolver, Status
from tests.helpers import DATA_DIR, csr_triple, to_wp


def _problem_files():
    data_folder = DATA_DIR / "maros_meszaros"
    if not data_folder.exists():
        return []
    return sorted(str(f) for f in data_folder.glob("*.mat"))


@pytest.mark.slow
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

    solver = SparseSolver()
    solver.settings.max_iter = 250
    solver.settings.eps_abs = 1e-6
    solver.settings.iterative_refinement_always_enabled = True
    solver.setup(P=csr_triple(P), c=to_wp(c),
        A=csr_triple(A) if A is not None else None,
        b=to_wp(b) if b is not None else None,
        G=csr_triple(G) if G is not None else None,
        h_u=to_wp(h_u) if h_u is not None else None,
        h_l=to_wp(h_l) if h_l is not None else None,
        x_u=to_wp(x_u) if x_u is not None else None,
        x_l=to_wp(x_l) if x_l is not None else None,
    )
    solver.solve()
    status = solver.result.info.to_host().status
    assert status[0] == Status.CUPIQP_SOLVED, f"Solver failed with status {status}"
