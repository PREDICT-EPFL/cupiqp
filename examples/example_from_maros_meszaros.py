
import sys, os
sys.path.append('./')
sys.path.append('../')
import cupy as cp
from cupyx.scipy.sparse import csr_matrix

import scipy.sparse as sp
import scipy.io
import numpy as np

from cupiqp import SolverBase


# PROBLEM_FILE = "PRIMALC1"
PROBLEM_FILE = "PRIMALC5"
# PROBLEM_FILE = "TAME"
PROBLEM_FILE = "QSHARE1B"
# PROBLEM_FILE = "DPKLO1"  # no constraints
# PROBLEM_FILE = "QADLITTL"
PROBLEM_FILE = "YAO"

SPARSE = True
# SPARSE = False


# Load problem data from .mat file
path = os.path.dirname(os.path.abspath(__file__))
data = scipy.io.loadmat(path + "/../tests/data/maros_meszaros/" + PROBLEM_FILE + ".mat")

if not SPARSE:
	P = data['P'].todense() if sp.issparse(data['P']) else data['P']
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
	c = np.array(data['c'].flatten(), dtype=np.float64)
	A = sp.csc_matrix(data['A']) if 'A' in data else None
	b = np.array(data['b'].flatten(), dtype=np.float64) if 'b' in data else None
	G = sp.csc_matrix(data['G']) if 'G' in data else None
	h_l = np.array(data['h_l'].flatten(), dtype=np.float64) if 'h_l' in data else None
	h_u = np.array(data['h_u'].flatten(), dtype=np.float64) if 'h_u' in data else None
	x_l = np.array(data['x_l'].flatten(), dtype=np.float64) if 'x_l' in data else None
	x_u = np.array(data['x_u'].flatten(), dtype=np.float64) if 'x_u' in data else None


solver = SolverBase()
solver.settings.kkt_solver = 'dense_cholesky' if not SPARSE else 'sparse_ldlt'
solver.settings.debug = False
solver.settings.verbose = True
solver.settings.max_iter = 200

with cp.cuda.Device(0):
	if SPARSE:
		solver.setup(
			P=csr_matrix(P),  # Convert scipy sparse to cupy sparse directly
			c=cp.array(c),
			A=csr_matrix(A),  # Convert scipy sparse to cupy sparse directly
			b=cp.array(b),
			G=csr_matrix(G),  # Convert scipy sparse to cupy sparse directly
			h_u=cp.array(h_u),
			h_l=cp.array(h_l),
			x_u=cp.array(x_u),
			x_l=cp.array(x_l)
		)
	else:
		solver.setup(
			P=cp.array(P),
			c=cp.array(c),
			A=cp.array(A),
			b=cp.array(b),
			G=cp.array(G),
			h_u=cp.array(h_u),
			h_l=cp.array(h_l),
			x_u=cp.array(x_u),
			x_l=cp.array(x_l)
		)
	status = solver.solve()

print("Solver status: ", status)
