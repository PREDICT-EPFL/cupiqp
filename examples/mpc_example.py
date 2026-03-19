
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import sparse	


from cupiqp import SolverBase


INF = np.inf

# Discrete time model of a quadcopter
Ad = sparse.csc_matrix([
  [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
  [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
  [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
  [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
  [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
  [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
  [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
  [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
  [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
])
Bd = sparse.csc_matrix([
  [0.,      -0.0726,  0.,     0.0726],
  [-0.0726,  0.,      0.0726, 0.    ],
  [-0.0152,  0.0152, -0.0152, 0.0152],
  [-0.,     -0.0006, -0.,     0.0006],
  [0.0006,   0.,     -0.0006, 0.0000],
  [0.0106,   0.0106,  0.0106, 0.0106],
  [0,       -1.4512,  0.,     1.4512],
  [-1.4512,  0.,      1.4512, 0.    ],
  [-0.3049,  0.3049, -0.3049, 0.3049],
  [-0.,     -0.0236,  0.,     0.0236],
  [0.0236,   0.,     -0.0236, 0.    ],
  [0.2107,   0.2107,  0.2107, 0.2107]])
[nx, nu] = Bd.shape

# Constraints
u0 = 10.5916
umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
umax = np.array([13., 13., 13., 13.]) - u0

xmin = np.array([-np.pi/6,-np.pi/6,-INF,-INF,-INF,-1.,
                 -INF,-INF,-INF,-INF,-INF,-INF])
xmax = np.array([ np.pi/6, np.pi/6, INF, INF, INF, INF,
                  INF, INF, INF, INF, INF, INF])

# Objective function
Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
QN = Q
R = 0.1*sparse.eye(4)

# Initial and reference states
x0 = np.zeros(12)
xr = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

# Prediction horizon
N = 10

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])
# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

idx_l_inf = np.where(lineq <= -1e5)[0]
idx_u_inf = np.where(uineq >= 1e5)[0]
lineq[idx_l_inf] = -1e5 * np.ones_like(idx_l_inf)
uineq[idx_u_inf] = 1e5 * np.ones_like(idx_u_inf)

x_l = -INF * np.ones_like(q)
x_u = INF * np.ones_like(q)

x_u = 1e3 * np.ones_like(q)
x_l = -1e3 * np.ones_like(q)

# Solve with OSQP:
import osqp
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_starting=True, verbose=True)
res = prob.solve()

print("Run time: ", res.info.run_time)

import piqp
solver_cpu = piqp.SparseSolver()
solver_cpu.settings.verbose = True
solver_cpu.settings.iterative_refinement_always_enabled = True
solver_cpu.setup(P, q, Aeq, leq, Aineq, lineq, uineq, x_l, x_u)
solver_cpu.solve()


import cupy as cp
from cupyx.scipy.sparse import csr_matrix

solver = SolverBase()
solver.settings.kkt_solver = 'sparse_ldlt'
# solver.settings.debug = True
solver.settings.verbose = True
solver.settings.max_iter = 30
solver.settings.iterative_refinement_always_enabled = False

with cp.cuda.Device(0):
	solver.setup(
		P=csr_matrix(P),  # Convert scipy sparse to cupy sparse directly
		c=cp.array(q),
		A=csr_matrix(Aeq),  # Convert scipy sparse to cupy sparse directly
		b=cp.array(leq),
		G=csr_matrix(Aineq),  # Convert scipy sparse to cupy sparse directly
		h_u=cp.array(uineq),
		h_l=cp.array(lineq),
		x_u=cp.array(x_u),
		x_l=cp.array(x_l)
	)

	result = solver.solve()

print("status: ", solver._result.info.status)



solver = SolverBase()
solver.settings.kkt_solver = 'dense_cholesky'
# solver.settings.debug = True
solver.settings.verbose = True
solver.settings.max_iter = 30
solver.settings.iterative_refinement_always_enabled = False

with cp.cuda.Device(0):
	solver.setup(
		P=cp.array(P.toarray()),  # Convert scipy sparse to cupy dense directly
		c=cp.array(q),
		A=cp.array(Aeq.toarray()),
		b=cp.array(leq),
		G=cp.array(Aineq.toarray()),  # Convert scipy sparse to cupy sparse directly
		h_u=cp.array(uineq),
		h_l=cp.array(lineq),
		x_u=cp.array(x_u),
		x_l=cp.array(x_l)
	)

	result = solver.solve()

print("status: ", solver._result.info.status)

	


