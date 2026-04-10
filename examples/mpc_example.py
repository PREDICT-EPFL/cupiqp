"""MPC benchmark: quadcopter stabilization using cupiqp (PyTorch/Warp backends)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import time
from scipy import sparse

INF = np.inf
PIQP_INF = 1e20

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

# Cast MPC problem to a QP
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])

Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])

Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])

# Clamp to PIQP sentinel
lineq = np.clip(lineq, -PIQP_INF, PIQP_INF)
uineq = np.clip(uineq, -PIQP_INF, PIQP_INF)

x_l = -1e3 * np.ones_like(q)
x_u = 1e3 * np.ones_like(q)

n_vars = P.shape[0]
print(f"QP: n={n_vars}, p={Aeq.shape[0]}, m={Aineq.shape[0]}")

# Helper: numpy -> torch GPU tensor
def to_gpu(arr):
    return torch.tensor(np.asarray(arr), dtype=torch.float64, device='cuda')

# ============================================================
# CPU PIQP baseline
# ============================================================
print("\n=== CPU PIQP (sparse) ===")
import piqp
solver_cpu = piqp.SparseSolver()
solver_cpu.settings.verbose = False
solver_cpu.setup(P, q, Aeq, leq, Aineq, lineq, uineq, x_l, x_u)
solver_cpu.solve()  # warmup

times_cpu = []
for _ in range(10):
    t0 = time.time()
    solver_cpu.solve()
    t1 = time.time()
    times_cpu.append((t1-t0)*1000)
print(f"CPU PIQP: {np.mean(times_cpu):.3f} +/- {np.std(times_cpu):.3f} ms (min={np.min(times_cpu):.3f})")

# ============================================================
# cupiqp sparse (cuDSS) — sparse backend still uses CuPy CSR
# ============================================================
print("\n=== cupiqp sparse (cuDSS) ===")
from cupyx.scipy.sparse import csr_matrix
from cupiqp import SolverBase

solver_sparse = SolverBase()
solver_sparse.settings.kkt_solver = 'sparse_ldlt'
solver_sparse.settings.verbose = False
solver_sparse.settings.max_iter = 100
solver_sparse.settings.enable_cuda_graph = False  # mixed CuPy CSR + torch tensors prevents graph capture
solver_sparse.setup(
    P=csr_matrix(P), c=to_gpu(q),
    A=csr_matrix(Aeq), b=to_gpu(leq),
    G=csr_matrix(Aineq), h_u=to_gpu(uineq), h_l=to_gpu(lineq),
    x_u=to_gpu(x_u), x_l=to_gpu(x_l))
solver_sparse.solve(); torch.cuda.synchronize()  # warmup

times_sparse = []
for _ in range(10):
    t0 = time.time()
    solver_sparse.solve()
    torch.cuda.synchronize()
    t1 = time.time()
    times_sparse.append((t1-t0)*1000)
print(f"cupiqp sparse: {np.mean(times_sparse):.3f} +/- {np.std(times_sparse):.3f} ms (min={np.min(times_sparse):.3f})")
print(f"  status={solver_sparse._result.info.status}, iter={solver_sparse._result.info.iter}")

# ============================================================
# cupiqp dense (Cholesky) — fully PyTorch
# ============================================================
print("\n=== cupiqp dense (Cholesky) ===")
solver_dense = SolverBase()
solver_dense.settings.kkt_solver = 'dense_cholesky'
solver_dense.settings.verbose = False
solver_dense.settings.max_iter = 100
solver_dense.setup(
    P=to_gpu(P.toarray()), c=to_gpu(q),
    A=to_gpu(Aeq.toarray()), b=to_gpu(leq),
    G=to_gpu(Aineq.toarray()), h_u=to_gpu(uineq), h_l=to_gpu(lineq),
    x_u=to_gpu(x_u), x_l=to_gpu(x_l))
solver_dense.solve(); torch.cuda.synchronize()  # warmup

times_dense = []
for _ in range(10):
    t0 = time.time()
    solver_dense.solve()
    torch.cuda.synchronize()
    t1 = time.time()
    times_dense.append((t1-t0)*1000)
print(f"cupiqp dense: {np.mean(times_dense):.3f} +/- {np.std(times_dense):.3f} ms (min={np.min(times_dense):.3f})")
print(f"  status={solver_dense._result.info.status}, iter={solver_dense._result.info.iter}")

# ============================================================
# Summary
# ============================================================
print(f"\n=== Summary ===")
print(f"CPU PIQP (sparse):   {np.mean(times_cpu):.3f} ms")
print(f"cupiqp sparse:       {np.mean(times_sparse):.3f} ms")
print(f"cupiqp dense:        {np.mean(times_dense):.3f} ms")
