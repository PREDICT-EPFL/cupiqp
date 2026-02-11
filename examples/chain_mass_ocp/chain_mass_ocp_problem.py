import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag
from .chain_mass_system import ChainMassSystem
from .qp_problem import QPProblem
from .ocp_problem import OCPProblem

class ChainMassOCPProblem(QPProblem, OCPProblem):
    def __init__(self, M, N, nu=None, use_u_diff_cost=False, use_u_diff_constr=False, randomize_x0=True):
        self.system = ChainMassSystem(M, N, nu)
        self.use_u_diff_cost = use_u_diff_cost
        self.use_u_diff_constr = use_u_diff_constr
        self._randomize_x0 = randomize_x0
        
        self._setup_qp()
        self._setup_ocp()

    def randomize_x0(self):
        x0_bound = np.random.uniform(0.5, 1.5)
        self.x0 = np.random.uniform(-x0_bound, x0_bound, self.system.nx)
        self.xlb[0:self.system.nx] = self.x0
        self.xub[0:self.system.nx] = self.x0

    def _setup_qp(self):
        N = self.system.N
        nx = self.system.nx
        nu = self.system.nu

        dim = N * (nx + nu) + nx

        self.P = sp.csc_matrix((dim, dim))
        self.c = np.zeros(dim)
        self.Aeq = sp.csc_matrix((N * nx, dim))
        self.beq = np.zeros(N * nx)
        if self.use_u_diff_constr:
            self.Aineq = sp.csc_matrix(((N - 1) * nu, dim))
            self.bineq_lb = np.zeros((N - 1) * nu)
            self.bineq_ub = np.zeros((N - 1) * nu)
        else:
            self.Aineq = sp.csc_matrix((0, dim))
            self.bineq_lb = np.zeros(0)
            self.bineq_ub = np.zeros(0)
        self.xlb = np.zeros(dim)
        self.xub = np.zeros(dim)
        
        # Initial condition
        if self._randomize_x0:
            self.randomize_x0()
        else:
            self.x0 = np.ones(self.system.nx)
            self.xlb[0:self.system.nx] = self.x0
            self.xub[0:self.system.nx] = self.x0
        
        for i in range(N):
            # Cost matrices
            self.P[i*(nx+nu):i*(nx+nu)+nx, 
                   i*(nx+nu):i*(nx+nu)+nx] = self.system.Q
            
            if self.use_u_diff_cost:
                self.P[i*(nx+nu)+nx:i*(nx+nu)+nx+nu, 
                       i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = self.system.R + self.system.R_diff
            else:
                self.P[i*(nx+nu)+nx:i*(nx+nu)+nx+nu, 
                       i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = self.system.R
                
            if self.use_u_diff_cost and i < N - 1:
                self.P[i*(nx+nu)+nx:i*(nx+nu)+nx+nu, 
                       (i+1)*(nx+nu)+nx:(i+1)*(nx+nu)+nx+nu] = -self.system.R_diff
            
            # Dynamics constraints
            self.Aeq[i*nx:(i+1)*nx, i*(nx+nu):i*(nx+nu)+nx] = self.system.Ad
            self.Aeq[i*nx:(i+1)*nx, i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = self.system.Bd
            self.Aeq[i*nx:(i+1)*nx, (i+1)*(nx+nu):(i+1)*(nx+nu)+nx] = -np.eye(nx)
            
            # Bounds
            self.xlb[i*(nx+nu)+nx:(i+1)*(nx+nu)] = -self.system.nu_max
            self.xub[i*(nx+nu)+nx:(i+1)*(nx+nu)] = self.system.nu_max
            self.xlb[(i+1)*(nx+nu):(i+1)*(nx+nu)+nx] = -self.system.nx_max
            self.xub[(i+1)*(nx+nu):(i+1)*(nx+nu)+nx] = self.system.nx_max
            
            # Input rate constraints
            if self.use_u_diff_constr and i < N - 1:
                self.Aineq[i*nu:i*nu+nu, 
                           i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = np.eye(nu)
                self.Aineq[i*nu:i*nu+nu, 
                           (i+1)*(nx+nu)+nx:(i+1)*(nx+nu)+nx+nu] = -np.eye(nu)
                self.bineq_lb[i*nu:i*nu+nu] = -self.system.nu_diff_max
                self.bineq_ub[i*nu:i*nu+nu] = self.system.nu_diff_max
        
        # Terminal cost
        self.P[N*(nx+nu):N*(nx+nu)+nx, 
               N*(nx+nu):N*(nx+nu)+nx] = self.system.P
        
    def _setup_ocp(self):
        self.N = self.system.N
        nx = self.system.nx
        nu = self.system.nu

        if self.use_u_diff_cost or self.use_u_diff_constr:
            self.nx = nx + nu
            self.nu = nu

            self.A = np.block([[self.system.Ad, np.zeros((nx, nu))],
                               [np.zeros((nu, nx)), np.zeros((nu, nu))]])
            self.B = np.block([[self.system.Bd], [np.eye(nu)]])

            if self.use_u_diff_cost:
                self.Q = block_diag(self.system.Q, self.system.R_diff)
                self.QN = block_diag(self.system.P, self.system.R_diff)
                self.R = self.system.R + self.system.R_diff
                self.S = np.zeros((nu, nx + nu))
                self.S[:, nx:] = -self.system.R_diff
            else:
                self.Q = block_diag(self.system.Q, np.zeros((nu, nu)))
                self.QN = block_diag(self.system.P, np.zeros((nu, nu)))
                self.R = self.system.R
                self.S = np.zeros((nu, nx))

            if self.use_u_diff_constr:
                self.C = np.zeros((nu, nx + nu))
                self.C[:, nx:] = -np.eye(nu)
                self.D = np.eye(nu)
                self.gl = -self.system.nu_diff_max * np.ones(nu)
                self.gu = self.system.nu_diff_max * np.ones(nu)
            else:
                self.C = np.zeros((0, nx + nu))
                self.D = np.zeros((0, nu))
                self.gl = np.zeros(0)
                self.gu = np.zeros(0)
        else:
            self.nx = nx
            self.nu = nu

            self.A = self.system.Ad
            self.B = self.system.Bd

            self.Q = self.system.Q
            self.QN = self.system.P
            self.R = self.system.R
            self.S = np.zeros((nu, nx))

            self.C = np.zeros((0, nx))
            self.D = np.zeros((0, nu))
            self.gl = np.zeros(0)
            self.gu = np.zeros(0)

        self.xl = -self.system.nx_max * np.ones(nx)
        self.xu = self.system.nx_max * np.ones(nx)

        self.ul = -self.system.nu_max * np.ones(nu)
        self.uu = self.system.nu_max * np.ones(nu)

    def get_solution_from_qp_solution(self, x: np.ndarray):
        N = self.system.N
        nx = self.system.nx
        nu = self.system.nu
        
        X = np.zeros((nx, N + 1))
        U = np.zeros((nu, N))
        for i in range(N):
            X[:, i] = x[i*(nx+nu):i*(nx+nu)+nx]
            U[:, i] = x[i*(nx+nu)+nx:i*(nx+nu)+nx+nu]
        X[:, N] = x[N*(nx+nu):N*(nx+nu)+nx]

        return X, U

    def get_solution_from_ocp_solution(self, X: np.ndarray, U: np.ndarray):
        return X[0:self.system.nx, :], U