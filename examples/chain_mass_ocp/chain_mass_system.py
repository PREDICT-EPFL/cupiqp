import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

class ChainMassSystem:
    def __init__(self, M, N, nu=None, m=1.0, c=0.1, k=1.0):
        self.M = M
        self.N = N

        # System parameters
        self.m = m # mass
        self.c = c # damping
        self.k = k # spring constant

        self.nx_max = 4.0
        self.nu_max = 0.5
        self.nu_diff_max = 0.1

        # System dimensions
        self.nx = 2 * M
        self.nu = M - 1 if nu is None else nu
        assert self.nu >= 1 and self.nu <= M, 'nu must be in range [1, M]'
        
        # Initialize system matrices
        self._setup_system()
        
    def _setup_system(self):
        # Build continuous system matrices
        L = np.eye(self.M, k=-1)
        self.A = np.block([
            [np.zeros((self.M, self.M)),                                          np.eye(self.M)],
            [(-2 * self.k * np.eye(self.M) + self.k * L + self.k * L.T) / self.m, (-2 * self.c * np.eye(self.M)) / self.m]
        ])
        self.B = np.block([[np.zeros((2 * self.M - self.nu, self.nu))], [np.eye(self.nu)]])
        self.C = np.eye(self.nx)
        self.D = np.zeros((self.nx, self.nu))
        
        # Discretize system
        dt = 0.5
        d_sys = cont2discrete((self.A, self.B, self.C, self.D), dt, method='zoh')
        self.Ad = d_sys[0]
        self.Bd = d_sys[1]
        
        # Cost matrices
        self.Q = 1e3 * np.eye(self.nx)
        self.R = 1e-1 * np.eye(self.nu)
        self.R_diff = 1e-1 * np.eye(self.nu)
        self.P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)