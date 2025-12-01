import numpy as np

from .settings import Settings
from .data import Data
from .results import Result, Status, Variables
from .kkt_systems import KKTSystem

class SolverBase:
    def __init__(self):

        self.settings = Settings()
        self._data: Data = None
        self._result = Result()  # store the values of primal and dual variables of current iteration
        self._step = Variables() # store the step direction of primal and dual variables

        self._kkt_system = None
    
    def setup(self, P, c, A, b, G, h_u, h_l, x_u, x_l):
        self._data = Data(P, c, A, b, G, h_u, h_l, x_u, x_l)

        self._result.x = np.zeros(self._data.n)
        self._result.y = np.zeros(self._data.p)
        self._result.z_l = -self._data.h_l
        self._result.z_u = self._data.h_u
        self._result.z_bl = -self._data.x_l
        self._result.z_bu = self._data.x_u
        self._result.s_u = np.zeros(self._data.m)
        self._result.s_l = np.zeros(self._data.m)
        self._result.s_bl = np.zeros(self._data.n)  # Slack variables for bound constraints (lower)
        self._result.s_bu = np.zeros(self._data.n)  # Slack variables for bound constraints (upper)

        self._result.rho = self.settings.rho_init
        self._result.delta = self.settings.delta_init

        self._kkt_system = KKTSystem(self._data)

        
        

    def solve(self):
        self._solve_impl()

    def _solve_impl(self):
        self._result.status = Status.PIQP_UNSOLVED

        ## ----------- first iteration --------------
        # eq(12) in Roland Schwan 2023 paper
        self._kkt_system.update_scalings_and_factor(
            self._data,
            self._result.rho,
            self._result.delta,
            self._result
        )

        res = Result()
        res.x = -self._data.c
        res.y = self._data.b
        res.z_l = -self._data.h_l
        res.z_u = self._data.h_u
        res.z_bl = -self._data.x_l
        res.z_bu = self._data.x_u
        res.s_l = np.zeros(self._data.m)
        res.s_u = np.zeros(self._data.m)
        res.s_bl = np.zeros(self._data.n)
        res.s_bu = np.zeros(self._data.n)

        self._result = self._kkt_system.solve(self._data, self.settings, res)

        # print(self._result.x)
        # print(self._result.y)
        # print(self._result.z_l)
        # print(self._result.z_u)
        # print(self._result.z_bl)
        # print(self._result.z_bu)
        # print(self._result.s_l)
        # print(self._result.s_u)
        # print(self._result.s_bl)
        # print(self._result.s_bu)

        ## ---------- remaining iterations -------------
        for iter in range(1, self.settings.max_iter):
            self._result.info.iter = iter

            self._result.info.prev_primal_res = ...



    def _update_residual_nr(self):
        """
        Compute the non-regularized primal and dual residuals:
        primal_residual = ||[A*x-b; G*x-h+s]||_inf
        dual_residual = ||P*x + c + AT*y + GT*z||_inf
        """
        # we calculate these term here first to be able to reuse temporary vectors
        # res_nr.y = -A * x
        # work_x = A^T * y
        A_x, At_xt = self._kkt_system.eval_A_xn_and_AT_xt(self._data, -1., 1., self._result.x, self._result.y)
        # res_nr.z_u = -G * x
        # res_nr.z_l = G * x
        # work_x += G^T * (z_u - z_l)
        # work_z.noalias() = m_result.z_u - m_result.z_l

        self._data.P @ self._result.x + self._data.c + self._data.A.transpose() * self._result.y + self._data.G.transpose() * (self._result.z_u - self._result.z_l) + bound_terms

        

# class DenseSolver(SolverBase):
#     def __init__(self):
#         super().__init__()
#         self._kkt_system = De

#     def _solve_impl(self):
#         # Implement the dense solver logic here
#         self._result.status = Status.PIQP_SOLVED
#         # Fill in other result fields as necessary