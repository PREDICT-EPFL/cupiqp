import numpy as np
from enum import Enum
from dataclasses import dataclass

class Status(Enum):
    PIQP_UNSOLVED = -1
    PIQP_SOLVED= 0
    PIQP_MAX_ITER_REACHED= 1
    PIQP_PRIMAL_INFEASIBLE= 2
    PIQP_DUAL_INFEASIBLE= 3
    PIQP_NUMERICAL_ISSUES= 4


class Variables:
    """
    Class to hold optimization variables.
    """
    def __init__(self, n: int, p: int, m: int):
        self.n = n        # Number of primal variables
        self.p = p        # Number of equality constraints
        self.m = m        # Number of inequality constraints

        self.x = np.zeros(n)        # Primal variables
        self.y = np.zeros(p)        # Dual variables for equality constraints
        self.z_u = np.ones(m)      # Dual variables for inequality constraints (upper)
        self.z_l = np.ones(m)      # Dual variables for inequality constraints (lower)
        self.z_bl = np.ones(n)     # Dual variables for bound constraints (lower)
        self.z_bu = np.ones(n)     # Dual variables for bound constraints (upper)
        self.s_u = np.ones(m)      # Slack variables for inequality constraints (upper)
        self.s_l = np.ones(m)      # Slack variables for inequality constraints (lower)
        self.s_bl = np.ones(n)     # Slack variables for bound constraints (lower)
        self.s_bu = np.ones(n)     # Slack variables for bound constraints (upper)
        # z_l, z_u are of size m because in the original KKT matrix we must have rows[G, ...; -G, ...] to efficiently handle double-sided inequalities

    def all_finite(self) -> bool:
        return (np.isfinite(self.x).all() and
                np.isfinite(self.y).all() and
                np.isfinite(self.z_u).all() and
                np.isfinite(self.z_l).all() and
                np.isfinite(self.z_bl).all() and
                np.isfinite(self.z_bu).all() and
                np.isfinite(self.s_u).all() and
                np.isfinite(self.s_l).all() and
                np.isfinite(self.s_bl).all() and
                np.isfinite(self.s_bu).all())
    
    def allclose(self, other: 'Variables', rtol: float = 1e-8, atol: float = 1e-8) -> bool:
        return (np.allclose(self.x, other.x, rtol=rtol, atol=atol) and
                np.allclose(self.y, other.y, rtol=rtol, atol=atol) and
                np.allclose(self.z_u, other.z_u, rtol=rtol, atol=atol) and
                np.allclose(self.z_l, other.z_l, rtol=rtol, atol=atol) and
                np.allclose(self.z_bl, other.z_bl, rtol=rtol, atol=atol) and
                np.allclose(self.z_bu, other.z_bu, rtol=rtol, atol=atol) and
                np.allclose(self.s_u, other.s_u, rtol=rtol, atol=atol) and
                np.allclose(self.s_l, other.s_l, rtol=rtol, atol=atol) and
                np.allclose(self.s_bl, other.s_bl, rtol=rtol, atol=atol) and
                np.allclose(self.s_bu, other.s_bu, rtol=rtol, atol=atol))
    
    def __copy__(self) -> 'Variables':
        new_vars = Variables(len(self.x), len(self.y), len(self.z_u))
        new_vars.x = self.x.copy()
        new_vars.y = self.y.copy()
        new_vars.z_u = self.z_u.copy()
        new_vars.z_l = self.z_l.copy()
        new_vars.z_bu = self.z_bu.copy()
        new_vars.z_bl = self.z_bl.copy()
        new_vars.s_u = self.s_u.copy()
        new_vars.s_l = self.s_l.copy()
        new_vars.s_bu = self.s_bu.copy()
        new_vars.s_bl = self.s_bl.copy()
        return new_vars
    
    def copy(self) -> 'Variables':
        """Create a copy of this Variables object."""
        return self.__copy__()
    
    def __sub__(self, other: 'Variables') -> 'Variables':
        if not (len(self.x) == len(other.x) and
                len(self.y) == len(other.y) and
                len(self.z_u) == len(other.z_u) and
                len(self.z_l) == len(other.z_l) and
                len(self.z_bl) == len(other.z_bl) and
                len(self.z_bu) == len(other.z_bu) and
                len(self.s_u) == len(other.s_u) and
                len(self.s_l) == len(other.s_l) and
                len(self.s_bl) == len(other.s_bl) and
                len(self.s_bu) == len(other.s_bu)):
            raise ValueError("Dimension mismatch in Variables subtraction.")
        result = Variables(self.n, self.p, self.m, self.num_xu, self.num_xl)
        result.x = self.x - other.x
        result.y = self.y - other.y
        result.z_u = self.z_u - other.z_u
        result.z_l = self.z_l - other.z_l
        result.z_bu = self.z_bu - other.z_bu
        result.z_bl = self.z_bl - other.z_bl
        result.s_u = self.s_u - other.s_u
        result.s_l = self.s_l - other.s_l
        result.s_bu = self.s_bu - other.s_bu
        result.s_bl = self.s_bl - other.s_bl
        return result
    
    def __str__(self) -> str:
        return (f"Variables:\n"
            f"  x:    {self.x}\n"
            f"  y:    {self.y}\n"
            f"  z_u:  {self.z_u}\n"
            f"  z_l:  {self.z_l}\n"
            f"  z_bu: {self.z_bu}\n"
            f"  z_bl: {self.z_bl}\n"
            f"  s_u:  {self.s_u}\n"
            f"  s_l:  {self.s_l}\n"
            f"  s_bu: {self.s_bu}\n"
            f"  s_bl: {self.s_bl}")
    
    def to_array(self) -> np.ndarray:
        return np.concatenate((self.x, self.y, self.z_u, self.z_l, self.z_bu, self.z_bl, self.s_u, self.s_l, self.s_bu, self.s_bl))
    
    def set_random(self):
        """Testing purpose only: set all variables to random values."""
        np.random.seed(0)
        self.x = np.random.randn(*self.x.shape)
        self.y = np.random.randn(*self.y.shape)
        self.z_u = np.random.rand(*self.z_u.shape) + 1.0  # ensure positivity
        self.z_l = np.random.rand(*self.z_l.shape) + 1.0
        self.z_bu = np.random.rand(*self.z_bu.shape) + 1.0
        self.z_bl = np.random.rand(*self.z_bl.shape) + 1.0
        self.s_u = np.random.rand(*self.s_u.shape) + 1.0
        self.s_l = np.random.rand(*self.s_l.shape) + 1.0
        self.s_bu = np.random.rand(*self.s_bu.shape) + 1.0
        self.s_bl = np.random.rand(*self.s_bl.shape) + 1.0
    
@dataclass
class Info:
    status: Status = Status.PIQP_UNSOLVED

    iter: int = 0
    rho: float = None
    delta: float = None
    mu: float = None
    sigma: float = None
    primal_step: float = 0.0
    dual_step: float = 0.0
    
    primal_res: float = None
    primal_res_rel: float = None
    dual_res: float = None
    dual_res_rel: float = None
    
    primal_res_reg: float = None
    primal_res_reg_rel: float = None  # relative primal residual with regularization
    dual_res_reg: float = None
    dual_res_reg_rel: float = None
    
    primal_prox_inf: float = None
    dual_prox_inf: float = None
    
    prev_primal_res: float = None  # primal residual from previous iteration
    prev_dual_res: float = None  # dual residual from previous iteration
    
    primal_obj: float = None
    dual_obj: float = None
    duality_gap: float = None      # duality gap
    duality_gap_rel: float = None  # relative duality gap
    
    factor_retires: int = 0
    reg_limit: float = None
    no_primal_update: int = 0  # dual infeasibility detection counter
    no_dual_update: int = 0    # primal infeasibility detection counter
    
    setup_time: float = None
    update_time: float = None
    solve_time: float = None
    kkt_factor_time: float = None
    kkt_solve_time: float = None
    run_time: float = None


class Result(Variables):
    def __init__(self):
        super().__init__(n=0, p=0, m=0)  # Initialize with default sizes; adjust as needed
        self.info = Info()