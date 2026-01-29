import cupy as cp
from enum import Enum
from dataclasses import dataclass

from .data import Data

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
    def __init__(self):
        pass

    def init(self, data: Data):
        self.n = data.n        # Number of primal variables
        self.p = data.p        # Number of equality constraints
        self.m = data.m        # Number of inequality constraints
        
        self.x = cp.zeros(data.n)            # Primal variables
        self.y = cp.zeros(data.p)            # Dual variables for equality constraints
        self.z_u = cp.ones(data.num_hu)      # Dual variables for inequality constraints (upper)
        self.z_l = cp.ones(data.num_hl)      # Dual variables for inequality constraints (lower)
        self.z_bl = cp.ones(data.num_xl)     # Dual variables for bound constraints (lower)
        self.z_bu = cp.ones(data.num_xu)     # Dual variables for bound constraints (upper)
        self.s_u = cp.ones(data.num_hu)      # Slack variables for inequality constraints (upper)
        self.s_l = cp.ones(data.num_hl)      # Slack variables for inequality constraints (lower)
        self.s_bl = cp.ones(data.num_xl)     # Slack variables for bound constraints (lower)
        self.s_bu = cp.ones(data.num_xu)     # Slack variables for bound constraints (upper)

    def all_finite(self) -> bool:
        return (cp.isfinite(self.x).all() and
                cp.isfinite(self.y).all() and
                cp.isfinite(self.z_u).all() and
                cp.isfinite(self.z_l).all() and
                cp.isfinite(self.z_bl).all() and
                cp.isfinite(self.z_bu).all() and
                cp.isfinite(self.s_u).all() and
                cp.isfinite(self.s_l).all() and
                cp.isfinite(self.s_bl).all() and
                cp.isfinite(self.s_bu).all())

    @property
    def buffer_ptr(self) -> tuple:
        """
        Returns a tuple of memory addresses of the underlying arrays.
        Used for CUDA graph caching keys.
        """
        return (
            self.x.data.ptr,
            self.y.data.ptr,
            self.z_u.data.ptr,
            self.z_l.data.ptr,
            self.z_bu.data.ptr,
            self.z_bl.data.ptr,
            self.s_u.data.ptr,
            self.s_l.data.ptr,
            self.s_bu.data.ptr,
            self.s_bl.data.ptr
        )
    
    def allclose(self, other: 'Variables', rtol: float = 1e-8, atol: float = 1e-8) -> bool:
        return (cp.allclose(self.x, other.x, rtol=rtol, atol=atol) and
                cp.allclose(self.y, other.y, rtol=rtol, atol=atol) and
                cp.allclose(self.z_u, other.z_u, rtol=rtol, atol=atol) and
                cp.allclose(self.z_l, other.z_l, rtol=rtol, atol=atol) and
                cp.allclose(self.z_bl, other.z_bl, rtol=rtol, atol=atol) and
                cp.allclose(self.z_bu, other.z_bu, rtol=rtol, atol=atol) and
                cp.allclose(self.s_u, other.s_u, rtol=rtol, atol=atol) and
                cp.allclose(self.s_l, other.s_l, rtol=rtol, atol=atol) and
                cp.allclose(self.s_bl, other.s_bl, rtol=rtol, atol=atol) and
                cp.allclose(self.s_bu, other.s_bu, rtol=rtol, atol=atol))
    
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
    
    def to_array(self) -> cp.ndarray:
        return cp.concatenate((self.x, self.y, self.z_u, self.z_l, self.z_bu, self.z_bl, self.s_u, self.s_l, self.s_bu, self.s_bl))
    
    def set_random(self):
        """Testing purpose only: set all variables to random values."""
        cp.random.seed(0)
        self.x = cp.random.randn(*self.x.shape)
        self.y = cp.random.randn(*self.y.shape)
        self.z_u = cp.random.rand(*self.z_u.shape) + 1.0  # ensure positivity
        self.z_l = cp.random.rand(*self.z_l.shape) + 1.0
        self.z_bu = cp.random.rand(*self.z_bu.shape) + 1.0
        self.z_bl = cp.random.rand(*self.z_bl.shape) + 1.0
        self.s_u = cp.random.rand(*self.s_u.shape) + 1.0
        self.s_l = cp.random.rand(*self.s_l.shape) + 1.0
        self.s_bu = cp.random.rand(*self.s_bu.shape) + 1.0
        self.s_bl = cp.random.rand(*self.s_bl.shape) + 1.0
    
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
        super().__init__()
        self.info = Info()

    def init(self, data: Data):
        super().init(data)