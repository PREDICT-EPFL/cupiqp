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

    All variables are backed by two contiguous buffers:
      _primal_buf: [x | s_l | s_u | s_bl | s_bu]
      _dual_buf:   [y | z_l | z_u | z_bl | z_bu]

    Individual variable attributes (x, y, s_l, z_u, ...) are zero-copy views
    into these buffers. Properties with custom setters ensure assignments
    always write in-place, preserving the contiguous layout.
    """
    def __init__(self):
        pass

    def init(self, data: Data):
        self.n = data.n        # Number of primal variables
        self.p = data.p        # Number of equality constraints
        self.m = data.m        # Number of inequality constraints
        self.num_ineq = data.num_hl + data.num_hu + data.num_xl + data.num_xu

        # Primal buffer: [x | s_l | s_u | s_bl | s_bu] (here we regard slacks as primal variables)
        self._primal_buffer = cp.empty(data.n + self.num_ineq)
        self._x = self._primal_buffer[:data.n]
        self._s_all = self._primal_buffer[data.n:]  # all slack variables
        offset = data.n
        self._s_l  = self._primal_buffer[offset : offset + data.num_hl]
        offset += data.num_hl
        self._s_u  = self._primal_buffer[offset : offset + data.num_hu]
        offset += data.num_hu
        self._s_bl = self._primal_buffer[offset : offset + data.num_xl]
        offset += data.num_xl
        self._s_bu = self._primal_buffer[offset : offset + data.num_xu]

        # Dual buffer: [y | z_l | z_u | z_bl | z_bu]
        self._dual_buffer = cp.empty(data.p + self.num_ineq)
        self._y = self._dual_buffer[:data.p]
        self._z_all = self._dual_buffer[data.p:]  # all dual variables (for inequalities)
        offset = data.p
        self._z_l  = self._dual_buffer[offset : offset + data.num_hl]
        offset += data.num_hl
        self._z_u  = self._dual_buffer[offset : offset + data.num_hu]
        offset += data.num_hu
        self._z_bl = self._dual_buffer[offset : offset + data.num_xl]
        offset += data.num_xl
        self._z_bu = self._dual_buffer[offset : offset + data.num_xu]

    # -- Properties: getters return views, setters copy in-place --

    @property
    def x(self): return self._x
    @x.setter
    def x(self, value): self._x[:] = value

    @property
    def y(self): return self._y
    @y.setter
    def y(self, value): self._y[:] = value

    @property
    def s_all(self): return self._s_all
    @s_all.setter
    def s_all(self, value): self._s_all[:] = value

    @property
    def s_l(self): return self._s_l
    @s_l.setter
    def s_l(self, value): self._s_l[:] = value

    @property
    def s_u(self): return self._s_u
    @s_u.setter
    def s_u(self, value): self._s_u[:] = value

    @property
    def s_bl(self): return self._s_bl
    @s_bl.setter
    def s_bl(self, value): self._s_bl[:] = value

    @property
    def s_bu(self): return self._s_bu
    @s_bu.setter
    def s_bu(self, value): self._s_bu[:] = value

    @property
    def z_all(self): return self._z_all
    @z_all.setter
    def z_all(self, value): self._z_all[:] = value

    @property
    def z_l(self): return self._z_l
    @z_l.setter
    def z_l(self, value): self._z_l[:] = value

    @property
    def z_u(self): return self._z_u
    @z_u.setter
    def z_u(self, value): self._z_u[:] = value

    @property
    def z_bl(self): return self._z_bl
    @z_bl.setter
    def z_bl(self, value): self._z_bl[:] = value

    @property
    def z_bu(self): return self._z_bu
    @z_bu.setter
    def z_bu(self, value): self._z_bu[:] = value

    @property
    def primals_all(self): return self._primal_buffer
    @primals_all.setter
    def primals_all(self, value): self._primal_buffer[:] = value

    @property
    def duals_all(self): return self._dual_buffer
    @duals_all.setter
    def duals_all(self, value): self._dual_buffer[:] = value

    def all_finite(self) -> bool:
        return (cp.isfinite(self._primal_buffer).all() and
                cp.isfinite(self._dual_buffer).all())

    @property
    def buffer_ptr(self) -> tuple:
        """
        Returns a tuple of memory addresses of the underlying arrays.
        Used for CUDA graph caching keys.
        """
        return (
            self._primal_buffer.data.ptr,
            self._dual_buffer.data.ptr,
        )

    def allclose(self, other: 'Variables', rtol: float = 1e-8, atol: float = 1e-8) -> bool:
        return (cp.allclose(self._primal_buffer, other._primal_buffer, rtol=rtol, atol=atol) and
                cp.allclose(self._dual_buffer, other._dual_buffer, rtol=rtol, atol=atol))

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
        return cp.concatenate((self._primal_buffer, self._dual_buffer))

    def set_random(self):
        """Testing purpose only: set all variables to random values."""
        cp.random.seed(0)
        self._x[:] = cp.random.randn(*self._x.shape)
        self._y[:] = cp.random.randn(*self._y.shape)
        self._z_u[:] = cp.random.rand(*self._z_u.shape) + 1.0  # ensure positivity
        self._z_l[:] = cp.random.rand(*self._z_l.shape) + 1.0
        self._z_bu[:] = cp.random.rand(*self._z_bu.shape) + 1.0
        self._z_bl[:] = cp.random.rand(*self._z_bl.shape) + 1.0
        self._s_u[:] = cp.random.rand(*self._s_u.shape) + 1.0
        self._s_l[:] = cp.random.rand(*self._s_l.shape) + 1.0
        self._s_bu[:] = cp.random.rand(*self._s_bu.shape) + 1.0
        self._s_bl[:] = cp.random.rand(*self._s_bl.shape) + 1.0
    
@dataclass
class Info:
    status: Status = Status.PIQP_UNSOLVED

    iter: int = 0
    rho: cp.ndarray = None # using array to allow in-place updates without CUDA graph cache misses
    delta: cp.ndarray = None
    mu: cp.ndarray = None
    sigma: cp.ndarray = None
    primal_step: cp.ndarray = None
    dual_step: cp.ndarray = None
    
    primal_res: cp.ndarray = None
    primal_res_rel: cp.ndarray = None
    dual_res: cp.ndarray = None
    dual_res_rel: cp.ndarray = None
    
    primal_res_reg: cp.ndarray = None
    primal_res_reg_rel: cp.ndarray = None  # relative primal residual with regularization
    dual_res_reg: cp.ndarray = None
    dual_res_reg_rel: cp.ndarray = None
    
    primal_prox_inf: cp.ndarray = None
    dual_prox_inf: cp.ndarray = None
    
    prev_primal_res: cp.ndarray = None  # primal residual from previous iteration
    prev_dual_res: cp.ndarray = None  # dual residual from previous iteration
    
    primal_obj: cp.ndarray = None
    dual_obj: cp.ndarray = None
    duality_gap: cp.ndarray = None      # duality gap
    duality_gap_rel: cp.ndarray = None  # relative duality gap
    
    factor_retires: int = 0
    reg_limit: cp.ndarray = None
    no_primal_update: int = 0  # dual infeasibility detection counter
    no_dual_update: int = 0    # primal infeasibility detection counter
    
    setup_time: cp.ndarray = None
    update_time: cp.ndarray = None
    solve_time: cp.ndarray = None
    kkt_factor_time: cp.ndarray = None
    kkt_solve_time: cp.ndarray = None
    run_time: cp.ndarray = None


class Result(Variables):
    def __init__(self):
        super().__init__()
        self.info = Info()

    def init(self, data: Data):
        super().init(data)
        self.info.rho = cp.zeros(1, dtype=cp.float64)
        self.info.delta = cp.zeros(1, dtype=cp.float64)
        self.info.mu = cp.zeros(1, dtype=cp.float64)
        self.info.sigma = cp.zeros(1, dtype=cp.float64)
        self.info.primal_step = cp.zeros(1, dtype=cp.float64)
        self.info.dual_step = cp.zeros(1, dtype=cp.float64)
        self.info.primal_res = cp.zeros(1, dtype=cp.float64)
        self.info.primal_res_rel = cp.zeros(1, dtype=cp.float64)
        self.info.dual_res = cp.zeros(1, dtype=cp.float64)
        self.info.dual_res_rel = cp.zeros(1, dtype=cp.float64)
        self.info.primal_res_reg = cp.zeros(1, dtype=cp.float64)
        self.info.primal_res_reg_rel = cp.zeros(1, dtype=cp.float64)
        self.info.dual_res_reg = cp.zeros(1, dtype=cp.float64)
        self.info.dual_res_reg_rel = cp.zeros(1, dtype=cp.float64)
        self.info.primal_prox_inf = cp.zeros(1, dtype=cp.float64)
        self.info.dual_prox_inf = cp.zeros(1, dtype=cp.float64)
        self.info.prev_primal_res = cp.zeros(1, dtype=cp.float64)
        self.info.prev_dual_res = cp.zeros(1, dtype=cp.float64)
        self.info.primal_obj = cp.zeros(1, dtype=cp.float64)
        self.info.dual_obj = cp.zeros(1, dtype=cp.float64)
        self.info.duality_gap = cp.zeros(1, dtype=cp.float64)
        self.info.duality_gap_rel = cp.zeros(1, dtype=cp.float64)
        self.info.reg_limit = cp.zeros(1, dtype=cp.float64)
        self.info.setup_time = cp.zeros(1, dtype=cp.float64)
        self.info.update_time = cp.zeros(1, dtype=cp.float64)
        self.info.solve_time = cp.zeros(1, dtype=cp.float64)
        self.info.kkt_factor_time = cp.zeros(1, dtype=cp.float64)
        self.info.kkt_solve_time = cp.zeros(1, dtype=cp.float64)
        self.info.run_time = cp.zeros(1, dtype=cp.float64)