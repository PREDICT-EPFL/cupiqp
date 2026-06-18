import cupy as cp
import numpy as np
from typing import List
from enum import Enum, IntEnum
import nvtx

from .data import Data

class Status(Enum):
    CUPIQP_UNSOLVED = -1
    CUPIQP_SOLVED = 0
    CUPIQP_MAX_ITER_REACHED = 1
    CUPIQP_PRIMAL_INFEASIBLE = 2
    CUPIQP_DUAL_INFEASIBLE = 3
    CUPIQP_NUMERICAL_ISSUES = 4


class Variables:
    """Optimization variables for a batch of B QPs.

    Two contiguous buffers per problem, laid out as::

        _primal_buffer : (B, num_var + num_ineq)  — [x | s_l | s_u | s_bl | s_bu]
        _dual_buffer   : (B, num_eq + num_ineq)   — [y | z_l | z_u | z_bl | z_bu]

    Individual variable attributes (x, y, s_l, z_u, ...) are zero-copy (B, *) views
    into these buffers. Properties with custom setters ensure assignments
    always write in-place to preserve the contiguous layout.
    """

    def __init__(self):
        pass

    def init(self, data: Data):
        self._batch_size = data.batch_size
        self.n = data.n
        self.p = data.p
        self.m = data.m
        n, m, p = data.n, data.m, data.p
        self.num_ineq = 2 * m + 2 * n

        B = self._batch_size
        dtype = data.dtype

        # Primal buffer: [x(n) | s_l(m) | s_u(m) | s_bl(n) | s_bu(n)]
        self._primal_buffer = cp.empty((B, n + self.num_ineq), dtype=dtype)
        offset = 0
        self._x = self._primal_buffer[:, offset : offset+n]
        self._s_all = self._primal_buffer[:, n:]
        offset += n
        self._s_l = self._primal_buffer[:, offset : offset+m]
        offset += m
        self._s_u = self._primal_buffer[:, offset : offset+m]
        offset += m
        self._s_bl = self._primal_buffer[:, offset : offset+n]
        offset += n
        self._s_bu = self._primal_buffer[:, offset : offset+n]

        # Dual buffer: [y(p) | z_l(m) | z_u(m) | z_bl(n) | z_bu(n)]
        self._dual_buffer = cp.empty((B, p + self.num_ineq), dtype=dtype)
        offset = 0
        self._y = self._dual_buffer[:, offset : offset+p]
        self._z_all = self._dual_buffer[:, p:]
        offset += p
        self._z_l = self._dual_buffer[:, offset : offset+m]
        offset += m
        self._z_u = self._dual_buffer[:, offset : offset+m]
        offset += m
        self._z_bl = self._dual_buffer[:, offset : offset+n]
        offset += n
        self._z_bu = self._dual_buffer[:, offset : offset+n]

    # -- Properties: getters return (batch_size, *) views, setters copy in-place --

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

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def all_finite(self) -> bool:
        """Test purpose only"""
        return bool(
            cp.isfinite(self._primal_buffer).all() and
            cp.isfinite(self._dual_buffer).all()
        )

    @property
    def buffer_ptr(self) -> tuple:
        """Memory addresses for CUDA graph cache keys."""
        return (
            self._primal_buffer.data.ptr,
            self._dual_buffer.data.ptr,
        )

    def allclose(self, other: 'Variables', rtol: float = 1e-8, atol: float = 1e-8) -> bool:
        return (cp.allclose(self._primal_buffer, other._primal_buffer, rtol=rtol, atol=atol) and
                cp.allclose(self._dual_buffer, other._dual_buffer, rtol=rtol, atol=atol))

    def __str__(self) -> str:
        return (f"Variables (B={self._batch_size}):\n"
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
        """Test purpose only."""
        return cp.concatenate((self._primal_buffer, self._dual_buffer), axis=1)

    def set_random(self):
        """Testing purpose only: set all variables to random values."""
        cp.random.seed(0)
        self._x[:] = cp.random.randn(*self._x.shape)
        self._y[:] = cp.random.randn(*self._y.shape)
        self._z_u[:] = cp.random.rand(*self._z_u.shape) + 1.0  # ensure positiveness
        self._z_l[:] = cp.random.rand(*self._z_l.shape) + 1.0
        self._z_bu[:] = cp.random.rand(*self._z_bu.shape) + 1.0
        self._z_bl[:] = cp.random.rand(*self._z_bl.shape) + 1.0
        self._s_u[:] = cp.random.rand(*self._s_u.shape) + 1.0
        self._s_l[:] = cp.random.rand(*self._s_l.shape) + 1.0
        self._s_bu[:] = cp.random.rand(*self._s_bu.shape) + 1.0
        self._s_bl[:] = cp.random.rand(*self._s_bl.shape) + 1.0

class InfoIdx(IntEnum):
    """Column index of each scalar field in the contiguous Info buffer."""
    rho = 0
    delta = 1
    mu = 2
    sigma = 3
    primal_step = 4
    dual_step = 5
    primal_res = 6
    primal_res_rel = 7
    dual_res = 8
    dual_res_rel = 9
    primal_res_reg = 10
    primal_res_reg_rel = 11
    dual_res_reg = 12
    dual_res_reg_rel = 13
    primal_prox_inf = 14
    dual_prox_inf = 15
    prev_primal_res = 16
    prev_dual_res = 17
    primal_obj = 18
    dual_obj = 19
    duality_gap = 20
    duality_gap_rel = 21
    reg_limit = 22
    setup_time = 23
    update_time = 24
    solve_time = 25
    kkt_factor_time = 26
    kkt_solve_time = 27
    run_time = 28

class Info:
    """Per-problem solver info — ``(B, num_fields)`` GPU buffer.

    Each problem independently tracks rho, delta, mu, residuals, etc.
    For B=1 this is a ``(1, num_fields)`` buffer.
    """

    # Each property is a (B,) view into the contiguous device buffer.
    # Getters return the view; setters copy in-place to preserve the layout.

    @property
    def rho(self): return self._buffer[:, InfoIdx.rho]
    @rho.setter
    def rho(self, value): self._buffer[:, InfoIdx.rho] = value

    @property
    def delta(self): return self._buffer[:, InfoIdx.delta]
    @delta.setter
    def delta(self, value): self._buffer[:, InfoIdx.delta] = value

    @property
    def mu(self): return self._buffer[:, InfoIdx.mu]
    @mu.setter
    def mu(self, value): self._buffer[:, InfoIdx.mu] = value

    @property
    def sigma(self): return self._buffer[:, InfoIdx.sigma]
    @sigma.setter
    def sigma(self, value): self._buffer[:, InfoIdx.sigma] = value

    @property
    def primal_step(self): return self._buffer[:, InfoIdx.primal_step]
    @primal_step.setter
    def primal_step(self, value): self._buffer[:, InfoIdx.primal_step] = value

    @property
    def dual_step(self): return self._buffer[:, InfoIdx.dual_step]
    @dual_step.setter
    def dual_step(self, value): self._buffer[:, InfoIdx.dual_step] = value

    @property
    def primal_res(self): return self._buffer[:, InfoIdx.primal_res]
    @primal_res.setter
    def primal_res(self, value): self._buffer[:, InfoIdx.primal_res] = value

    @property
    def primal_res_rel(self): return self._buffer[:, InfoIdx.primal_res_rel]
    @primal_res_rel.setter
    def primal_res_rel(self, value): self._buffer[:, InfoIdx.primal_res_rel] = value

    @property
    def dual_res(self): return self._buffer[:, InfoIdx.dual_res]
    @dual_res.setter
    def dual_res(self, value): self._buffer[:, InfoIdx.dual_res] = value

    @property
    def dual_res_rel(self): return self._buffer[:, InfoIdx.dual_res_rel]
    @dual_res_rel.setter
    def dual_res_rel(self, value): self._buffer[:, InfoIdx.dual_res_rel] = value

    @property
    def primal_res_reg(self): return self._buffer[:, InfoIdx.primal_res_reg]
    @primal_res_reg.setter
    def primal_res_reg(self, value): self._buffer[:, InfoIdx.primal_res_reg] = value

    @property
    def primal_res_reg_rel(self): return self._buffer[:, InfoIdx.primal_res_reg_rel]
    @primal_res_reg_rel.setter
    def primal_res_reg_rel(self, value): self._buffer[:, InfoIdx.primal_res_reg_rel] = value

    @property
    def dual_res_reg(self): return self._buffer[:, InfoIdx.dual_res_reg]
    @dual_res_reg.setter
    def dual_res_reg(self, value): self._buffer[:, InfoIdx.dual_res_reg] = value

    @property
    def dual_res_reg_rel(self): return self._buffer[:, InfoIdx.dual_res_reg_rel]
    @dual_res_reg_rel.setter
    def dual_res_reg_rel(self, value): self._buffer[:, InfoIdx.dual_res_reg_rel] = value

    @property
    def primal_prox_inf(self): return self._buffer[:, InfoIdx.primal_prox_inf]
    @primal_prox_inf.setter
    def primal_prox_inf(self, value): self._buffer[:, InfoIdx.primal_prox_inf] = value

    @property
    def dual_prox_inf(self): return self._buffer[:, InfoIdx.dual_prox_inf]
    @dual_prox_inf.setter
    def dual_prox_inf(self, value): self._buffer[:, InfoIdx.dual_prox_inf] = value

    @property
    def prev_primal_res(self): return self._buffer[:, InfoIdx.prev_primal_res]
    @prev_primal_res.setter
    def prev_primal_res(self, value): self._buffer[:, InfoIdx.prev_primal_res] = value

    @property
    def prev_dual_res(self): return self._buffer[:, InfoIdx.prev_dual_res]
    @prev_dual_res.setter
    def prev_dual_res(self, value): self._buffer[:, InfoIdx.prev_dual_res] = value

    @property
    def primal_obj(self): return self._buffer[:, InfoIdx.primal_obj]
    @primal_obj.setter
    def primal_obj(self, value): self._buffer[:, InfoIdx.primal_obj] = value

    @property
    def dual_obj(self): return self._buffer[:, InfoIdx.dual_obj]
    @dual_obj.setter
    def dual_obj(self, value): self._buffer[:, InfoIdx.dual_obj] = value

    @property
    def duality_gap(self): return self._buffer[:, InfoIdx.duality_gap]
    @duality_gap.setter
    def duality_gap(self, value): self._buffer[:, InfoIdx.duality_gap] = value

    @property
    def duality_gap_rel(self): return self._buffer[:, InfoIdx.duality_gap_rel]
    @duality_gap_rel.setter
    def duality_gap_rel(self, value): self._buffer[:, InfoIdx.duality_gap_rel] = value

    @property
    def reg_limit(self): return self._buffer[:, InfoIdx.reg_limit]
    @reg_limit.setter
    def reg_limit(self, value): self._buffer[:, InfoIdx.reg_limit] = value

    @property
    def setup_time(self): return self._buffer[:, InfoIdx.setup_time]
    @setup_time.setter
    def setup_time(self, value): self._buffer[:, InfoIdx.setup_time] = value

    @property
    def update_time(self): return self._buffer[:, InfoIdx.update_time]
    @update_time.setter
    def update_time(self, value): self._buffer[:, InfoIdx.update_time] = value

    @property
    def solve_time(self): return self._buffer[:, InfoIdx.solve_time]
    @solve_time.setter
    def solve_time(self, value): self._buffer[:, InfoIdx.solve_time] = value

    @property
    def kkt_factor_time(self): return self._buffer[:, InfoIdx.kkt_factor_time]
    @kkt_factor_time.setter
    def kkt_factor_time(self, value): self._buffer[:, InfoIdx.kkt_factor_time] = value

    @property
    def kkt_solve_time(self): return self._buffer[:, InfoIdx.kkt_solve_time]
    @kkt_solve_time.setter
    def kkt_solve_time(self, value): self._buffer[:, InfoIdx.kkt_solve_time] = value

    @property
    def run_time(self): return self._buffer[:, InfoIdx.run_time]
    @run_time.setter
    def run_time(self, value): self._buffer[:, InfoIdx.run_time] = value

    def __init__(self, batch_size: int = 1):
        self._batch_size = batch_size
        self._status_value = np.full(batch_size, Status.CUPIQP_UNSOLVED.value, dtype=np.int32)
        self.iter = np.zeros(batch_size, dtype=np.int32)  # individual iter counts for each problem in the batch
        self.iter_total = 0  # total iterations the solver runs for this batch (the slowest problem's count)
        self.factor_retires = np.zeros(batch_size, dtype=np.int32)
        # Per-batch "no update" counters live on device (int32). Source of truth;
        # the rho/delta kernels reset on improved, increment on stagnated.
        # to_host() syncs them into the InfoHost mirror once per IPM iteration.
        self.no_primal_update = cp.zeros(batch_size, dtype=cp.int32)
        self.no_dual_update = cp.zeros(batch_size, dtype=cp.int32)

    def init(self, dtype=cp.float64):
        self._buffer = cp.zeros((self._batch_size, len(InfoIdx)), dtype=dtype)

    @property
    def status(self) -> List[Status]:
        """Per-problem status as a list of Status enums."""
        return [Status(v) for v in self._status_value]

    @property
    def status_value(self) -> np.ndarray:
        """Per-problem status as a writable (B,) int32 array of Status values."""
        return self._status_value

    @nvtx.annotate("Info:to_host")
    def to_host(self, info_host: 'InfoHost'):
        cp.asnumpy(self._buffer, out=info_host._buffer)
        cp.asnumpy(self.no_primal_update, out=info_host.no_primal_update)
        cp.asnumpy(self.no_dual_update,   out=info_host.no_dual_update)

    @property
    def batch_size(self) -> int:
        return self._batch_size


class InfoHost:
    """
    A mirror of Info on the host side (CPU). The purpose is to fetch all device-side info to host all at once, instead of multiple time to reduce overhead.

    Each property returns a ``(B,)`` NumPy array.
    """
    __slots__ = ('_buffer', '_batch_size', 'no_primal_update', 'no_dual_update')

    # Each property is a read-only (B,) view into the contiguous host buffer.

    @property
    def rho(self): return self._buffer[:, InfoIdx.rho]

    @property
    def delta(self): return self._buffer[:, InfoIdx.delta]

    @property
    def mu(self): return self._buffer[:, InfoIdx.mu]

    @property
    def sigma(self): return self._buffer[:, InfoIdx.sigma]

    @property
    def primal_step(self): return self._buffer[:, InfoIdx.primal_step]

    @property
    def dual_step(self): return self._buffer[:, InfoIdx.dual_step]

    @property
    def primal_res(self): return self._buffer[:, InfoIdx.primal_res]

    @property
    def primal_res_rel(self): return self._buffer[:, InfoIdx.primal_res_rel]

    @property
    def dual_res(self): return self._buffer[:, InfoIdx.dual_res]

    @property
    def dual_res_rel(self): return self._buffer[:, InfoIdx.dual_res_rel]

    @property
    def primal_res_reg(self): return self._buffer[:, InfoIdx.primal_res_reg]

    @property
    def primal_res_reg_rel(self): return self._buffer[:, InfoIdx.primal_res_reg_rel]

    @property
    def dual_res_reg(self): return self._buffer[:, InfoIdx.dual_res_reg]

    @property
    def dual_res_reg_rel(self): return self._buffer[:, InfoIdx.dual_res_reg_rel]

    @property
    def primal_prox_inf(self): return self._buffer[:, InfoIdx.primal_prox_inf]

    @property
    def dual_prox_inf(self): return self._buffer[:, InfoIdx.dual_prox_inf]

    @property
    def prev_primal_res(self): return self._buffer[:, InfoIdx.prev_primal_res]

    @property
    def prev_dual_res(self): return self._buffer[:, InfoIdx.prev_dual_res]

    @property
    def primal_obj(self): return self._buffer[:, InfoIdx.primal_obj]

    @property
    def dual_obj(self): return self._buffer[:, InfoIdx.dual_obj]

    @property
    def duality_gap(self): return self._buffer[:, InfoIdx.duality_gap]

    @property
    def duality_gap_rel(self): return self._buffer[:, InfoIdx.duality_gap_rel]

    @property
    def reg_limit(self): return self._buffer[:, InfoIdx.reg_limit]

    @property
    def setup_time(self): return self._buffer[:, InfoIdx.setup_time]

    @property
    def update_time(self): return self._buffer[:, InfoIdx.update_time]

    @property
    def solve_time(self): return self._buffer[:, InfoIdx.solve_time]

    @property
    def kkt_factor_time(self): return self._buffer[:, InfoIdx.kkt_factor_time]

    @property
    def kkt_solve_time(self): return self._buffer[:, InfoIdx.kkt_solve_time]

    @property
    def run_time(self): return self._buffer[:, InfoIdx.run_time]

    def __init__(self, batch_size: int = 1, dtype=np.float64):
        self._batch_size = batch_size
        self._buffer = np.empty((batch_size, len(InfoIdx)), dtype=dtype)
        self.no_primal_update = np.zeros(batch_size, dtype=np.int32)
        self.no_dual_update = np.zeros(batch_size, dtype=np.int32)



class Result(Variables):
    """Combined variables + per-problem info."""
    def __init__(self, batch_size: int = 1):
        super().__init__()
        self.info = Info(batch_size)

    def init(self, data):
        assert data.batch_size == self.info.batch_size, \
            f"batch_size mismatch: Result({self.info.batch_size}) vs data({data.batch_size})"
        super().init(data)
        self.info.init(dtype=data.dtype)
