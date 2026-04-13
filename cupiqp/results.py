import cupy as cp
import numpy as np
from enum import Enum, IntEnum, auto
import nvtx

from .data import Data

class Status(Enum):
    PIQP_UNSOLVED = -1
    PIQP_SOLVED = 0
    PIQP_MAX_ITER_REACHED = 1
    PIQP_PRIMAL_INFEASIBLE = 2
    PIQP_DUAL_INFEASIBLE = 3
    PIQP_NUMERICAL_ISSUES = 4


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

    def init(self, data):
        self._batch_size = data.batch_size
        self.n = data.n
        self.p = data.p
        self.m = data.m
        self.num_ineq = data.num_hl + data.num_hu + data.num_xl + data.num_xu

        B = self._batch_size

        # Primal buffer: [x | s_l | s_u | s_bl | s_bu]
        self._primal_buffer = cp.empty((B, data.n + self.num_ineq), dtype=cp.float64)
        offset = 0
        self._x = self._primal_buffer[:, offset : offset+data.n]
        self._s_all = self._primal_buffer[:, data.n:]
        offset += data.n
        self._s_l = self._primal_buffer[:, offset : offset+data.num_hl]
        offset += data.num_hl
        self._s_u = self._primal_buffer[:, offset : offset+data.num_hu]
        offset += data.num_hu
        self._s_bl = self._primal_buffer[:, offset : offset+data.num_xl]
        offset += data.num_xl
        self._s_bu = self._primal_buffer[:, offset : offset+data.num_xu]

        # Dual buffer: [y | z_l | z_u | z_bl | z_bu]
        self._dual_buffer = cp.empty((B, data.p + self.num_ineq), dtype=cp.float64)
        offset = 0
        self._y = self._dual_buffer[:, offset : offset+data.p]
        self._z_all = self._dual_buffer[:, data.p:]
        offset += data.p
        self._z_l = self._dual_buffer[:, offset : offset+data.num_hl]
        offset += data.num_hl
        self._z_u = self._dual_buffer[:, offset : offset+data.num_hu]
        offset += data.num_hu
        self._z_bl = self._dual_buffer[:, offset : offset+data.num_xl]
        offset += data.num_xl
        self._z_bu = self._dual_buffer[:, offset : offset+data.num_xu]

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
    """Index mapping for the contiguous Info scalar buffer."""
    def _generate_next_value_(name, start, count, last_values):
        return count   # 0, 1, 2, ...

    rho = auto()
    delta = auto()
    mu = auto()
    sigma = auto()
    primal_step = auto()
    dual_step = auto()
    primal_res = auto()
    primal_res_rel = auto()
    dual_res = auto()
    dual_res_rel = auto()
    primal_res_reg = auto()
    primal_res_reg_rel = auto()
    dual_res_reg = auto()
    dual_res_reg_rel = auto()
    primal_prox_inf = auto()
    dual_prox_inf = auto()
    prev_primal_res = auto()
    prev_dual_res = auto()
    primal_obj = auto()
    dual_obj = auto()
    duality_gap = auto()
    duality_gap_rel = auto()
    reg_limit = auto()
    setup_time = auto()
    update_time = auto()
    solve_time = auto()
    kkt_factor_time = auto()
    kkt_solve_time = auto()
    run_time = auto()

class Info:
    """Per-problem solver info — ``(B, num_fields)`` GPU buffer.

    Each problem independently tracks rho, delta, mu, residuals, etc.
    For B=1 this is a ``(1, num_fields)`` buffer.
    """

    # Auto-generate properties from InfoIdx: getter returns (B,) view,
    # setter copies in-place.
    for idx in InfoIdx:
        def _make(i=idx):
            def getter(self):
                return self._buffer[:, i]
            def setter(self, value):
                self._buffer[:, i] = value
            return property(getter, setter)
        locals()[idx.name] = _make()
    del idx, _make

    def __init__(self, batch_size: int = 1):
        self._batch_size = batch_size
        self.status = np.full(batch_size, Status.PIQP_UNSOLVED.value, dtype=np.int32)
        self.iter = np.zeros(batch_size, dtype=np.int32)
        self.factor_retires = np.zeros(batch_size, dtype=np.int32)
        self.no_primal_update = np.zeros(batch_size, dtype=np.int32)
        self.no_dual_update = np.zeros(batch_size, dtype=np.int32)

    def init(self):
        self._buffer = cp.zeros((self._batch_size, len(InfoIdx)), dtype=cp.float64)

    @nvtx.annotate("Info:to_host")
    def to_host(self, info_host: 'InfoHost'):
        cp.asnumpy(self._buffer, out=info_host._buffer)

    @property
    def batch_size(self) -> int:
        return self._batch_size


class InfoHost:
    """
    A mirror of Info on the host side (CPU). The purpose is to fetch all device-side info to host all at once, instead of multiple time to reduce overhead.

    Each property returns a ``(B,)`` NumPy array.
    """
    __slots__ = ('_buffer', '_batch_size')

    for _idx in InfoIdx:
        locals()[_idx.name] = property(lambda self, i=_idx: self._buffer[:, i])
    del _idx

    def __init__(self, batch_size: int = 1):
        self._batch_size = batch_size
        self._buffer = np.empty((batch_size, len(InfoIdx)), dtype=np.float64)



class Result(Variables):
    """Combined variables + per-problem info."""
    def __init__(self, batch_size: int = 1):
        super().__init__()
        self.info = Info(batch_size)

    def init(self, data):
        assert data.batch_size == self.info.batch_size, \
            f"batch_size mismatch: Result({self.info.batch_size}) vs data({data.batch_size})"
        super().init(data)
        self.info.init()
