import numpy as np
import warp as wp
from typing import List
from enum import Enum, IntEnum
import nvtx

from .data import Data
from .utils import to_warp_dtype, as_warp_array, column_slice


class Status(Enum):
    CUPIQP_UNSOLVED = -1
    CUPIQP_SOLVED = 0
    CUPIQP_MAX_ITER_REACHED = 1
    CUPIQP_PRIMAL_INFEASIBLE = 2
    CUPIQP_DUAL_INFEASIBLE = 3
    CUPIQP_NUMERICAL_ISSUES = 4


class Variables:
    """Optimization variables for a batch of B QPs.

    Two contiguous Warp buffers per problem, laid out as::

        _primal_buffer : (B, num_var + num_ineq)  -- [x | s_l | s_u | s_bl | s_bu]
        _dual_buffer   : (B, num_eq + num_ineq)   -- [y | z_l | z_u | z_bl | z_bu]

    Individual variable attributes (x, y, s_l, z_u, ...) are zero-copy (B, *)
    Warp views into these buffers. Assigning to an attribute copies the given
    GPU array into the view in place, so the contiguous layout is preserved.
    """

    def __init__(self):
        pass

    def init(self, data: Data):
        self._batch_size = data.batch_size
        self.n = data.n
        self.p = data.p
        self.m = data.m
        n, m, p = data.n, data.m, data.p
        num_hl, num_hu = data.num_hl, data.num_hu
        num_xl, num_xu = data.num_xl, data.num_xu
        self.num_ineq = num_hl + num_hu + num_xl + num_xu

        B = self._batch_size
        self._dtype = data.dtype
        device = data.device

        # Each inequality/box block is full width (m or n) when provided at
        # setup() and zero width (B, 0) when omitted; the following blocks slide
        # up. Primal: [x(n) | s_l(num_hl) | s_u(num_hu) | s_bl(num_xl) | s_bu(num_xu)]
        self._primal_buffer = wp.empty((B, n + self.num_ineq), dtype=self._dtype, device=device)
        offset = 0
        self._x = column_slice(self._primal_buffer, offset, offset+n)
        self._s_all = column_slice(self._primal_buffer, n, None)
        offset += n
        self._s_l = column_slice(self._primal_buffer, offset, offset+num_hl)
        offset += num_hl
        self._s_u = column_slice(self._primal_buffer, offset, offset+num_hu)
        offset += num_hu
        self._s_bl = column_slice(self._primal_buffer, offset, offset+num_xl)
        offset += num_xl
        self._s_bu = column_slice(self._primal_buffer, offset, offset+num_xu)

        # Dual: [y(p) | z_l(num_hl) | z_u(num_hu) | z_bl(num_xl) | z_bu(num_xu)]
        self._dual_buffer = wp.empty((B, p + self.num_ineq), dtype=self._dtype, device=device)
        offset = 0
        self._y = column_slice(self._dual_buffer, offset, offset+p)
        self._z_all = column_slice(self._dual_buffer, p, None)
        offset += p
        self._z_l = column_slice(self._dual_buffer, offset, offset+num_hl)
        offset += num_hl
        self._z_u = column_slice(self._dual_buffer, offset, offset+num_hu)
        offset += num_hu
        self._z_bl = column_slice(self._dual_buffer, offset, offset+num_xl)
        offset += num_xl
        self._z_bu = column_slice(self._dual_buffer, offset, offset+num_xu)

    @staticmethod
    def _assign(view: wp.array, value) -> None:
        """Copy a GPU array of the view's shape and dtype into the view, or fill it with a scalar."""
        if view.size == 0:
            return
        if isinstance(value, (int, float, np.floating, np.integer)):
            view.fill_(float(value))
            return
        src = as_warp_array(value)
        if tuple(src.shape) != tuple(view.shape):
            raise ValueError(f"shape mismatch: expected {tuple(view.shape)}, got {tuple(src.shape)}")
        if src.dtype != view.dtype:
            raise TypeError(f"dtype mismatch: expected {view.dtype.__name__}, got {src.dtype.__name__}")
        wp.copy(view, src)

    # -- Properties: getters return (batch_size, *) views, setters copy in-place --

    @property
    def x(self) -> wp.array: return self._x
    @x.setter
    def x(self, value): self._assign(self._x, value)

    @property
    def y(self) -> wp.array: return self._y
    @y.setter
    def y(self, value): self._assign(self._y, value)

    @property
    def s_all(self) -> wp.array: return self._s_all
    @s_all.setter
    def s_all(self, value): self._assign(self._s_all, value)

    @property
    def s_l(self) -> wp.array: return self._s_l
    @s_l.setter
    def s_l(self, value): self._assign(self._s_l, value)

    @property
    def s_u(self) -> wp.array: return self._s_u
    @s_u.setter
    def s_u(self, value): self._assign(self._s_u, value)

    @property
    def s_bl(self) -> wp.array: return self._s_bl
    @s_bl.setter
    def s_bl(self, value): self._assign(self._s_bl, value)

    @property
    def s_bu(self) -> wp.array: return self._s_bu
    @s_bu.setter
    def s_bu(self, value): self._assign(self._s_bu, value)

    @property
    def z_all(self) -> wp.array: return self._z_all
    @z_all.setter
    def z_all(self, value): self._assign(self._z_all, value)

    @property
    def z_l(self) -> wp.array: return self._z_l
    @z_l.setter
    def z_l(self, value): self._assign(self._z_l, value)

    @property
    def z_u(self) -> wp.array: return self._z_u
    @z_u.setter
    def z_u(self, value): self._assign(self._z_u, value)

    @property
    def z_bl(self) -> wp.array: return self._z_bl
    @z_bl.setter
    def z_bl(self, value): self._assign(self._z_bl, value)

    @property
    def z_bu(self) -> wp.array: return self._z_bu
    @z_bu.setter
    def z_bu(self, value): self._assign(self._z_bu, value)

    @property
    def primals_all(self) -> wp.array: return self._primal_buffer
    @primals_all.setter
    def primals_all(self, value): self._assign(self._primal_buffer, value)

    @property
    def duals_all(self) -> wp.array: return self._dual_buffer
    @duals_all.setter
    def duals_all(self, value): self._assign(self._dual_buffer, value)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def copy_from(self, other: 'Variables') -> None:
        """Copy every variable of ``other`` (same layout) into this object."""
        wp.copy(self._primal_buffer, other._primal_buffer)
        wp.copy(self._dual_buffer, other._dual_buffer)

    def all_finite(self) -> bool:
        """Test purpose only"""
        return bool(
            np.isfinite(self._primal_buffer.numpy()).all() and
            np.isfinite(self._dual_buffer.numpy()).all()
        )

    @property
    def buffer_ptr(self) -> tuple:
        """Memory addresses for CUDA graph cache keys."""
        return (
            self._primal_buffer.ptr,
            self._dual_buffer.ptr
        )

    def allclose(self, other: 'Variables', rtol: float = 1e-8, atol: float = 1e-8) -> bool:
        return (np.allclose(self._primal_buffer.numpy(), other._primal_buffer.numpy(), rtol=rtol, atol=atol) and
                np.allclose(self._dual_buffer.numpy(), other._dual_buffer.numpy(), rtol=rtol, atol=atol))

    def __str__(self) -> str:
        return (f"Variables (B={self._batch_size}):\n"
            f"  x:    {self.x.numpy()}\n"
            f"  y:    {self.y.numpy()}\n"
            f"  z_u:  {self.z_u.numpy()}\n"
            f"  z_l:  {self.z_l.numpy()}\n"
            f"  z_bu: {self.z_bu.numpy()}\n"
            f"  z_bl: {self.z_bl.numpy()}\n"
            f"  s_u:  {self.s_u.numpy()}\n"
            f"  s_l:  {self.s_l.numpy()}\n"
            f"  s_bu: {self.s_bu.numpy()}\n"
            f"  s_bl: {self.s_bl.numpy()}")

    def to_array(self) -> np.ndarray:
        """Test purpose only: host copy of ``[primals | duals]``, shape ``(B, ...)``."""
        return np.concatenate((self._primal_buffer.numpy(), self._dual_buffer.numpy()), axis=1)

    def set_random(self):
        """Testing purpose only: set all variables to random values."""
        rng = np.random.RandomState(0)
        dtype = wp.dtype_to_numpy(self._dtype)

        def put(view, values):
            if view.size:
                wp.copy(view, wp.array(np.ascontiguousarray(values, dtype=dtype), device=view.device))

        put(self._x, rng.randn(*self._x.shape))
        put(self._y, rng.randn(*self._y.shape))
        put(self._z_u, rng.rand(*self._z_u.shape) + 1.0)  # ensure positiveness
        put(self._z_l, rng.rand(*self._z_l.shape) + 1.0)
        put(self._z_bu, rng.rand(*self._z_bu.shape) + 1.0)
        put(self._z_bl, rng.rand(*self._z_bl.shape) + 1.0)
        put(self._s_u, rng.rand(*self._s_u.shape) + 1.0)
        put(self._s_l, rng.rand(*self._s_l.shape) + 1.0)
        put(self._s_bu, rng.rand(*self._s_bu.shape) + 1.0)
        put(self._s_bl, rng.rand(*self._s_bl.shape) + 1.0)

class InfoIdx(IntEnum):
    """Row index of each scalar field in the contiguous Info buffer."""
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


def _info_field(idx: InfoIdx):
    """A ``(B,)`` Warp view of one field; assignment fills or copies in place."""
    def getter(self):
        return self._views[idx]

    def setter(self, value):
        view = self._views[idx]
        if isinstance(value, (int, float, np.floating, np.integer)):
            view.fill_(float(value))
        else:
            wp.copy(view, as_warp_array(value))

    return property(getter, setter)


class Info:
    """Per-problem solver info: a ``(num_fields, B)`` GPU buffer.

    Each problem independently tracks rho, delta, mu, residuals, etc. Every
    field is a contiguous ``(B,)`` Warp view of one row of the buffer, so a
    single device-to-host copy fetches all of them.
    """

    rho = _info_field(InfoIdx.rho)
    delta = _info_field(InfoIdx.delta)
    mu = _info_field(InfoIdx.mu)
    sigma = _info_field(InfoIdx.sigma)
    primal_step = _info_field(InfoIdx.primal_step)
    dual_step = _info_field(InfoIdx.dual_step)
    primal_res = _info_field(InfoIdx.primal_res)
    primal_res_rel = _info_field(InfoIdx.primal_res_rel)
    dual_res = _info_field(InfoIdx.dual_res)
    dual_res_rel = _info_field(InfoIdx.dual_res_rel)
    primal_res_reg = _info_field(InfoIdx.primal_res_reg)
    primal_res_reg_rel = _info_field(InfoIdx.primal_res_reg_rel)
    dual_res_reg = _info_field(InfoIdx.dual_res_reg)
    dual_res_reg_rel = _info_field(InfoIdx.dual_res_reg_rel)
    primal_prox_inf = _info_field(InfoIdx.primal_prox_inf)
    dual_prox_inf = _info_field(InfoIdx.dual_prox_inf)
    prev_primal_res = _info_field(InfoIdx.prev_primal_res)
    prev_dual_res = _info_field(InfoIdx.prev_dual_res)
    primal_obj = _info_field(InfoIdx.primal_obj)
    dual_obj = _info_field(InfoIdx.dual_obj)
    duality_gap = _info_field(InfoIdx.duality_gap)
    duality_gap_rel = _info_field(InfoIdx.duality_gap_rel)
    reg_limit = _info_field(InfoIdx.reg_limit)
    setup_time = _info_field(InfoIdx.setup_time)
    update_time = _info_field(InfoIdx.update_time)
    solve_time = _info_field(InfoIdx.solve_time)
    kkt_factor_time = _info_field(InfoIdx.kkt_factor_time)
    kkt_solve_time = _info_field(InfoIdx.kkt_solve_time)
    run_time = _info_field(InfoIdx.run_time)

    def __init__(self, batch_size: int = 1):
        self._batch_size = batch_size
        self._status_value = np.full(batch_size, Status.CUPIQP_UNSOLVED.value, dtype=np.int32)
        self.iter = np.zeros(batch_size, dtype=np.int32)  # individual iter counts for each problem in the batch
        self.iter_total = 0  # total iterations the solver runs for this batch (the slowest problem's count)
        self.factor_retires = np.zeros(batch_size, dtype=np.int32)
        # Per-batch "no update" counters live on device (int32). Source of truth;
        # the rho/delta kernels reset on improved, increment on stagnated.
        # to_host() syncs them into the InfoHost mirror once per IPM iteration.
        self._counters = wp.zeros((2, batch_size), dtype=wp.int32, device="cuda")
        self.no_primal_update = self._counters[0]
        self.no_dual_update = self._counters[1]

    def init(self, dtype="float64", device: str = "cuda"):
        self._buffer = wp.zeros((len(InfoIdx), self._batch_size), dtype=to_warp_dtype(dtype), device=device)
        self._views = {idx: self._buffer[int(idx)] for idx in InfoIdx}

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
        """Copy every device field into the pinned host mirror and wait for it."""
        stream = wp.get_stream("cuda")
        wp.copy(info_host._buffer, self._buffer)
        wp.copy(info_host._counters, self._counters)
        wp.synchronize_stream(stream)

    @property
    def batch_size(self) -> int:
        return self._batch_size


def _host_field(idx: InfoIdx):
    def getter(self):
        return self._np[int(idx)]
    return property(getter)


class InfoHost:
    """
    A mirror of Info on the host side (CPU). The purpose is to fetch all device-side info to host all at once, instead of multiple time to reduce overhead.

    Each property returns a ``(B,)`` NumPy array (a view of pinned host memory).
    """
    __slots__ = ('_buffer', '_counters', '_np', '_np_counters', '_batch_size')

    rho = _host_field(InfoIdx.rho)
    delta = _host_field(InfoIdx.delta)
    mu = _host_field(InfoIdx.mu)
    sigma = _host_field(InfoIdx.sigma)
    primal_step = _host_field(InfoIdx.primal_step)
    dual_step = _host_field(InfoIdx.dual_step)
    primal_res = _host_field(InfoIdx.primal_res)
    primal_res_rel = _host_field(InfoIdx.primal_res_rel)
    dual_res = _host_field(InfoIdx.dual_res)
    dual_res_rel = _host_field(InfoIdx.dual_res_rel)
    primal_res_reg = _host_field(InfoIdx.primal_res_reg)
    primal_res_reg_rel = _host_field(InfoIdx.primal_res_reg_rel)
    dual_res_reg = _host_field(InfoIdx.dual_res_reg)
    dual_res_reg_rel = _host_field(InfoIdx.dual_res_reg_rel)
    primal_prox_inf = _host_field(InfoIdx.primal_prox_inf)
    dual_prox_inf = _host_field(InfoIdx.dual_prox_inf)
    prev_primal_res = _host_field(InfoIdx.prev_primal_res)
    prev_dual_res = _host_field(InfoIdx.prev_dual_res)
    primal_obj = _host_field(InfoIdx.primal_obj)
    dual_obj = _host_field(InfoIdx.dual_obj)
    duality_gap = _host_field(InfoIdx.duality_gap)
    duality_gap_rel = _host_field(InfoIdx.duality_gap_rel)
    reg_limit = _host_field(InfoIdx.reg_limit)
    setup_time = _host_field(InfoIdx.setup_time)
    update_time = _host_field(InfoIdx.update_time)
    solve_time = _host_field(InfoIdx.solve_time)
    kkt_factor_time = _host_field(InfoIdx.kkt_factor_time)
    kkt_solve_time = _host_field(InfoIdx.kkt_solve_time)
    run_time = _host_field(InfoIdx.run_time)

    @property
    def no_primal_update(self):
        return self._np_counters[0]

    @property
    def no_dual_update(self):
        return self._np_counters[1]

    def __init__(self, batch_size: int = 1, dtype=np.float64):
        self._batch_size = batch_size
        # Pinned host memory: the device-to-host copy is asynchronous on the
        # solver stream and completed with one stream synchronization.
        self._buffer = wp.zeros((len(InfoIdx), batch_size), dtype=to_warp_dtype(dtype), device="cpu", pinned=True)
        self._counters = wp.zeros((2, batch_size), dtype=wp.int32, device="cpu", pinned=True)
        self._np = self._buffer.numpy()
        self._np_counters = self._counters.numpy()



class Result(Variables):
    """Combined variables + per-problem info."""
    def __init__(self, batch_size: int = 1):
        super().__init__()
        self.info = Info(batch_size)

    def init(self, data):
        assert data.batch_size == self.info.batch_size, \
            f"batch_size mismatch: Result({self.info.batch_size}) vs data({data.batch_size})"
        super().init(data)
        self.info.init(dtype=data.dtype, device=data.device)
