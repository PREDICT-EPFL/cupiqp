import numpy as np
import warp as wp
from typing import List
from enum import Enum, IntEnum
import nvtx

from .data import Data
from .utils import to_warp_dtype, as_warp_array, column_slice
from .typedef import (
    STATUS_UNSOLVED,
    STATUS_SOLVED,
    STATUS_MAX_ITER_REACHED,
    STATUS_PRIMAL_INFEASIBLE,
    STATUS_DUAL_INFEASIBLE,
    STATUS_NUMERICAL_ISSUES,
)


class Status(Enum):
    """Per-problem solver status; values are the codes in ``cupiqp.typedef``."""
    CUPIQP_UNSOLVED = STATUS_UNSOLVED
    CUPIQP_SOLVED = STATUS_SOLVED
    CUPIQP_MAX_ITER_REACHED = STATUS_MAX_ITER_REACHED
    CUPIQP_PRIMAL_INFEASIBLE = STATUS_PRIMAL_INFEASIBLE
    CUPIQP_DUAL_INFEASIBLE = STATUS_DUAL_INFEASIBLE
    CUPIQP_NUMERICAL_ISSUES = STATUS_NUMERICAL_ISSUES


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

class InfoFloatIdx(IntEnum):
    """Column index of each scalar field in the ``(B, num_fields)`` Info buffer."""
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
    # setup_time = 23
    # update_time = 24
    # solve_time = 25
    # kkt_factor_time = 26
    # kkt_solve_time = 27
    # run_time = 28


class InfoIntIdx(IntEnum):
    """Column index of each int32 field in the ``(B, num_int_fields)`` Info counter buffer."""
    no_primal_update = 0
    no_dual_update = 1
    status = 2
    iter = 3
    factor_retries = 4


def _info_field(idx: InfoFloatIdx):
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


def _info_counter(idx: InfoIntIdx):
    """A ``(B,)`` int32 Warp view of one counter column."""
    def getter(self):
        return self._counter_views[idx]
    return property(getter)


class Info:
    """Per-problem solver info, resident on the GPU.

    Two buffers with batch as the leading dimension, so row ``b`` is the
    complete record of problem ``b``: a ``(B, num_fields)`` buffer of the
    solver dtype for rho, delta, mu, residuals, objectives and so on, and a
    ``(B, num_int_fields)`` int32 buffer for the stagnation counters, the
    status code (a ``Status`` value), the termination iteration and the
    factorization retries. Every named attribute (``info.rho``,
    ``info.status_device``, ...) is a ``(B,)`` Warp view into one column, so
    kernels read and write the fields in place and nothing is copied.

    The solver decides everything on the device and never reads these
    buffers during a solve. ``solve()`` copies them to a pinned host snapshot
    once at the end; the host-side accessors ``status``, ``status_value``,
    ``iter``, ``factor_retires`` and ``host`` read that snapshot and do not
    touch the GPU. Call ``to_host()`` to refresh the snapshot yourself after
    launching your own kernels on the solver stream; it synchronizes the
    current stream.
    """

    rho = _info_field(InfoFloatIdx.rho)
    delta = _info_field(InfoFloatIdx.delta)
    mu = _info_field(InfoFloatIdx.mu)
    sigma = _info_field(InfoFloatIdx.sigma)
    primal_step = _info_field(InfoFloatIdx.primal_step)
    dual_step = _info_field(InfoFloatIdx.dual_step)
    primal_res = _info_field(InfoFloatIdx.primal_res)
    primal_res_rel = _info_field(InfoFloatIdx.primal_res_rel)
    dual_res = _info_field(InfoFloatIdx.dual_res)
    dual_res_rel = _info_field(InfoFloatIdx.dual_res_rel)
    primal_res_reg = _info_field(InfoFloatIdx.primal_res_reg)
    primal_res_reg_rel = _info_field(InfoFloatIdx.primal_res_reg_rel)
    dual_res_reg = _info_field(InfoFloatIdx.dual_res_reg)
    dual_res_reg_rel = _info_field(InfoFloatIdx.dual_res_reg_rel)
    primal_prox_inf = _info_field(InfoFloatIdx.primal_prox_inf)
    dual_prox_inf = _info_field(InfoFloatIdx.dual_prox_inf)
    prev_primal_res = _info_field(InfoFloatIdx.prev_primal_res)
    prev_dual_res = _info_field(InfoFloatIdx.prev_dual_res)
    primal_obj = _info_field(InfoFloatIdx.primal_obj)
    dual_obj = _info_field(InfoFloatIdx.dual_obj)
    duality_gap = _info_field(InfoFloatIdx.duality_gap)
    duality_gap_rel = _info_field(InfoFloatIdx.duality_gap_rel)
    reg_limit = _info_field(InfoFloatIdx.reg_limit)

    no_primal_update = _info_counter(InfoIntIdx.no_primal_update)
    no_dual_update = _info_counter(InfoIntIdx.no_dual_update)
    status_device = _info_counter(InfoIntIdx.status)
    iter_device = _info_counter(InfoIntIdx.iter)
    factor_retries_device = _info_counter(InfoIntIdx.factor_retries)

    def __init__(self, batch_size: int = 1):
        self._batch_size = batch_size
        self.iter_total = 0  # iterations the solver ran for this batch (the slowest problem's count)

    def init(self, dtype=wp.float64, device: str = "cuda"):
        dtype = to_warp_dtype(dtype)
        B = self._batch_size
        self._buffer = wp.zeros((B, len(InfoFloatIdx)), dtype=dtype, device=device)
        self._views = {idx: self._buffer[:, int(idx)] for idx in InfoFloatIdx}
        self._counters = wp.zeros((B, len(InfoIntIdx)), dtype=wp.int32, device=device)
        self._counter_views = {idx: self._counters[:, int(idx)] for idx in InfoIntIdx}
        # Pinned host snapshot of both buffers: the D2H copies are asynchronous
        # on the solver stream and completed with one stream synchronization.
        self._buffer_host = wp.zeros((B, len(InfoFloatIdx)), dtype=dtype, device="cpu", pinned=True)
        self._counters_host = wp.zeros((B, len(InfoIntIdx)), dtype=wp.int32, device="cpu", pinned=True)
        self._np = self._buffer_host.numpy()
        self._np_counters = self._counters_host.numpy()
        self.reset_status()

    def reset_status(self) -> None:
        """Mark every problem ``CUPIQP_UNSOLVED`` on the device and in the snapshot."""
        self.status_device.fill_(Status.CUPIQP_UNSOLVED.value)
        self._np_counters[:, int(InfoIntIdx.status)] = Status.CUPIQP_UNSOLVED.value

    @nvtx.annotate("Info:to_host")
    def to_host(self) -> None:
        """Copy both device buffers into the host snapshot and wait for them."""
        wp.copy(self._buffer_host, self._buffer)
        wp.copy(self._counters_host, self._counters)
        wp.synchronize_stream(wp.get_stream("cuda"))

    @property
    def host(self) -> np.ndarray:
        """Snapshot of the float fields as a ``(B, num_fields)`` NumPy array,
        indexed by ``InfoFloatIdx`` along the last axis."""
        return self._np

    @property
    def status(self) -> List[Status]:
        """Per-problem status as a list of Status enums (from the snapshot)."""
        return [Status(v) for v in self._np_counters[:, int(InfoIntIdx.status)]]

    @property
    def status_value(self) -> np.ndarray:
        """Per-problem status as a ``(B,)`` int32 array of Status values (from the snapshot)."""
        return self._np_counters[:, int(InfoIntIdx.status)]

    @property
    def iter(self) -> np.ndarray:
        """Per-problem iteration at which each problem terminated (from the snapshot)."""
        return self._np_counters[:, int(InfoIntIdx.iter)]

    @property
    def factor_retires(self) -> np.ndarray:
        """Per-problem number of factorization retries (from the snapshot)."""
        return self._np_counters[:, int(InfoIntIdx.factor_retries)]

    @property
    def batch_size(self) -> int:
        return self._batch_size


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
