from abc import ABC, abstractmethod
import cupy as cp
import warp as wp

from .typedef import PIQP_INF


def _wp_from_cupy_int32(cp_arr, device: str = "cuda") -> "wp.array":
    """Allocate a warp int32 array and copy the contents of a cupy int32 array.

    Used for bound-index arrays. cp_arr must be int32.
    """
    if cp_arr.size == 0:
        return wp.empty(0, dtype=wp.int32, device=device)
    out = wp.empty(cp_arr.shape, dtype=wp.int32, device=device)
    cp.asarray(out)[:] = cp_arr
    return out


class Data(ABC):
    """Abstract base class for QP problem data.

    All arrays carry a leading batch dimension ``(B, ...)``.
    For single-problem inputs, ``B = 1``.
    """

    def __init__(self, dtype=wp.float64, device: str = "cuda"):
        self._dtype = dtype
        self._device = device

    @property
    def dtype(self):
        return self._dtype

    @abstractmethod
    def _disable_inf_constraints(self):
        """Zero out G rows where both h_l and h_u are infinite."""
        ...

    @abstractmethod
    def extract_P_diag(self, diag_P: cp.ndarray) -> None:
        """Extract the diagonal of P into *diag_P* — shape ``(B, n)``."""
        ...

    @abstractmethod
    def set_P(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_c(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_A(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_b(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_G(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_h_l(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_h_u(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_x_l(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_x_u(self, value, check: bool = True):
        ...

    def _finalize(self):
        """Shared post-init: preprocessing.

        Must be called by every subclass ``__init__`` after ``_batch_size``,
        ``_n``, ``_P``, ``_c``, ``_A``, ``_b``, ``_G``, ``_h_l``, ``_h_u``,
        ``_x_l``, ``_x_u`` have been populated.
        """
        self._preprocess()

    def _preprocess(self):
        self._init_h_l()
        self._init_h_u()
        self._disable_inf_constraints()
        self._init_h_l()
        self._init_h_u()
        self._init_x_l()
        self._init_x_u()

    # ------------------------------------------------------------------
    # Bound index initialization — works on (B, k) dense vectors
    # ------------------------------------------------------------------

    def _init_h_l(self):
        B, m = self._batch_size, self.m
        if m > 0 and self._h_l.shape[-1] == m:
            mask0 = cp.asarray(self._h_l)[0] > -PIQP_INF
            self._validate_bound_consistency(self._h_l, -PIQP_INF, '>', 'h_l')
            self._idx_hl = _wp_from_cupy_int32(cp.where(mask0)[0].astype(cp.int32), device=self._device)
        else:
            self._idx_hl = wp.empty(0, dtype=wp.int32, device=self._device)
            if self._h_l.shape[-1] == 0:
                self._h_l = wp.empty((B, m), dtype=self._dtype, device=self._device)
                cp.asarray(self._h_l)[:] = -2 * PIQP_INF

    def _init_h_u(self):
        B, m = self._batch_size, self.m
        if m > 0 and self._h_u.shape[-1] == m:
            mask0 = cp.asarray(self._h_u)[0] < PIQP_INF
            self._validate_bound_consistency(self._h_u, PIQP_INF, '<', 'h_u')
            self._idx_hu = _wp_from_cupy_int32(cp.where(mask0)[0].astype(cp.int32), device=self._device)
        else:
            self._idx_hu = wp.empty(0, dtype=wp.int32, device=self._device)
            if self._h_u.shape[-1] == 0:
                self._h_u = wp.empty((B, m), dtype=self._dtype, device=self._device)
                cp.asarray(self._h_u)[:] = 2 * PIQP_INF

    def _init_x_l(self):
        n = self._n
        if self._x_l.shape[-1] == n:
            mask0 = cp.asarray(self._x_l)[0] > -PIQP_INF
            self._validate_bound_consistency(self._x_l, -PIQP_INF, '>', 'x_l')
            self._idx_xl = _wp_from_cupy_int32(cp.where(mask0)[0].astype(cp.int32), device=self._device)
        else:
            self._idx_xl = wp.empty(0, dtype=wp.int32, device=self._device)
            self._x_l = wp.empty((self._batch_size, n), dtype=self._dtype, device=self._device)
            cp.asarray(self._x_l)[:] = -2 * PIQP_INF

    def _init_x_u(self):
        n = self._n
        if self._x_u.shape[-1] == n:
            mask0 = cp.asarray(self._x_u)[0] < PIQP_INF
            self._validate_bound_consistency(self._x_u, PIQP_INF, '<', 'x_u')
            self._idx_xu = _wp_from_cupy_int32(cp.where(mask0)[0].astype(cp.int32), device=self._device)
        else:
            self._idx_xu = wp.empty(0, dtype=wp.int32, device=self._device)
            self._x_u = wp.empty((self._batch_size, n), dtype=self._dtype, device=self._device)
            cp.asarray(self._x_u)[:] = 2 * PIQP_INF

    @staticmethod
    def _validate_bound_consistency(bounds, threshold: float,
                                    direction: str, name: str):
        """Ensure all problems in the batch have the same finite/infinite pattern."""
        if bounds.shape[0] <= 1:
            return  # nothing to validate for B=1
        bounds_cp = cp.asarray(bounds)
        if direction == '>':
            mask = bounds_cp > -abs(threshold)
        else:
            mask = bounds_cp < abs(threshold)
        if not bool(cp.all(mask == mask[0:1])):
            raise ValueError(
                f"Bound structure mismatch in '{name}': all problems in the batch "
                f"must have the same set of finite bounds."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def n(self) -> int:
        """Number of variables."""
        return self._n

    @property
    def p(self) -> int:
        """Number of equality constraints."""
        return self._A.shape[-2] if hasattr(self._A, 'shape') else 0

    @property
    def m(self) -> int:
        """Number of inequality constraints."""
        return self._G.shape[-2] if hasattr(self._G, 'shape') else 0

    @property
    def P(self):
        return self._P

    @property
    def c(self):
        return self._c

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def G(self):
        return self._G

    @property
    def h_u(self):
        return self._h_u

    @property
    def h_l(self):
        return self._h_l

    @property
    def x_u(self):
        return self._x_u

    @property
    def x_l(self):
        return self._x_l

    @property
    def num_hl(self) -> int:
        return int(self._idx_hl.size)

    @property
    def idx_hl(self):
        """Indices of lower inequality constraints."""
        return self._idx_hl

    @property
    def num_hu(self) -> int:
        return int(self._idx_hu.size)

    @property
    def idx_hu(self):
        """Indices of upper inequality constraints."""
        return self._idx_hu

    @property
    def num_xl(self) -> int:
        """Number of lower bound constraints."""
        return int(self._idx_xl.size)

    @property
    def idx_xl(self):
        """Indices of lower bound constraints."""
        return self._idx_xl

    @property
    def num_xu(self) -> int:
        """Number of upper bound constraints."""
        return int(self._idx_xu.size)

    @property
    def idx_xu(self):
        """Indices of upper bound constraints."""
        return self._idx_xu

    @property
    def num_ineq(self) -> int:
        return self.num_hl + self.num_hu + self.num_xl + self.num_xu
