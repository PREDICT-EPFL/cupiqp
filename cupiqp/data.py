from abc import ABC, abstractmethod
import cupy as cp
import warp as wp

from .typedef import PIQP_INF
from .data_kernels import create_finite_bound_masks_kernel


class Data(ABC):
    """Abstract base class for QP problem data.

    All arrays carry a leading batch dimension ``(B, ...)``.
    For single-problem inputs, ``B = 1``.
    """

    def __init__(self, dtype=cp.float64, device: str = "cuda"):
        self._dtype = dtype
        self._device = device
        self._finite_masks_kernel = create_finite_bound_masks_kernel(dtype)
        self._finite_mask_all = None

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self) -> str:
        return self._device

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
        self._init_x_l()
        self._init_x_u()
        self._update_finite_bound_masks()

    def _update_finite_bound_masks(self):
        """Build per-batch finite-bound masks of inequality bounds.

        For each bound class, a ``(B, m)`` or ``(B, n)`` float mask holds 1.0
        where the bound is finite and 0.0 where it is +/-inf. ``num_finite_bounds``
        is the per-problem count of finite bounds (the divisor used for mu/sigma).
        Unlike the compressed ``idx_*`` arrays these are full-length and may
        differ across batch elements.
        """
        B, m, n = self._batch_size, self.m, self._n
        dtype = self._dtype
        num_ineq = 2 * m + 2 * n

        # allocate once; reuse buffers on later calls
        if self._finite_mask_all is None or self._finite_mask_all.shape != (B, num_ineq):
            self._finite_mask_all = cp.empty((B, num_ineq), dtype=dtype)
            self._finite_mask_hl = self._finite_mask_all[:, 0:m]
            self._finite_mask_hu = self._finite_mask_all[:, m:2 * m]
            self._finite_mask_xl = self._finite_mask_all[:, 2 * m:2 * m + n]
            self._finite_mask_xu = self._finite_mask_all[:, 2 * m + n:2 * m + 2 * n]
            self._active_G_row = cp.empty((B, m), dtype=dtype)
            self._active_x_bound = cp.empty((B, n), dtype=dtype)
            self._num_finite_bounds = cp.empty((B,), dtype=dtype)

        wp.launch(
            kernel=self._finite_masks_kernel,
            dim=(B,),
            inputs=[
                self._h_l, self._h_u, self._x_l, self._x_u,
                self._finite_mask_all, self._active_G_row,
                self._active_x_bound, self._num_finite_bounds,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    # ------------------------------------------------------------------
    # Bound index initialization — works on (B, k) dense vectors
    # ------------------------------------------------------------------

    def _init_h_l(self):
        B, m = self._batch_size, self.m
        if m == 0:
            self._h_l = cp.empty((B, 0), dtype=self._dtype)
            return
        if self._h_l.shape[-1] == 0:
            self._h_l = -2 * PIQP_INF * cp.ones((B, m), dtype=self._dtype)
        if self._h_l.shape != (B, m):
            raise ValueError(f"h_l shape mismatch: expected {(B, m)}, got {self._h_l.shape}")

    def _init_h_u(self):
        B, m = self._batch_size, self.m
        if m == 0:
            self._h_u = cp.empty((B, 0), dtype=self._dtype)
            return
        if self._h_u.shape[-1] == 0:
            self._h_u = 2 * PIQP_INF * cp.ones((B, m), dtype=self._dtype)
        if self._h_u.shape != (B, m):
            raise ValueError(f"h_u shape mismatch: expected {(B, m)}, got {self._h_u.shape}")

    def _init_x_l(self):
        n = self._n
        if self._x_l.shape[-1] == 0:
            self._x_l = -2 * PIQP_INF * cp.ones((self._batch_size, n), dtype=self._dtype)
        if self._x_l.shape != (self._batch_size, n):
            raise ValueError(f"x_l shape mismatch: expected {(self._batch_size, n)}, got {self._x_l.shape}")

    def _init_x_u(self):
        n = self._n
        if self._x_u.shape[-1] == 0:
            self._x_u = 2 * PIQP_INF * cp.ones((self._batch_size, n), dtype=self._dtype)
        if self._x_u.shape != (self._batch_size, n):
            raise ValueError(f"x_u shape mismatch: expected {(self._batch_size, n)}, got {self._x_u.shape}")

    @staticmethod
    def _validate_bound_consistency(bounds: cp.ndarray, threshold: float,
                                    direction: str, name: str):
        """Ensure all problems in the batch have the same finite/infinite pattern."""
        if bounds.shape[0] <= 1:
            return  # nothing to validate for B=1
        if direction == '>':
            mask = bounds > -abs(threshold)
        else:
            mask = bounds < abs(threshold)
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
        return self.m

    @property
    def num_hu(self) -> int:
        return self.m

    @property
    def num_xl(self) -> int:
        """Number of lower bound constraints."""
        return self.n

    @property
    def num_xu(self) -> int:
        """Number of upper bound constraints."""
        return self.n

    @property
    def num_ineq(self) -> int:
        return 2 * self.m + 2 * self.n

    @property
    def finite_mask_hl(self):
        """(B, m) float mask, 1.0 where the lower inequality bound is finite."""
        return self._finite_mask_hl

    @property
    def finite_mask_hu(self):
        """(B, m) float mask, 1.0 where the upper inequality bound is finite."""
        return self._finite_mask_hu

    @property
    def finite_mask_xl(self):
        """(B, n) float mask, 1.0 where the lower box bound is finite."""
        return self._finite_mask_xl

    @property
    def finite_mask_xu(self):
        """(B, n) float mask, 1.0 where the upper box bound is finite."""
        return self._finite_mask_xu

    @property
    def active_G_row(self):
        """(B, m) float mask, 1.0 where inequality row i has any finite bound."""
        return self._active_G_row

    @property
    def active_x_bound(self):
        """(B, n) float mask, 1.0 where variable i has any finite box bound."""
        return self._active_x_bound

    @property
    def num_finite_bounds(self):
        """(B,) per-problem count of finite bounds (mu/sigma divisor)."""
        return self._num_finite_bounds

    @property
    def finite_mask_all(self):
        """(B, 2*m+2*n) finite-bound mask in [l, u, bl, bu] layout."""
        return self._finite_mask_all
