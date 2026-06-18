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
        # box-bound block presence is structural and fixed at setup(): the
        # subclass init sets these from whether x_l / x_u were provided. An
        # omitted block occupies no storage and produces empty (B, 0) views.
        self._has_x_l = False
        self._has_x_u = False
        self._finite_masks_kernel = None
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
        self._finite_masks_kernel = create_finite_bound_masks_kernel(
            self._dtype, has_x_l=self._has_x_l, has_x_u=self._has_x_u
        )
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
        num_hl, num_hu = self.num_hl, self.num_hu
        num_xl, num_xu = self.num_xl, self.num_xu
        num_ineq = num_hl + num_hu + num_xl + num_xu

        # Running offsets in the packed [hl | hu | xl? | xu?] layout. An absent
        # box block has zero width, so the following block slides up and its
        # mask view becomes (B, 0).
        off_hl = 0
        off_hu = off_hl + num_hl
        off_xl = off_hu + num_hu
        off_xu = off_xl + num_xl

        # allocate once; reuse buffers on later calls
        if self._finite_mask_all is None or self._finite_mask_all.shape != (B, num_ineq):
            self._finite_mask_all = cp.empty((B, num_ineq), dtype=dtype)
            self._finite_mask_hl = self._finite_mask_all[:, off_hl:off_hl + num_hl]
            self._finite_mask_hu = self._finite_mask_all[:, off_hu:off_hu + num_hu]
            self._finite_mask_xl = self._finite_mask_all[:, off_xl:off_xl + num_xl]
            self._finite_mask_xu = self._finite_mask_all[:, off_xu:off_xu + num_xu]
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
        B, n = self._batch_size, self._n
        if not self._has_x_l:
            self._x_l = cp.empty((B, 0), dtype=self._dtype)
            return
        if self._x_l.shape[-1] == 0:
            self._x_l = -2 * PIQP_INF * cp.ones((B, n), dtype=self._dtype)
        if self._x_l.shape != (B, n):
            raise ValueError(f"x_l shape mismatch: expected {(B, n)}, got {self._x_l.shape}")

    def _init_x_u(self):
        B, n = self._batch_size, self._n
        if not self._has_x_u:
            self._x_u = cp.empty((B, 0), dtype=self._dtype)
            return
        if self._x_u.shape[-1] == 0:
            self._x_u = 2 * PIQP_INF * cp.ones((B, n), dtype=self._dtype)
        if self._x_u.shape != (B, n):
            raise ValueError(f"x_u shape mismatch: expected {(B, n)}, got {self._x_u.shape}")

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
        """Length of the lower-inequality dual/slack block, always ``m``.

        With the full-length layout there is one slot per row of ``G``
        regardless of how many lower bounds are finite; infinite bounds are
        masked, not dropped.
        """
        return self.m

    @property
    def num_hu(self) -> int:
        """Length of the upper-inequality dual/slack block, always ``m``.

        One slot per row of ``G``; infinite bounds are masked, not dropped.
        """
        return self.m

    @property
    def has_x_l(self) -> bool:
        """Whether a lower box-bound block was provided at setup().

        Box-block presence is structural and fixed at setup(); when ``False``
        the lower box block occupies no storage and ``x_l`` / ``z_bl`` / ``s_bl``
        are empty ``(B, 0)`` views.
        """
        return self._has_x_l

    @property
    def has_x_u(self) -> bool:
        """Whether an upper box-bound block was provided at setup().

        See :attr:`has_x_l`; when ``False`` the upper box block has no storage.
        """
        return self._has_x_u

    @property
    def num_xl(self) -> int:
        """Length of the lower box-bound dual/slack block.

        ``n`` when a lower box block was provided at setup(), else ``0``. When
        present there is one slot per variable; infinite bounds inside the
        block are masked, not dropped.
        """
        return self.n if self._has_x_l else 0

    @property
    def num_xu(self) -> int:
        """Length of the upper box-bound dual/slack block.

        ``n`` when an upper box block was provided at setup(), else ``0``. When
        present there is one slot per variable; infinite bounds inside the
        block are masked, not dropped.
        """
        return self.n if self._has_x_u else 0

    @property
    def num_ineq(self) -> int:
        return self.num_hl + self.num_hu + self.num_xl + self.num_xu

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
        """(B, num_ineq) finite-bound mask in packed [hl | hu | xl? | xu?] layout.

        Absent box blocks contribute zero width, so ``num_ineq`` is
        ``2*m + num_xl + num_xu``.
        """
        return self._finite_mask_all
