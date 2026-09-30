from abc import ABC, abstractmethod

import warp as wp

from .typedef import PIQP_INF
from .utils import to_warp_dtype, batch_broadcast_view, column_slice
from .data_kernels import create_finite_bound_masks_kernel


class Data(ABC):
    """Abstract base class for QP problem data.

    All arrays carry a leading batch dimension ``(B, ...)`` and are Warp
    arrays owned by this object. For single-problem inputs, ``B = 1``.
    Every input is a Warp array of this object's dtype: the solver converts
    and checks user arrays once at its public boundary, and nothing here
    re-checks or casts them.
    """

    def __init__(self, dtype="float64", device: str = "cuda"):
        self._dtype = to_warp_dtype(dtype)
        self._device = device

        self._has_h_l = False
        self._has_h_u = False
        self._has_x_l = False
        self._has_x_u = False
        self._finite_masks_kernel = None
        self._finite_mask_all = None

    @property
    def dtype(self):
        """Floating-point type of every array, a Warp scalar type (``wp.float32`` or ``wp.float64``)."""
        return self._dtype

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Storage helpers shared by every backend
    # ------------------------------------------------------------------

    def _write(self, dst: wp.array, src: wp.array, name: str, allow_batched: bool = True) -> None:
        """Copy ``src`` into the batched buffer ``dst`` in place.

        ``src`` has the single-problem shape ``dst.shape[1:]`` (copied to
        every problem) or, when ``allow_batched``, the full shape
        ``dst.shape``. One device-to-device copy, no allocation.
        """
        full, single = tuple(dst.shape), tuple(dst.shape[1:])
        shape = tuple(src.shape)
        if allow_batched and shape == full:
            pass
        elif shape == single:
            src = batch_broadcast_view(src, full[0])
        else:
            allowed = [single, full] if allow_batched else [single]
            raise ValueError(
                f"{name} must have shape {' or '.join(str(s) for s in allowed)}; got {shape}."
            )
        if dst.size == 0:
            return
        wp.copy(dst, src)

    @staticmethod
    def _resolve_batch_size(batch_dims: dict) -> int:
        """Batch size ``B`` from the leading batch axes of the inputs.

        ``batch_dims`` maps each input name to its batch size, or ``None``
        for an input in the single-problem shape (shared by every problem).
        All batched inputs must agree; with none, ``B = 1``.
        """
        batched = {name: B for name, B in batch_dims.items() if B is not None}
        sizes = set(batched.values())
        if len(sizes) > 1:
            listed = ", ".join(f"{name}: {B}" for name, B in batched.items())
            raise ValueError(f"Batched inputs disagree on the batch size ({listed}).")
        return sizes.pop() if sizes else 1

    def _init_vec(self, value: wp.array, k: int, name: str, batch_size: int) -> wp.array:
        """A new ``(B, k)`` buffer holding ``value``: ``(k,)`` shared by every
        problem, or ``(B, k)``."""
        out = wp.zeros((batch_size, k), dtype=self._dtype, device=self._device)
        self._write(out, value, name)
        return out

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

    def _set_vec(self, dst: wp.array, value, name: str, check: bool) -> None:
        """Shared body of the vector setters: shape check, then in-place copy."""
        self._write(dst, value, name)

    def _set_bound(self, dst: wp.array, has_block: bool, value, name: str, check: bool) -> None:
        if not has_block:
            raise ValueError(
                f"Cannot set {name}: no {name} block was provided at setup(). "
                "Adding a bound block requires a new setup()."
            )
        self._write(dst, value, name)
        self._update_finite_bound_masks()

    def _finalize(self):
        """Shared post-init: preprocessing.

        Must be called by every subclass ``init`` after ``_batch_size``,
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
            has_h_l=self._has_h_l, has_h_u=self._has_h_u,
            has_x_l=self._has_x_l, has_x_u=self._has_x_u,
            dtype=self._dtype
        )
        self._update_finite_bound_masks()

    def _update_finite_bound_masks(self):
        """Build per-batch finite-bound masks of inequality bounds.

        For each bound class, a ``(B, m)`` or ``(B, n)`` float mask holds 1.0
        where the bound is finite and 0.0 where it is +/-inf. ``num_finite_bounds``
        is the per-problem count of finite bounds (the divisor used for mu/sigma).
        These are full-length and may differ across batch elements.
        """
        B, m, n = self._batch_size, self.m, self._n
        num_hl, num_hu = self.num_hl, self.num_hu
        num_xl, num_xu = self.num_xl, self.num_xu
        num_ineq = num_hl + num_hu + num_xl + num_xu

        # Running offsets in the packed [hl? | hu? | xl? | xu?] layout. An
        # absent block (any of the four) has zero width, so the following block
        # slides up and its mask view becomes (B, 0).
        off_hl = 0
        off_hu = off_hl + num_hl
        off_xl = off_hu + num_hu
        off_xu = off_xl + num_xl

        # allocate once; reuse buffers on later calls
        if self._finite_mask_all is None or tuple(self._finite_mask_all.shape) != (B, num_ineq):
            self._finite_mask_all = wp.zeros((B, num_ineq), dtype=self._dtype, device=self._device)
            self._finite_mask_hl = column_slice(self._finite_mask_all, off_hl, off_hl + num_hl)
            self._finite_mask_hu = column_slice(self._finite_mask_all, off_hu, off_hu + num_hu)
            self._finite_mask_xl = column_slice(self._finite_mask_all, off_xl, off_xl + num_xl)
            self._finite_mask_xu = column_slice(self._finite_mask_all, off_xu, off_xu + num_xu)
            self._active_G_row = wp.zeros((B, m), dtype=self._dtype, device=self._device)
            self._active_x_bound = wp.zeros((B, n), dtype=self._dtype, device=self._device)
            self._num_finite_bounds = wp.zeros((B,), dtype=self._dtype, device=self._device)

        wp.launch(
            kernel=self._finite_masks_kernel,
            dim=(B,),
            inputs=[
                self._h_l, self._h_u, self._x_l, self._x_u,
                self._finite_mask_all, self._active_G_row,
                self._active_x_bound, self._num_finite_bounds,
            ],
            device=self._device
        )

    # ------------------------------------------------------------------
    # Bound initialization: an absent block has no storage, (B, 0).
    # ------------------------------------------------------------------

    def _init_h_l(self):
        B, m = self._batch_size, self.m
        if not self._has_h_l:
            self._h_l = wp.zeros((B, 0), dtype=self._dtype, device=self._device)
            return
        if tuple(self._h_l.shape) != (B, m):
            raise ValueError(f"h_l shape mismatch: expected {(B, m)}, got {tuple(self._h_l.shape)}")

    def _init_h_u(self):
        B, m = self._batch_size, self.m
        if not self._has_h_u:
            self._h_u = wp.zeros((B, 0), dtype=self._dtype, device=self._device)
            return
        if tuple(self._h_u.shape) != (B, m):
            raise ValueError(f"h_u shape mismatch: expected {(B, m)}, got {tuple(self._h_u.shape)}")

    def _init_x_l(self):
        B, n = self._batch_size, self._n
        if not self._has_x_l:
            self._x_l = wp.zeros((B, 0), dtype=self._dtype, device=self._device)
            return
        if tuple(self._x_l.shape) != (B, n):
            raise ValueError(f"x_l shape mismatch: expected {(B, n)}, got {tuple(self._x_l.shape)}")

    def _init_x_u(self):
        B, n = self._batch_size, self._n
        if not self._has_x_u:
            self._x_u = wp.zeros((B, 0), dtype=self._dtype, device=self._device)
            return
        if tuple(self._x_u.shape) != (B, n):
            raise ValueError(f"x_u shape mismatch: expected {(B, n)}, got {tuple(self._x_u.shape)}")

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
    def c(self) -> wp.array:
        """Linear cost, a ``(B, n)`` Warp array."""
        return self._c

    @property
    def A(self):
        return self._A

    @property
    def b(self) -> wp.array:
        """Equality right-hand side, a ``(B, p)`` Warp array (``(B, 0)`` without equalities)."""
        return self._b

    @property
    def G(self):
        return self._G

    @property
    def h_u(self) -> wp.array:
        """Upper inequality bounds, ``(B, m)``; ``(B, 0)`` when the side was omitted."""
        return self._h_u

    @property
    def h_l(self) -> wp.array:
        """Lower inequality bounds, ``(B, m)``; ``(B, 0)`` when the side was omitted."""
        return self._h_l

    @property
    def x_u(self) -> wp.array:
        """Upper box bounds, ``(B, n)``; ``(B, 0)`` when the side was omitted."""
        return self._x_u

    @property
    def x_l(self) -> wp.array:
        """Lower box bounds, ``(B, n)``; ``(B, 0)`` when the side was omitted."""
        return self._x_l

    @property
    def has_h_l(self) -> bool:
        """Whether a lower-inequality bound block was provided at setup().

        Block presence is structural and fixed at setup(); when ``False`` the
        lower-inequality block occupies no storage and ``h_l`` / ``z_l`` /
        ``s_l`` are empty ``(B, 0)`` views.
        """
        return self._has_h_l

    @property
    def has_h_u(self) -> bool:
        """Whether an upper-inequality bound block was provided at setup().

        See :attr:`has_h_l`; when ``False`` the upper-inequality block has no
        storage.
        """
        return self._has_h_u

    @property
    def num_hl(self) -> int:
        """Length of the lower-inequality dual/slack block.

        ``m`` when a lower-inequality block was provided at setup(), else ``0``.
        When present there is one slot per row of ``G`` regardless of how many
        lower bounds are finite; infinite bounds inside the block are masked,
        not dropped.
        """
        return self.m if self._has_h_l else 0

    @property
    def num_hu(self) -> int:
        """Length of the upper-inequality dual/slack block.

        ``m`` when an upper-inequality block was provided at setup(), else
        ``0``. When present there is one slot per row of ``G``; infinite bounds
        inside the block are masked, not dropped.
        """
        return self.m if self._has_h_u else 0

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
    def finite_mask_hl(self) -> wp.array:
        """(B, m) float mask, 1.0 where the lower inequality bound is finite."""
        return self._finite_mask_hl

    @property
    def finite_mask_hu(self) -> wp.array:
        """(B, m) float mask, 1.0 where the upper inequality bound is finite."""
        return self._finite_mask_hu

    @property
    def finite_mask_xl(self) -> wp.array:
        """(B, n) float mask, 1.0 where the lower box bound is finite."""
        return self._finite_mask_xl

    @property
    def finite_mask_xu(self) -> wp.array:
        """(B, n) float mask, 1.0 where the upper box bound is finite."""
        return self._finite_mask_xu

    @property
    def active_G_row(self) -> wp.array:
        """(B, m) float mask, 1.0 where inequality row i has any finite bound."""
        return self._active_G_row

    @property
    def active_x_bound(self) -> wp.array:
        """(B, n) float mask, 1.0 where variable i has any finite box bound."""
        return self._active_x_bound

    @property
    def num_finite_bounds(self) -> wp.array:
        """(B,) per-problem count of finite bounds (mu/sigma divisor)."""
        return self._num_finite_bounds

    @property
    def finite_mask_all(self) -> wp.array:
        """(B, num_ineq) finite-bound mask in packed [hl? | hu? | xl? | xu?] layout.

        Any absent block (inequality or box) contributes zero width, so
        ``num_ineq`` is ``num_hl + num_hu + num_xl + num_xu``.
        """
        return self._finite_mask_all
