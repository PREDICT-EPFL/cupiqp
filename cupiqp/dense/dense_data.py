from typing import Any, Optional

import cupy as cp
import warp as wp

from ..data import Data
from ..typedef import PIQP_INF


# Type alias for every dense-input form ``init`` accepts. Must be a GPU-resident
# array exposing ``__cuda_array_interface__``: cupy ndarrays, CUDA torch
# tensors, JAX GPU arrays. CPU inputs (numpy, torch CPU, JAX CPU) are rejected
# — the caller is responsible for staging data to GPU before passing it in.
DenseInput = Any


def _ensure_3d(array: wp.array) -> wp.array:
    """(rows, cols) -> (1, rows, cols); no-op if already 3-D. Warp-native."""
    if array.ndim == 2:
        return array.reshape((1, array.shape[0], array.shape[1]))
    return array


def _ensure_2d(array: wp.array) -> wp.array:
    """(k,) -> (1, k); no-op if already 2-D. Warp-native."""
    if array.ndim == 1:
        return array.reshape((1, array.shape[0]))
    return array


def _to_warp(src: Any, copy: bool = True,
             dtype=wp.float64, device: str = "cuda") -> wp.array:
    """Wrap a GPU array (cupy, torch CUDA, jax GPU) as a warp array.

    The source must expose ``__cuda_array_interface__``. CPU inputs (numpy,
    torch CPU, jax CPU) are rejected — silently H2D-copying them here would
    hide the fact that the caller is feeding host data into a GPU solver.

    copy=True (default): allocate a fresh ``dtype`` buffer on ``device`` and
        D2D-memcpy the source in. Safe — caller can mutate the source without
        affecting the returned array. cupy's slice-assign coerces dtype if it
        differs from ``dtype``.
    copy=False: zero-copy adoption via DLPack. The returned warp array views
        the source memory; mutating one mutates the other. ``dtype`` and
        ``device`` are advisory — the result inherits both from the source.
        Caller must ensure the source is contiguous and outlives the view.
    """
    if not hasattr(src, '__cuda_array_interface__'):
        raise TypeError(
            f"Expected a GPU array exposing __cuda_array_interface__; "
            f"got {type(src).__name__}. "
        )
    if not copy:
        return wp.from_dlpack(src)
    src = cp.asarray(src)
    out = wp.empty(src.shape, dtype=dtype, device=device)
    if src.size > 0:
        cp.asarray(out)[:] = src
    return out


class DenseData(Data):
    """Dense data for one or more QPs with identical dimensions and bound structure.

    Two-phase construction:
        ``DenseData(dtype=wp.float64, device="cuda")`` stores config only;
        ``init(P, c, A, b, G, h_u, h_l, x_u, x_l)`` accepts the user-provided
        matrices/vectors and copies them into freshly allocated warp buffers
        of the configured dtype/device.

    Every matrix/vector argument to ``init`` must be a GPU-resident array
    exposing ``__cuda_array_interface__`` (cupy ndarray, CUDA torch tensor,
    JAX GPU array). CPU inputs are rejected. The cupy slice-assign inside
    ``_to_warp`` handles dtype coercion to ``self._dtype``.
    """
    def __init__(self, dtype=wp.float64, device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)

    def init(self,
             P: DenseInput,
             c: DenseInput,
             A: Optional[DenseInput] = None,
             b: Optional[DenseInput] = None,
             G: Optional[DenseInput] = None,
             h_u: Optional[DenseInput] = None,
             h_l: Optional[DenseInput] = None,
             x_u: Optional[DenseInput] = None,
             x_l: Optional[DenseInput] = None):
        """Populate storage from user inputs, converting to ``self._dtype``."""

        dtype, device = self._dtype, self._device

        # --- validate and store P, c ---
        # Pipeline: input → cupy (input-type normalization) → _to_warp
        # (allocate fresh warp buffer + memcpy with dtype coercion) →
        # _ensure_3d/2d (warp reshape) → validate shape.
        P = _ensure_3d(_to_warp(P, copy=True, dtype=dtype, device=device))
        c = _ensure_2d(_to_warp(c, copy=True, dtype=dtype, device=device))

        if P.ndim != 3 or P.shape[1] != P.shape[2]:
            raise ValueError(f"P must have shape (B, n, n), got {P.shape}")
        if c.ndim != 2:
            raise ValueError(f"c must have shape (B, n), got {c.shape}")
        if P.shape[0] != c.shape[0]:
            raise ValueError("Batch size mismatch between P and c.")
        if P.shape[1] != c.shape[1]:
            raise ValueError("Dimension mismatch between P and c.")

        B = P.shape[0]
        n = P.shape[1]
        self._batch_size = B
        self._n = n

        self._P = P
        self._c = c

        # --- equality constraints ---
        if A is not None and b is not None:
            A = _ensure_3d(_to_warp(A, copy=True, dtype=dtype, device=device))
            b = _ensure_2d(_to_warp(b, copy=True, dtype=dtype, device=device))
            if A.ndim != 3:
                raise ValueError(f"A must have shape (B, p, n), got {A.shape}")
            if b.ndim != 2:
                raise ValueError(f"b must have shape (B, p), got {b.shape}")
            if A.shape[0] != B or b.shape[0] != B:
                raise ValueError("Batch size mismatch in A or b.")
            if A.shape[1] != b.shape[1]:
                raise ValueError("Row mismatch between A and b.")
            if A.shape[2] != n:
                raise ValueError("Column mismatch between A and P.")
            self._A = A
            self._b = b
        else:
            self._A = wp.zeros((B, 0, n), dtype=dtype, device=device)
            self._b = wp.zeros((B, 0), dtype=dtype, device=device)

        # --- inequality constraints ---
        if G is not None:
            G = _ensure_3d(_to_warp(G, copy=True, dtype=dtype, device=device))
            if G.ndim != 3:
                raise ValueError(f"G must have shape (B, m, n), got {G.shape}")
            if G.shape[0] != B or G.shape[2] != n:
                raise ValueError("Shape mismatch in G.")
            if h_l is None and h_u is None:
                raise ValueError("Either h_l or h_u must be provided when G is given.")
            m = G.shape[1]
            if h_l is not None:
                h_l = _ensure_2d(_to_warp(h_l, copy=True, dtype=dtype, device=device))
                if h_l.shape != (B, m):
                    raise ValueError(f"h_l must have shape ({B}, {m}), got {h_l.shape}")
            if h_u is not None:
                h_u = _ensure_2d(_to_warp(h_u, copy=True, dtype=dtype, device=device))
                if h_u.shape != (B, m):
                    raise ValueError(f"h_u must have shape ({B}, {m}), got {h_u.shape}")
            self._G = G
        else:
            if h_u is not None or h_l is not None:
                raise ValueError("h_l and h_u must be None when G is None.")
            self._G = wp.zeros((B, 0, n), dtype=dtype, device=device)

        self._h_u = self._as_batched_vec(h_u)
        self._h_l = self._as_batched_vec(h_l)

        # --- variable bounds ---
        if x_l is not None:
            x_l = _ensure_2d(_to_warp(x_l, copy=True, dtype=dtype, device=device))
        if x_u is not None:
            x_u = _ensure_2d(_to_warp(x_u, copy=True, dtype=dtype, device=device))
        if x_l is not None and x_u is not None:
            if x_l.shape != (B, n) or x_u.shape != (B, n):
                raise ValueError(f"x_l and x_u must have shape ({B}, {n}).")
        self._x_u = self._as_batched_vec(x_u)
        self._x_l = self._as_batched_vec(x_l)

        self._finalize()
        return self

    def _as_batched_vec(self, v: Optional[wp.array]) -> wp.array:
        # v has already been converted to a (B, k) wp array (or None) by the
        # init() pipeline.
        if v is not None:
            return v
        return wp.zeros((self._batch_size, 0), dtype=self._dtype, device=self._device)

    def _disable_inf_constraints(self):
        """Zero out G rows where both h_l and h_u are infinite (fully free)."""
        m = self.m
        if m == 0:
            return
        h_l_cp = cp.asarray(self._h_l)
        h_u_cp = cp.asarray(self._h_u)
        free = (h_l_cp[0] <= -PIQP_INF) & (h_u_cp[0] >= PIQP_INF)
        if not bool(cp.any(free)):
            return
        cp.asarray(self._G)[:, free, :] = 0.0
        h_l_cp[:, free] = -1.0
        h_u_cp[:, free] = 1.0

    def extract_P_diag(self, out):
        """Extract diagonal of each P — shape (B, n)."""
        idx = cp.arange(self._n)
        cp.asarray(out)[:] = cp.asarray(self._P)[:, idx, idx]

    # ------------------------------------------------------------------
    # In-place setters
    # ------------------------------------------------------------------

    def set_P(self, value, check: bool = True):
        if check and value.shape != self._P.shape:
            raise ValueError(f"P shape mismatch: expected {self._P.shape}, got {value.shape}")
        cp.asarray(self._P)[:] = value

    def set_c(self, value, check: bool = True):
        if check and value.shape != self._c.shape:
            raise ValueError(f"c shape mismatch: expected {self._c.shape}, got {value.shape}")
        cp.asarray(self._c)[:] = value

    def set_A(self, value, check: bool = True):
        if check and value.shape != self._A.shape:
            raise ValueError(f"A shape mismatch: expected {self._A.shape}, got {value.shape}")
        cp.asarray(self._A)[:] = value

    def set_b(self, value, check: bool = True):
        if check and value.shape != self._b.shape:
            raise ValueError(f"b shape mismatch: expected {self._b.shape}, got {value.shape}")
        cp.asarray(self._b)[:] = value

    def set_G(self, value, check: bool = True):
        if check and value.shape != self._G.shape:
            raise ValueError(f"G shape mismatch: expected {self._G.shape}, got {value.shape}")
        cp.asarray(self._G)[:] = value

    def set_h_l(self, value, check: bool = True):
        if check and value.shape != self._h_l.shape:
            raise ValueError(f"h_l shape mismatch: expected {self._h_l.shape}, got {value.shape}")
        cp.asarray(self._h_l)[:] = value

    def set_h_u(self, value, check: bool = True):
        if check and value.shape != self._h_u.shape:
            raise ValueError(f"h_u shape mismatch: expected {self._h_u.shape}, got {value.shape}")
        cp.asarray(self._h_u)[:] = value

    def set_x_l(self, value, check: bool = True):
        if check and value.shape != self._x_l.shape:
            raise ValueError(f"x_l shape mismatch: expected {self._x_l.shape}, got {value.shape}")
        cp.asarray(self._x_l)[:] = value

    def set_x_u(self, value, check: bool = True):
        if check and value.shape != self._x_u.shape:
            raise ValueError(f"x_u shape mismatch: expected {self._x_u.shape}, got {value.shape}")
        cp.asarray(self._x_u)[:] = value

