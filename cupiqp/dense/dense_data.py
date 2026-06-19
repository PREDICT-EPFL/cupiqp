from typing import Any, Optional

import cupy as cp

from ..data import Data


# Type alias for every dense-input form the constructor accepts. ``cp.asarray``
# normalizes all of these: cupy arrays are returned as-is; CUDA torch tensors
# and JAX GPU arrays are adopted zero-copy via the ``__cuda_array_interface__``
# / DLPack protocols; CPU torch tensors, JAX CPU arrays, and numpy arrays are
# copied onto the current CUDA device.
DenseInput = Any


def _ensure_3d(array: cp.ndarray) -> cp.ndarray:
    """(rows, cols) -> (1, rows, cols); no-op if already 3-D."""
    if array.ndim == 2:
        return array.reshape(1, array.shape[0], array.shape[1])
    return array


def _ensure_2d(array: cp.ndarray) -> cp.ndarray:
    """(k,) -> (1, k); no-op if already 2-D."""
    if array.ndim == 1:
        return array.reshape(1, array.shape[0])
    return array


class DenseData(Data):
    """Dense data for one or more QPs with identical dimensions and bound structure.

    Every matrix/vector argument may be a cupy ndarray, a CUDA torch tensor,
    a JAX GPU array, a numpy array, or anything ``cupy.asarray`` understands.
    Inputs are normalized via ``_to_cupy`` — see its docstring for the zero-
    copy paths and the CPU-tensor error behavior.
    """
    def __init__(self, dtype=cp.float64, device: str = "cuda"):
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
        """Allocate and populate buffers from user inputs. Returns self for chaining."""

        # --- validate and store P, c ---
        P = _ensure_3d(cp.asarray(P))
        c = _ensure_2d(cp.asarray(c))

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

        self._P = P.astype(self._dtype, copy=True)
        self._c = c.astype(self._dtype, copy=True)

        # --- equality constraints ---
        if A is not None and b is not None:
            A = _ensure_3d(cp.asarray(A))
            b = _ensure_2d(cp.asarray(b))
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
            self._A = A.astype(self._dtype, copy=True)
            self._b = b.astype(self._dtype, copy=True)
        else:
            self._A = cp.zeros((B, 0, n), dtype=self._dtype)
            self._b = cp.zeros((B, 0), dtype=self._dtype)

        # --- inequality constraints ---
        if G is not None:
            G = _ensure_3d(cp.asarray(G))
            if G.ndim != 3:
                raise ValueError(f"G must have shape (B, m, n), got {G.shape}")
            if G.shape[0] != B or G.shape[2] != n:
                raise ValueError("Shape mismatch in G.")
            if h_l is None and h_u is None:
                raise ValueError("Either h_l or h_u must be provided when G is given.")
            m = G.shape[1]
            if h_l is not None:
                h_l = _ensure_2d(cp.asarray(h_l))
                if h_l.shape != (B, m):
                    raise ValueError(f"h_l must have shape ({B}, {m}), got {h_l.shape}")
            if h_u is not None:
                h_u = _ensure_2d(cp.asarray(h_u))
                if h_u.shape != (B, m):
                    raise ValueError(f"h_u must have shape ({B}, {m}), got {h_u.shape}")
            self._G = G.astype(self._dtype, copy=True)
        else:
            if h_u is not None or h_l is not None:
                raise ValueError("h_l and h_u must be None when G is None.")
            self._G = cp.zeros((B, 0, n), dtype=self._dtype)

        # Inequality-block presence is structural and fixed here: an omitted
        # side gets no storage (empty (B, 0)); a provided one is a full (B, m)
        # block. Both omitted is only valid when G is absent (handled above).
        self._has_h_l = h_l is not None
        self._has_h_u = h_u is not None
        self._h_u = self._as_batched_vec(h_u)
        self._h_l = self._as_batched_vec(h_l)

        # --- variable bounds ---
        if x_l is not None:
            x_l = _ensure_2d(cp.asarray(x_l))
        if x_u is not None:
            x_u = _ensure_2d(cp.asarray(x_u))
        if x_l is not None and x_u is not None:
            if x_l.shape != (B, n) or x_u.shape != (B, n):
                raise ValueError(f"x_l and x_u must have shape ({B}, {n}).")
        # Box-block presence is structural and fixed here: an omitted bound
        # gets no storage (empty (B, 0)); a provided one is a full (B, n) block.
        self._has_x_l = x_l is not None
        self._has_x_u = x_u is not None
        self._x_u = self._as_batched_vec(x_u)
        self._x_l = self._as_batched_vec(x_l)

        self._finalize()

    def _as_batched_vec(self, v: Optional[cp.ndarray]) -> cp.ndarray:
        if v is not None:
            return v.astype(self._dtype, copy=True)
        return cp.zeros((self._batch_size, 0), dtype=self._dtype)

    def extract_P_diag(self, diag_P: cp.ndarray):
        """Extract diagonal of each P into diag_P — shape (B, n)."""
        idx = cp.arange(self._n)
        diag_P[:] = self._P[:, idx, idx]

    # ------------------------------------------------------------------
    # In-place setters
    # ------------------------------------------------------------------

    def set_P(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._P.shape:
            raise ValueError(f"P shape mismatch: expected {self._P.shape}, got {value.shape}")
        self._P[:] = value

    def set_c(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._c.shape:
            raise ValueError(f"c shape mismatch: expected {self._c.shape}, got {value.shape}")
        self._c[:] = value

    def set_A(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._A.shape:
            raise ValueError(f"A shape mismatch: expected {self._A.shape}, got {value.shape}")
        self._A[:] = value

    def set_b(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._b.shape:
            raise ValueError(f"b shape mismatch: expected {self._b.shape}, got {value.shape}")
        self._b[:] = value

    def set_G(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._G.shape:
            raise ValueError(f"G shape mismatch: expected {self._G.shape}, got {value.shape}")
        self._G[:] = value

    def set_h_l(self, value: cp.ndarray, check: bool = True):
        if not self._has_h_l:
            raise ValueError(
                "Cannot set h_l: no lower-inequality block was provided at setup(). "
                "Adding an inequality block requires a new setup()."
            )
        if check and value.shape != self._h_l.shape:
            raise ValueError(f"h_l shape mismatch: expected {self._h_l.shape}, got {value.shape}")
        self._h_l[:] = value
        self._update_finite_bound_masks()

    def set_h_u(self, value: cp.ndarray, check: bool = True):
        if not self._has_h_u:
            raise ValueError(
                "Cannot set h_u: no upper-inequality block was provided at setup(). "
                "Adding an inequality block requires a new setup()."
            )
        if check and value.shape != self._h_u.shape:
            raise ValueError(f"h_u shape mismatch: expected {self._h_u.shape}, got {value.shape}")
        self._h_u[:] = value
        self._update_finite_bound_masks()

    def set_x_l(self, value: cp.ndarray, check: bool = True):
        if not self._has_x_l:
            raise ValueError(
                "Cannot set x_l: no lower box-bound block was provided at setup(). "
                "Adding a box-bound block requires a new setup()."
            )
        if check and value.shape != self._x_l.shape:
            raise ValueError(f"x_l shape mismatch: expected {self._x_l.shape}, got {value.shape}")
        self._x_l[:] = value
        self._update_finite_bound_masks()

    def set_x_u(self, value: cp.ndarray, check: bool = True):
        if not self._has_x_u:
            raise ValueError(
                "Cannot set x_u: no upper box-bound block was provided at setup(). "
                "Adding a box-bound block requires a new setup()."
            )
        if check and value.shape != self._x_u.shape:
            raise ValueError(f"x_u shape mismatch: expected {self._x_u.shape}, got {value.shape}")
        self._x_u[:] = value
        self._update_finite_bound_masks()

