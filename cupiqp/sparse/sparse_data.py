from typing import Any, Optional
import cupy as cp

from ..data import Data
from ..utils import is_cuda_array
from .batched_csr import UniformBatchedCsrMatrix


# Type alias for the accepted matrix input forms.
# - For batched (B > 1): UniformBatchedCsrMatrix, 3-D torch.sparse_csr_tensor, or List[cupy csr_matrix].
# - For single (B = 1): cupy csr_matrix or 2-D torch.sparse_csr_tensor.
SparseMatrixInput = Any


class SparseData(Data):
    """Sparse data structure for batched QP problems.

    Matrices ``P``, ``A``, ``G`` are stored internally as
    :class:`UniformBatchedCsrMatrix` instances — one shared ``indptr``/``indices``
    pair plus a packed ``(B, nnz)`` values buffer. Callers may pass any of
    the following for each matrix:

    * :class:`UniformBatchedCsrMatrix` (already has a uniform structure)
    * ``torch.sparse_csr_tensor`` — 3-D ``(B, M, N)`` for batched,
      2-D ``(M, N)`` for single
    * ``list[cupy csr_matrix]`` sharing the same sparsity pattern
    * a single cupy ``csr_matrix`` (B = 1)

    Whatever the input, the normalized storage is a ``UniformBatchedCsrMatrix``
    accessible via ``self.P`` / ``self.A`` / ``self.G``. Dense vectors
    (``c``, ``b``, ``h_l``, ``h_u``, ``x_l``, ``x_u``) carry a leading
    batch dimension ``(B, k)``.
    """

    def __init__(self, dtype=cp.float64, device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)

    def init(
        self,
        P: SparseMatrixInput,
        c: cp.ndarray,
        A: Optional[SparseMatrixInput] = None,
        b: Optional[cp.ndarray] = None,
        G: Optional[SparseMatrixInput] = None,
        h_u: Optional[cp.ndarray] = None,
        h_l: Optional[cp.ndarray] = None,
        x_u: Optional[cp.ndarray] = None,
        x_l: Optional[cp.ndarray] = None,
    ):
        dtype = self._dtype

        # -- P (determines B and n) -------------------------------------
        self._P = UniformBatchedCsrMatrix.from_input(
            P, dtype=dtype, validate_shared_sparsity=True,
        )
        B = self._P.batch_size
        if self._P.rows != self._P.cols:
            raise ValueError("P must be square.")
        n = self._P.rows
        self._batch_size = B
        self._n = n

        # -- c -----------------------------------------------------------
        self._c = self._to_batched_vec(c, B, n, "c", dtype=dtype)

        # -- A, b --------------------------------------------------------
        if (A is None) != (b is None):
            raise ValueError("A and b must either both be provided or both be None.")
        if A is not None and b is not None:
            self._A = UniformBatchedCsrMatrix.from_input(
                A, dtype=dtype, validate_shared_sparsity=True,
            )
            if self._A.batch_size != B:
                raise ValueError(
                    f"A batch size ({self._A.batch_size}) != P batch size ({B})"
                )
            if self._A.cols != n:
                raise ValueError(
                    f"A.cols ({self._A.cols}) != n ({n})"
                )
            self._b = self._to_batched_vec(b, B, self._A.rows, "b", dtype=dtype)
        else:
            self._A = UniformBatchedCsrMatrix.empty(B, 0, n, dtype=dtype)
            self._b = cp.zeros((B, 0), dtype=dtype)

        # -- G, h_u, h_l ------------------------------------------------
        if G is not None:
            if h_l is None and h_u is None:
                raise ValueError("Either h_l or h_u must be provided when G is given.")
            self._G = UniformBatchedCsrMatrix.from_input(
                G, dtype=dtype, validate_shared_sparsity=True,
            )
            if self._G.batch_size != B:
                raise ValueError(
                    f"G batch size ({self._G.batch_size}) != P batch size ({B})"
                )
            if self._G.cols != n:
                raise ValueError(
                    f"G.cols ({self._G.cols}) != n ({n})"
                )
        else:
            if h_u is not None or h_l is not None:
                raise ValueError("h_l and h_u must be None when G is None.")
            self._G = UniformBatchedCsrMatrix.empty(B, 0, n, dtype=dtype)

        m = self._G.rows
        self._has_h_l = h_l is not None
        self._has_h_u = h_u is not None
        self._h_u = self._to_batched_vec(h_u, B, m, "h_u", dtype=dtype) if h_u is not None else cp.zeros((B, 0), dtype=dtype)
        self._h_l = self._to_batched_vec(h_l, B, m, "h_l", dtype=dtype) if h_l is not None else cp.zeros((B, 0), dtype=dtype)

        self._has_x_l = x_l is not None
        self._has_x_u = x_u is not None
        self._x_u = self._to_batched_vec(x_u, B, n, "x_u", dtype=dtype) if x_u is not None else cp.zeros((B, 0), dtype=dtype)
        self._x_l = self._to_batched_vec(x_l, B, n, "x_l", dtype=dtype) if x_l is not None else cp.zeros((B, 0), dtype=dtype)

        self._finalize()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def p(self) -> int:
        return self._A.rows

    @property
    def m(self) -> int:
        return self._G.rows

    # ------------------------------------------------------------------
    # Vector normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _to_batched_vec(
        v: cp.ndarray, B: int, k: int, name: str, dtype=cp.float64,
    ) -> cp.ndarray:
        """Ensure *v* has shape ``(B, k)``."""
        v = cp.asarray(v, dtype=dtype)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        elif v.ndim != 2:
            raise ValueError(
                f"{name} must have shape ({B}, {k}) or ({k},), got {v.shape}"
            )
        if v.shape[1] != k:
            raise ValueError(
                f"{name} width mismatch: expected {k}, got {v.shape[1]}"
            )
        if v.shape[0] == 1 and B > 1:
            # broadcast_to + .copy() already produces an owned buffer.
            return cp.broadcast_to(v, (B, k)).copy()
        if v.shape[0] != B:
            raise ValueError(
                f"{name} batch size ({v.shape[0]}) != expected ({B})"
            )
        # Copy so the solver owns the buffer (preconditioner mutates in place).
        return v.copy()

    # ------------------------------------------------------------------
    # Overrides for batched CSR storage
    # ------------------------------------------------------------------

    def extract_P_diag(self, out: cp.ndarray):
        """Extract diagonal of each P into *out* — shape ``(B, n)``.
        """
        out[:] = self._P.diagonal()

    # ------------------------------------------------------------------
    # In-place setters
    # ------------------------------------------------------------------

    def _set_matrix_values(
        self,
        target: UniformBatchedCsrMatrix,
        value,
        name: str,
    ):
        """Common helper for set_P / set_A / set_G.

        ``value`` holds only the nonzero values, laid out against the CSR
        pattern fixed at setup(): a dense GPU array of shape ``(B, nnz)``, or
        ``(nnz,)`` to broadcast one set of values over the whole batch. The
        pattern itself cannot change, so no structural comparison (and no
        device-to-host sync) is needed. The shape check only reads host-side
        metadata and is therefore always performed.
        """
        if not is_cuda_array(value):
            raise TypeError(
                f"{name} must be a dense GPU array holding the nonzero values, "
                f"of shape ({target.batch_size}, {target.nnz}) or "
                f"({target.nnz},), in the CSR order of the matrix passed to "
                f"setup(); got {type(value).__name__}. The sparsity pattern is "
                "fixed at setup() and cannot be updated."
            )
        values = cp.asarray(value)
        if values.shape not in ((target.batch_size, target.nnz), (target.nnz,)):
            raise ValueError(
                f"{name} values shape mismatch: expected "
                f"({target.batch_size}, {target.nnz}) or ({target.nnz},), "
                f"got {values.shape}."
            )
        # In-place write keeps the buffer address stable for cuDSS / cuSPARSE.
        target.data[...] = values

    def set_P(self, value: cp.ndarray, check: bool = True):
        self._set_matrix_values(self._P, value, "P")

    def set_c(self, value: cp.ndarray, check: bool = True):
        if check and value.shape not in (self._c.shape, self._c.shape[1:]):
            raise ValueError(f"c shape mismatch: expected {self._c.shape} or {self._c.shape[1:]}, got {value.shape}")
        self._c[:] = self._as_gpu_array(value, "c")

    def set_A(self, value: cp.ndarray, check: bool = True):
        self._set_matrix_values(self._A, value, "A")

    def set_b(self, value: cp.ndarray, check: bool = True):
        if check and value.shape not in (self._b.shape, self._b.shape[1:]):
            raise ValueError(f"b shape mismatch: expected {self._b.shape} or {self._b.shape[1:]}, got {value.shape}")
        self._b[:] = self._as_gpu_array(value, "b")

    def set_G(self, value: cp.ndarray, check: bool = True):
        self._set_matrix_values(self._G, value, "G")

    def set_h_l(self, value: cp.ndarray, check: bool = True):
        if not self._has_h_l:
            raise ValueError(
                "Cannot set h_l: no lower-inequality block was provided at setup(). "
                "Adding an inequality block requires a new setup()."
            )
        if check and value.shape not in (self._h_l.shape, self._h_l.shape[1:]):
            raise ValueError(f"h_l shape mismatch: expected {self._h_l.shape} or {self._h_l.shape[1:]}, got {value.shape}")
        self._h_l[:] = self._as_gpu_array(value, "h_l")
        self._update_finite_bound_masks()

    def set_h_u(self, value: cp.ndarray, check: bool = True):
        if not self._has_h_u:
            raise ValueError(
                "Cannot set h_u: no upper-inequality block was provided at setup(). "
                "Adding an inequality block requires a new setup()."
            )
        if check and value.shape not in (self._h_u.shape, self._h_u.shape[1:]):
            raise ValueError(f"h_u shape mismatch: expected {self._h_u.shape} or {self._h_u.shape[1:]}, got {value.shape}")
        self._h_u[:] = self._as_gpu_array(value, "h_u")
        self._update_finite_bound_masks()

    def set_x_l(self, value: cp.ndarray, check: bool = True):
        if not self._has_x_l:
            raise ValueError(
                "Cannot set x_l: no lower box-bound block was provided at setup(). "
                "Adding a box-bound block requires a new setup()."
            )
        if check and value.shape not in (self._x_l.shape, self._x_l.shape[1:]):
            raise ValueError(f"x_l shape mismatch: expected {self._x_l.shape} or {self._x_l.shape[1:]}, got {value.shape}")
        self._x_l[:] = self._as_gpu_array(value, "x_l")
        self._update_finite_bound_masks()

    def set_x_u(self, value: cp.ndarray, check: bool = True):
        if not self._has_x_u:
            raise ValueError(
                "Cannot set x_u: no upper box-bound block was provided at setup(). "
                "Adding a box-bound block requires a new setup()."
            )
        if check and value.shape not in (self._x_u.shape, self._x_u.shape[1:]):
            raise ValueError(f"x_u shape mismatch: expected {self._x_u.shape} or {self._x_u.shape[1:]}, got {value.shape}")
        self._x_u[:] = self._as_gpu_array(value, "x_u")
        self._update_finite_bound_masks()
