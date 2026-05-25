from typing import Any, Optional
import cupy as cp
from cupyx.scipy.sparse import csr_matrix

from ..data import Data
from ..typedef import PIQP_INF
from .batched_csr import UniformBatchedCsrMatrix


# Type alias for the accepted matrix input forms.
# - For batched (B > 1): BatchedCsrMatrix, 3-D torch.sparse_csr_tensor, or List[cupy csr_matrix].
# - For single (B = 1): cupy csr_matrix or 2-D torch.sparse_csr_tensor.
SparseMatrixInput = Any


def _is_torch_sparse_csr(obj) -> bool:
    """Duck-typed check for a torch.sparse_csr_tensor without importing torch
    at module load.  We only touch ``torch`` when needed."""
    if not (hasattr(obj, "layout") and hasattr(obj, "crow_indices")
            and hasattr(obj, "values")):
        return False
    try:
        import torch
    except ImportError:
        return False
    return isinstance(obj, torch.Tensor) and obj.layout == torch.sparse_csr


class SparseData(Data):
    """Sparse data structure for batched QP problems.

    Matrices ``P``, ``A``, ``G`` are stored internally as
    :class:`BatchedCsrMatrix` instances — one shared ``indptr``/``indices``
    pair plus a packed ``(B, nnz)`` values buffer. Callers may pass any of
    the following for each matrix:

    * :class:`BatchedCsrMatrix` (used as-is, zero extra packing)
    * ``torch.sparse_csr_tensor`` — 3-D ``(B, M, N)`` for batched,
      2-D ``(M, N)`` for single
    * ``list[cupy csr_matrix]`` sharing the same sparsity pattern
    * a single cupy ``csr_matrix`` (B = 1)

    Whatever the input, the normalized storage is a ``BatchedCsrMatrix``
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
        self._P = self._to_batched_csr(P, "P", dtype=dtype)
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
            self._A = self._to_batched_csr(A, "A", dtype=dtype)
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
            self._A = self._empty_batched_csr(B, 0, n, dtype=dtype)
            self._b = cp.zeros((B, 0), dtype=dtype)

        # -- G, h_u, h_l ------------------------------------------------
        if G is not None:
            if h_l is None and h_u is None:
                raise ValueError("Either h_l or h_u must be provided when G is given.")
            self._G = self._to_batched_csr(G, "G", dtype=dtype)
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
            self._G = self._empty_batched_csr(B, 0, n, dtype=dtype)

        m = self._G.rows
        self._h_u = self._to_batched_vec(h_u, B, m, "h_u", dtype=dtype) if h_u is not None else cp.zeros((B, 0), dtype=dtype)
        self._h_l = self._to_batched_vec(h_l, B, m, "h_l", dtype=dtype) if h_l is not None else cp.zeros((B, 0), dtype=dtype)
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
    # Input normalization
    # ------------------------------------------------------------------

    @classmethod
    def _to_batched_csr(cls, mat: SparseMatrixInput, name: str, dtype=cp.float64) -> UniformBatchedCsrMatrix:
        """Normalize any accepted matrix input form to ``BatchedCsrMatrix``."""
        # Already a BatchedCsrMatrix — reuse the shared sparsity pattern but
        # clone the values buffer so the solver (preconditioner) can mutate
        # without touching the caller's matrix.
        if isinstance(mat, UniformBatchedCsrMatrix):
            return UniformBatchedCsrMatrix(
                batch_size=mat.batch_size,
                indices=mat.indices,
                indptr=mat.indptr,
                data=mat.data,  # BatchedCsrMatrix.__init__ allocates + copies
                shape=(mat.rows, mat.cols),
                dtype=dtype,
            )

        # torch.sparse_csr_tensor (2-D single or 3-D batched).
        if _is_torch_sparse_csr(mat):
            return cls._from_torch_sparse_csr(mat, name, dtype=dtype)

        # List/tuple of cupy csr_matrix — stack per-batch data.
        if isinstance(mat, (list, tuple)):
            if len(mat) == 0:
                raise ValueError(f"{name} cannot be an empty list.")
            mats = [csr_matrix(m, dtype=dtype) for m in mat]
            tpl = mats[0]  # template matrix
            if tpl.nnz == 0:
                data = cp.empty((len(mats), 0), dtype=dtype)
            else:
                data = cp.stack([m.data for m in mats])
            return UniformBatchedCsrMatrix(
                len(mats), tpl.indices, tpl.indptr, data, shape=tpl.shape,
                dtype=dtype,
            )

        # Single cupy csr_matrix (or convertible) — wrap as B = 1.
        single = csr_matrix(mat, dtype=dtype)
        if single.nnz == 0:
            data = cp.empty((1, 0), dtype=dtype)
        else:
            data = single.data.reshape(1, -1)
        return UniformBatchedCsrMatrix(
            1, single.indices, single.indptr, data, shape=single.shape,
            dtype=dtype,
        )

    @staticmethod
    def _from_torch_sparse_csr(tensor, name: str, dtype=cp.float64) -> UniformBatchedCsrMatrix:
        """Wrap a torch.sparse_csr_tensor into a BatchedCsrMatrix.

        Handles both 2-D (single) and 3-D (batched) tensors. In both
        cases the per-batch sparsity pattern is assumed shared (for 3-D
        the pattern is read from batch 0 only).
        """
        if tensor.dim() == 3:
            return UniformBatchedCsrMatrix.from_torch_sparse_csr_tensor(tensor, dtype=dtype)
        if tensor.dim() != 2:
            raise ValueError(
                f"{name} torch.sparse_csr_tensor must be 2-D or 3-D; "
                f"got shape {tuple(tensor.shape)}."
            )
        if not tensor.is_cuda:
            raise ValueError(f"{name} tensor must reside on a CUDA device.")
        indptr = cp.from_dlpack(tensor.crow_indices().contiguous()).astype(cp.int32, copy=False)
        indices = cp.from_dlpack(tensor.col_indices().contiguous()).astype(cp.int32, copy=False)
        values = cp.from_dlpack(tensor.values().contiguous())
        data = values.reshape(1, -1) if values.size > 0 else cp.empty((1, 0), dtype=dtype)
        rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
        return UniformBatchedCsrMatrix(1, indices, indptr, data, shape=(rows, cols), dtype=dtype)

    @staticmethod
    def _empty_batched_csr(B: int, rows: int, cols: int, dtype=cp.float64) -> UniformBatchedCsrMatrix:
        """Placeholder for an omitted matrix block — (B, rows, cols), nnz = 0."""
        return UniformBatchedCsrMatrix(
            batch_size=B,
            indices=cp.empty(0, dtype=cp.int32),
            indptr=cp.zeros(rows + 1, dtype=cp.int32),
            data=cp.empty((B, 0), dtype=dtype),
            shape=(rows, cols),
            dtype=dtype,
        )

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

    def _disable_inf_constraints(self):
        """Zero G-rows where both h_l and h_u are infinite (shared sparsity)."""
        m = self.m
        if m == 0:
            return
        # Bound structure is consistent across the batch — check batch 0
        free = (self._h_l[0] <= -PIQP_INF) & (self._h_u[0] >= PIQP_INF)
        if not bool(cp.any(free)):
            return
        # Zero the values for each free row across all batches (sparsity stays).
        indptr_host = cp.asnumpy(self._G.indptr)
        free_idx = cp.asnumpy(cp.where(free)[0])
        g_data = self._G.data  # (B, nnz) view
        for i in free_idx:
            start, end = int(indptr_host[i]), int(indptr_host[i + 1])
            if end > start:
                g_data[:, start:end] = 0.0
        self._h_l[:, free] = -1.0
        self._h_u[:, free] = 1.0

    # ------------------------------------------------------------------
    # In-place setters
    # ------------------------------------------------------------------

    @staticmethod
    def _same_sparsity_pattern(
        target: UniformBatchedCsrMatrix,
        new: UniformBatchedCsrMatrix,
        value: SparseMatrixInput,
    ) -> bool:
        """Compare CSR structure values, including supplied batch members."""
        same = (
            cp.array_equal(new.indices, target.indices)
            & cp.array_equal(new.indptr, target.indptr)
        )
        if isinstance(value, (list, tuple)):
            for mat in value[1:]:
                same &= (
                    cp.array_equal(mat.indices, target.indices)
                    & cp.array_equal(mat.indptr, target.indptr)
                )
        elif _is_torch_sparse_csr(value) and value.dim() == 3:
            indices = cp.from_dlpack(value.col_indices().contiguous()).astype(
                target.indices.dtype, copy=False
            )
            indptr = cp.from_dlpack(value.crow_indices().contiguous()).astype(
                target.indptr.dtype, copy=False
            )
            same &= (
                cp.array_equal(indices, cp.broadcast_to(target.indices, indices.shape))
                & cp.array_equal(indptr, cp.broadcast_to(target.indptr, indptr.shape))
            )
        return bool(same)

    def _set_matrix_values(
        self,
        target: UniformBatchedCsrMatrix,
        value: SparseMatrixInput,
        check: bool,
        name: str,
    ):
        """Common helper for set_P / set_A / set_G.

        When ``check`` is true, compare actual CSR ``indices`` / ``indptr``
        values. Resolving that device-side comparison to a Python decision
        synchronizes the update path. When ``check`` is false, the caller
        must preserve the existing sparsity pattern.
        """
        new = self._to_batched_csr(value, name, dtype=self._dtype)
        if check:
            if new.batch_size != target.batch_size:
                raise ValueError(
                    f"{name} batch size mismatch: expected {target.batch_size}, "
                    f"got {new.batch_size}"
                )
            if new.nnz != target.nnz:
                raise ValueError(
                    f"{name} nnz mismatch: expected {target.nnz}, got {new.nnz} "
                    "(sparsity pattern must be preserved)."
                )
            if new.rows != target.rows or new.cols != target.cols:
                raise ValueError(
                    f"{name} shape mismatch: expected ({target.rows}, {target.cols}), "
                    f"got ({new.rows}, {new.cols})"
                )
            if not self._same_sparsity_pattern(target, new, value):
                raise ValueError(
                    f"{name} sparsity pattern differs from setup(): "
                    "indices/indptr values must be unchanged."
                )
        target.update_data(new.data)

    def set_P(self, value: SparseMatrixInput, check: bool = True):
        self._set_matrix_values(self._P, value, check, "P")

    def set_c(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._c.shape:
            raise ValueError(f"c shape mismatch: expected {self._c.shape}, got {value.shape}")
        self._c[:] = value

    def set_A(self, value: SparseMatrixInput, check: bool = True):
        self._set_matrix_values(self._A, value, check, "A")

    def set_b(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._b.shape:
            raise ValueError(f"b shape mismatch: expected {self._b.shape}, got {value.shape}")
        self._b[:] = value

    def set_G(self, value: SparseMatrixInput, check: bool = True):
        self._set_matrix_values(self._G, value, check, "G")

    def set_h_l(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._h_l.shape:
            raise ValueError(f"h_l shape mismatch: expected {self._h_l.shape}, got {value.shape}")
        self._h_l[:] = value

    def set_h_u(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._h_u.shape:
            raise ValueError(f"h_u shape mismatch: expected {self._h_u.shape}, got {value.shape}")
        self._h_u[:] = value

    def set_x_l(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._x_l.shape:
            raise ValueError(f"x_l shape mismatch: expected {self._x_l.shape}, got {value.shape}")
        self._x_l[:] = value

    def set_x_u(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._x_u.shape:
            raise ValueError(f"x_u shape mismatch: expected {self._x_u.shape}, got {value.shape}")
        self._x_u[:] = value
