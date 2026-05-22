from typing import Any, Optional
import cupy as cp
import warp as wp
from cupyx.scipy.sparse import csr_matrix

from ..data import Data, _to_warp
from ..typedef import PIQP_INF
from .batched_csr import BatchedCsrMatrix


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

    Two-phase construction (mirrors :class:`DenseData`):
        ``SparseData(dtype=wp.float64, device="cuda")`` stores config only;
        ``init(P, c, A, b, G, h_u, h_l, x_u, x_l)`` accepts the user inputs
        and copies them into fresh warp buffers (dense vectors) plus a
        :class:`BatchedCsrMatrix` (sparse matrices).

    Matrices ``P``, ``A``, ``G`` are stored internally as
    :class:`BatchedCsrMatrix` instances — one shared ``indptr``/``indices``
    pair plus a packed ``(B, nnz)`` values buffer. Callers may pass any of
    the following for each matrix:

    * :class:`BatchedCsrMatrix` (used as-is, zero extra packing)
    * ``torch.sparse_csr_tensor`` — 3-D ``(B, M, N)`` for batched,
      2-D ``(M, N)`` for single
    * ``list[cupy csr_matrix]`` sharing the same sparsity pattern
    * a single cupy ``csr_matrix`` (B = 1)

    Dense vectors (``c``, ``b``, ``h_l``, ``h_u``, ``x_l``, ``x_u``) carry
    a leading batch dimension ``(B, k)`` and are stored as warp arrays.
    Each must be a GPU-resident array exposing
    ``__cuda_array_interface__``; CPU inputs are rejected.
    """

    def __init__(self, dtype=wp.float64, device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)

    def init(
        self,
        P: SparseMatrixInput,
        c: Any,
        A: Optional[SparseMatrixInput] = None,
        b: Optional[Any] = None,
        G: Optional[SparseMatrixInput] = None,
        h_u: Optional[Any] = None,
        h_l: Optional[Any] = None,
        x_u: Optional[Any] = None,
        x_l: Optional[Any] = None,
    ):
        """Populate storage from user inputs."""

        dtype, device = self._dtype, self._device

        # -- P (determines B and n) -------------------------------------
        self._P = self._to_batched_csr(P, "P")
        B = self._P.batch_size
        if self._P.rows != self._P.cols:
            raise ValueError("P must be square.")
        n = self._P.rows
        self._batch_size = B
        self._n = n

        # -- c -----------------------------------------------------------
        self._c = self._to_batched_vec(c, B, n, "c", dtype, device)

        # -- A, b --------------------------------------------------------
        if A is not None and b is not None:
            self._A = self._to_batched_csr(A, "A")
            if self._A.batch_size != B:
                raise ValueError(
                    f"A batch size ({self._A.batch_size}) != P batch size ({B})"
                )
            if self._A.cols != n:
                raise ValueError(
                    f"A.cols ({self._A.cols}) != n ({n})"
                )
            self._b = self._to_batched_vec(b, B, self._A.rows, "b", dtype, device)
        else:
            self._A = self._empty_batched_csr(B, 0, n)
            self._b = wp.zeros((B, 0), dtype=dtype, device=device)

        # -- G, h_u, h_l ------------------------------------------------
        if G is not None:
            self._G = self._to_batched_csr(G, "G")
            if self._G.batch_size != B:
                raise ValueError(
                    f"G batch size ({self._G.batch_size}) != P batch size ({B})"
                )
            if self._G.cols != n:
                raise ValueError(
                    f"G.cols ({self._G.cols}) != n ({n})"
                )
        else:
            self._G = self._empty_batched_csr(B, 0, n)

        m = self._G.rows
        self._h_u = (self._to_batched_vec(h_u, B, m, "h_u", dtype, device) if h_u is not None
                    else wp.zeros((B, 0), dtype=dtype, device=device))
        self._h_l = (self._to_batched_vec(h_l, B, m, "h_l", dtype, device) if h_l is not None
                    else wp.zeros((B, 0), dtype=dtype, device=device))
        self._x_u = (self._to_batched_vec(x_u, B, n, "x_u", dtype, device) if x_u is not None
                    else wp.zeros((B, 0), dtype=dtype, device=device))
        self._x_l = (self._to_batched_vec(x_l, B, n, "x_l", dtype, device) if x_l is not None
                    else wp.zeros((B, 0), dtype=dtype, device=device))

        self._finalize()
        return self

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
    def _to_batched_csr(cls, mat: SparseMatrixInput, name: str) -> BatchedCsrMatrix:
        """Normalize any accepted matrix input form to ``BatchedCsrMatrix``."""
        # Already a BatchedCsrMatrix — reuse the shared sparsity pattern but
        # clone the values buffer so the solver (preconditioner) can mutate
        # without touching the caller's matrix.
        if isinstance(mat, BatchedCsrMatrix):
            return BatchedCsrMatrix(
                batch_size=mat.batch_size,
                indices=mat.indices,
                indptr=mat.indptr,
                data=mat.data,  # BatchedCsrMatrix.__init__ allocates + copies
                shape=(mat.rows, mat.cols),
            )

        # torch.sparse_csr_tensor (2-D single or 3-D batched).
        if _is_torch_sparse_csr(mat):
            return cls._from_torch_sparse_csr(mat, name)

        # List/tuple of cupy csr_matrix — stack per-batch data.
        if isinstance(mat, (list, tuple)):
            if len(mat) == 0:
                raise ValueError(f"{name} cannot be an empty list.")
            mats = [csr_matrix(m, dtype=cp.float64) for m in mat]
            tpl = mats[0]  # template matrix
            if tpl.nnz == 0:
                data = cp.empty((len(mats), 0), dtype=cp.float64)
            else:
                data = cp.stack([m.data for m in mats])
            return BatchedCsrMatrix(
                len(mats), tpl.indices, tpl.indptr, data, shape=tpl.shape,
            )

        # Single cupy csr_matrix (or convertible) — wrap as B = 1.
        single = csr_matrix(mat, dtype=cp.float64)
        if single.nnz == 0:
            data = cp.empty((1, 0), dtype=cp.float64)
        else:
            data = single.data.reshape(1, -1)
        return BatchedCsrMatrix(
            1, single.indices, single.indptr, data, shape=single.shape,
        )

    @staticmethod
    def _from_torch_sparse_csr(tensor, name: str) -> BatchedCsrMatrix:
        """Wrap a torch.sparse_csr_tensor into a BatchedCsrMatrix.

        Handles both 2-D (single) and 3-D (batched) tensors. In both
        cases the per-batch sparsity pattern is assumed shared (for 3-D
        the pattern is read from batch 0 only).
        """
        if tensor.dim() == 3:
            return BatchedCsrMatrix.from_torch_sparse_csr_tensor(tensor)
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
        data = values.reshape(1, -1) if values.size > 0 else cp.empty((1, 0), dtype=cp.float64)
        rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
        return BatchedCsrMatrix(1, indices, indptr, data, shape=(rows, cols))

    @staticmethod
    def _empty_batched_csr(B: int, rows: int, cols: int) -> BatchedCsrMatrix:
        """Placeholder for an omitted matrix block — (B, rows, cols), nnz = 0."""
        return BatchedCsrMatrix(
            batch_size=B,
            indices=cp.empty(0, dtype=cp.int32),
            indptr=cp.zeros(rows + 1, dtype=cp.int32),
            data=cp.empty((B, 0), dtype=cp.float64),
            shape=(rows, cols),
        )

    @staticmethod
    def _to_batched_vec(
        v: Any, B: int, k: int, name: str, dtype, device: str,
    ) -> wp.array:
        """Reshape, broadcast, and copy ``v`` into a fresh ``(B, k)`` warp
        array. Input must be GPU-resident (CAI-exposing)."""
        if not hasattr(v, '__cuda_array_interface__'):
            raise TypeError(
                f"{name} must be a GPU array exposing __cuda_array_interface__; "
                f"got {type(v).__name__}."
            )
        v = cp.asarray(v)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        if v.shape[0] == 1 and B > 1:
            v = cp.broadcast_to(v, (B, v.shape[1]))
        if v.shape[0] != B:
            raise ValueError(
                f"{name} batch size ({v.shape[0]}) != expected ({B})"
            )
        # _to_warp allocates a fresh warp buffer and copies the source in.
        return _to_warp(v, copy=True, dtype=dtype, device=device)

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
        h_l_cp = cp.asarray(self._h_l)
        h_u_cp = cp.asarray(self._h_u)
        # Bound structure is consistent across the batch — check batch 0
        free = (h_l_cp[0] <= -PIQP_INF) & (h_u_cp[0] >= PIQP_INF)
        if not bool(cp.any(free)):
            return
        # Zero the values for each free row across all batches (sparsity stays).
        indptr_host = cp.asnumpy(cp.asarray(self._G.indptr))
        free_idx = cp.asnumpy(cp.where(free)[0])
        g_data = cp.asarray(self._G.data)  # cupy view of (B, nnz) warp buffer
        for i in free_idx:
            start, end = int(indptr_host[i]), int(indptr_host[i + 1])
            if end > start:
                g_data[:, start:end] = 0.0
        h_l_cp[:, free] = -1.0
        h_u_cp[:, free] = 1.0

    # ------------------------------------------------------------------
    # In-place setters
    # ------------------------------------------------------------------

    def _set_matrix_values(self, target: BatchedCsrMatrix, value: SparseMatrixInput, check: bool, name: str):
        """Common helper for set_P / set_A / set_G."""
        new = self._to_batched_csr(value, name)
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
        target.update_data(new.data)

    def set_P(self, value: SparseMatrixInput, check: bool = True):
        self._set_matrix_values(self._P, value, check, "P")

    def set_c(self, value, check: bool = True):
        if check and value.shape != self._c.shape:
            raise ValueError(f"c shape mismatch: expected {self._c.shape}, got {value.shape}")
        cp.asarray(self._c)[:] = value

    def set_A(self, value: SparseMatrixInput, check: bool = True):
        self._set_matrix_values(self._A, value, check, "A")

    def set_b(self, value, check: bool = True):
        if check and value.shape != self._b.shape:
            raise ValueError(f"b shape mismatch: expected {self._b.shape}, got {value.shape}")
        cp.asarray(self._b)[:] = value

    def set_G(self, value: SparseMatrixInput, check: bool = True):
        self._set_matrix_values(self._G, value, check, "G")

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
