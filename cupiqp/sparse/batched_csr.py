from typing import Optional, Sequence, Tuple, Union
import cupy as cp
from cupyx.scipy.sparse import csr_matrix



class BatchedCsrMatrix:
    """Batched storage of CSR matrices with the same sparsity."""
    def __init__(
        self,
        batch_size: int,
        indices: Sequence[int],
        indptr: Sequence[int],
        data: cp.ndarray,
        shape: Optional[Tuple[int, int]] = None,
        dtype=cp.float64,
    ):
        if not batch_size > 0:
            raise ValueError("batch_size must be a positive integer.")

        self._dtype = dtype
        data_cp = cp.asarray(data, dtype=dtype)
        try:
            if shape is None:
                self._template_matrix = csr_matrix((data_cp[0], indices, indptr))
            else:
                self._template_matrix = csr_matrix(
                    (data_cp[0], indices, indptr), shape=shape,
                )
        except Exception as e:
            raise ValueError(
                "Invalid indices, indptr, and data combination."
            ) from e

        self._batch_size = batch_size
        self._indices = self._template_matrix.indices
        self._indptr = self._template_matrix.indptr
        self._nnz = self._template_matrix.nnz

        if data_cp.shape != (batch_size, self._nnz):
            raise ValueError(
                f"data must have shape ({batch_size}, {self._nnz}), got {data_cp.shape}."
            )
        self.data = cp.empty((batch_size, self._nnz), dtype=dtype)
        self.data[:] = data_cp

    # TODO: consider not allocate self.data but directly point to the provided data. Also, change update_data() as a setter to point to new data
    def update_data(self, new_data: cp.ndarray) -> None:
        """Overwrite the internal values buffer in place.

        ``new_data`` must be a cupy array of shape ``(batch_size, nnz)``.
        Copies element-wise into ``self.data`` — preserves the buffer's
        device pointer, so any cuSPARSE descriptor built on it stays valid.
        """
        new_data_cp = cp.asarray(new_data, dtype=self._dtype)
        if new_data_cp.shape != (self._batch_size, self._nnz):
            raise ValueError(
                f"new_data must have shape ({self._batch_size}, {self._nnz}); "
                f"got {new_data_cp.shape}."
            )
        self.data[:] = new_data_cp

    @classmethod
    def from_torch_sparse_csr_tensor(cls, tensor, dtype=cp.float64) -> "BatchedCsrMatrix":
        """Build a ``BatchedCsrMatrix`` from a batched torch ``sparse_csr_tensor``.

        ``tensor`` must be 3-D with shape ``(B, M, N)``, in ``torch.sparse_csr``
        layout, residing on a CUDA device. Every batch is expected to share the
        same sparsity pattern; the shared ``indptr``/``indices`` are read from
        batch 0 (no cross-batch consistency check — trusted by convention), and
        the per-batch ``values`` (shape ``(B, nnz)``) become the ``_data`` buffer.

        Device-side data is moved zero-copy via DLPack.
        """
        import torch  # local import to avoid a hard dependency on torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected a torch.Tensor, got {type(tensor).__name__}."
            )
        if tensor.layout != torch.sparse_csr:
            raise ValueError(
                f"Expected torch.sparse_csr layout, got {tensor.layout}."
            )
        if tensor.dim() != 3:
            raise ValueError(
                "Expected a 3-D batched sparse CSR tensor of shape (B, M, N); "
                f"got shape {tuple(tensor.shape)}."
            )
        if not tensor.is_cuda:
            raise ValueError("tensor must reside on a CUDA device.")

        B = tensor.shape[0]
        crow = tensor.crow_indices()   # (B, M+1)
        col = tensor.col_indices()     # (B, nnz)
        values = tensor.values()       # (B, nnz)

        # Zero-copy views into torch device memory, then dtype-cast as needed.
        # Slicing crow[0] / col[0] picks the shared per-batch pattern; those
        # rows are contiguous in the underlying 2-D buffer so DLPack works
        # without a copy.
        indptr = cp.from_dlpack(crow[0].contiguous()).astype(cp.int32, copy=False)
        indices = cp.from_dlpack(col[0].contiguous()).astype(cp.int32, copy=False)
        data = cp.from_dlpack(values.contiguous())

        return cls(batch_size=B, indices=indices, indptr=indptr, data=data, dtype=dtype)

    @property
    def batch_size(self) -> int: return self._batch_size

    @property
    def nnz(self) -> int: return self._nnz

    @property
    def indices(self): return self._indices

    @property
    def indptr(self): return self._indptr

    @property
    def rows(self): return self._template_matrix.shape[0]

    @property
    def cols(self): return self._template_matrix.shape[1]

    @property
    def shape(self) -> Tuple[int]: 
        return (self.batch_size, self._template_matrix.shape[0], self._template_matrix.shape[1])

    def __getitem__(self, key: int) -> csr_matrix:
        return csr_matrix(
            (self.data[key], self._indices, self._indptr),
            shape=self._template_matrix.shape,
        )

    def diagonal(self) -> cp.ndarray:
        """Batched analogue of ``cupyx.scipy.sparse.csr_matrix.diagonal``.

        Returns a ``(batch_size, min(rows, cols))`` cupy float64 array whose
        ``[b, i]`` entry is the value at position ``(i, i)`` of the ``b``-th
        matrix, or ``0`` if that entry is not structurally stored.

        Fully vectorized — no Python loop over batches. The row/column
        mapping depends only on the shared sparsity pattern, so it is
        computed lazily on first call and cached for reuse.
        """
        var_idx, csr_idx = self._diag_index_map()
        k = min(self._template_matrix.shape)
        out = cp.zeros((self._batch_size, k), dtype=self.data.dtype)
        if var_idx.size > 0:
            out[:, var_idx] = self.data[:, csr_idx]
        return out

    def _diag_index_map(self):
        """Cached ``(var_idx, csr_idx)`` locating stored diagonal entries.

        ``var_idx[j]`` is the row/col index of the j-th stored diagonal;
        ``csr_idx[j]`` is its column in ``self.data``. Rows with no stored
        diagonal entry are simply absent from both arrays (``diagonal()``
        leaves those slots at 0). Computed on-device via ``cp.searchsorted``
        + a mask — no Python loop over rows.
        """
        cached = getattr(self, "_diag_idx_cache", None)
        if cached is not None:
            return cached
        if self._nnz == 0:
            empty = cp.empty(0, dtype=cp.int32)
            self._diag_idx_cache = (empty, empty)
            return self._diag_idx_cache
        idx_dtype = self._indptr.dtype
        # Row index of each CSR entry (indptr is non-decreasing).
        rows = cp.searchsorted(
            self._indptr[1:], cp.arange(self._nnz, dtype=idx_dtype),
            side='right',
        )
        is_diag = rows == self._indices
        csr_idx = cp.where(is_diag)[0].astype(cp.int32)
        var_idx = rows[csr_idx].astype(cp.int32)
        self._diag_idx_cache = (var_idx, csr_idx)
        return self._diag_idx_cache

    def __setitem__(self, key: int, value: Union[csr_matrix, cp.ndarray]):
        """Overwrite the ``key``-th matrix. ``value`` is either a csr_matrix
        sharing the template's sparsity, or a 1D cupy array of the nnz values."""
        if isinstance(value, csr_matrix):
            if value.shape != self._template_matrix.shape:
                raise ValueError(
                    f"Shape mismatch. Expected {self._template_matrix.shape}, "
                    f"got {value.shape}."
                )
            if value.nnz != self._nnz:
                raise ValueError(
                    f"nnz mismatch. Expected {self._nnz}, got {value.nnz}. "
                    "Input must share the template's sparsity pattern."
                )
            self.data[key] = value.data
        elif isinstance(value, cp.ndarray):
            if value.shape != (self._nnz,):
                raise ValueError(
                    f"Expected cupy.ndarray of shape ({self._nnz},), "
                    f"got {value.shape}."
                )
            self.data[key] = value
        else:
            raise TypeError(
                f"Unsupported value type {type(value).__name__}; expected "
                "cupyx.scipy.sparse.csr_matrix or cupy.ndarray."
            )
