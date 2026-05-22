from typing import Any, Optional, Tuple, Union
import cupy as cp
import warp as wp
from cupyx.scipy.sparse import csr_matrix

from ..data import _to_warp


class BatchedCsrMatrix:
    """Batched storage of CSR matrices with the same sparsity pattern.

    Storage is warp-native: ``_indices``, ``_indptr`` are warp ``int32``
    arrays; ``data`` is a warp ``(B, nnz)`` array of the configured dtype
    on the configured device. The sparsity pattern is shared across the
    batch.

    Inputs to ``__init__`` may be any CAI-exposing GPU arrays (cupy,
    warp, CUDA torch, JAX GPU); each is copied into a freshly allocated
    warp buffer.
    """
    def __init__(
        self,
        batch_size: int,
        indices: Any,
        indptr: Any,
        data: Any,
        shape: Optional[Tuple[int, int]] = None,
        dtype=wp.float64,
        device: str = "cuda",
    ):
        if not batch_size > 0:
            raise ValueError("batch_size must be a positive integer.")

        self._batch_size = batch_size
        self._dtype = dtype
        self._device = device

        # Index buffers — kept as int32 warp arrays.
        self._indices = _to_warp(indices, copy=True, dtype=wp.int32, device=device)
        self._indptr = _to_warp(indptr, copy=True, dtype=wp.int32, device=device)

        rows = int(self._indptr.shape[0]) - 1
        nnz = int(self._indices.shape[0])
        self._nnz = nnz

        # Cupy view of the warp index buffers — used for shape inference
        # and for cupy-side fancy indexing (e.g. ``cp.searchsorted``).
        indices_cp = cp.asarray(self._indices)
        indptr_cp = cp.asarray(self._indptr)

        if shape is None:
            cols = int(indices_cp.max()) + 1 if nnz > 0 else rows
            shape = (rows, cols)
        else:
            if shape[0] != rows:
                raise ValueError(
                    f"shape[0] ({shape[0]}) != len(indptr)-1 ({rows})"
                )
        self._shape = (int(shape[0]), int(shape[1]))

        # Values buffer — warp (B, nnz) of configured dtype.
        if hasattr(data, "shape") and tuple(data.shape) != (batch_size, nnz):
            raise ValueError(
                f"data must have shape ({batch_size}, {nnz}), got {tuple(data.shape)}."
            )
        self.data = _to_warp(data, copy=True, dtype=dtype, device=device)

    def update_data(self, new_data: Any) -> None:
        """Overwrite the internal values buffer in place.

        ``new_data`` must be a GPU array of shape ``(batch_size, nnz)``
        exposing ``__cuda_array_interface__``. Copies element-wise into
        ``self.data`` — preserves the buffer's device pointer, so any
        cuSPARSE/cuDSS descriptor built on it stays valid.
        """
        if tuple(new_data.shape) != (self._batch_size, self._nnz):
            raise ValueError(
                f"new_data must have shape ({self._batch_size}, {self._nnz}); "
                f"got {tuple(new_data.shape)}."
            )
        cp.asarray(self.data)[:] = cp.asarray(new_data)

    @classmethod
    def from_torch_sparse_csr_tensor(cls, tensor,
                                     dtype=wp.float64,
                                     device: str = "cuda") -> "BatchedCsrMatrix":
        """Build a ``BatchedCsrMatrix`` from a batched torch ``sparse_csr_tensor``.

        ``tensor`` must be 3-D with shape ``(B, M, N)``, in ``torch.sparse_csr``
        layout, residing on a CUDA device. Every batch is expected to share the
        same sparsity pattern; the shared ``indptr``/``indices`` are read from
        batch 0 (no cross-batch consistency check — trusted by convention), and
        the per-batch ``values`` (shape ``(B, nnz)``) become the values buffer.

        Device-side data is moved zero-copy via DLPack before being copied into
        the configured warp buffers.
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

        indptr = cp.from_dlpack(crow[0].contiguous())
        indices = cp.from_dlpack(col[0].contiguous())
        data = cp.from_dlpack(values.contiguous())

        return cls(batch_size=B, indices=indices, indptr=indptr, data=data,
                   dtype=dtype, device=device)

    @property
    def batch_size(self) -> int: return self._batch_size

    @property
    def nnz(self) -> int: return self._nnz

    @property
    def indices(self): return self._indices

    @property
    def indptr(self): return self._indptr

    @property
    def rows(self): return self._shape[0]

    @property
    def cols(self): return self._shape[1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._batch_size, self._shape[0], self._shape[1])

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key: int) -> csr_matrix:
        """Return a cupyx ``csr_matrix`` view onto batch ``key``.

        The returned view aliases the underlying warp ``data`` buffer via
        ``__cuda_array_interface__`` — in-place updates to the warp buffer
        are visible to the view, and vice versa.
        """
        data_cp = cp.asarray(self.data)[key]
        indices_cp = cp.asarray(self._indices)
        indptr_cp = cp.asarray(self._indptr)
        return csr_matrix(
            (data_cp, indices_cp, indptr_cp),
            shape=self._shape,
        )

    def diagonal(self) -> cp.ndarray:
        """Batched analogue of ``cupyx.scipy.sparse.csr_matrix.diagonal``.

        Returns a ``(batch_size, min(rows, cols))`` cupy array whose
        ``[b, i]`` entry is the value at position ``(i, i)`` of the ``b``-th
        matrix, or ``0`` if that entry is not structurally stored.

        Fully vectorized — no Python loop over batches. The row/column
        mapping depends only on the shared sparsity pattern, so it is
        computed lazily on first call and cached for reuse.
        """
        var_idx, csr_idx = self._diag_index_map()
        k = min(self._shape)
        data_cp = cp.asarray(self.data)
        out = cp.zeros((self._batch_size, k), dtype=data_cp.dtype)
        if var_idx.size > 0:
            out[:, var_idx] = data_cp[:, csr_idx]
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
        indptr_cp = cp.asarray(self._indptr)
        indices_cp = cp.asarray(self._indices)
        idx_dtype = indptr_cp.dtype
        # Row index of each CSR entry (indptr is non-decreasing).
        rows = cp.searchsorted(
            indptr_cp[1:], cp.arange(self._nnz, dtype=idx_dtype),
            side='right',
        )
        is_diag = rows == indices_cp
        csr_idx = cp.where(is_diag)[0].astype(cp.int32)
        var_idx = rows[csr_idx].astype(cp.int32)
        self._diag_idx_cache = (var_idx, csr_idx)
        return self._diag_idx_cache

    def __setitem__(self, key: int, value: Union[csr_matrix, cp.ndarray, wp.array]):
        """Overwrite the ``key``-th matrix. ``value`` is either a csr_matrix
        sharing the sparsity, or a 1-D array of length ``nnz``."""
        data_cp = cp.asarray(self.data)
        if isinstance(value, csr_matrix):
            if value.shape != self._shape:
                raise ValueError(
                    f"Shape mismatch. Expected {self._shape}, "
                    f"got {value.shape}."
                )
            if value.nnz != self._nnz:
                raise ValueError(
                    f"nnz mismatch. Expected {self._nnz}, got {value.nnz}. "
                    "Input must share the sparsity pattern."
                )
            data_cp[key] = value.data
        elif hasattr(value, "shape"):
            if tuple(value.shape) != (self._nnz,):
                raise ValueError(
                    f"Expected array of shape ({self._nnz},), "
                    f"got {tuple(value.shape)}."
                )
            data_cp[key] = cp.asarray(value)
        else:
            raise TypeError(
                f"Unsupported value type {type(value).__name__}; expected "
                "cupyx.scipy.sparse.csr_matrix or array-like."
            )
