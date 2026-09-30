from typing import Optional, Sequence, Tuple

import cupy as cp
import numpy as np
import warp as wp
from cupyx.scipy.sparse import csr_matrix

from ..utils import to_warp_dtype


class UniformBatchedCsrMatrix:
    """A batch of CSR matrices that share one sparsity pattern.

    A batched extension of ``cupyx.scipy.sparse.csr_matrix``: where a single
    CSR matrix stores ``indptr``, ``indices`` and a 1-D ``data`` array of
    length ``nnz``, this type stores **one shared** ``indptr`` / ``indices``
    pair plus a **2-D** ``data`` buffer of shape ``(batch_size, nnz)`` - the
    values of all matrices stacked along a leading batch axis. Every matrix
    in the batch therefore has the same nonzero structure and differs only
    in its values.

    The storage is Warp arrays owned by this object: ``indptr`` and
    ``indices`` are ``int32`` and ``data`` has the value dtype. cuSPARSE and
    cuDSS read them through their device pointers and the solver's Warp
    kernels index them directly. cupy is used only while constructing the
    pattern; ``mat[b]`` returns a zero-copy ``csr_matrix`` view of one matrix
    for pattern algebra at setup time.

    Parameters
    ----------
    batch_size : int
        Number of matrices in the batch, ``B`` (must be positive).
    indptr, indices : int array
        The shared CSR row-pointer and column-index arrays - the single
        sparsity pattern used by every matrix in the batch. Any host or GPU
        integer array; copied into ``int32`` Warp arrays.
    data : GPU array
        Values of shape ``(batch_size, nnz)``, row ``i`` holding the nonzeros
        of the ``i``-th matrix in the order of ``indices``, or ``(nnz,)`` for
        the same values in every matrix. Copied into the owned buffer.
    shape : tuple of (int, int)
        Dense ``(rows, cols)`` shape of each matrix.
    dtype : Warp scalar type, default: ``wp.float64``
        Value dtype (``wp.float32`` or ``wp.float64``).
    device : str, default: ``"cuda"``
        Warp device of the storage.

    Attributes
    ----------
    batch_size, nnz, rows, cols, shape :
        Batch size ``B``, nonzeros per matrix, and the shared dense shape.
    indptr, indices : warp.array of int32
        The shared CSR sparsity pattern.
    data : warp.array2d
        The ``(batch_size, nnz)`` values buffer.
    """
    def __init__(
        self,
        batch_size: int,
        indptr,
        indices,
        data,
        shape: Tuple[int, int],
        dtype=wp.float64,
        device: str = "cuda"
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        batch_size = int(batch_size)
        rows, cols = int(shape[0]), int(shape[1])
        dtype = to_warp_dtype(dtype)

        # The pattern is validated and normalized with cupy (setup time only).
        indptr_cp = cp.asarray(indptr, dtype=cp.int32).ravel()
        indices_cp = cp.asarray(indices, dtype=cp.int32).ravel()
        nnz = int(indices_cp.size)
        if int(indptr_cp.size) != rows + 1:
            raise ValueError(
                f"indptr must have length rows + 1 = {rows + 1}; got {int(indptr_cp.size)}."
            )
        data_cp = cp.asarray(data, dtype=wp.dtype_to_numpy(dtype))
        if data_cp.shape == (nnz,):
            data_cp = cp.broadcast_to(data_cp, (batch_size, nnz))
        if data_cp.shape != (batch_size, nnz):
            raise ValueError(
                f"data must have shape ({batch_size}, {nnz}) or ({nnz},), got {data_cp.shape}."
            )

        self._batch_size = batch_size
        self._nnz = nnz
        self._rows = rows
        self._cols = cols
        self._dtype = dtype
        self._device = device
        self._indptr = to_wp_int32(indptr_cp, device)
        self._indices = to_wp_int32(indices_cp, device)
        self._data = wp.empty((batch_size, nnz), dtype=dtype, device=device)
        if nnz > 0:
            wp.copy(self._data, wp.array(data_cp, dtype=dtype, copy=False))
        self._row_indices: Optional[wp.array] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def nnz(self) -> int:
        return self._nnz

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._batch_size, self._rows, self._cols)

    @property
    def dtype(self):
        """Value dtype, a Warp scalar type."""
        return self._dtype

    @property
    def device(self) -> str:
        return self._device

    @property
    def indptr(self) -> wp.array:
        """Shared CSR row pointers, ``(rows + 1,)`` int32."""
        return self._indptr

    @property
    def indices(self) -> wp.array:
        """Shared CSR column indices, ``(nnz,)`` int32."""
        return self._indices

    @property
    def data(self) -> wp.array:
        """Values, ``(batch_size, nnz)``."""
        return self._data

    @property
    def row_indices(self) -> wp.array:
        """Row index of every stored entry, ``(nnz,)`` int32 (the COO row array).

        Depends only on the pattern, so it is computed once, with cupy, on
        first use.
        """
        if self._row_indices is None:
            if self._nnz == 0:
                self._row_indices = wp.zeros(0, dtype=wp.int32, device=self._device)
            else:
                indptr_cp = cp.asarray(self._indptr)
                rows = cp.searchsorted(indptr_cp[1:], cp.arange(self._nnz, dtype=cp.int32), side="right")
                self._row_indices = to_wp_int32(rows, self._device)
        return self._row_indices

    def __getitem__(self, key: int) -> csr_matrix:
        """Zero-copy ``csr_matrix`` view of the ``key``-th matrix (cupy views of
        the Warp buffers), for pattern algebra and library setup calls."""
        return csr_matrix(
            (cp.asarray(self._data)[key], cp.asarray(self._indices), cp.asarray(self._indptr)),
            shape=(self._rows, self._cols)
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def is_torch_sparse_csr_tensor(obj) -> bool:
        """Return whether *obj* is a Torch CSR tensor."""
        if not (hasattr(obj, "layout") and hasattr(obj, "crow_indices")
                and hasattr(obj, "values")):
            return False
        try:
            import torch
        except ImportError:
            return False
        return isinstance(obj, torch.Tensor) and obj.layout == torch.sparse_csr

    @classmethod
    def from_input(
        cls,
        matrix,
        dtype=wp.float64,
        device: str = "cuda",
        validate_shared_sparsity: bool = True
    ) -> "UniformBatchedCsrMatrix":
        """Normalize one accepted sparse input form to uniform batched CSR:
        a ``UniformBatchedCsrMatrix`` (copied), a 2-D or 3-D torch CSR
        tensor, a list of cupy CSR matrices sharing one pattern, or a single
        cupy CSR matrix."""
        if isinstance(matrix, cls):
            return cls(
                batch_size=matrix.batch_size,
                indptr=matrix.indptr,
                indices=matrix.indices,
                data=matrix.data,
                shape=(matrix.rows, matrix.cols),
                dtype=dtype,
                device=device
            )
        if cls.is_torch_sparse_csr_tensor(matrix):
            return cls.from_torch_sparse_csr_tensor(
                matrix, dtype=dtype, device=device,
                validate_shared_sparsity=validate_shared_sparsity
            )
        if isinstance(matrix, (list, tuple)):
            return cls.from_cupy_csr_matrix_sequence(
                matrix, dtype=dtype, device=device,
                validate_shared_sparsity=validate_shared_sparsity
            )
        return cls.from_cupy_csr_matrix(matrix, dtype=dtype, device=device)

    @classmethod
    def from_cupy_csr_matrix(
        cls, matrix, batch_size: int = 1, dtype=wp.float64, device: str = "cuda"
    ) -> "UniformBatchedCsrMatrix":
        """Build from a single cupy CSR (or convertible) matrix; its values are
        replicated into ``batch_size`` identical rows."""
        matrix = csr_matrix(matrix, dtype=wp.dtype_to_numpy(to_warp_dtype(dtype)))
        return cls(
            batch_size=batch_size,
            indptr=matrix.indptr,
            indices=matrix.indices,
            data=matrix.data,
            shape=matrix.shape,
            dtype=dtype,
            device=device
        )

    @classmethod
    def from_cupy_csr_matrix_sequence(
        cls,
        matrices: Sequence[csr_matrix],
        dtype=wp.float64,
        device: str = "cuda",
        validate_shared_sparsity: bool = True
    ) -> "UniformBatchedCsrMatrix":
        """Build from a non-empty list or tuple of cupy CSR matrices sharing one pattern."""
        if not isinstance(matrices, (list, tuple)) or len(matrices) == 0:
            raise ValueError("matrices must be a non-empty list or tuple of CuPy csr matrices.")
        np_dtype = wp.dtype_to_numpy(to_warp_dtype(dtype))
        matrices = [csr_matrix(matrix, dtype=np_dtype) for matrix in matrices]
        template = matrices[0]
        if validate_shared_sparsity:
            cls._require_uniform_sparsity(matrices)
        data = (
            cp.stack([matrix.data for matrix in matrices])
            if template.nnz > 0 else cp.empty((len(matrices), 0), dtype=np_dtype)
        )
        return cls(
            batch_size=len(matrices),
            indptr=template.indptr,
            indices=template.indices,
            data=data,
            shape=template.shape,
            dtype=dtype,
            device=device
        )

    @staticmethod
    def _require_uniform_sparsity(matrices: Sequence[csr_matrix]) -> None:
        """Raise unless all CSR matrices share one shape and structure."""
        template = matrices[0]
        error = "All matrices must share the same CSR sparsity pattern."
        if any(
            matrix.shape != template.shape or matrix.nnz != template.nnz
            for matrix in matrices[1:]
        ):
            raise ValueError(error)
        if len(matrices) == 1:
            return

        indices = cp.stack([matrix.indices for matrix in matrices[1:]])
        indptr = cp.stack([matrix.indptr for matrix in matrices[1:]])
        same = (
            cp.array_equal(indices, cp.broadcast_to(template.indices, indices.shape))
            & cp.array_equal(indptr, cp.broadcast_to(template.indptr, indptr.shape))
        )
        if not bool(same):
            raise ValueError(error)

    @classmethod
    def empty(
        cls, batch_size: int, rows: int, cols: int, dtype=wp.float64, device: str = "cuda"
    ) -> "UniformBatchedCsrMatrix":
        """Build a batch of ``(rows, cols)`` matrices with no stored entries."""
        return cls(
            batch_size=batch_size,
            indptr=np.zeros(rows + 1, dtype=np.int32),
            indices=np.zeros(0, dtype=np.int32),
            data=cp.empty((batch_size, 0), dtype=wp.dtype_to_numpy(to_warp_dtype(dtype))),
            shape=(rows, cols),
            dtype=dtype,
            device=device
        )

    @classmethod
    def from_torch_sparse_csr_tensor(
        cls, tensor, batch_size: int = 1, dtype=wp.float64, device: str = "cuda",
        validate_shared_sparsity: bool = True
    ) -> "UniformBatchedCsrMatrix":
        """Build from a CUDA torch CSR tensor.

        ``tensor`` may be 2-D with shape ``(M, N)``, whose values are then
        replicated into ``batch_size`` rows, or 3-D with shape ``(B, M, N)``.
        For 3-D inputs every batch must share one CSR pattern; by default
        this is checked before using the first batch's ``indptr`` /
        ``indices`` as the shared structure. Values are copied into
        solver-owned storage.
        """
        import torch  # local import to avoid a hard dependency on torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.layout != torch.sparse_csr:
            raise ValueError(f"Expected torch.sparse_csr layout, got {tensor.layout}.")
        if tensor.dim() not in (2, 3):
            raise ValueError(
                "Expected a sparse CSR tensor of shape (M, N) or (B, M, N); "
                f"got shape {tuple(tensor.shape)}."
            )
        if not tensor.is_cuda:
            raise ValueError("tensor must reside on a CUDA device.")

        if tensor.dim() == 2:
            rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
            return cls(
                batch_size=batch_size,
                indptr=cp.from_dlpack(tensor.crow_indices().contiguous()),
                indices=cp.from_dlpack(tensor.col_indices().contiguous()),
                data=cp.from_dlpack(tensor.values().contiguous()),
                shape=(rows, cols),
                dtype=dtype,
                device=device
            )

        B, rows, cols = (
            int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2])
        )
        if B < 1:
            raise ValueError(
                "A batched sparse CSR tensor must contain at least one matrix."
            )
        crow = tensor.crow_indices()   # (B, M+1)
        col = tensor.col_indices()     # (B, nnz)
        values = tensor.values()       # (B, nnz)

        if validate_shared_sparsity and B > 1:
            crow_all = cp.from_dlpack(crow.contiguous())
            col_all = cp.from_dlpack(col.contiguous())
            same = (
                cp.array_equal(crow_all, cp.broadcast_to(crow_all[0], crow_all.shape))
                & cp.array_equal(col_all, cp.broadcast_to(col_all[0], col_all.shape))
            )
            if not bool(same):
                raise ValueError(
                    "All batch matrices must share the same CSR sparsity pattern."
                )

        # crow[0] / col[0] are contiguous rows of the 2-D index buffers, so
        # DLPack views them without a copy; the constructor copies from them.
        return cls(
            batch_size=B,
            indptr=cp.from_dlpack(crow[0].contiguous()),
            indices=cp.from_dlpack(col[0].contiguous()),
            data=cp.from_dlpack(values.contiguous()),
            shape=(rows, cols),
            dtype=dtype,
            device=device
        )


def to_wp_int32(values, device: str = "cuda") -> wp.array:
    """Owned ``int32`` Warp copy of an integer array (host or GPU); a
    zero-length input gives an empty array instead of a null-pointer view."""
    values_cp = cp.asarray(values, dtype=cp.int32).ravel()
    if values_cp.size == 0:
        return wp.zeros(0, dtype=wp.int32, device=device)
    return wp.array(cp.ascontiguousarray(values_cp), dtype=wp.int32, device=device)
