from typing import Optional

import cupy as cp
import warp as wp

from .sparse_data import SparseData
from ..preconditioner import RuizEquilibration
from .batched_csr import BatchedCsrMatrix
from .sparse_preconditioner_kernels import (
    create_sparse_scale_matrices_kernel,
    create_sparse_compute_kkt_norms_kernel,
)


class SparseRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for the sparse CSR backend.

    After the ``SparseData`` refactor, ``data.P`` / ``data.A`` / ``data.G``
    are always :class:`BatchedCsrMatrix` instances (B = 1 is just a 1-batch
    batched matrix), so this class only has a single batched code path —
    no ``isinstance(P, list)`` branching.

    All in-place scaling and norm computations act on the shared
    ``(B, nnz)`` values buffer via ``M.data`` (zero-copy), and on the
    shared ``indptr`` / ``indices`` via ``M.indptr`` / ``M.indices``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sparse_scale_matrices_kernel = create_sparse_scale_matrices_kernel(self.n, self.p, self.m, dtype=self._dtype)
        self._sparse_compute_row_inf_norm_kernel, self._sparse_compute_col_inf_norm_kernel = \
            create_sparse_compute_kkt_norms_kernel(self.n, self.p, self.m, dtype=self._dtype)

        self._ones = cp.ones(self.B, dtype=self._dtype)

    # ------------------------------------------------------------------
    # 3-hook backend API
    # ------------------------------------------------------------------

    def compute_kkt_norms(self, data: SparseData,
                          d_iter: cp.ndarray, d_b_iter: cp.ndarray):
        rows_max = max(self.n, self.p, self.m)
        if rows_max == 0:
            return
        stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
        wp.launch(
            kernel=self._sparse_compute_row_inf_norm_kernel,
            dim=(self.B, rows_max),
            inputs=[
                data.P.data, data.P.indptr,
                data.A.data, data.A.indptr,
                data.G.data, data.G.indptr,
                self._x_b_scaling, d_iter, d_b_iter,
            ],
            device="cuda", stream=stream,
        )
        
        nnz_max = max(int(data.A.nnz), int(data.G.nnz))
        if nnz_max > 0 and (self.p > 0 or self.m > 0):
            wp.launch(
                kernel=self._sparse_compute_col_inf_norm_kernel,
                dim=(self.B, nnz_max),
                inputs=[
                    data.A.data, data.A.indices,
                    data.G.data, data.G.indices,
                    d_iter,
                ],
                device="cuda", stream=stream,
            )

    def scale_matrices(self, data: SparseData,
                       d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray,
                       cost_scaling_factor: Optional[cp.ndarray] = None):
        cost_factor = cost_scaling_factor if cost_scaling_factor is not None else self._ones
        rows_max = max(self.n, self.p, self.m)
        if rows_max == 0:
            return
        wp.launch(
            kernel=self._sparse_scale_matrices_kernel,
            dim=(self.B, rows_max),
            inputs=[
                data.P.data, data.P.indptr, data.P.indices,
                data.A.data, data.A.indptr, data.A.indices,
                data.G.data, data.G.indptr, data.G.indices,
                data.c, d_x, d_y, d_z, cost_factor,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    def apply_cost_scaling(self, data: SparseData):
        P_norms = self._batched_utri_symmetric_col_inf_norms(data.P)  # (B, n)
        gamma = cp.mean(P_norms, axis=1)                                # (B,)
        gamma = cp.clip(gamma, self.min_scaling, self.max_scaling)
        c_norm = cp.max(cp.abs(data.c), axis=1)                        # (B,)
        gamma = cp.maximum(gamma, c_norm)
        gamma = cp.clip(gamma, self.min_scaling, self.max_scaling)
        gamma = 1.0 / gamma                                             # (B,)
        data.P.data[:] *= gamma[:, None]
        data.c[:] *= gamma[:, None]
        self._cost_scaling *= gamma

    # ------------------------------------------------------------------
    # Batched CSR primitives — operate on a BatchedCsrMatrix's shared
    # sparsity + (B, nnz) values buffer
    # ------------------------------------------------------------------

    @staticmethod
    def _row_indices_from_indptr(indptr: cp.ndarray, nnz: int) -> cp.ndarray:
        """nnz-length array mapping each CSR entry to its row index."""
        if nnz == 0:
            return cp.zeros(0, dtype=cp.int32)
        nz = cp.arange(nnz, dtype=cp.int32)
        return cp.searchsorted(indptr[1:], nz, side='right').astype(cp.int32)

    def _batched_row_inf_norms(self, M: BatchedCsrMatrix, out: cp.ndarray):
        """Row inf-norms: out[b, r] = max_j |M[b, r, j]|. out shape: (B, rows)."""
        out.fill(0.0)
        if M.nnz == 0 or M.rows == 0:
            return
        row_idx = self._row_indices_from_indptr(M.indptr, M.nnz)
        cp.maximum.at(out, (slice(None), row_idx), cp.abs(M.data))

    def _batched_col_inf_norms(self, M: BatchedCsrMatrix, out: cp.ndarray):
        """Column inf-norms: out[b, c] = max_i |M[b, i, c]|. out shape: (B, cols)."""
        out.fill(0.0)
        if M.nnz == 0 or M.cols == 0:
            return
        cp.maximum.at(out, (slice(None), M.indices), cp.abs(M.data))

    def _batched_utri_symmetric_col_inf_norms(self, P: BatchedCsrMatrix) -> cp.ndarray:
        """For a symmetric P stored as upper triangle only, per-batch column
        inf-norms treat missing lower-triangle entries via row/col max."""
        B, n = P.batch_size, P.rows
        if P.nnz == 0:
            return cp.zeros((B, n), dtype=cp.float64)
        row = cp.empty((B, n), dtype=cp.float64)
        col = cp.empty((B, n), dtype=cp.float64)
        self._batched_row_inf_norms(P, row)
        self._batched_col_inf_norms(P, col)
        return cp.maximum(row, col)
