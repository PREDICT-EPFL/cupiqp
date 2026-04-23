import cupy as cp

from ..data import Data
from ..preconditioner import RuizEquilibration
from .batched_csr import BatchedCsrMatrix


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

    # ------------------------------------------------------------------
    # Norm evaluation
    # ------------------------------------------------------------------

    def eval_P_row_inf_norms(self, P: BatchedCsrMatrix, out: cp.ndarray):
        self._batched_row_inf_norms(P, out)

    def eval_A_row_inf_norms(self, A: BatchedCsrMatrix, out: cp.ndarray):
        self._batched_row_inf_norms(A, out)

    def eval_A_col_inf_norms(self, A: BatchedCsrMatrix, out: cp.ndarray):
        self._batched_col_inf_norms(A, out)

    def eval_G_row_inf_norms(self, G: BatchedCsrMatrix, out: cp.ndarray):
        self._batched_row_inf_norms(G, out)

    def eval_G_col_inf_norms(self, G: BatchedCsrMatrix, out: cp.ndarray):
        self._batched_col_inf_norms(G, out)

    # ------------------------------------------------------------------
    # Scaling — in-place updates to each matrix's .data buffer
    # ------------------------------------------------------------------

    def _scale_matrices(self, data: Data,
                        d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray):
        self._batched_row_scale(data._P, d_x)
        self._batched_col_scale(data._P, d_x)
        data._c *= d_x

        if self.p > 0:
            self._batched_row_scale(data._A, d_y)
            self._batched_col_scale(data._A, d_x)
        if self.m > 0:
            self._batched_row_scale(data._G, d_z)
            self._batched_col_scale(data._G, d_x)

    def _apply_cost_scaling(self, data: Data):
        P_norms = self._batched_utri_symmetric_col_inf_norms(data._P)  # (B, n)
        gamma = cp.mean(P_norms, axis=1)                                # (B,)
        gamma = cp.clip(gamma, self.min_scaling, self.max_scaling)
        c_norm = cp.max(cp.abs(data._c), axis=1)                        # (B,)
        gamma = cp.maximum(gamma, c_norm)
        gamma = cp.clip(gamma, self.min_scaling, self.max_scaling)
        gamma = 1.0 / gamma                                             # (B,)
        data._P.data *= gamma[:, None]
        data._c *= gamma[:, None]
        self._c_scaling *= gamma

    def _unscale_matrices(self, data: Data,
                          d_x_inv: cp.ndarray, d_y_inv: cp.ndarray, d_z_inv: cp.ndarray):
        data._P.data *= self._cost_scaling_inv[:, None]
        self._batched_row_scale(data._P, d_x_inv)
        self._batched_col_scale(data._P, d_x_inv)
        data._c *= self._cost_scaling_inv[:, None] * d_x_inv

        if self.p > 0:
            self._batched_row_scale(data._A, d_y_inv)
            self._batched_col_scale(data._A, d_x_inv)
        if self.m > 0:
            self._batched_row_scale(data._G, d_z_inv)
            self._batched_col_scale(data._G, d_x_inv)

    def _apply_stored_scaling(self, data: Data,
                              d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray):
        data._P.data *= self._c_scaling[:, None]
        self._batched_row_scale(data._P, d_x)
        self._batched_col_scale(data._P, d_x)
        data._c *= self._c_scaling[:, None] * d_x

        if self.p > 0:
            self._batched_row_scale(data._A, d_y)
            self._batched_col_scale(data._A, d_x)
        if self.m > 0:
            self._batched_row_scale(data._G, d_z)
            self._batched_col_scale(data._G, d_x)

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

    def _batched_row_scale(self, M: BatchedCsrMatrix, d: cp.ndarray):
        """In-place row scaling: M[b] := D(d[b]) @ M[b].  d shape: (B, rows)."""
        if M.nnz == 0:
            return
        row_idx = self._row_indices_from_indptr(M.indptr, M.nnz)
        M.data *= d[:, row_idx]

    def _batched_col_scale(self, M: BatchedCsrMatrix, d: cp.ndarray):
        """In-place col scaling: M[b] := M[b] @ D(d[b]).  d shape: (B, cols)."""
        if M.nnz == 0:
            return
        M.data *= d[:, M.indices]

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
