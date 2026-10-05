import warp as wp

from .sparse_data import SparseData
from ..preconditioner import RuizEquilibration
from ..preconditioner_kernels import REDUCTION_BLOCK_DIM
from .sparse_preconditioner_kernels import (
    create_sparse_scale_matrices_kernel,
    create_sparse_compute_kkt_norms_kernel,
    create_sparse_P_norms_kernel,
    create_sparse_compute_gamma_kernel,
    create_sparse_apply_gamma_kernel
)


class SparseRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for the sparse CSR backend.

    ``data.P`` / ``data.A`` / ``data.G`` are :class:`UniformBatchedCsrMatrix`
    instances (B = 1 is a one-element batch), so there is a single batched
    code path. All norm and scaling computations are Warp kernels over the
    shared ``(B, nnz)`` values buffer ``M.data`` and the shared ``indptr`` /
    ``indices``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sparse_scale_matrices_kernel = create_sparse_scale_matrices_kernel(self.n, self.p, self.m, dtype=self._dtype)
        self._sparse_compute_row_inf_norm_kernel, self._sparse_compute_col_inf_norm_kernel = \
            create_sparse_compute_kkt_norms_kernel(self.n, self.p, self.m, dtype=self._dtype)
        self._sparse_P_norms_kernel = create_sparse_P_norms_kernel(dtype=self._dtype)
        self._sparse_compute_gamma_kernel = create_sparse_compute_gamma_kernel(
            self.min_scaling, self.max_scaling, dtype=self._dtype)
        self._sparse_apply_gamma_kernel = create_sparse_apply_gamma_kernel(dtype=self._dtype)

        self._ones = wp.full(self.B, 1.0, dtype=self._dtype, device=self._device)
        # Per-column inf-norms of P (row and column max) and the resulting
        # cost-scaling factor, reused by every apply_cost_scaling call.
        self._P_norms = wp.zeros((self.B, self.n), dtype=self._dtype, device=self._device)
        self._gamma_buf = wp.empty(self.B, dtype=self._dtype, device=self._device)

    # ------------------------------------------------------------------
    # 3-hook backend API
    # ------------------------------------------------------------------

    def compute_kkt_norms(self, data: SparseData, d_iter: wp.array, d_b_iter: wp.array):
        rows_max = max(self.n, self.p, self.m)
        if rows_max == 0:
            return
        wp.launch(
            kernel=self._sparse_compute_row_inf_norm_kernel,
            dim=(self.B, rows_max),
            inputs=[
                data.P.data, data.P.indptr,
                data.A.data, data.A.indptr,
                data.G.data, data.G.indptr,
                self._x_b_scaling, d_iter, d_b_iter,
            ],
            device=self._device
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
                device=self._device
            )

    def scale_matrices(self, data: SparseData, d_x: wp.array, d_y: wp.array, d_z: wp.array,
                       cost_scaling_factor: wp.array = None):
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
            device=self._device
        )

    def apply_cost_scaling(self, data: SparseData):
        """Per-problem cost scaling gamma = 1 / clip(max(mean(||P_cols||), ||c||)).

        Scales P and c by gamma and accumulates gamma into the cost scaling.
        The column norms of P treat the stored triangle symmetrically (row and
        column max per index).
        """
        P = data.P
        self._P_norms.zero_()
        if P.nnz > 0:
            wp.launch(
                kernel=self._sparse_P_norms_kernel,
                dim=(self.B, P.nnz),
                inputs=[P.data, P.row_indices, P.indices, self._P_norms],
                device=self._device
            )
        wp.launch_tiled(
            kernel=self._sparse_compute_gamma_kernel,
            dim=[self.B],
            inputs=[self._P_norms, data.c, self._gamma_buf],
            block_dim=REDUCTION_BLOCK_DIM,
            device=self._device
        )
        wp.launch(
            kernel=self._sparse_apply_gamma_kernel,
            dim=(self.B, max(P.nnz, self.n)),
            inputs=[P.data, data.c, self._cost_scaling, self._gamma_buf],
            device=self._device
        )
