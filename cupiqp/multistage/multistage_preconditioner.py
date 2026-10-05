from typing import Optional

import warp as wp

from .multistage_data import MultistageData
from ..preconditioner import RuizEquilibration
from ..utils import column_slice
from .multistage_preconditioner_kernels import (
    create_multistage_scale_matrices_kernel,
    create_multistage_compute_kkt_norms_kernel,
    create_multistage_P_col_norms_kernel,
    create_gamma_from_norms_kernel
)




class MultistageRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for the multistage backend, batched.

    P is block-tridiagonal and A, G are block lower-bidiagonal, stored as
    ``(diag, offdiag)`` Warp block arrays on the data object. Norms and
    scaling are fused Warp kernels over that per-block storage, with a
    leading batch axis throughout.
    """

    def __init__(self, *args, data: MultistageData, **kwargs):
        super().__init__(*args, **kwargs)

        # Block layout is fixed at setup; pull it once and bake into the
        # specialized warp kernels here.
        N = data.num_blocks
        d = data.block_size
        A_rows_per_block, G_rows_per_block = data.A_rows_per_block, data.G_rows_per_block
        self._N, self._d, self._A_rows_per_block, self._G_rows_per_block = N, d, A_rows_per_block, G_rows_per_block

        self._multistage_scale_matrices_kernel = create_multistage_scale_matrices_kernel(
            N, d, A_rows_per_block, G_rows_per_block, dtype=self._dtype)
        self._multistage_compute_kkt_norms_kernel = create_multistage_compute_kkt_norms_kernel(
            N, d, A_rows_per_block, G_rows_per_block, dtype=self._dtype)
        self._multistage_P_col_norms_kernel = create_multistage_P_col_norms_kernel(N, d, dtype=self._dtype)
        self._gamma_from_norms_kernel = create_gamma_from_norms_kernel(
            self.min_scaling, self.max_scaling, dtype=self._dtype)

        self._dummy_4d = wp.zeros((self.B, 1, 1, 1), dtype=self._dtype, device=self._device)
        self._P_D = data.P_diag
        self._P_E = data.P_offdiag
        self._A_D = data.A_diag if self.p > 0 else self._dummy_4d
        self._A_E = data.A_offdiag if self.p > 0 else self._dummy_4d
        self._G_D = data.G_diag if self.m > 0 else self._dummy_4d
        self._G_E = data.G_offdiag if self.m > 0 else self._dummy_4d
        self._c = data.c

        self._ones = wp.full(self.B, 1.0, dtype=self._dtype, device=self._device)
        # Unit row/column factors for applying a pure cost scaling through scale_matrices.
        self._ones_npm = wp.full((self.B, self.n + self.p + self.m), 1.0, dtype=self._dtype, device=self._device)
        self._gamma_buf = wp.empty(self.B, dtype=self._dtype, device=self._device)

    # ------------------------------------------------------------------
    # 3-hook backend API
    # ------------------------------------------------------------------

    def compute_kkt_norms(self, data: MultistageData, d_iter: wp.array, d_b_iter: wp.array):
        wp.launch(
            kernel=self._multistage_compute_kkt_norms_kernel,
            dim=(self.B, self.n + self.p + self.m),
            inputs=[
                self._P_D, self._P_E,
                self._A_D, self._A_E,
                self._G_D, self._G_E,
                self._x_b_scaling, d_iter, d_b_iter,
            ],
            device=self._device
        )

    def scale_matrices(self, data: MultistageData,
                       d_x: wp.array, d_y: wp.array, d_z: wp.array,
                       cost_scaling_factor: Optional[wp.array] = None):
        cf = cost_scaling_factor if cost_scaling_factor is not None else self._ones
        N, d, A_rows_per_block, G_rows_per_block = self._N, self._d, self._A_rows_per_block, self._G_rows_per_block
        max_rows = max(d, A_rows_per_block, G_rows_per_block)
        wp.launch(
            kernel=self._multistage_scale_matrices_kernel,
            dim=(self.B, N, max_rows, d),
            inputs=[
                self._P_D, self._P_E,
                self._A_D, self._A_E,
                self._G_D, self._G_E,
                self._c, d_x, d_y, d_z, cf,
            ],
            device=self._device
        )

    def apply_cost_scaling(self, data: MultistageData):
        """Per-problem cost scaling gamma = 1/max(mean(||P_cols||), ||c||).

        Scales P and c by gamma, accumulates gamma into self._cost_scaling.
        """
        n, p, m = self.n, self.p, self.m
        # (B, n) column norms of P -> per-batch gamma (delta_iter is free scratch
        # here: it is recomputed by compute_kkt_norms at the next iteration).
        col_norms = column_slice(self._delta_iter, 0, n)
        wp.launch(
            kernel=self._multistage_P_col_norms_kernel,
            dim=(self.B, n),
            inputs=[self._P_D, self._P_E, col_norms],
            device=self._device
        )
        wp.launch(
            kernel=self._gamma_from_norms_kernel,
            dim=(self.B,),
            inputs=[col_norms, self._c, self._gamma_buf],
            device=self._device
        )
        # P *= gamma, c *= gamma (unit row/column factors), cost_scaling *= gamma.
        self.scale_matrices(
            data,
            column_slice(self._ones_npm, 0, n), column_slice(self._ones_npm, n, n + p), column_slice(self._ones_npm, n + p, None),
            cost_scaling_factor=self._gamma_buf
        )
        wp.map(wp.mul, self._cost_scaling, self._gamma_buf, out=self._cost_scaling)
