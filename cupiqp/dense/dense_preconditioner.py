from typing import Optional
import warp as wp


from .dense_data import DenseData
from ..preconditioner import RuizEquilibration
from .dense_preconditioner_kernels import (
    create_dense_compute_kkt_norms_kernel,
    create_dense_scale_P_and_c_kernel,
    create_dense_scale_A_or_G_kernel,
    create_dense_compute_gamma_kernel,
    create_dense_apply_gamma_kernel
)



class DenseRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for dense matrix backends.

    All matrices are (B, rows, cols), all vectors are (B, k).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dense_compute_kkt_norms_kernel = create_dense_compute_kkt_norms_kernel(
            self.n, self.p, self.m, dtype=self._dtype)
        self._dense_scale_P_c_kernel = create_dense_scale_P_and_c_kernel(self.n, dtype=self._dtype)
        if self.p > 0:
            self._dense_scale_A_kernel = create_dense_scale_A_or_G_kernel(self.p, self.n, dtype=self._dtype)
        if self.m > 0:
            self._dense_scale_G_kernel = create_dense_scale_A_or_G_kernel(self.m, self.n, dtype=self._dtype)
        # (B,)-ones buffer used when scale_matrices is called with cost_scaling_factor=None.
        self._cost_factor_ones = wp.full(self.B, 1.0, dtype=self._dtype, device=self._device)

        self._dense_compute_gamma_kernel = create_dense_compute_gamma_kernel(
            self.n, self.min_scaling, self.max_scaling, dtype=self._dtype)
        self._dense_apply_gamma_kernel = create_dense_apply_gamma_kernel(self.n, dtype=self._dtype)
        self._gamma_buf = wp.empty(self.B, dtype=self._dtype, device=self._device)

    # ------------------------------------------------------------------
    # 3-hook backend API
    # ------------------------------------------------------------------

    def compute_kkt_norms(self, data: DenseData, d_iter: wp.array, d_b_iter: wp.array):
        """Fill d_iter (B, n+p+m) with Ruiz row/col inf-norms; d_b_iter = x_b_scaling."""
        n, p, m = self.n, self.p, self.m

        wp.launch(
            kernel=self._dense_compute_kkt_norms_kernel,
            dim=(self.B, n + p + m),
            inputs=[data.P, data.A, data.G, self._x_b_scaling, d_iter, d_b_iter],
            device=self._device
        )
    def scale_matrices(self, data: DenseData,
                       d_x: wp.array, d_y: wp.array, d_z: wp.array,
                       cost_scaling_factor: Optional[wp.array] = None):
        """Apply row/col scaling in-place to P, c, A, G."""
        cf = cost_scaling_factor if cost_scaling_factor is not None else self._cost_factor_ones
        wp.launch(
            kernel=self._dense_scale_P_c_kernel,
            dim=(self.B, self.n, self.n),
            inputs=[data.P, data.c, d_x, cf],
            device=self._device
        )
        if self.p > 0:
            wp.launch(
                kernel=self._dense_scale_A_kernel,
                dim=(self.B, self.p, self.n),
                inputs=[data.A, d_y, d_x],
                device=self._device
            )
        if self.m > 0:
            wp.launch(
                kernel=self._dense_scale_G_kernel,
                dim=(self.B, self.m, self.n),
                inputs=[data.G, d_z, d_x],
                device=self._device
            )
    def apply_cost_scaling(self, data: DenseData):
        """Per-problem cost scaling gamma = 1/max(mean(||P_cols||), ||c||).

        Scales P and c by gamma, accumulates gamma into self._cost_scaling.
        """
        wp.launch(
            kernel=self._dense_compute_gamma_kernel,
            dim=(self.B,),
            inputs=[data.P, data.c, self._gamma_buf],
            device=self._device
        )
        wp.launch(
            kernel=self._dense_apply_gamma_kernel,
            dim=(self.B, self.n, self.n),
            inputs=[data.P, data.c, self._cost_scaling, self._gamma_buf],
            device=self._device
        )
