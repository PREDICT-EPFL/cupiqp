from typing import Optional
import cupy as cp
import warp as wp

from .dense_data import DenseData
from ..preconditioner import RuizEquilibration
from .dense_preconditioner_kernels import (
    create_dense_compute_kkt_norms_kernel,
    create_dense_scale_P_and_c_kernel,
    create_dense_scale_A_or_G_kernel,
    create_dense_compute_gamma_kernel,
    create_dense_apply_gamma_kernel,
)


USE_WARP = True


class DenseRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for dense matrix backends.

    All matrices are (B, rows, cols), all vectors are (B, k).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dense_compute_kkt_norms_kernel = create_dense_compute_kkt_norms_kernel(
            self.n, self.p, self.m,
        )
        self._dense_scale_P_c_kernel = create_dense_scale_P_and_c_kernel(self.n)
        if self.p > 0:
            self._dense_scale_A_kernel = create_dense_scale_A_or_G_kernel(self.p, self.n)
        if self.m > 0:
            self._dense_scale_G_kernel = create_dense_scale_A_or_G_kernel(self.m, self.n)
        # (B,)-ones buffer used when scale_matrices is called with cost_scaling_factor=None.
        self._cost_factor_ones = cp.ones(self.B, dtype=cp.float64)

        self._dense_compute_gamma_kernel = create_dense_compute_gamma_kernel(
            self.n, self.min_scaling, self.max_scaling,
        )
        self._dense_apply_gamma_kernel = create_dense_apply_gamma_kernel(self.n)
        self._gamma_buf = cp.empty(self.B, dtype=cp.float64)

    # ------------------------------------------------------------------
    # 3-hook backend API
    # ------------------------------------------------------------------

    def compute_kkt_norms(self, data: DenseData,
                          d_iter: cp.ndarray, d_b_iter: cp.ndarray):
        """Fill d_iter (B, n+p+m) with Ruiz row/col inf-norms; d_b_iter = x_b_scaling."""
        n, p, m = self.n, self.p, self.m

        if USE_WARP:
            wp.launch(
                kernel=self._dense_compute_kkt_norms_kernel,
                dim=(self.B, n + p + m),
                inputs=[data.P, data.A, data.G, self._x_b_scaling, d_iter, d_b_iter],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )
            return

        # --- cupy fallback (kept for A/B verification) ---
        P, A, G = data.P, data.A, data.G
        cp.max(cp.abs(P), axis=2, out=d_iter[:, :n])
        if p > 0:
            cp.max(cp.abs(A), axis=1, out=self._work_n)
            d_iter[:, :n] = cp.maximum(d_iter[:, :n], self._work_n)
            cp.max(cp.abs(A), axis=2, out=d_iter[:, n:n+p])
        if m > 0:
            cp.max(cp.abs(G), axis=1, out=self._work_n)
            d_iter[:, :n] = cp.maximum(d_iter[:, :n], self._work_n)
            cp.max(cp.abs(G), axis=2, out=d_iter[:, n+p:n+p+m])
        cp.maximum(d_iter[:, :n], self._x_b_scaling, out=d_iter[:, :n])
        d_b_iter[:] = self._x_b_scaling

    def scale_matrices(self, data: DenseData,
                       d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray,
                       cost_scaling_factor: Optional[cp.ndarray] = None):
        """Apply row/col scaling in-place to P, c, A, G."""
        if USE_WARP:
            stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
            cf = cost_scaling_factor if cost_scaling_factor is not None else self._cost_factor_ones
            wp.launch(
                kernel=self._dense_scale_P_c_kernel,
                dim=(self.B, self.n, self.n),
                inputs=[data.P, data.c, d_x, cf],
                device="cuda", stream=stream,
            )
            if self.p > 0:
                wp.launch(
                    kernel=self._dense_scale_A_kernel,
                    dim=(self.B, self.p, self.n),
                    inputs=[data.A, d_y, d_x],
                    device="cuda", stream=stream,
                )
            if self.m > 0:
                wp.launch(
                    kernel=self._dense_scale_G_kernel,
                    dim=(self.B, self.m, self.n),
                    inputs=[data.G, d_z, d_x],
                    device="cuda", stream=stream,
                )
            return

        # --- cupy fallback ---
        data.P[:] *= d_x[:, None, :]
        data.P[:] *= d_x[:, :, None]
        data.c[:] *= d_x
        if cost_scaling_factor is not None:
            data.P[:] *= cost_scaling_factor[:, None, None]
            data.c[:] *= cost_scaling_factor[:, None]
        if self.p > 0:
            data.A[:] *= d_x[:, None, :]
            data.A[:] *= d_y[:, :, None]
        if self.m > 0:
            data.G[:] *= d_x[:, None, :]
            data.G[:] *= d_z[:, :, None]

    def apply_cost_scaling(self, data: DenseData):
        """Per-problem cost scaling gamma = 1/max(mean(||P_cols||), ||c||).

        Scales P and c by gamma, accumulates gamma into self._cost_scaling.
        """
        if USE_WARP:
            stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
            wp.launch(
                kernel=self._dense_compute_gamma_kernel,
                dim=(self.B,),
                inputs=[data.P, data.c, self._gamma_buf],
                device="cuda", stream=stream,
            )
            wp.launch(
                kernel=self._dense_apply_gamma_kernel,
                dim=(self.B, self.n, self.n),
                inputs=[data.P, data.c, self._cost_scaling, self._gamma_buf],
                device="cuda", stream=stream,
            )
            return

        # --- cupy fallback ---
        n = self.n
        P_abs = cp.abs(data.P)                                       # (B, n, n)
        P_utri = cp.triu(P_abs.reshape(-1, n, n)).reshape(data.P.shape)
        col_max = cp.max(P_utri, axis=1)                              # (B, n)
        row_max = cp.max(P_utri, axis=2)                              # (B, n)
        gamma = cp.mean(cp.maximum(col_max, row_max), axis=1)         # (B,)
        gamma = cp.clip(gamma, self.min_scaling, self.max_scaling)
        c_norm = cp.max(cp.abs(data.c), axis=1)                      # (B,)
        gamma = cp.maximum(gamma, c_norm)
        gamma = cp.clip(gamma, self.min_scaling, self.max_scaling)
        gamma = 1.0 / gamma                                           # (B,)
        data.P[:] *= gamma[:, None, None]
        data.c[:] *= gamma[:, None]
        self._cost_scaling *= gamma
