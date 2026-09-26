from typing import Optional

import cupy as cp
import warp as wp

from .multistage_data import MultistageData
from ..preconditioner import RuizEquilibration
from ..utils import to_warp_dtype
from .multistage_preconditioner_kernels import (
    create_multistage_scale_matrices_kernel,
    create_multistage_compute_kkt_norms_kernel,
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
        rows_A, rows_G = data.A_rows, data.G_rows
        self._N, self._d, self._rows_A, self._rows_G = N, d, rows_A, rows_G

        self._multistage_scale_matrices_kernel = create_multistage_scale_matrices_kernel(
            N, d, rows_A, rows_G, dtype=self._dtype)
        self._multistage_compute_kkt_norms_kernel = create_multistage_compute_kkt_norms_kernel(
            N, d, rows_A, rows_G, dtype=self._dtype)


        self._dummy_4d = wp.zeros(
            (self.B, 1, 1, 1), dtype=to_warp_dtype(self._dtype), device="cuda"
        )
        self._P_D = data.P_diag
        self._P_E = data.P_offdiag
        self._A_D = data.A_diag if self.p > 0 else self._dummy_4d
        self._A_E = data.A_offdiag if self.p > 0 else self._dummy_4d
        self._G_D = data.G_diag if self.m > 0 else self._dummy_4d
        self._G_E = data.G_offdiag if self.m > 0 else self._dummy_4d
        self._c = data.c

        self._ones = cp.ones(self.B, dtype=self._dtype)

    # ------------------------------------------------------------------
    # 3-hook backend API
    # ------------------------------------------------------------------

    def compute_kkt_norms(self, data: MultistageData,
                          d_iter: cp.ndarray, d_b_iter: cp.ndarray):
        wp.launch(
            kernel=self._multistage_compute_kkt_norms_kernel,
            dim=(self.B, self.n + self.p + self.m),
            inputs=[
                self._P_D, self._P_E,
                self._A_D, self._A_E,
                self._G_D, self._G_E,
                self._x_b_scaling, d_iter, d_b_iter,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    def scale_matrices(self, data: MultistageData,
                       d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray,
                       cost_scaling_factor: Optional[cp.ndarray] = None):
        cf = cost_scaling_factor if cost_scaling_factor is not None else self._ones
        N, d, rows_A, rows_G = self._N, self._d, self._rows_A, self._rows_G
        max_rows = max(d, rows_A, rows_G)
        wp.launch(
            kernel=self._multistage_scale_matrices_kernel,
            dim=(self.B, N, max_rows, d),
            inputs=[
                self._P_D, self._P_E,
                self._A_D, self._A_E,
                self._G_D, self._G_E,
                self._c, d_x, d_y, d_z, cf,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    def apply_cost_scaling(self, data: MultistageData):
        B = self.B
        N = data.num_blocks
        # (B, N, d, d) and (B, N-1, d, d).
        P_D = cp.from_dlpack(wp.to_dlpack(data.P_diag))
        P_E = cp.from_dlpack(wp.to_dlpack(data.P_offdiag))

        # Column inf-norms of upper-triangular P (symmetric → col_norm == row_norm).
        # cp.triu broadcasts over the leading (B, N) axes.
        P_D_abs = cp.abs(P_D)
        P_D_utri = cp.triu(P_D_abs)                 # (B, N, d, d)
        col_norms = cp.maximum(
            cp.max(P_D_utri, axis=2),               # rowwise max → (B, N, d)
            cp.max(P_D_utri, axis=3),               # colwise max → (B, N, d)
        )                                            # (B, N, d)
        if N > 1:
            P_E_abs = cp.abs(P_E)
            cp.maximum(col_norms[:, :N - 1], cp.max(P_E_abs, axis=3), out=col_norms[:, :N - 1])
            cp.maximum(col_norms[:, 1:N],     cp.max(P_E_abs, axis=2), out=col_norms[:, 1:N])

        # gamma per batch: 1 / max(mean(col_norms_per_batch), max_abs_c_per_batch).
        gamma = cp.mean(col_norms.reshape(B, -1), axis=1)         # (B,)
        gamma = self._limit_scaling_array(gamma)
        gamma = cp.maximum(gamma, cp.max(cp.abs(data.c), axis=1))
        gamma = self._limit_scaling_array(gamma)
        gamma = 1.0 / gamma                                        # (B,)

        P_D *= gamma[:, None, None, None]
        if N > 1:
            P_E *= gamma[:, None, None, None]
        data.c[:] *= gamma[:, None]
        self._cost_scaling *= gamma

    # ------------------------------------------------------------------
    # Per-batch helper (avoids the scalar ``_limit_scaling_scalar``)
    # ------------------------------------------------------------------

    def _limit_scaling_array(self, d: cp.ndarray) -> cp.ndarray:
        """Element-wise clamp like ``_limit_scaling`` but pure functional —
        returns a new array; the input is not modified.
        """
        # below the floor → reset to 1; above the ceiling → clamp to max.
        out = cp.where(d < self.min_scaling, 1.0, d)
        return cp.minimum(out, self.max_scaling)
