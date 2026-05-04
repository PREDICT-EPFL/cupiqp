from typing import Optional

import cupy as cp
import warp as wp

from .multistage_data import MultistageData
from ..preconditioner import RuizEquilibration
from .multistage_utils import BlockTridiagMat, BlockBidiagMat


class MultistageRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for the multistage backend, batched.

    P is block-tridiagonal (BlockTridiagMat); A, G are block lower-bidiagonal
    (BlockBidiagMat). Norms and scaling operate on the dense per-block storage
    via DLPack bridges to Warp buffers, with a leading batch axis throughout.
    """

    # ------------------------------------------------------------------
    # 3-hook backend API
    # ------------------------------------------------------------------

    def compute_kkt_norms(self, data: MultistageData,
                          d_iter: cp.ndarray, d_b_iter: cp.ndarray):
        n, p, m = self.n, self.p, self.m

        # x-block (B, n): row inf-norms of P (symmetric, so row ≡ col).
        self._tridiag_row_inf_norms(data.P, d_iter[:, :n])
        if p > 0:
            self._bidiag_col_inf_norms(data.A, self._work_n)
            cp.maximum(d_iter[:, :n], self._work_n, out=d_iter[:, :n])
            self._bidiag_row_inf_norms(data.A, d_iter[:, n:n + p])
        if m > 0:
            self._bidiag_col_inf_norms(data.G, self._work_n)
            cp.maximum(d_iter[:, :n], self._work_n, out=d_iter[:, :n])
            self._bidiag_row_inf_norms(data.G, d_iter[:, n + p:n + p + m])
        cp.maximum(d_iter[:, :n], self._x_b_scaling, out=d_iter[:, :n])

        d_b_iter[:] = self._x_b_scaling

    def scale_matrices(self, data: MultistageData,
                       d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray,
                       cost_scaling_factor: Optional[cp.ndarray] = None):
        B = self.B
        N = data.num_blocks
        bs = data.block_size

        # (B, n) -> (B, N, bs) — for broadcast against (B, N, d, d) blocks.
        d_x_2d = d_x.reshape(B, N, bs)

        # P_D shape (B, N, d, d); P_E shape (B, N-1, d, d).
        P_D = cp.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = cp.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))
        # P <- D_x P D_x: scale columns then rows of each batch's diag blocks.
        P_D *= d_x_2d[:, :, None, :]
        P_D *= d_x_2d[:, :, :, None]
        if N > 1:
            P_E *= d_x_2d[:, :N - 1, None, :]    # column scaling (block k)
            P_E *= d_x_2d[:, 1:N, :, None]       # row scaling (block k+1)

        data._c *= d_x

        if cost_scaling_factor is not None:
            cf = cost_scaling_factor[:, None, None, None]   # (B, 1, 1, 1)
            P_D *= cf
            if N > 1:
                P_E *= cf
            data._c *= cost_scaling_factor[:, None]

        if self.p > 0:
            r_a = data._A.rows_of_blocks
            d_y_2d = d_y.reshape(B, N + 1, r_a)
            A_D = cp.from_dlpack(wp.to_dlpack(data._A.D))
            A_E = cp.from_dlpack(wp.to_dlpack(data._A.E))
            A_D *= d_x_2d[:, :, None, :]            # column scale by d_x
            A_D *= d_y_2d[:, :N, :, None]           # row scale by d_y[block k]
            A_E *= d_x_2d[:, :, None, :]
            A_E *= d_y_2d[:, 1:N + 1, :, None]      # row scale by d_y[block k+1]

        if self.m > 0:
            r_g = data._G.rows_of_blocks
            d_z_2d = d_z.reshape(B, N + 1, r_g)
            G_D = cp.from_dlpack(wp.to_dlpack(data._G.D))
            G_E = cp.from_dlpack(wp.to_dlpack(data._G.E))
            G_D *= d_x_2d[:, :, None, :]
            G_D *= d_z_2d[:, :N, :, None]
            G_E *= d_x_2d[:, :, None, :]
            G_E *= d_z_2d[:, 1:N + 1, :, None]

    def apply_cost_scaling(self, data: MultistageData):
        B = self.B
        N = data.num_blocks
        # (B, N, d, d) and (B, N-1, d, d).
        P_D = cp.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = cp.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))

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
        gamma = cp.maximum(gamma, cp.max(cp.abs(data._c), axis=1))
        gamma = self._limit_scaling_array(gamma)
        gamma = 1.0 / gamma                                        # (B,)

        P_D *= gamma[:, None, None, None]
        if N > 1:
            P_E *= gamma[:, None, None, None]
        data._c *= gamma[:, None]
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

    # ------------------------------------------------------------------
    # Block-matrix primitives (batched)
    # ------------------------------------------------------------------

    @staticmethod
    def _tridiag_row_inf_norms(P: BlockTridiagMat, out: cp.ndarray):
        """Row inf-norms of a batched block-tridiagonal matrix.

        out: shape (B, N*d).
        """
        B = out.shape[0]
        N, d = P.num_diag_blocks, P.block_size

        # (B, N, d, d) — row inf-norm over the trailing axis → (B, N, d).
        P_D = cp.from_dlpack(wp.to_dlpack(P.diag_blocks.data))
        out[:] = cp.linalg.norm(P_D, ord=cp.inf, axis=-1).reshape(B, N * d)

        if N > 1:
            # (B, N-1, d, d). Lower contributes to rows of blocks [1..N-1];
            # upper (= lower transposed) contributes to rows of blocks [0..N-2].
            P_E = cp.from_dlpack(wp.to_dlpack(P.off_diag_blocks_lower.data))
            row_lower = cp.linalg.norm(P_E, ord=cp.inf, axis=-1)                # (B, N-1, d)
            row_upper = cp.linalg.norm(P_E.swapaxes(-1, -2), ord=cp.inf, axis=-1)  # (B, N-1, d)

            out_2d = out.reshape(B, N, d)
            cp.maximum(out_2d[:, 1:], row_lower, out=out_2d[:, 1:])
            cp.maximum(out_2d[:, :-1], row_upper, out=out_2d[:, :-1])

    @staticmethod
    def _bidiag_row_inf_norms(mat: BlockBidiagMat, out: cp.ndarray):
        """Row inf-norms of a batched block lower-bidiagonal matrix.

        out: shape (B, (N+1)*r).
        """
        B = out.shape[0]
        N = mat.N
        r = mat.rows_of_blocks

        D = cp.from_dlpack(wp.to_dlpack(mat.D))   # (B, N, r, c)
        E = cp.from_dlpack(wp.to_dlpack(mat.E))   # (B, N, r, c)
        row_D = cp.linalg.norm(D, ord=cp.inf, axis=-1)   # (B, N, r)
        row_E = cp.linalg.norm(E, ord=cp.inf, axis=-1)   # (B, N, r)

        out_2d = out.reshape(B, N + 1, r)
        out_2d[:, 0] = row_D[:, 0]
        if N > 1:
            cp.maximum(row_D[:, 1:], row_E[:, :N - 1], out=out_2d[:, 1:N])
        out_2d[:, N] = row_E[:, N - 1]

    @staticmethod
    def _bidiag_col_inf_norms(mat: BlockBidiagMat, out: cp.ndarray):
        """Column inf-norms of a batched block lower-bidiagonal matrix.

        out: shape (B, N*c).
        """
        B = out.shape[0]
        N = mat.N
        c = mat.cols_of_blocks

        D = cp.from_dlpack(wp.to_dlpack(mat.D))   # (B, N, r, c)
        E = cp.from_dlpack(wp.to_dlpack(mat.E))   # (B, N, r, c)

        # column inf-norm over the second-to-last axis -> (B, N, c).
        col_D = cp.linalg.norm(D, ord=cp.inf, axis=-2)
        col_E = cp.linalg.norm(E, ord=cp.inf, axis=-2)
        cp.maximum(col_D, col_E, out=out.reshape(B, N, c))
