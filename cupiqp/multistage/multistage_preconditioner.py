from typing import Optional

import cupy as cp
import warp as wp

from .multistage_data import MultistageData
from ..preconditioner import RuizEquilibration
from .multistage_utils import BlockTridiagMat, BlockBidiagMat


class MultistageRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for multistage backend (single problem, B = 1).

    P is block-tridiagonal (BlockTridiagMat); A, G are block lower-bidiagonal
    (BlockBidiagMat). Norms and scaling operate on the dense per-block
    storage via DLPack bridges to warp buffers.
    """

    # ------------------------------------------------------------------
    # 3-hook backend API
    # ------------------------------------------------------------------

    def compute_kkt_norms(self, data: MultistageData,
                          d_iter: cp.ndarray, d_b_iter: cp.ndarray):
        n, p, m = self.n, self.p, self.m

        # d_iter[:, :n] — x-block: P row-norms (P symmetric, so row ≡ col).
        self._tridiag_row_inf_norms(data.P, d_iter[0, :n])
        if p > 0:
            self._bidiag_col_inf_norms(data.A, self._work_n[0])
            cp.maximum(d_iter[0, :n], self._work_n[0], out=d_iter[0, :n])
            self._bidiag_row_inf_norms(data.A, d_iter[0, n:n+p])
        if m > 0:
            self._bidiag_col_inf_norms(data.G, self._work_n[0])
            cp.maximum(d_iter[0, :n], self._work_n[0], out=d_iter[0, :n])
            self._bidiag_row_inf_norms(data.G, d_iter[0, n+p:n+p+m])
        cp.maximum(d_iter[:, :n], self._x_b_scaling, out=d_iter[:, :n])

        d_b_iter[:] = self._x_b_scaling

    def scale_matrices(self, data: MultistageData,
                       d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray,
                       cost_scaling_factor: Optional[cp.ndarray] = None):
        N = data.num_blocks
        bs = data.block_size
        d_x_2d = d_x[0].reshape(N, bs)           # (N, bs) — strip the B=1 dim

        P_D = cp.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = cp.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))
        P_D *= d_x_2d[:, None, :]                # scale columns
        P_D *= d_x_2d[:, :, None]                # scale rows
        if N > 1:
            P_E *= d_x_2d[:N-1, None, :]         # col scale
            P_E *= d_x_2d[1:N, :, None]          # row scale

        data._c *= d_x

        if cost_scaling_factor is not None:
            cf = float(cost_scaling_factor[0])
            P_D *= cf
            if N > 1:
                P_E *= cf
            data._c *= cf

        if self.p > 0:
            r_a = data._A.rows_of_blocks
            d_y_2d = d_y[0].reshape(N + 1, r_a)
            A_D = cp.from_dlpack(wp.to_dlpack(data._A.D))
            A_E = cp.from_dlpack(wp.to_dlpack(data._A.E))
            A_D *= d_x_2d[:, None, :]
            A_D *= d_y_2d[:N, :, None]
            A_E *= d_x_2d[:, None, :]
            A_E *= d_y_2d[1:N+1, :, None]

        if self.m > 0:
            r_g = data._G.rows_of_blocks
            d_z_2d = d_z[0].reshape(N + 1, r_g)
            G_D = cp.from_dlpack(wp.to_dlpack(data._G.D))
            G_E = cp.from_dlpack(wp.to_dlpack(data._G.E))
            G_D *= d_x_2d[:, None, :]
            G_D *= d_z_2d[:N, :, None]
            G_E *= d_x_2d[:, None, :]
            G_E *= d_z_2d[1:N+1, :, None]

    def apply_cost_scaling(self, data: MultistageData):
        N = data.num_blocks
        P_D = cp.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = cp.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))

        # Column inf-norms of upper-triangular P (symmetric => col_norm == row_norm)
        P_D_abs = cp.abs(P_D)
        P_D_utri = cp.array([cp.triu(P_D_abs[k]) for k in range(N)])
        col_norms = cp.maximum(
            cp.max(P_D_utri, axis=1),
            cp.max(P_D_utri, axis=2),
        )  # (N, d)
        if N > 1:
            P_E_abs = cp.abs(P_E)
            cp.maximum(col_norms[:N-1], cp.max(P_E_abs, axis=2), out=col_norms[:N-1])
            cp.maximum(col_norms[1:N], cp.max(P_E_abs, axis=1), out=col_norms[1:N])

        gamma = float(cp.mean(col_norms))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = max(gamma, float(cp.max(cp.abs(data._c))))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = 1.0 / gamma

        P_D *= gamma
        P_E *= gamma
        data._c *= gamma
        self._cost_scaling *= gamma

    # ------------------------------------------------------------------
    # Block-matrix primitives (single-problem)
    # ------------------------------------------------------------------

    @staticmethod
    def _tridiag_row_inf_norms(P: BlockTridiagMat, out: cp.ndarray):
        """Row inf-norms of a block-tridiagonal matrix. out shape: (n,)."""
        N, d = P.num_diag_blocks, P.block_size
        P_D = cp.from_dlpack(wp.to_dlpack(P.diag_blocks.data))  # (N, d, d)
        out[:] = cp.linalg.norm(P_D.reshape(-1, d), ord=cp.inf, axis=1)

        if N > 1:
            P_E = cp.from_dlpack(wp.to_dlpack(P.off_diag_blocks_lower.data))  # (N-1, d, d)
            cp.maximum(out[d:], cp.linalg.norm(P_E.reshape(-1, d), ord=cp.inf, axis=1), out=out[d:])
            cp.maximum(out[:-d], cp.linalg.norm(P_E.transpose(0, 2, 1).reshape(-1, d), ord=cp.inf, axis=1), out=out[:-d])

    @staticmethod
    def _bidiag_row_inf_norms(mat: BlockBidiagMat, out: cp.ndarray):
        """Row inf-norms of a block lower-bidiagonal matrix. out shape: ((N+1)*r,)."""
        N = mat.N
        r = mat.rows_of_blocks
        c = mat.cols_of_blocks
        D = cp.from_dlpack(wp.to_dlpack(mat.D))
        E = cp.from_dlpack(wp.to_dlpack(mat.E))
        row_D = cp.linalg.norm(D.reshape(-1, c), ord=cp.inf, axis=1).reshape(N, r)
        row_E = cp.linalg.norm(E.reshape(-1, c), ord=cp.inf, axis=1).reshape(N, r)

        out_2d = out.reshape(N + 1, r)
        out_2d[0] = row_D[0]
        if N > 1:
            cp.maximum(row_D[1:], row_E[:N-1], out=out_2d[1:N])
        out_2d[N] = row_E[N - 1]

    @staticmethod
    def _bidiag_col_inf_norms(mat: BlockBidiagMat, out: cp.ndarray):
        """Column inf-norms of a block lower-bidiagonal matrix. out shape: (N*c,)."""
        r = mat.rows_of_blocks
        D = cp.from_dlpack(wp.to_dlpack(mat.D))
        E = cp.from_dlpack(wp.to_dlpack(mat.E))
        col_norms_D = cp.linalg.norm(D.reshape(r, -1), ord=cp.inf, axis=0)
        col_norms_E = cp.linalg.norm(E.reshape(r, -1), ord=cp.inf, axis=0)
        cp.maximum(col_norms_D, col_norms_E, out=out)
