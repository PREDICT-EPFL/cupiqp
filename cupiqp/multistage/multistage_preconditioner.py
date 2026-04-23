import cupy as cp
import warp as wp

from ..data import Data
from ..preconditioner import RuizEquilibration
from .multistage_utils import BlockTridiagMat, BlockBidiagMat

class MultistageRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for multistage backend."""

    def eval_P_row_inf_norms(self, P: BlockTridiagMat, out: cp.ndarray):
        self._tridiag_row_inf_norms(P, out)
    
    def eval_A_row_inf_norms(self, A: BlockBidiagMat, out: cp.ndarray):
        self._bidiag_row_inf_norms(A, out)

    def eval_A_col_inf_norms(self, A: BlockBidiagMat, out: cp.ndarray):
        self._bidiag_col_inf_norms(A, out)

    def eval_G_row_inf_norms(self, G: BlockBidiagMat, out: cp.ndarray):
        self._bidiag_row_inf_norms(G, out)

    def eval_G_col_inf_norms(self, G: BlockBidiagMat, out: cp.ndarray):
        self._bidiag_col_inf_norms(G, out)

    @staticmethod
    def _tridiag_row_inf_norms(P: BlockTridiagMat, out: cp.ndarray):
        """Row inf-norms of a block-tridiagonal matrix."""
        N, d = P.num_diag_blocks, P.block_size
        P_D = cp.from_dlpack(wp.to_dlpack(P.diag_blocks.data))  # (N, d, d)
        # NOTE: we assume P_D stores the full diagonal blocks, not just the upper-triangular part. 
        out[:] = cp.linalg.norm(P_D.reshape(-1, d), ord=cp.inf, axis=1)

        if N > 1:
            P_E = cp.from_dlpack(wp.to_dlpack(P.off_diag_blocks_lower.data))  # (N-1, d, d)
            # lower diagonal blocks
            cp.maximum(out[d:], cp.linalg.norm(P_E.reshape(-1, d), ord=cp.inf, axis=1), out=out[d:])
            # upper diagonal blocks (transpose)
            cp.maximum(out[:-d], cp.linalg.norm(P_E.transpose(0, 2, 1).reshape(-1, d), ord=cp.inf, axis=1), out=out[:-d])

    @staticmethod
    def _bidiag_row_inf_norms(mat: BlockBidiagMat, out: cp.ndarray):
        """Row inf-norms of a block lower-bidiagonal matrix."""
        N = mat.N
        r = mat.rows_of_blocks
        c = mat.cols_of_blocks
        D = cp.from_dlpack(wp.to_dlpack(mat.D))   # (N, r, c)
        E = cp.from_dlpack(wp.to_dlpack(mat.E))   # (N, r, c)
        row_D = cp.linalg.norm(D.reshape(-1, c), ord=cp.inf, axis=1).reshape(N, r)  # (N, r)
        row_E = cp.linalg.norm(E.reshape(-1, c), ord=cp.inf, axis=1).reshape(N, r)  # (N, r)

        out_2d = out.reshape(N + 1, r)
        out_2d[0] = row_D[0]
        if N > 1:
            cp.maximum(row_D[1:], row_E[:N-1], out=out_2d[1:N])
        out_2d[N] = row_E[N - 1]

    def _bidiag_col_inf_norms(self, mat: BlockBidiagMat, out: cp.ndarray):
        """Column inf-norms of a block lower-bidiagonal matrix."""
        r = mat.rows_of_blocks
        D = cp.from_dlpack(wp.to_dlpack(mat.D))   # (N, r, c)
        E = cp.from_dlpack(wp.to_dlpack(mat.E))   # (N, r, c)
        col_norms_D = cp.linalg.norm(D.reshape(r, -1), ord=cp.inf, axis=0)          # (N, c)
        col_norms_E = cp.linalg.norm(E.reshape(r, -1), ord=cp.inf, axis=0)          # (N, c)
        cp.maximum(col_norms_D, col_norms_E, out=out)

    def _scale_matrices(self, data: Data,
                        d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray):
        N = data.num_blocks
        bs = data.block_size
        d_x_2d = d_x.reshape(N, bs)

        P_D = cp.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = cp.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))
        P_D *= d_x_2d[:, None, :]   # scale columns
        P_D *= d_x_2d[:, :, None]   # scale rows
        if N > 1:
            P_E *= d_x_2d[:N-1, None, :]   # scale columns (block-col k)
            P_E *= d_x_2d[1:N, :, None]    # scale rows (block-row k+1)

        data._c *= d_x

        if self.p > 0:
            r_a = data._A.rows_of_blocks
            d_y_2d = d_y.reshape(N + 1, r_a)
            A_D = cp.from_dlpack(wp.to_dlpack(data._A.D))
            A_E = cp.from_dlpack(wp.to_dlpack(data._A.E))
            A_D *= d_x_2d[:, None, :]      # scale columns
            A_D *= d_y_2d[:N, :, None]     # scale rows
            A_E *= d_x_2d[:, None, :]      # scale columns (block-col k)
            A_E *= d_y_2d[1:N+1, :, None]  # scale rows (block-row k+1)

        if self.m > 0:
            r_g = data._G.rows_of_blocks
            d_z_2d = d_z.reshape(N + 1, r_g)
            G_D = cp.from_dlpack(wp.to_dlpack(data._G.D))
            G_E = cp.from_dlpack(wp.to_dlpack(data._G.E))
            G_D *= d_x_2d[:, None, :]
            G_D *= d_z_2d[:N, :, None]
            G_E *= d_x_2d[:, None, :]
            G_E *= d_z_2d[1:N+1, :, None]

    def _apply_cost_scaling(self, data: Data):
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
        self.cost_scaling *= gamma

    def _unscale_matrices(self, data: Data,
                          d_x_inv: cp.ndarray, d_y_inv: cp.ndarray, d_z_inv: cp.ndarray):
        cost_inv = float(self._cost_scaling_inv)
        N = data.num_blocks
        bs = data.block_size
        d_x_inv_2d = d_x_inv.reshape(N, bs)

        P_D = cp.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = cp.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))
        P_D *= cost_inv
        P_D *= d_x_inv_2d[:, None, :]
        P_D *= d_x_inv_2d[:, :, None]
        if N > 1:
            P_E *= cost_inv
            P_E *= d_x_inv_2d[:N-1, None, :]
            P_E *= d_x_inv_2d[1:N, :, None]

        data._c *= cost_inv * d_x_inv

        if self.p > 0:
            r_a = data._A.rows_of_blocks
            d_y_inv_2d = d_y_inv.reshape(N + 1, r_a)
            A_D = cp.from_dlpack(wp.to_dlpack(data._A.D))
            A_E = cp.from_dlpack(wp.to_dlpack(data._A.E))
            A_D *= d_x_inv_2d[:, None, :]
            A_D *= d_y_inv_2d[:N, :, None]
            A_E *= d_x_inv_2d[:, None, :]
            A_E *= d_y_inv_2d[1:N+1, :, None]

        if self.m > 0:
            r_g = data._G.rows_of_blocks
            d_z_inv_2d = d_z_inv.reshape(N + 1, r_g)
            G_D = cp.from_dlpack(wp.to_dlpack(data._G.D))
            G_E = cp.from_dlpack(wp.to_dlpack(data._G.E))
            G_D *= d_x_inv_2d[:, None, :]
            G_D *= d_z_inv_2d[:N, :, None]
            G_E *= d_x_inv_2d[:, None, :]
            G_E *= d_z_inv_2d[1:N+1, :, None]

    def _apply_stored_scaling(self, data: Data,
                              d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray):
        c = float(self.cost_scaling)
        N = data.num_blocks
        bs = data.block_size
        d_x_2d = d_x.reshape(N, bs)

        P_D = cp.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = cp.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))
        P_D *= c
        P_D *= d_x_2d[:, None, :]
        P_D *= d_x_2d[:, :, None]
        if N > 1:
            P_E *= c
            P_E *= d_x_2d[:N-1, None, :]
            P_E *= d_x_2d[1:N, :, None]

        data._c *= c * d_x

        if self.p > 0:
            r_a = data._A.rows_of_blocks
            d_y_2d = d_y.reshape(N + 1, r_a)
            A_D = cp.from_dlpack(wp.to_dlpack(data._A.D))
            A_E = cp.from_dlpack(wp.to_dlpack(data._A.E))
            A_D *= d_x_2d[:, None, :]
            A_D *= d_y_2d[:N, :, None]
            A_E *= d_x_2d[:, None, :]
            A_E *= d_y_2d[1:N+1, :, None]

        if self.m > 0:
            r_g = data._G.rows_of_blocks
            d_z_2d = d_z.reshape(N + 1, r_g)
            G_D = cp.from_dlpack(wp.to_dlpack(data._G.D))
            G_E = cp.from_dlpack(wp.to_dlpack(data._G.E))
            G_D *= d_x_2d[:, None, :]
            G_D *= d_z_2d[:N, :, None]
            G_E *= d_x_2d[:, None, :]
            G_E *= d_z_2d[1:N+1, :, None]
