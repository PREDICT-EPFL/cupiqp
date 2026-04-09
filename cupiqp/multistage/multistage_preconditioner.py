import torch
import warp as wp

from ..data import Data
from ..preconditioner import RuizEquilibration
from .multistage_utils import BlockTridiagMat, BlockBidiagMat

class MultistageRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for multistage backend."""

    def eval_P_row_inf_norms(self, P: BlockTridiagMat, out: torch.Tensor):
        self._tridiag_row_inf_norms(P, out)

    def eval_A_row_inf_norms(self, A: BlockBidiagMat, out: torch.Tensor):
        self._bidiag_row_inf_norms(A, out)

    def eval_A_col_inf_norms(self, A: BlockBidiagMat, out: torch.Tensor):
        self._bidiag_col_inf_norms(A, out)

    def eval_G_row_inf_norms(self, G: BlockBidiagMat, out: torch.Tensor):
        self._bidiag_row_inf_norms(G, out)

    def eval_G_col_inf_norms(self, G: BlockBidiagMat, out: torch.Tensor):
        self._bidiag_col_inf_norms(G, out)

    @staticmethod
    def _tridiag_row_inf_norms(P: BlockTridiagMat, out: torch.Tensor):
        """Row inf-norms of a block-tridiagonal matrix."""
        N, d = P.num_diag_blocks, P.block_size
        P_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(P.diag_blocks.data))  # (N, d, d)
        # NOTE: we assume P_D stores the full diagonal blocks, not just the upper-triangular part.
        out[:] = torch.linalg.norm(P_D.reshape(-1, d), ord=float('inf'), dim=1)

        if N > 1:
            P_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(P.off_diag_blocks_lower.data))  # (N-1, d, d)
            # lower diagonal blocks
            torch.maximum(out[d:], torch.linalg.norm(P_E.reshape(-1, d), ord=float('inf'), dim=1), out=out[d:])
            # upper diagonal blocks (transpose)
            torch.maximum(out[:-d], torch.linalg.norm(P_E.transpose(1, 2).reshape(-1, d), ord=float('inf'), dim=1), out=out[:-d])

    @staticmethod
    def _bidiag_row_inf_norms(mat: BlockBidiagMat, out: torch.Tensor):
        """Row inf-norms of a block lower-bidiagonal matrix."""
        N = mat.N
        r = mat.rows_of_blocks
        c = mat.cols_of_blocks
        D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(mat.D))   # (N, r, c)
        E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(mat.E))   # (N, r, c)
        row_D = torch.linalg.norm(D.reshape(-1, c), ord=float('inf'), dim=1).reshape(N, r)  # (N, r)
        row_E = torch.linalg.norm(E.reshape(-1, c), ord=float('inf'), dim=1).reshape(N, r)  # (N, r)

        out_2d = out.reshape(N + 1, r)
        out_2d[0] = row_D[0]
        if N > 1:
            torch.maximum(row_D[1:], row_E[:N-1], out=out_2d[1:N])
        out_2d[N] = row_E[N - 1]

    def _bidiag_col_inf_norms(self, mat: BlockBidiagMat, out: torch.Tensor):
        """Column inf-norms of a block lower-bidiagonal matrix."""
        r = mat.rows_of_blocks
        D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(mat.D))   # (N, r, c)
        E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(mat.E))   # (N, r, c)
        col_norms_D = torch.linalg.norm(D.reshape(r, -1), ord=float('inf'), dim=0)          # (N, c)
        col_norms_E = torch.linalg.norm(E.reshape(r, -1), ord=float('inf'), dim=0)          # (N, c)
        torch.maximum(col_norms_D, col_norms_E, out=out)

    def _scale_matrices(self, data: Data,
                        d_x: torch.Tensor, d_y: torch.Tensor, d_z: torch.Tensor):
        N = data.num_blocks
        bs = data.block_size
        d_x_2d = d_x.reshape(N, bs)

        P_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))
        P_D *= d_x_2d[:, None, :]   # scale columns
        P_D *= d_x_2d[:, :, None]   # scale rows
        if N > 1:
            P_E *= d_x_2d[:N-1, None, :]   # scale columns (block-col k)
            P_E *= d_x_2d[1:N, :, None]    # scale rows (block-row k+1)

        data._c *= d_x

        if self.p > 0:
            r_a = data._A.rows_of_blocks
            d_y_2d = d_y.reshape(N + 1, r_a)
            A_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._A.D))
            A_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._A.E))
            A_D *= d_x_2d[:, None, :]      # scale columns
            A_D *= d_y_2d[:N, :, None]     # scale rows
            A_E *= d_x_2d[:, None, :]      # scale columns (block-col k)
            A_E *= d_y_2d[1:N+1, :, None]  # scale rows (block-row k+1)

        if self.m > 0:
            r_g = data._G.rows_of_blocks
            d_z_2d = d_z.reshape(N + 1, r_g)
            G_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._G.D))
            G_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._G.E))
            G_D *= d_x_2d[:, None, :]
            G_D *= d_z_2d[:N, :, None]
            G_E *= d_x_2d[:, None, :]
            G_E *= d_z_2d[1:N+1, :, None]

    def _apply_cost_scaling(self, data: Data):
        N = data.num_blocks
        P_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))

        # Column inf-norms of upper-triangular P (symmetric => col_norm == row_norm)
        P_D_abs = torch.abs(P_D)
        P_D_utri = torch.stack([torch.triu(P_D_abs[k]) for k in range(N)])
        col_norms = torch.maximum(
            torch.max(P_D_utri, dim=1).values,
            torch.max(P_D_utri, dim=2).values,
        )  # (N, d)
        if N > 1:
            P_E_abs = torch.abs(P_E)
            torch.maximum(col_norms[:N-1], torch.max(P_E_abs, dim=2).values, out=col_norms[:N-1])
            torch.maximum(col_norms[1:N], torch.max(P_E_abs, dim=1).values, out=col_norms[1:N])

        gamma = float(torch.mean(col_norms))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = max(gamma, float(torch.max(torch.abs(data._c))))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = 1.0 / gamma

        P_D *= gamma
        P_E *= gamma
        data._c *= gamma
        self.c_scaling *= gamma

    def _unscale_matrices(self, data: Data,
                          d_x_inv: torch.Tensor, d_y_inv: torch.Tensor, d_z_inv: torch.Tensor):
        c_inv = float(self._c_scaling_inv)
        N = data.num_blocks
        bs = data.block_size
        d_x_inv_2d = d_x_inv.reshape(N, bs)

        P_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))
        P_D *= c_inv
        P_D *= d_x_inv_2d[:, None, :]
        P_D *= d_x_inv_2d[:, :, None]
        if N > 1:
            P_E *= c_inv
            P_E *= d_x_inv_2d[:N-1, None, :]
            P_E *= d_x_inv_2d[1:N, :, None]

        data._c *= c_inv * d_x_inv

        if self.p > 0:
            r_a = data._A.rows_of_blocks
            d_y_inv_2d = d_y_inv.reshape(N + 1, r_a)
            A_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._A.D))
            A_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._A.E))
            A_D *= d_x_inv_2d[:, None, :]
            A_D *= d_y_inv_2d[:N, :, None]
            A_E *= d_x_inv_2d[:, None, :]
            A_E *= d_y_inv_2d[1:N+1, :, None]

        if self.m > 0:
            r_g = data._G.rows_of_blocks
            d_z_inv_2d = d_z_inv.reshape(N + 1, r_g)
            G_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._G.D))
            G_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._G.E))
            G_D *= d_x_inv_2d[:, None, :]
            G_D *= d_z_inv_2d[:N, :, None]
            G_E *= d_x_inv_2d[:, None, :]
            G_E *= d_z_inv_2d[1:N+1, :, None]

    def _apply_stored_scaling(self, data: Data,
                              d_x: torch.Tensor, d_y: torch.Tensor, d_z: torch.Tensor):
        c = float(self.c_scaling)
        N = data.num_blocks
        bs = data.block_size
        d_x_2d = d_x.reshape(N, bs)

        P_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._P.diag_blocks.data))
        P_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._P.off_diag_blocks_lower.data))
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
            A_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._A.D))
            A_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._A.E))
            A_D *= d_x_2d[:, None, :]
            A_D *= d_y_2d[:N, :, None]
            A_E *= d_x_2d[:, None, :]
            A_E *= d_y_2d[1:N+1, :, None]

        if self.m > 0:
            r_g = data._G.rows_of_blocks
            d_z_2d = d_z.reshape(N + 1, r_g)
            G_D = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._G.D))
            G_E = torch.utils.dlpack.from_dlpack(wp.to_dlpack(data._G.E))
            G_D *= d_x_2d[:, None, :]
            G_D *= d_z_2d[:N, :, None]
            G_E *= d_x_2d[:, None, :]
            G_E *= d_z_2d[1:N+1, :, None]
