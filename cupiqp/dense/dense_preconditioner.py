import torch

from ..data import Data
from ..preconditioner import RuizEquilibration


class DenseRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for dense matrix backends."""

    def eval_P_row_inf_norms(self, P: torch.Tensor, out: torch.Tensor):
        out[:] = torch.linalg.norm(P, ord=float('inf'), dim=1)

    def eval_A_row_inf_norms(self, A: torch.Tensor, out: torch.Tensor):
        out[:] = torch.linalg.norm(A, ord=float('inf'), dim=1)

    def eval_A_col_inf_norms(self, A: torch.Tensor, out: torch.Tensor):
        out[:] = torch.linalg.norm(A, ord=float('inf'), dim=0)

    def eval_G_row_inf_norms(self, G: torch.Tensor, out: torch.Tensor):
        out[:] = torch.linalg.norm(G, ord=float('inf'), dim=1)

    def eval_G_col_inf_norms(self, G: torch.Tensor, out: torch.Tensor):
        out[:] = torch.linalg.norm(G, ord=float('inf'), dim=0)

    def _scale_matrices(self, data: Data,
                        d_x: torch.Tensor, d_y: torch.Tensor, d_z: torch.Tensor):
        data._P *= d_x[None, :]
        data._P *= d_x[:, None]
        data._c *= d_x

        if self.p > 0:
            data._A *= d_x[None, :]
            data._A *= d_y[:, None]

        if self.m > 0:
            data._G *= d_x[None, :]
            data._G *= d_z[:, None]

    def _apply_cost_scaling(self, data: Data):
        P_abs = torch.abs(data._P)
        P_utri = torch.triu(P_abs)
        gamma = float(torch.mean(torch.maximum(torch.max(P_utri, dim=0).values, torch.max(P_utri, dim=1).values)))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = max(gamma, float(torch.max(torch.abs(data._c))))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = 1.0 / gamma
        data._P *= gamma
        data._c *= gamma
        self.c_scaling *= gamma

    def _unscale_matrices(self, data: Data,
                          d_x_inv: torch.Tensor, d_y_inv: torch.Tensor, d_z_inv: torch.Tensor):
        c_inv = self._c_scaling_inv

        data._P *= c_inv
        data._P *= d_x_inv[None, :]
        data._P *= d_x_inv[:, None]
        data._c *= float(c_inv) * d_x_inv

        if self.p > 0:
            data._A *= d_x_inv[None, :]
            data._A *= d_y_inv[:, None]
        if self.m > 0:
            data._G *= d_x_inv[None, :]
            data._G *= d_z_inv[:, None]

    def _apply_stored_scaling(self, data: Data,
                              d_x: torch.Tensor, d_y: torch.Tensor, d_z: torch.Tensor):
        c = self.c_scaling

        data._P *= c
        data._P *= d_x[None, :]
        data._P *= d_x[:, None]
        data._c *= float(c) * d_x

        if self.p > 0:
            data._A *= d_x[None, :]
            data._A *= d_y[:, None]
        if self.m > 0:
            data._G *= d_x[None, :]
            data._G *= d_z[:, None]
