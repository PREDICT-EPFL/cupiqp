import cupy as cp

from ..data import Data
from ..preconditioner import RuizEquilibration


class DenseRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for dense matrix backends."""

    def eval_P_row_inf_norms(self, P: cp.ndarray, out: cp.ndarray):
        out[:] = cp.linalg.norm(P, ord=cp.inf, axis=1)
    
    def eval_A_row_inf_norms(self, A: cp.ndarray, out: cp.ndarray):
        out[:] = cp.linalg.norm(A, ord=cp.inf, axis=1)
    
    def eval_A_col_inf_norms(self, A: cp.ndarray, out: cp.ndarray):
        out[:] = cp.linalg.norm(A, ord=cp.inf, axis=0)
    
    def eval_G_row_inf_norms(self, G: cp.ndarray, out: cp.ndarray):
        out[:] = cp.linalg.norm(G, ord=cp.inf, axis=1)
    
    def eval_G_col_inf_norms(self, G: cp.ndarray, out: cp.ndarray):
        out[:] = cp.linalg.norm(G, ord=cp.inf, axis=0)

    def _scale_matrices(self, data: Data,
                        d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray):
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
        P_abs = cp.abs(data._P)
        P_utri = cp.triu(P_abs)
        gamma = float(cp.mean(cp.maximum(cp.max(P_utri, axis=0), cp.max(P_utri, axis=1))))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = max(gamma, float(cp.max(cp.abs(data._c))))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = 1.0 / gamma
        data._P *= gamma
        data._c *= gamma
        self.c_scaling *= gamma

    def _unscale_matrices(self, data: Data,
                          d_x_inv: cp.ndarray, d_y_inv: cp.ndarray, d_z_inv: cp.ndarray):
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
                              d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray):
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
