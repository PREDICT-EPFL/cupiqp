import cupy as cp

from ..preconditioner import RuizEquilibration


class DenseRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for dense matrix backends.

    All matrices are (B, rows, cols), all vectors are (B, k).
    """

    def eval_P_row_inf_norms(self, P: cp.ndarray, out: cp.ndarray):
        """P: (B, n, n), out: (B, n)."""
        cp.max(cp.abs(P), axis=2, out=out)

    def eval_A_row_inf_norms(self, A: cp.ndarray, out: cp.ndarray):
        """A: (B, p, n), out: (B, p)."""
        cp.max(cp.abs(A), axis=2, out=out)

    def eval_A_col_inf_norms(self, A: cp.ndarray, out: cp.ndarray):
        """A: (B, p, n), out: (B, n)."""
        cp.max(cp.abs(A), axis=1, out=out)

    def eval_G_row_inf_norms(self, G: cp.ndarray, out: cp.ndarray):
        """G: (B, m, n), out: (B, m)."""
        cp.max(cp.abs(G), axis=2, out=out)

    def eval_G_col_inf_norms(self, G: cp.ndarray, out: cp.ndarray):
        """G: (B, m, n), out: (B, n)."""
        cp.max(cp.abs(G), axis=1, out=out)

    def _scale_matrices(self, data,
                        d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray):
        """d_x: (B, n), d_y: (B, p), d_z: (B, m).  Data arrays are (B, *, *)."""
        # P: (B, n, n) *= (B, 1, n) then *= (B, n, 1)
        data._P *= d_x[:, None, :]
        data._P *= d_x[:, :, None]
        data._c *= d_x
        if self.p > 0:
            data._A *= d_x[:, None, :]
            data._A *= d_y[:, :, None]
        if self.m > 0:
            data._G *= d_x[:, None, :]
            data._G *= d_z[:, :, None]

    def _apply_cost_scaling(self, data):
        """Per-problem cost scaling gamma.  Stores (B,) in self.c_scaling."""
        P_abs = cp.abs(data._P)                          # (B, n, n)
        # triu per-batch: use einsum or just compute max along rows and cols
        P_utri = cp.triu(P_abs.reshape(-1, self.n, self.n)).reshape(data._P.shape)
        col_max = cp.max(P_utri, axis=1)                 # (B, n)
        row_max = cp.max(P_utri, axis=2)                 # (B, n)
        gamma = cp.mean(cp.maximum(col_max, row_max), axis=1)  # (B,)
        gamma = cp.clip(gamma, self.min_scaling, self.max_scaling)
        c_norm = cp.max(cp.abs(data._c), axis=1)         # (B,)
        gamma = cp.maximum(gamma, c_norm)
        gamma = cp.clip(gamma, self.min_scaling, self.max_scaling)
        gamma = 1.0 / gamma                              # (B,)
        data._P *= gamma[:, None, None]
        data._c *= gamma[:, None]
        self._c_scaling *= gamma

    def _unscale_matrices(self, data,
                          d_x_inv: cp.ndarray, d_y_inv: cp.ndarray, d_z_inv: cp.ndarray):
        c_inv = self._cost_scaling_inv[:, None, None]  # (B, 1, 1) for P, reuse as needed
        data._P *= c_inv
        data._P *= d_x_inv[:, None, :]
        data._P *= d_x_inv[:, :, None]
        data._c *= self._cost_scaling_inv[:, None] * d_x_inv
        if self.p > 0:
            data._A *= d_x_inv[:, None, :]
            data._A *= d_y_inv[:, :, None]
        if self.m > 0:
            data._G *= d_x_inv[:, None, :]
            data._G *= d_z_inv[:, :, None]

    def _apply_stored_scaling(self, data,
                              d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray):
        c = self._c_scaling
        data._P *= c[:, None, None]
        data._P *= d_x[:, None, :]
        data._P *= d_x[:, :, None]
        data._c *= c[:, None] * d_x
        if self.p > 0:
            data._A *= d_x[:, None, :]
            data._A *= d_y[:, :, None]
        if self.m > 0:
            data._G *= d_x[:, None, :]
            data._G *= d_z[:, :, None]
