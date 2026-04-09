from abc import ABC, abstractmethod
import nvtx
import torch

from .data import Data
from .results import Variables


class RuizEquilibration(ABC):
    """Ruiz equilibration preconditioner for QP problems.

    Iteratively scale the following matrix so that each row/column has inf-norm close to 1:

        K = [ P    A'   G'   D_b ]
            [ A    0    0    0   ]
            [ G    0    0    0   ]
            [ D_b  0    0    0   ]

    where D_b = diag(x_b_scaling) is the box constraint block, with entries
    initialized to 1 for bounded variables and 0 for unbounded ones.

    The algorithm iterates:
        1. Compute inf-norm of each row/column of K:
           - d_x[i] = max(||P_col_i||, ||A_col_i||, ||G_col_i||, x_b_scaling[i])
           - d_y[j] = ||A_row_j||,  d_z[l] = ||G_row_l||
           - d_b[i] = x_b_scaling[i]
        2. Clamp to [MIN_SCALING, MAX_SCALING], then d <- 1/sqrt(d)
        3. Scale: P <- D_x P D_x,  A <- D_y A D_x,  G <- D_z G D_x,  c <- D_x c
        4. Update box scaling: x_b_scaling *= d_b * d_x
        5. Accumulate: delta *= d,  delta_b *= d_b
        6. (Optional) Cost scaling: gamma = 1/max(mean(||P_cols||), ||c||),
           then P *= gamma, c *= gamma, c_scaling *= gamma
        7. Converge when max(||1 - d||_inf, ||1 - d_b||_inf) < 1e-3

    After convergence, bounds are scaled: b *= d_y, h *= d_z, x_l/x_u *= delta_b.

    Solution unscaling recovers original coordinates:
        x_orig = delta_x * x_scaled
        y_orig = c_inv * delta_y * y_scaled
        z_orig = c_inv * delta_z * z_scaled
        z_b_orig = c_inv * delta_b * z_b_scaled
    """

    def __init__(self, n: int, p: int, m: int,
                 idx_xl: torch.Tensor,
                 idx_xu: torch.Tensor,
                 min_scaling: float = 1e-4,
                 max_scaling: float = 1e4,
                 convergence_tol: float = 1e-3,
                 max_iter: int = 10
                 ):
        self.n = n
        self.p = p
        self.m = m
        self.max_iter = max_iter
        self.min_scaling = min_scaling
        self.max_scaling = max_scaling
        self.convergence_tol = convergence_tol

        # Combined scaling: delta[0:n] for x, delta[n:n+p] for y, delta[n+p:] for z
        self._delta = torch.ones(n + p + m, dtype=torch.float64, device='cuda')
        self._delta_inv = torch.ones(n + p + m, dtype=torch.float64, device='cuda')

        # Box constraint scaling (accumulated product of delta_b_iter across Ruiz iterations)
        self._delta_b = torch.ones(n, dtype=torch.float64, device='cuda')
        self._delta_b_inv = torch.ones(n, dtype=torch.float64, device='cuda')

        # Cost scaling (scalar stored as 1-element array)
        self._c_scaling = torch.ones(1, dtype=torch.float64, device='cuda')
        self._c_scaling_inv = torch.ones(1, dtype=torch.float64, device='cuda')

        # x_b_scaling: starts at {0,1} (1 for bounded variables, 0 otherwise)
        # and evolves as x_b_scaling *= delta_b_iter * delta_x each Ruiz iteration.
        # This tracks the diagonal of the box constraint block in the KKT matrix.
        self._x_b_scaling_init = torch.zeros(n, dtype=torch.float64, device='cuda')
        bounded = torch.zeros(n, dtype=torch.bool, device='cuda')
        if idx_xl.numel() > 0:
            bounded[idx_xl.long()] = True
        if idx_xu.numel() > 0:
            bounded[idx_xu.long()] = True
        self._x_b_scaling_init[bounded] = 1.0
        self._x_b_scaling = self._x_b_scaling_init.clone()

        self._delta_iter = torch.empty(n + p + m, dtype=torch.float64, device='cuda')  # used to store current Ruiz iteration scaling factors
        self._delta_b_iter = torch.empty(n, dtype=torch.float64, device='cuda') # used to store current Ruiz iteration box scaling factors

        self._work_n = torch.empty(n, dtype=torch.float64, device='cuda')  # temp workspace of size n

    @property
    def c_scaling_inv(self) -> torch.Tensor:
        return self._c_scaling_inv

    @property
    def delta(self) -> torch.Tensor:
        return self._delta

    @property
    def delta_inv(self) -> torch.Tensor:
        return self._delta_inv

    @property
    def delta_b(self) -> torch.Tensor:
        return self._delta_b

    @property
    def delta_b_inv(self) -> torch.Tensor:
        return self._delta_b_inv

    @property
    def x_b_scaling(self) -> torch.Tensor:
        return self._x_b_scaling

    def reset(self):
        self._delta.fill_(1.0)
        self._delta_inv.fill_(1.0)
        self._delta_b.fill_(1.0)
        self._delta_b_inv.fill_(1.0)
        self._c_scaling.fill_(1.0)
        self._c_scaling_inv.fill_(1.0)
        self._x_b_scaling.copy_(self._x_b_scaling_init)

    @nvtx.annotate("RuizEquilibration::scale_data")
    def scale_data(self, data: Data, scale_cost: bool, max_iter: int):
        """Run Ruiz equilibration iterations to scale the problem data."""
        n, p = self.n, self.p

        for _ in range(max_iter):
            # Compute KKT row/column inf-norms
            self._compute_kkt_norms(data, self._delta_iter, self._delta_b_iter)
            torch.maximum(self._delta_iter[:n], self._x_b_scaling, out=self._delta_iter[:n])
            self._delta_b_iter[:] = self._x_b_scaling

            self._limit_scaling(self._delta_iter)
            self._limit_scaling(self._delta_b_iter)
            torch.sqrt(self._delta_iter, out=self._delta_iter)
            torch.reciprocal(self._delta_iter, out=self._delta_iter)
            torch.sqrt(self._delta_b_iter, out=self._delta_b_iter)
            torch.reciprocal(self._delta_b_iter, out=self._delta_b_iter)

            d_x = self._delta_iter[:n]
            d_y = self._delta_iter[n:n+p]
            d_z = self._delta_iter[n+p:]

            # Apply scaling
            self._scale_matrices(data, d_x, d_y, d_z)
            self._x_b_scaling *= self._delta_b_iter * d_x

            # Accumulate
            self._delta *= self._delta_iter
            self._delta_b *= self._delta_b_iter

            if scale_cost:
                self._apply_cost_scaling(data)

            # Check convergence
            conv = max(
                float(torch.max(torch.abs(1.0 - self._delta_iter))),
                float(torch.max(torch.abs(1.0 - self._delta_b_iter)))
            )
            if conv < self.convergence_tol:
                break

        # Compute inverses
        torch.reciprocal(self._delta, out=self._delta_inv)
        torch.reciprocal(self._delta_b, out=self._delta_b_inv)
        torch.reciprocal(self._c_scaling, out=self._c_scaling_inv)

        # Write x_b_scaling to data for use by KKT system and solver
        data._x_b_scaling.copy_(self._x_b_scaling)

        # Scale bounds
        self._scale_bounds(data)

    @nvtx.annotate("RuizEquilibration::unscale_data")
    def unscale_data(self, data: Data):
        """Reverse all scaling transformations on the problem data."""
        n, p = self.n, self.p
        d_x_inv = self._delta_inv[:n]
        d_y_inv = self._delta_inv[n:n+p]
        d_z_inv = self._delta_inv[n+p:]

        self._unscale_matrices(data, d_x_inv, d_y_inv, d_z_inv)
        self._unscale_bounds(data)
        data._x_b_scaling.fill_(1.0)
        self._x_b_scaling *= self._delta_b_inv * d_x_inv
        self.reset()

    @nvtx.annotate("RuizEquilibration::apply_scaling")
    def apply_scaling(self, data: Data):
        """Re-apply stored scaling to fresh (unscaled) data."""
        n, p = self.n, self.p
        d_x = self._delta[:n]
        d_y = self._delta[n:n + p]
        d_z = self._delta[n + p:]

        self._apply_stored_scaling(data, d_x, d_y, d_z)

        data._x_b_scaling.copy_(self._x_b_scaling)
        self._scale_bounds(data)

    @nvtx.annotate("RuizEquilibration::unscale_solution")
    def unscale_solution(self, result: Variables, data: Data):
        """Transform scaled IPM solution back to original coordinates.
        Matches C++ PIQP SolverBase::unscale_results()."""
        result.x[:] = self.unscale_primal(result.x)

        if self.p > 0:
            result.y[:] = self.unscale_dual_eq(result.y)

        if data.num_hu > 0:
            result.z_u[:] = self.unscale_dual_ineq(result.z_u, data.idx_hu)
            result.s_u[:] = self.unscale_slack_ineq(result.s_u, data.idx_hu)
        if data.num_hl > 0:
            result.z_l[:] = self.unscale_dual_ineq(result.z_l, data.idx_hl)
            result.s_l[:] = self.unscale_slack_ineq(result.s_l, data.idx_hl)
        if data.num_xu > 0:
            result.z_bu[:] = self.unscale_dual_b(result.z_bu, data.idx_xu)
            result.s_bu[:] = self.unscale_slack_b(result.s_bu, data.idx_xu)
        if data.num_xl > 0:
            result.z_bl[:] = self.unscale_dual_b(result.z_bl, data.idx_xl)
            result.s_bl[:] = self.unscale_slack_b(result.s_bl, data.idx_xl)

    # ------------------------------------------------------------------
    # Primal / dual / slack scaling and unscaling
    # ------------------------------------------------------------------

    def unscale_primal(self, x: torch.Tensor) -> torch.Tensor:
        """x_orig = delta_x * x_scaled"""
        return x * self._delta[:self.n]

    def scale_primal(self, x: torch.Tensor) -> torch.Tensor:
        """x_scaled = delta_inv_x * x_orig"""
        return x * self._delta_inv[:self.n]

    def unscale_dual_eq(self, y: torch.Tensor) -> torch.Tensor:
        """y_orig = c_inv * delta_y * y_scaled"""
        return y * self._c_scaling_inv * self._delta[self.n:self.n + self.p]

    def scale_dual_eq(self, y: torch.Tensor) -> torch.Tensor:
        """y_scaled = c * delta_inv_y * y_orig"""
        return y * self._c_scaling * self._delta_inv[self.n:self.n + self.p]

    def unscale_dual_ineq(self, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """z_orig = c_inv * delta_z[idx] * z_scaled"""
        return z * self._c_scaling_inv * self._delta[self.n + self.p + idx.long()]

    def unscale_dual_b(self, z_b: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """z_b_orig = c_inv * delta_b[idx] * z_b_scaled"""
        return z_b * self._c_scaling_inv * self._delta_b[idx.long()]

    def unscale_slack_ineq(self, s: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """s_orig = delta_inv_z[idx] * s_scaled"""
        return s * self._delta_inv[self.n + self.p + idx.long()]

    def unscale_slack_b(self, s_b: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """s_b_orig = delta_b_inv[idx] * s_b_scaled"""
        return s_b * self._delta_b_inv[idx.long()]

    # ------------------------------------------------------------------
    # Residual unscaling (used every iteration for convergence checks)
    # ------------------------------------------------------------------

    def unscale_dual_res(self, v: torch.Tensor) -> torch.Tensor:
        """v_orig = c_inv * delta_inv_x * v_scaled"""
        return v * self._c_scaling_inv * self._delta_inv[:self.n]

    def unscale_primal_res_eq(self, v: torch.Tensor) -> torch.Tensor:
        """v_orig = delta_inv_y * v_scaled"""
        return v * self._delta_inv[self.n:self.n + self.p]

    def unscale_primal_res_ineq(self, v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """v_orig = delta_inv_z[idx] * v_scaled"""
        return v * self._delta_inv[self.n + self.p + idx.long()]

    def unscale_primal_res_b(self, v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """v_orig = delta_b_inv[idx] * v_scaled"""
        return v * self._delta_b_inv[idx.long()]

    # ------------------------------------------------------------------
    # Cost unscaling
    # ------------------------------------------------------------------

    def unscale_cost(self, cost: float) -> float:
        """cost_orig = c_inv * cost_scaled"""
        return float(cost * self._c_scaling_inv)

    def _compute_kkt_norms(self, data: Data, d: torch.Tensor, d_b: torch.Tensor):
        """Compute inf-norms of each KKT row/column into d[0:n+p+m].

        d[:n]     = max over P columns, A columns, G columns
        d[n:n+p]  = A row norms
        d[n+p:]   = G row norms
        d_b is NOT set here (handled by the base class).
        """
        n, p, m = self.n, self.p, self.m
        self.eval_P_row_inf_norms(data.P, d[:n])
        if p > 0:
            self.eval_A_col_inf_norms(data.A, self._work_n)
            d[:n] = torch.maximum(d[:n], self._work_n)
            self.eval_A_row_inf_norms(data.A, d[n:n+p])
        if m > 0:
            self.eval_G_col_inf_norms(data.G, self._work_n)
            d[:n] = torch.maximum(d[:n], self._work_n)
            self.eval_G_row_inf_norms(data.G, d[n+p:n+p+m])

    @abstractmethod
    def eval_P_row_inf_norms(self, P, out: torch.Tensor):
        """Compute infinity norms of rows of P. The shape of P is (n, n). Return shape is (n,)."""
        pass

    @abstractmethod
    def eval_A_row_inf_norms(self, A, out: torch.Tensor):
        """Compute infinity norms of rows of A. The shape of A is (p, n). Return shape is (p,)."""
        pass

    @abstractmethod
    def eval_A_col_inf_norms(self, A, out: torch.Tensor):
        """Compute infinity norms of columns of A. The shape of A is (p, n). Return shape is (n,)."""
        pass

    @abstractmethod
    def eval_G_row_inf_norms(self, G, out: torch.Tensor):
        """Compute infinity norms of rows of G. The shape of G is (m, n). Return shape is (m,)."""
        pass

    @abstractmethod
    def eval_G_col_inf_norms(self, G, out: torch.Tensor):
        """Compute infinity norms of columns of G. The shape of G is (m, n). Return shape is (n,)."""
        pass

    @abstractmethod
    def _scale_matrices(self, data: Data,
                        d_x: torch.Tensor, d_y: torch.Tensor, d_z: torch.Tensor):
        """Apply one Ruiz iteration scaling: P = D_x*P*D_x, A = D_y*A*D_x, G = D_z*G*D_x."""
        pass

    @abstractmethod
    def _apply_cost_scaling(self, data: Data):
        """Compute gamma from P norms and ||c||, scale P and c by gamma to avoid the cost is dominated by P term or c term"""
        pass

    @abstractmethod
    def _unscale_matrices(self, data: Data,
                          d_x_inv: torch.Tensor, d_y_inv: torch.Tensor, d_z_inv: torch.Tensor):
        """Reverse all matrix scaling using stored inverses."""
        pass

    @abstractmethod
    def _apply_stored_scaling(self, data: Data,
                              d_x: torch.Tensor, d_y: torch.Tensor, d_z: torch.Tensor):
        """Re-apply stored scaling to fresh data (reuse_prev_scaling path)."""
        pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _limit_scaling(self, d: torch.Tensor):
        d[d < self.min_scaling] = 1.0
        torch.minimum(d, torch.tensor(self.max_scaling, dtype=d.dtype, device=d.device), out=d)

    def _limit_scaling_scalar(self, d: float) -> float:
        if d < self.min_scaling:
            return 1.0
        elif d > self.max_scaling:
            return self.max_scaling
        return d

    def _scale_bounds(self, data: Data):
        n, p, m = self.n, self.p, self.m
        d_y = self._delta[n:n + p]
        d_z = self._delta[n + p:]

        if p > 0:
            data._b *= d_y
        if m > 0:
            data._h_l *= d_z
            data._h_u *= d_z
        if data.num_xl > 0:
            data._x_l[data.idx_xl.long()] *= self._delta_b[data.idx_xl.long()]
        if data.num_xu > 0:
            data._x_u[data.idx_xu.long()] *= self._delta_b[data.idx_xu.long()]

    def _unscale_bounds(self, data: Data):
        n, p, m = self.n, self.p, self.m
        d_y_inv = self._delta_inv[n:n + p]
        d_z_inv = self._delta_inv[n + p:]

        if p > 0:
            data._b *= d_y_inv
        if m > 0:
            data._h_l *= d_z_inv
            data._h_u *= d_z_inv
        if data.num_xl > 0:
            data._x_l[data.idx_xl.long()] *= self._delta_b_inv[data.idx_xl.long()]
        if data.num_xu > 0:
            data._x_u[data.idx_xu.long()] *= self._delta_b_inv[data.idx_xu.long()]
