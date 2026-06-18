from abc import ABC, abstractmethod
from typing import Optional
import nvtx
import cupy as cp
import warp as wp

from .data import Data
from .results import Variables
from .preconditioner_kernels import (
    create_clamp_and_rsqrt_kernel,
    create_accumulate_deltas_kernel,
    create_ruiz_conv_check_kernel,
    create_calc_scaling_inv_and_scale_bounds_kernel,
    create_scale_bounds_kernel,
    create_unscale_bounds_kernel,
    create_compute_constraints_rhs_inf_norm_unscaled_kernel,
)



class PreconditionerBase(ABC):
    """Abstract preconditioner interface for QP problems — batched.

    Defines only the operations the solver depends on. Concrete preconditioners
    (Ruiz equilibration, identity, block-Jacobi, ...) choose their own internal
    representation (diagonal vectors, sparse factors, ...).

    Contract with the solver:
        * scale_data / unscale_data / reuse_scaling — transform the problem data.
        * scale_primal / unscale_primal, scale_dual_eq / unscale_dual_eq, ...
          — map individual vectors between scaled and original coordinates.
        * unscale_solution — convenience that unscales a full Variables struct.
        * x_b_scaling, cost_scaling, cost_scaling_inv — state the KKT solver and
          solver read directly.
        * reset — restore to identity scaling.
    """

    # ------------------------------------------------------------------
    # Data-level scaling
    # ------------------------------------------------------------------

    @abstractmethod
    def scale_data(self, data: Data, *args, **kwargs):
        """Compute scalings from data and apply them to the problem matrices/vectors."""
        ...

    @abstractmethod
    def unscale_data(self, data: Data):
        """Reverse all scaling transformations on the problem data."""
        ...

    @abstractmethod
    def reuse_scaling(self, data: Data):
        """Re-apply stored scaling to fresh (unscaled) data."""
        ...

    @abstractmethod
    def reset(self):
        """Restore the preconditioner to identity (no) scaling."""
        ...

    # ------------------------------------------------------------------
    # State exposed to the solver / KKT system
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def cost_scaling(self) -> cp.ndarray:
        """(B,) scalar objective scaling."""
        ...

    @property
    @abstractmethod
    def cost_scaling_inv(self) -> cp.ndarray:
        """(B,) inverse of cost_scaling."""
        ...

    @property
    @abstractmethod
    def x_b_scaling(self) -> cp.ndarray:
        """(B, n) diagonal of the box block in the scaled KKT matrix.

        Zero for unbounded variables, nonzero for bounded ones.
        """
        ...

    # ------------------------------------------------------------------
    # Full-solution unscaling — generic, dispatches to per-component methods
    # ------------------------------------------------------------------

    @nvtx.annotate("Preconditioner::unscale_solution")
    def unscale_solution(self, result: Variables, data: Data):
        """Transform scaled IPM solution back to original coordinates, in place."""
        self.unscale_primal(result.x, out=result.x)

        if data.p > 0:
            self.unscale_dual_eq(result.y, out=result.y)

        if data.num_hu > 0:
            self.unscale_dual_ineq(result.z_u, out=result.z_u)
            self.unscale_slack_ineq(result.s_u, out=result.s_u)
        if data.num_hl > 0:
            self.unscale_dual_ineq(result.z_l, out=result.z_l)
            self.unscale_slack_ineq(result.s_l, out=result.s_l)
        if data.num_xu > 0:
            self.unscale_dual_b(result.z_bu, out=result.z_bu)
            self.unscale_slack_b(result.s_bu, out=result.s_bu)
        if data.num_xl > 0:
            self.unscale_dual_b(result.z_bl, out=result.z_bl)
            self.unscale_slack_b(result.s_bl, out=result.s_bl)

    # ------------------------------------------------------------------
    # Per-component scaling / unscaling (abstract).
    # ------------------------------------------------------------------

    @abstractmethod
    def unscale_primal(self, x: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def scale_primal(self, x: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def unscale_dual_eq(self, y: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def scale_dual_eq(self, y: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def unscale_dual_ineq(self, z: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def unscale_dual_b(self, z_b: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def unscale_slack_ineq(self, s: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def unscale_slack_b(self, s_b: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def unscale_dual_res(self, v: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def unscale_primal_res_eq(self, v: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def unscale_primal_res_ineq(self, v: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def unscale_primal_res_b(self, v: cp.ndarray, out: cp.ndarray): ...

    @abstractmethod
    def unscale_cost(self, cost: cp.ndarray, out: cp.ndarray): ...


class RuizEquilibration(PreconditionerBase):
    """Ruiz equilibration preconditioner for QP problems — batched.

    Diagonal preconditioner: stores (delta, delta_b, cost_scaling) as vectors
    and uses them multiplicatively in every scale/unscale operation.

    All scaling vectors carry a leading batch dimension ``(B, ...)``. For
    single problems, ``B = 1``.

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
           then P *= gamma, c *= gamma, cost_scaling *= gamma
        7. Converge when max(||1 - d||_inf, ||1 - d_b||_inf) < 1e-3

    After convergence, bounds are scaled: b *= d_y, h *= d_z, x_l/x_u *= delta_b.

    x_b_scaling = x_b_scaling_init * delta_b * delta_x

    Solution unscaling recovers original coordinates:
        x_orig   = delta_x * x_scaled
        y_orig   = c_inv * delta_y * y_scaled
        z_orig   = c_inv * delta_z * z_scaled
        z_b_orig = c_inv * delta_b * z_b_scaled
    """

    def __init__(self, B: int, n: int, p: int, m: int,
                 active_x_bound: Optional[cp.ndarray] = None,
                 min_scaling: float = 1e-4,
                 max_scaling: float = 1e4,
                 convergence_tol: float = 1e-3,
                 use_warp_tile_kernels: bool = True,
                 dtype=cp.float64,
                 ):
        self._use_warp_tile_kernels = use_warp_tile_kernels
        self.B = B
        self.n = n
        self.p = p
        self.m = m
        self._dtype = dtype

        self.min_scaling = min_scaling
        self.max_scaling = max_scaling
        self.convergence_tol = convergence_tol

        # Combined scaling: (B, n+p+m) — delta[:, :n] for x, delta[:, n:n+p] for y, etc.
        self._delta = cp.ones((B, n + p + m), dtype=dtype)
        self._delta_inv = cp.ones((B, n + p + m), dtype=dtype)

        # Box constraint scaling: (B, n)
        self._delta_b = cp.ones((B, n), dtype=dtype)
        self._delta_b_inv = cp.ones((B, n), dtype=dtype)

        # Cost scaling: (B,)
        self._cost_scaling = cp.ones(B, dtype=dtype)
        self._cost_scaling_inv = cp.ones(B, dtype=dtype)

        # Pre-combined residual unscaling factors — materialized once per
        # Ruiz update in _refresh_unscale_factors(). The solver's
        # update_residuals_r_kernel reads these directly so its stage-2
        # reductions stay pure tile_max(tile_map(_abs_mul, ...)) calls with
        # no in-kernel gather / cost_scaling_inv multiply.
        #
        #   dual_res_unscale_factor[b, i]   = cost_scaling_inv[b] * delta_inv[b, i]
        #                                     for i ∈ [0, n)
        #       — multiplies res.x (the dual residual on the x-block)
        #         element-wise to convert it back to original units.
        #
        #   primal_res_unscale_factor[b, j] = unscaling factor for res.duals_all
        #       — shape (B, num_duals); packed in Variables._dual_buffer
        #         order [y | z_l | z_u | z_bl | z_bu], with per-segment
        #         content (see _refresh_unscale_factors for the exact slices):
        #
        #             [y]:    delta_inv[:, n : n+p]
        #             [z_l]:  delta_inv[:, n + p : n + p + m]   (full-length)
        #             [z_u]:  delta_inv[:, n + p : n + p + m]   (full-length)
        #             [z_bl]: delta_b_inv[:, :n]                (full-length)
        #             [z_bu]: delta_b_inv[:, :n]                (full-length)
        num_duals = p + m + m + n + n
        self._dual_res_unscale_factor = cp.ones((B, n), dtype=dtype)
        self._primal_res_unscale_factor = cp.ones((B, num_duals), dtype=dtype)

        # x_b_scaling: (B, n) -- 1 for variables with any finite box bound,
        # 0 for variables without finite box bounds. Full-length dual storage
        # allows this mask to differ across the batch.
        if active_x_bound is None:
            # No per-variable bound mask supplied: with the full-length dual
            # layout every variable carries a box-bound slot, so the default
            # scaling mask is all-ones.
            self._x_b_scaling_init = cp.ones((B, n), dtype=dtype)
        else:
            self._x_b_scaling_init = active_x_bound.astype(dtype, copy=True)
        self._x_b_scaling = cp.copy(self._x_b_scaling_init)

        # Per-iteration workspace: (B, n+p+m) and (B, n)
        self._delta_iter = cp.empty((B, n + p + m), dtype=dtype)
        self._delta_b_iter = cp.empty((B, n), dtype=dtype)

        self._work_n = cp.empty((B, n), dtype=dtype)

        # Precompile warp kernels (one specialization per (n, p, m,
        # min_scaling, max_scaling) tuple).
        self._clamp_and_rsqrt_kernel = create_clamp_and_rsqrt_kernel(
            n, p, m, min_scaling, max_scaling, dtype=self._dtype,
        )
        self._accumulate_deltas_kernel = create_accumulate_deltas_kernel(n, p, m, dtype=self._dtype)
        # Gate the tile-factory call — calling it triggers shape-specialized
        # warp tile codegen. ``LargeProblemSolver`` constructs this class
        # with ``use_warp_kernels=False`` to skip the compile entirely; the
        # cupy fallback at the launch site is used instead.
        if use_warp_tile_kernels:
            self._conv_check_kernel = create_ruiz_conv_check_kernel(n, p, m, dtype=self._dtype)
            self._compute_rhs_inf_norm_unscaled_kernel = (
                create_compute_constraints_rhs_inf_norm_unscaled_kernel(
                    n, p, m, m, m, n, n,
                    dtype=self._dtype,
                )
            )
        else:
            self._conv_check_kernel = None
            self._compute_rhs_inf_norm_unscaled_kernel = None
        self._calc_scaling_inv_and_scale_bounds_kernel = create_calc_scaling_inv_and_scale_bounds_kernel(
            n, p, m, m, m, n, n,
            dtype=self._dtype,
        )
        self._scale_bounds_kernel = create_scale_bounds_kernel(n, p, m, dtype=self._dtype)
        self._unscale_bounds_kernel = create_unscale_bounds_kernel(n, p, m, dtype=self._dtype)

        # (2B,) buffer for per-batch (max_d, max_db) pairs — cp.max gives global.
        self._conv_buf = cp.empty(2 * B, dtype=dtype)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def cost_scaling(self) -> cp.ndarray:
        return self._cost_scaling

    @cost_scaling.setter
    def cost_scaling(self, value):
        self._cost_scaling[:] = value

    @property
    def cost_scaling_inv(self) -> cp.ndarray:
        return self._cost_scaling_inv

    @property
    def delta(self) -> cp.ndarray:
        return self._delta

    @property
    def delta_inv(self) -> cp.ndarray:
        return self._delta_inv

    @property
    def delta_b(self) -> cp.ndarray:
        return self._delta_b

    @property
    def delta_b_inv(self) -> cp.ndarray:
        return self._delta_b_inv

    @property
    def x_b_scaling(self) -> cp.ndarray:
        """x_b_scaling = x_b_scaling_init * delta_b * delta_x

        where x_b_scaling_init is a mask of 0/1 indicating whether x[i] has a finite bound.
        """
        return self._x_b_scaling

    @property
    def dual_res_unscale_factor(self) -> cp.ndarray:
        """(B, n). Per-element multiplier that converts the scaled dual
        residual on the x-block back to original units::

            unscaled_dual_res[b, i] = res.x[b, i] * dual_res_unscale_factor[b, i]

        Computed from the preconditioner's inverse scalings as::

            dual_res_unscale_factor[b, i] = cost_scaling_inv[b] * delta_inv[b, i]
                                            for i ∈ [0, n)

        where ``delta_inv[:, :n] = 1 / delta[:, :n]`` is the x-block of the
        combined scaling vector and ``cost_scaling_inv = 1 / cost_scaling``.

        Refreshed by ``_refresh_unscale_factors`` whenever ``delta`` or
        ``cost_scaling`` changes (end of ``scale_data``, in ``reset``).
        """
        return self._dual_res_unscale_factor

    @property
    def primal_res_unscale_factor(self) -> cp.ndarray:
        """(B, num_duals). Per-element multiplier that converts the scaled
        primal residuals on the dual variables back to original units::

            unscaled_primal_res[b, j] = res.duals_all[b, j]
                                        * primal_res_unscale_factor[b, j]

        Laid out in ``Variables._dual_buffer`` order
        ``[y | z_l | z_u | z_bl | z_bu]``. Per-segment computation from
        the preconditioner's inverse scalings (``delta_inv = 1 / delta``,
        ``delta_b_inv = 1 / delta_b``):

            offset range                                  content
            -----------------------------------           ----------------------------
            [0 : p]                                       delta_inv[:, n : n+p]
            [p : p+m]                                     delta_inv[:, n + p : n + p + m]
            [p+m : p+2m]                                  delta_inv[:, n + p : n + p + m]
            [p+2m : p+2m+n]                               delta_b_inv[:, :n]
            [p+2m+n : num_duals]                          delta_b_inv[:, :n]

        Note there's no ``cost_scaling_inv`` factor on the primal side — only
        the dual residual is scaled by the cost factor.

        Refreshed by ``_refresh_unscale_factors`` whenever ``delta`` or
        ``delta_b`` changes (end of ``scale_data``, in ``reset``).
        """
        return self._primal_res_unscale_factor

    def _refresh_unscale_factors(self):
        """Re-materialize dual_res_unscale_factor and primal_res_unscale_factor
        from the current ``delta_inv`` / ``delta_b_inv`` / ``cost_scaling_inv``.

        Call whenever any of those three arrays is updated (end of
        ``_scale_data_warp`` / ``_scale_data_cupy``, and in ``reset``).
        Not needed in ``reuse_scaling`` because it leaves delta/delta_b/
        cost_scaling untouched.

        Formulas:

            dual_res_unscale_factor[b, i]    =  cost_scaling_inv[b]
                                              * delta_inv[b, i]            i ∈ [0, n)

            primal_res_unscale_factor[b, j]  =  see layout in the property
                                                docstring — concatenated
                                                slices of delta_inv and
                                                delta_b_inv, in
                                                _dual_buffer order.
        """
        n, p, m = self.n, self.p, self.m

        # x-block for the dual residual:
        #     dual_res_unscale_factor = cost_scaling_inv[:, None] * delta_inv[:, :n]
        cp.multiply(self._cost_scaling_inv[:, None], self._delta_inv[:, :n],
                    out=self._dual_res_unscale_factor)

        # Duals-block for the primal residual, packed in Variables._dual_buffer
        # order [y | z_l | z_u | z_bl | z_bu]. With the full-length dual layout
        # the inequality / box segments are contiguous slices of delta_inv /
        # delta_b_inv (identity index map), so no gather is needed.
        off = 0
        if p > 0:
            # [y]: delta_inv[:, n : n+p]
            self._primal_res_unscale_factor[:, off:off + p] = self._delta_inv[:, n:n + p]
            off += p
        if m > 0:
            # [z_l]: delta_inv[:, n + p : n + p + m]
            self._primal_res_unscale_factor[:, off:off + m] = self._delta_inv[:, n + p:n + p + m]
            off += m
            # [z_u]: delta_inv[:, n + p : n + p + m]
            self._primal_res_unscale_factor[:, off:off + m] = self._delta_inv[:, n + p:n + p + m]
            off += m
        if n > 0:
            # [z_bl]: delta_b_inv[:, :n]
            self._primal_res_unscale_factor[:, off:off + n] = self._delta_b_inv[:, :n]
            off += n
            # [z_bu]: delta_b_inv[:, :n]
            self._primal_res_unscale_factor[:, off:off + n] = self._delta_b_inv[:, :n]

    def reset(self):
        self._delta.fill(1.0)
        self._delta_inv.fill(1.0)
        self._delta_b.fill(1.0)
        self._delta_b_inv.fill(1.0)
        self._cost_scaling.fill(1.0)
        self._cost_scaling_inv.fill(1.0)
        cp.copyto(self._x_b_scaling, self._x_b_scaling_init)
        self._dual_res_unscale_factor.fill(1.0)
        self._primal_res_unscale_factor.fill(1.0)

    # ------------------------------------------------------------------
    # Data-level scaling
    # ------------------------------------------------------------------

    @nvtx.annotate("RuizEquilibration::scale_data")
    def scale_data(self, data: Data, scale_cost: bool, max_iter: int):
        """Run Ruiz equilibration iterations to scale the problem data."""
        if self._use_warp_tile_kernels:
            self._scale_data_warp(data, scale_cost, max_iter)
        else:
            self._scale_data_cupy(data, scale_cost, max_iter)

    def _scale_data_warp(self, data: Data, scale_cost: bool, max_iter: int):
        n, p, m = self.n, self.p, self.m
        stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)

        for _ in range(max_iter):
            # backend specific
            self.compute_kkt_norms(data, self._delta_iter, self._delta_b_iter)

            wp.launch(
                kernel=self._clamp_and_rsqrt_kernel,
                dim=(self.B, n + p + m),
                inputs=[self._delta_iter, self._delta_b_iter],
                device="cuda", stream=stream,
            )

            # backend specific
            self.scale_matrices(
                data,
                self._delta_iter[:, :n],
                self._delta_iter[:, n:n+p],
                self._delta_iter[:, n+p:n+p+m],
                cost_scaling_factor=None
                )

            wp.launch(
                kernel=self._accumulate_deltas_kernel,
                dim=(self.B, n + p + m),
                inputs=[
                    self._delta, self._delta_b, self._x_b_scaling,
                    self._delta_iter, self._delta_b_iter,
                ],
                device="cuda", stream=stream,
            )

            if scale_cost:
                self.apply_cost_scaling(data)

            if self._use_warp_tile_kernels:
                _CONV_CHECK_BLOCK_DIM = 256
                wp.launch_tiled(
                    kernel=self._conv_check_kernel,
                    dim=[self.B],
                    inputs=[self._delta_iter, self._delta_b_iter, self._conv_buf],
                    block_dim=_CONV_CHECK_BLOCK_DIM,
                    device="cuda", stream=stream,
                )
            else:
                # _conv_buf shape (2B,) layout matching the warp kernel:
                #   [d_max_b0, db_max_b0, d_max_b1, db_max_b1, ...]
                cp.max(cp.abs(self._delta_iter   - 1.0), axis=1, out=self._conv_buf[0::2])
                cp.max(cp.abs(self._delta_b_iter - 1.0), axis=1, out=self._conv_buf[1::2])
            if float(cp.max(self._conv_buf)) < self.convergence_tol:
                break

        num_duals = self._primal_res_unscale_factor.shape[1]
        wp.launch(
            kernel=self._calc_scaling_inv_and_scale_bounds_kernel,
            dim=(self.B, n + p + m + 1 + num_duals),
            inputs=[
                self._delta, self._delta_inv,
                self._delta_b, self._delta_b_inv,
                self._cost_scaling, self._cost_scaling_inv,
                data.b, data.h_l, data.h_u, data.x_l, data.x_u,
                self._dual_res_unscale_factor, self._primal_res_unscale_factor,
            ],
            device="cuda", stream=stream,
        )

    def _scale_data_cupy(self, data: Data, scale_cost: bool, max_iter: int):
        n, p = self.n, self.p

        for _ in range(max_iter):
            self.compute_kkt_norms(data, self._delta_iter, self._delta_b_iter)

            self._limit_scaling(self._delta_iter)
            self._limit_scaling(self._delta_b_iter)
            cp.sqrt(self._delta_iter, out=self._delta_iter)
            cp.reciprocal(self._delta_iter, out=self._delta_iter)
            cp.sqrt(self._delta_b_iter, out=self._delta_b_iter)
            cp.reciprocal(self._delta_b_iter, out=self._delta_b_iter)

            d_x = self._delta_iter[:, :n]
            d_y = self._delta_iter[:, n:n+p]
            d_z = self._delta_iter[:, n+p:]

            self.scale_matrices(data, d_x, d_y, d_z, cost_scaling_factor=None)

            self._x_b_scaling *= self._delta_b_iter * d_x
            self._delta *= self._delta_iter
            self._delta_b *= self._delta_b_iter

            if scale_cost:
                self.apply_cost_scaling(data)

            conv = max(
                float(cp.max(cp.abs(1.0 - self._delta_iter))),
                float(cp.max(cp.abs(1.0 - self._delta_b_iter))),
            )
            if conv < self.convergence_tol:
                break

        cp.reciprocal(self._delta, out=self._delta_inv)
        cp.reciprocal(self._delta_b, out=self._delta_b_inv)
        cp.reciprocal(self._cost_scaling, out=self._cost_scaling_inv)

        self._scale_bounds(data)
        self._refresh_unscale_factors()

    @nvtx.annotate("RuizEquilibration::unscale_data")
    def unscale_data(self, data: Data):
        """Reverse scaling on the problem data; leave internal factors intact."""
        n, p = self.n, self.p
        d_x_inv = self._delta_inv[:, :n]
        d_y_inv = self._delta_inv[:, n:n+p]
        d_z_inv = self._delta_inv[:, n+p:]

        # Applies D_x^-1 P D_x^-1, D_y^-1 A D_x^-1, D_z^-1 G D_x^-1, D_x^-1 c,
        # plus cost_scaling_inv on P and c.
        self.scale_matrices(data, d_x_inv, d_y_inv, d_z_inv,
                            cost_scaling_factor=self._cost_scaling_inv)
        self._unscale_bounds(data)
        self._x_b_scaling *= self._delta_b_inv * d_x_inv

    @nvtx.annotate("RuizEquilibration::reuse_scaling")
    def reuse_scaling(self, data: Data):
        """Re-apply stored scaling to fresh (unscaled) data."""
        n, p = self.n, self.p
        d_x = self._delta[:, :n]
        d_y = self._delta[:, n:n + p]
        d_z = self._delta[:, n + p:]

        # Applies D_x P D_x, D_y A D_x, D_z G D_x, D_x c, plus cost_scaling on P, c.
        self.scale_matrices(data, d_x, d_y, d_z, cost_scaling_factor=self._cost_scaling)
        self._scale_bounds(data)
        # x_b_scaling = x_b_scaling_init * delta_b * d_x.
        cp.multiply(self._x_b_scaling_init, self._delta_b, out=self._x_b_scaling)
        self._x_b_scaling *= d_x

    # ------------------------------------------------------------------
    # Primal / dual / slack scaling and unscaling
    # ------------------------------------------------------------------

    def unscale_primal(self, x: cp.ndarray, out: cp.ndarray):
        """out = delta_x * x_scaled"""
        cp.multiply(x, self._delta[:, :self.n], out=out)

    def scale_primal(self, x: cp.ndarray, out: cp.ndarray):
        """out = delta_inv_x * x_orig"""
        cp.multiply(x, self._delta_inv[:, :self.n], out=out)

    def unscale_dual_eq(self, y: cp.ndarray, out: cp.ndarray):
        """out = c_inv * delta_y * y_scaled"""
        cp.multiply(y, self._delta[:, self.n:self.n + self.p], out=out)
        out *= self._cost_scaling_inv[:, None]

    def scale_dual_eq(self, y: cp.ndarray, out: cp.ndarray):
        """out = c * delta_inv_y * y_orig"""
        cp.multiply(y, self._delta_inv[:, self.n:self.n + self.p], out=out)
        out *= self._cost_scaling[:, None]

    def unscale_dual_ineq(self, z: cp.ndarray, out: cp.ndarray):
        """out = c_inv * delta_z * z_scaled (full-length inequality duals)"""
        cp.multiply(z, self._delta[:, self.n + self.p:], out=out)
        out *= self._cost_scaling_inv[:, None]

    def unscale_dual_b(self, z_b: cp.ndarray, out: cp.ndarray):
        """out = c_inv * delta_b * z_b_scaled (full-length box duals)"""
        cp.multiply(z_b, self._delta_b, out=out)
        out *= self._cost_scaling_inv[:, None]

    def unscale_slack_ineq(self, s: cp.ndarray, out: cp.ndarray):
        """out = delta_inv_z * s_scaled (full-length inequality slacks)"""
        cp.multiply(s, self._delta_inv[:, self.n + self.p:], out=out)

    def unscale_slack_b(self, s_b: cp.ndarray, out: cp.ndarray):
        """out = delta_b_inv * s_b_scaled (full-length box slacks)"""
        cp.multiply(s_b, self._delta_b_inv, out=out)

    # ------------------------------------------------------------------
    # Residual unscaling (used every iteration for convergence checks)
    # ------------------------------------------------------------------

    def unscale_dual_res(self, v: cp.ndarray, out: cp.ndarray):
        """out = c_inv * delta_inv_x * v_scaled"""
        cp.multiply(v, self._delta_inv[:, :self.n], out=out)
        out *= self._cost_scaling_inv[:, None]

    def unscale_primal_res_eq(self, v: cp.ndarray, out: cp.ndarray):
        """out = delta_inv_y * v_scaled"""
        cp.multiply(v, self._delta_inv[:, self.n:self.n + self.p], out=out)

    def unscale_primal_res_ineq(self, v: cp.ndarray, out: cp.ndarray):
        """out = delta_inv_z * v_scaled (full-length inequality residual)"""
        cp.multiply(v, self._delta_inv[:, self.n + self.p:], out=out)

    def unscale_primal_res_b(self, v: cp.ndarray, out: cp.ndarray):
        """out = delta_b_inv * v_scaled (full-length box residual)"""
        cp.multiply(v, self._delta_b_inv, out=out)

    @nvtx.annotate("RuizEquilibration::compute_constraints_rhs_inf_norm_unscaled")
    def compute_constraints_rhs_inf_norm_unscaled(self, data: Data, out: cp.ndarray) -> None:
        """Fill ``out`` (B,) with the inf-norm of the user-space constraint RHS.

        Recovers the unscaled b / h_l / h_u / x_l / x_u inf-norm by passing
        the currently-scaled buffers through ``unscale_primal_res_*``. Caller
        must invoke this after the preconditioner has been (re-)applied so
        that ``delta_inv`` / ``delta_b_inv`` reflect current scaling, and
        must own the ``out`` buffer (allocated once in solver setup).
        """
        if self._use_warp_tile_kernels:
            BLOCK_DIM = 256
            wp.launch_tiled(
                kernel=self._compute_rhs_inf_norm_unscaled_kernel,
                dim=[self.B],
                inputs=[
                    self._delta_inv, self._delta_b_inv,
                    data.b, data.h_l, data.h_u, data.x_l, data.x_u,
                    data.finite_mask_hl, data.finite_mask_hu, data.finite_mask_xl, data.finite_mask_xu,
                    out,
                ],
                block_dim=BLOCK_DIM,
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )
        
        else:
            # cupy fallback (use_warp_tile_kernels=False; LargeProblemSolver path).
            out.fill(0.0)
            if data.p > 0:
                tmp = cp.empty_like(data.b)
                self.unscale_primal_res_eq(data.b, out=tmp)
                cp.maximum(out, cp.max(cp.abs(tmp), axis=1), out=out)
            if data.num_hu > 0:
                tmp = cp.where(data.finite_mask_hu > 0.5, data.h_u, 0.0)
                self.unscale_primal_res_ineq(tmp, out=tmp)
                cp.maximum(out, cp.max(cp.abs(tmp), axis=1), out=out)
            if data.num_hl > 0:
                tmp = cp.where(data.finite_mask_hl > 0.5, data.h_l, 0.0)
                self.unscale_primal_res_ineq(tmp, out=tmp)
                cp.maximum(out, cp.max(cp.abs(tmp), axis=1), out=out)
            if data.num_xu > 0:
                tmp = cp.where(data.finite_mask_xu > 0.5, data.x_u, 0.0)
                self.unscale_primal_res_b(tmp, out=tmp)
                cp.maximum(out, cp.max(cp.abs(tmp), axis=1), out=out)
            if data.num_xl > 0:
                tmp = cp.where(data.finite_mask_xl > 0.5, data.x_l, 0.0)
                self.unscale_primal_res_b(tmp, out=tmp)
                cp.maximum(out, cp.max(cp.abs(tmp), axis=1), out=out)

    # ------------------------------------------------------------------
    # Cost unscaling
    # ------------------------------------------------------------------

    def unscale_cost(self, cost: cp.ndarray, out: cp.ndarray):
        """out = c_inv * cost_scaled"""
        cp.multiply(cost, self._cost_scaling_inv, out=out)

    # ------------------------------------------------------------------
    # Backend hooks — implemented by DenseRuiz / SparseRuiz / MultistageRuiz
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_kkt_norms(self, data: Data,
                          d_iter: cp.ndarray, d_b_iter: cp.ndarray):
        """Fill the Ruiz row/col inf-norms.

        d_iter  (B, n+p+m): row/col inf-norms of the Ruiz KKT matrix
            [:, :n]      = max(max over P rows/cols, A cols, G cols, x_b_scaling)
            [:, n:n+p]   = A row inf-norms
            [:, n+p:]    = G row inf-norms
        d_b_iter (B, n)  : copy of the current x_b_scaling.

        Backends are expected to write both outputs in a single fused pass.
        """
        ...

    @abstractmethod
    def scale_matrices(self, data: Data,
                       d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray,
                       cost_scaling_factor: cp.ndarray | None = None):
        """Apply row/col scaling to P, A, G, c.

        P <- D_x P D_x,   c <- D_x c,   A <- D_y A D_x,   G <- D_z G D_x

        If ``cost_scaling_factor`` is provided (shape (B,)), additionally 
        multiply P and c by it. Used by all three call sites:
          - One Ruiz iter       : (d_iter_x,  d_iter_y,  d_iter_z,  None)
          - Unscaling            : (d_x_inv,   d_y_inv,   d_z_inv,   cost_scaling_inv)
          - Re-apply stored      : (delta_x,   delta_y,   delta_z,   cost_scaling)
        """
        ...

    @abstractmethod
    def apply_cost_scaling(self, data: Data):
        """Compute gamma from |P| (triu) and |c|; multiply P and c by gamma;
        multiply self._cost_scaling by gamma."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers (for cupy implementation)
    # ------------------------------------------------------------------

    def _limit_scaling(self, d: cp.ndarray):
        d[d < self.min_scaling] = 1.0
        cp.minimum(d, self.max_scaling, out=d)

    def _limit_scaling_scalar(self, d: float) -> float:
        if d < self.min_scaling:
            return 1.0
        elif d > self.max_scaling:
            return self.max_scaling
        return d

    def _scale_bounds(self, data: Data):
        wp.launch(
            kernel=self._scale_bounds_kernel,
            dim=(self.B, self.n + self.p + self.m),
            inputs=[
                self._delta, self._delta_b,
                data.b, data.h_l, data.h_u,
                data.x_l, data.x_u,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    def _unscale_bounds(self, data: Data):
        wp.launch(
            kernel=self._unscale_bounds_kernel,
            dim=(self.B, self.n + self.p + self.m),
            inputs=[
                self._delta_inv, self._delta_b_inv,
                data.b, data.h_l, data.h_u,
                data.x_l, data.x_u,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )
