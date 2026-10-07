from abc import ABC, abstractmethod
from typing import Any, Optional
import nvtx
import numpy as np
import warp as wp

from .utils import as_warp_array, column_slice
from .data import Data
from .results import Variables
from .preconditioner_kernels import (
    REDUCTION_BLOCK_DIM,
    create_clamp_and_rsqrt_kernel,
    create_accumulate_deltas_kernel,
    create_ruiz_conv_check_kernel,
    create_calc_scaling_inv_and_scale_bounds_kernel,
    create_scale_bounds_kernel,
    create_unscale_bounds_kernel,
    create_compute_constraints_rhs_inf_norm_unscaled_kernel,
    create_unscale_solution_kernel
)




@wp.func
def _product3(x: Any, y: Any, z: Any):
    # element-wise x * y * z, applied with wp.map. A Warp function with generic
    # argument types works for float32 and float64 and, unlike a plain Python
    # function, is not re-parsed on every wp.map call.
    return x * y * z

class PreconditionerBase(ABC):
    """Abstract preconditioner interface for QP problems -- batched.

    Defines only the operations the solver depends on. Concrete preconditioners
    (Ruiz equilibration, identity, block-Jacobi, ...) choose their own internal
    representation (diagonal vectors, sparse factors, ...).

    Contract with the solver:
        * scale_data / unscale_data / reuse_scaling -- transform the problem data.
        * unscale_solution -- map a full Variables struct back to original coordinates.
        * x_b_scaling, cost_scaling, cost_scaling_inv -- state the KKT solver and
          solver read directly.
        * reset -- restore to identity scaling.
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

    @abstractmethod
    def unscale_solution(self, result: Variables, data: Data):
        """Transform a scaled IPM solution back to original coordinates, in place."""
        ...

    # ------------------------------------------------------------------
    # State exposed to the solver / KKT system
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def cost_scaling(self) -> wp.array:
        """(B,) scalar objective scaling."""
        ...

    @property
    @abstractmethod
    def cost_scaling_inv(self) -> wp.array:
        """(B,) inverse of cost_scaling."""
        ...

    @property
    @abstractmethod
    def x_b_scaling(self) -> wp.array:
        """(B, n) diagonal of the box block in the scaled KKT matrix.

        Zero for unbounded variables, nonzero for bounded ones.
        """
        ...


class RuizEquilibration(PreconditionerBase):
    """Ruiz equilibration preconditioner for QP problems -- batched.

    Diagonal preconditioner: stores (delta, delta_b, cost_scaling) as vectors
    and uses them multiplicatively in every scale/unscale operation.

    All scaling vectors are Warp arrays with a leading batch dimension
    ``(B, ...)``. For single problems, ``B = 1``.

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
                 has_h_l: bool, has_h_u: bool,
                 has_x_l: bool, has_x_u: bool,
                 active_x_bound=None,
                 min_scaling: float = 1e-4,
                 max_scaling: float = 1e4,
                 convergence_tol: float = 1e-3,
                 dtype=wp.float64,
                 device: str = "cuda"
                 ):
        self.B = B
        self.n = n
        self.p = p
        self.m = m

        self._has_h_l = has_h_l
        self._has_h_u = has_h_u
        self._num_hl = m if has_h_l else 0
        self._num_hu = m if has_h_u else 0
        self._has_x_l = has_x_l
        self._has_x_u = has_x_u
        self._num_xl = n if has_x_l else 0
        self._num_xu = n if has_x_u else 0
        self._dtype = dtype
        self._device = device

        self.min_scaling = min_scaling
        self.max_scaling = max_scaling
        self.convergence_tol = convergence_tol


        # Combined scaling: (B, n+p+m) -- delta[:, :n] for x, delta[:, n:n+p] for y, etc.
        self._delta = wp.full((B, n + p + m), 1.0, dtype=self._dtype, device=device)
        self._delta_inv = wp.full((B, n + p + m), 1.0, dtype=self._dtype, device=device)

        # Box constraint scaling: (B, n)
        self._delta_b = wp.full((B, n), 1.0, dtype=self._dtype, device=device)
        self._delta_b_inv = wp.full((B, n), 1.0, dtype=self._dtype, device=device)

        # Cost scaling: (B,)
        self._cost_scaling = wp.full((B,), 1.0, dtype=self._dtype, device=device)
        self._cost_scaling_inv = wp.full((B,), 1.0, dtype=self._dtype, device=device)

        # Pre-combined residual unscaling factors -- materialized once per
        # Ruiz update by the finalize kernel. The solver's residual kernels
        # read these directly so their reductions need no gather.
        #
        #   dual_res_unscale_factor[b, i]   = cost_scaling_inv[b] * delta_inv[b, i]
        #                                     for i in [0, n)
        #       -- multiplies res.x (the dual residual on the x-block)
        #         element-wise to convert it back to original units.
        #
        #   primal_res_unscale_factor[b, j] = unscaling factor for res.duals_all
        #       -- shape (B, num_duals); packed in Variables._dual_buffer
        #         order [y | z_l | z_u | z_bl | z_bu], with per-segment
        #         content:
        #
        #             [y]:    delta_inv[:, n:n+p]
        #             [z_l]:  delta_inv[:, n + p:n + p + m]   (full-length)
        #             [z_u]:  delta_inv[:, n + p:n + p + m]   (full-length)
        #             [z_bl]: delta_b_inv[:, :n]                (full-length)
        #             [z_bu]: delta_b_inv[:, :n]                (full-length)
        num_duals = p + self._num_hl + self._num_hu + self._num_xl + self._num_xu
        self._dual_res_unscale_factor = wp.full((B, n), 1.0, dtype=self._dtype, device=device)
        self._primal_res_unscale_factor = wp.full((B, num_duals), 1.0, dtype=self._dtype, device=device)

        # x_b_scaling: (B, n) -- 1 for variables with any finite box bound,
        # 0 for variables without finite box bounds. Full-length dual storage
        # allows this mask to differ across the batch.
        self._x_b_scaling_init = wp.full((B, n), 1.0, dtype=self._dtype, device=device)
        if active_x_bound is not None:
            # No per-variable bound mask supplied means every variable carries
            # a box-bound slot, so the default scaling mask is all-ones.
            wp.copy(self._x_b_scaling_init, as_warp_array(active_x_bound, "active_x_bound"))
        self._x_b_scaling = wp.clone(self._x_b_scaling_init)

        # Per-iteration workspace: (B, n+p+m) and (B, n)
        self._delta_iter = wp.empty((B, n + p + m), dtype=self._dtype, device=device)
        self._delta_b_iter = wp.empty((B, n), dtype=self._dtype, device=device)

        # One-element accumulator for the Ruiz convergence check.
        self._conv_buf = wp.zeros(1, dtype=self._dtype, device=device)
        self._conv_host = wp.zeros(1, dtype=self._dtype, device="cpu", pinned=True)

        # Precompile warp kernels (one specialization per (n, p, m,
        # min_scaling, max_scaling) tuple).
        self._clamp_and_rsqrt_kernel = create_clamp_and_rsqrt_kernel(
            n, p, m, min_scaling, max_scaling, dtype=self._dtype
        )
        self._accumulate_deltas_kernel = create_accumulate_deltas_kernel(n, p, m, dtype=self._dtype)
        self._conv_check_kernel = create_ruiz_conv_check_kernel(dtype=self._dtype)
        self._compute_rhs_inf_norm_unscaled_kernel = (
            create_compute_constraints_rhs_inf_norm_unscaled_kernel(dtype=self._dtype)
        )
        self._calc_scaling_inv_and_scale_bounds_kernel = create_calc_scaling_inv_and_scale_bounds_kernel(
            n, p, m, self._num_hl, self._num_hu, self._num_xl, self._num_xu,
            has_h_l=self._has_h_l, has_h_u=self._has_h_u,
            has_x_l=self._has_x_l, has_x_u=self._has_x_u,
            dtype=self._dtype
        )
        self._scale_bounds_kernel = create_scale_bounds_kernel(n, p, m, has_h_l=self._has_h_l, has_h_u=self._has_h_u, has_x_l=self._has_x_l, has_x_u=self._has_x_u, dtype=self._dtype)
        self._unscale_bounds_kernel = create_unscale_bounds_kernel(n, p, m, has_h_l=self._has_h_l, has_h_u=self._has_h_u, has_x_l=self._has_x_l, has_x_u=self._has_x_u, dtype=self._dtype)
        self._unscale_solution_kernel = create_unscale_solution_kernel(dtype=self._dtype)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def cost_scaling(self) -> wp.array:
        return self._cost_scaling

    @property
    def cost_scaling_inv(self) -> wp.array:
        return self._cost_scaling_inv

    @property
    def delta(self) -> wp.array:
        return self._delta

    @property
    def delta_inv(self) -> wp.array:
        return self._delta_inv

    @property
    def delta_b(self) -> wp.array:
        return self._delta_b

    @property
    def delta_b_inv(self) -> wp.array:
        return self._delta_b_inv

    @property
    def x_b_scaling(self) -> wp.array:
        """x_b_scaling = x_b_scaling_init * delta_b * delta_x

        where x_b_scaling_init is a mask of 0/1 indicating whether x[i] has a finite bound.
        """
        return self._x_b_scaling

    @property
    def dual_res_unscale_factor(self) -> wp.array:
        """(B, n). Per-element multiplier that converts the scaled dual
        residual on the x-block back to original units::

            unscaled_dual_res[b, i] = res.x[b, i] * dual_res_unscale_factor[b, i]

        Computed from the preconditioner's inverse scalings as::

            dual_res_unscale_factor[b, i] = cost_scaling_inv[b] * delta_inv[b, i]
                                            for i in [0, n)

        Refreshed at the end of ``scale_data`` and in ``reset``.
        """
        return self._dual_res_unscale_factor

    @property
    def primal_res_unscale_factor(self) -> wp.array:
        """(B, num_duals). Per-element multiplier that converts the scaled
        primal residuals on the dual variables back to original units::

            unscaled_primal_res[b, j] = res.duals_all[b, j]
                                        * primal_res_unscale_factor[b, j]

        Laid out in ``Variables._dual_buffer`` order
        ``[y | z_l | z_u | z_bl | z_bu]``. Each inequality/box segment has its
        full width (``m`` or ``n``) when the block was provided at setup() and
        zero width when omitted, so the following segment slides up:

            segment (width)                               content
            -----------------------------------           ----------------------------
            [y]    (p)                                    delta_inv[:, n:n+p]
            [z_l]  (num_hl)                               delta_inv[:, n + p : n + p + m]
            [z_u]  (num_hu)                               delta_inv[:, n + p : n + p + m]
            [z_bl] (num_xl)                               delta_b_inv[:, :n]
            [z_bu] (num_xu)                               delta_b_inv[:, :n]

        Note there's no ``cost_scaling_inv`` factor on the primal side -- only
        the dual residual is scaled by the cost factor.

        Refreshed at the end of ``scale_data`` and in ``reset``.
        """
        return self._primal_res_unscale_factor

    def reset(self):
        self._delta.fill_(1.0)
        self._delta_inv.fill_(1.0)
        self._delta_b.fill_(1.0)
        self._delta_b_inv.fill_(1.0)
        self._cost_scaling.fill_(1.0)
        self._cost_scaling_inv.fill_(1.0)
        wp.copy(self._x_b_scaling, self._x_b_scaling_init)
        self._dual_res_unscale_factor.fill_(1.0)
        self._primal_res_unscale_factor.fill_(1.0)

    # ------------------------------------------------------------------
    # Data-level scaling
    # ------------------------------------------------------------------

    @nvtx.annotate("RuizEquilibration::scale_data")
    def scale_data(self, data: Data, scale_cost: bool, max_iter: int):
        """Run Ruiz equilibration iterations to scale the problem data."""
        n, p, m = self.n, self.p, self.m
        stream = wp.get_stream("cuda")

        for _ in range(max_iter):
            # backend specific
            self.compute_kkt_norms(data, self._delta_iter, self._delta_b_iter)

            wp.launch(
                kernel=self._clamp_and_rsqrt_kernel,
                dim=(self.B, n + p + m),
                inputs=[self._delta_iter, self._delta_b_iter],
                device=self._device
            )

            # backend specific
            self.scale_matrices(
                data,
                column_slice(self._delta_iter, 0, n),
                column_slice(self._delta_iter, n, n+p),
                column_slice(self._delta_iter, n+p, n+p+m),
                cost_scaling_factor=None
                )

            wp.launch(
                kernel=self._accumulate_deltas_kernel,
                dim=(self.B, n + p + m),
                inputs=[
                    self._delta, self._delta_b, self._x_b_scaling,
                    self._delta_iter, self._delta_b_iter,
                ],
                device=self._device
            )

            if scale_cost:
                self.apply_cost_scaling(data)

            # Batch-wide convergence measure; one device-to-host read per iteration.
            self._conv_buf.zero_()
            wp.launch_tiled(
                kernel=self._conv_check_kernel,
                dim=[self.B],
                inputs=[self._delta_iter, self._delta_b_iter, self._conv_buf],
                block_dim=REDUCTION_BLOCK_DIM,
                device=self._device
            )
            wp.copy(self._conv_host, self._conv_buf)
            wp.synchronize_stream(stream)
            if float(self._conv_host.numpy()[0]) < self.convergence_tol:
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
            device=self._device
        )

    @nvtx.annotate("RuizEquilibration::unscale_data")
    def unscale_data(self, data: Data):
        """Reverse scaling on the problem data; leave internal factors intact."""
        n, p = self.n, self.p
        d_x_inv = column_slice(self._delta_inv, 0, n)
        d_y_inv = column_slice(self._delta_inv, n, n+p)
        d_z_inv = column_slice(self._delta_inv, n+p, None)

        # Applies D_x^-1 P D_x^-1, D_y^-1 A D_x^-1, D_z^-1 G D_x^-1, D_x^-1 c,
        # plus cost_scaling_inv on P and c.
        self.scale_matrices(data, d_x_inv, d_y_inv, d_z_inv,
                            cost_scaling_factor=self._cost_scaling_inv)
        self._unscale_bounds(data)
        # x_b_scaling *= delta_b_inv * d_x_inv
        wp.map(_product3, self._x_b_scaling, self._delta_b_inv, d_x_inv, out=self._x_b_scaling)

    @nvtx.annotate("RuizEquilibration::reuse_scaling")
    def reuse_scaling(self, data: Data):
        """Re-apply stored scaling to fresh (unscaled) data."""
        n, p = self.n, self.p
        d_x = column_slice(self._delta, 0, n)
        d_y = column_slice(self._delta, n, n + p)
        d_z = column_slice(self._delta, n + p, None)

        # Applies D_x P D_x, D_y A D_x, D_z G D_x, D_x c, plus cost_scaling on P, c.
        self.scale_matrices(data, d_x, d_y, d_z, cost_scaling_factor=self._cost_scaling)
        self._scale_bounds(data)
        # x_b_scaling = x_b_scaling_init * delta_b * d_x.
        wp.map(_product3, self._x_b_scaling_init, self._delta_b, d_x, out=self._x_b_scaling)

    # ------------------------------------------------------------------
    # Solution unscaling
    # ------------------------------------------------------------------

    @nvtx.annotate("Preconditioner::unscale_solution")
    def unscale_solution(self, result: Variables, data: Data):
        """Transform scaled IPM solution back to original coordinates, in place."""
        total = (result.primals_all.shape[1] + result.duals_all.shape[1])
        wp.launch(
            kernel=self._unscale_solution_kernel,
            dim=(self.B, total),
            inputs=[
                result.x, result.s_l, result.s_u, result.s_bl, result.s_bu,
                result.y, result.z_l, result.z_u, result.z_bl, result.z_bu,
                self._delta, self._delta_inv, self._delta_b, self._delta_b_inv,
                self._cost_scaling_inv,
            ],
            device=self._device
        )

    @nvtx.annotate("RuizEquilibration::compute_constraints_rhs_inf_norm_unscaled")
    def compute_constraints_rhs_inf_norm_unscaled(self, data: Data, out: wp.array) -> None:
        """Fill ``out`` (B,) with the inf-norm of the user-space constraint RHS.

        Recovers the unscaled b / h_l / h_u / x_l / x_u inf-norm from the
        currently-scaled buffers and the inverse scalings. Caller must invoke
        this after the preconditioner has been (re-)applied so that
        ``delta_inv`` / ``delta_b_inv`` reflect current scaling, and must own
        the ``out`` buffer (allocated once in solver setup).
        """
        wp.launch_tiled(
            kernel=self._compute_rhs_inf_norm_unscaled_kernel,
            dim=[self.B],
            inputs=[
                self._delta_inv, self._delta_b_inv,
                data.b, data.h_l, data.h_u, data.x_l, data.x_u,
                data.finite_mask_hl, data.finite_mask_hu, data.finite_mask_xl, data.finite_mask_xu,
                out,
            ],
            block_dim=REDUCTION_BLOCK_DIM,
            device=self._device
        )

    # ------------------------------------------------------------------
    # Backend hooks -- implemented by DenseRuiz / SparseRuiz / MultistageRuiz
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_kkt_norms(self, data: Data, d_iter: wp.array, d_b_iter: wp.array):
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
                       d_x: wp.array, d_y: wp.array, d_z: wp.array,
                       cost_scaling_factor: Optional[wp.array] = None):
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
    # Shared helpers
    # ------------------------------------------------------------------

    def _scale_bounds(self, data: Data):
        wp.launch(
            kernel=self._scale_bounds_kernel,
            dim=(self.B, self.n + self.p + self.m),
            inputs=[
                self._delta, self._delta_b,
                data.b, data.h_l, data.h_u,
                data.x_l, data.x_u,
            ],
            device=self._device
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
            device=self._device
        )
