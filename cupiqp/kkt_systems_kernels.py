import warp as wp
from .utils import to_warp_dtype


def create_update_regularizations_step_1_kernel(num_ineq: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Create kernel operating on contiguous s_all/z_all buffers. Performs:

        self._rho = rho
        self._delta = delta

        self._m_s_u = vars.s_u
        self._m_s_l = vars.s_l
        self._m_s_bu = vars.s_bu
        self._m_s_bl = vars.s_bl
        self._m_z_u_inv = 1. / vars.z_u
        self._m_z_l_inv = 1. / vars.z_l
        self._m_z_bu_inv = 1. / vars.z_bu
        self._m_z_bl_inv = 1. / vars.z_bl

        self._w_bu_delta_inv = 1. / (self._m_s_bu * self._m_z_bu_inv + delta)
        self._w_bl_delta_inv = 1. / (self._m_s_bl * self._m_z_bl_inv + delta)
        self._w_u_delta_inv = 1. / (self._m_s_u * self._m_z_u_inv + delta)
        self._w_l_delta_inv = 1. / (self._m_s_l * self._m_z_l_inv + delta)

        Since s and z are stored contiguously, it becomes:

        rho_out[b]            = rho_in[b]                          (when i == 0)
        delta_out[b]          = delta_in[b]                        (when i == 0)
        m_s_all[b, i]         = vars_s_all[b, i]                   (when i < num_ineq)
        m_z_inv_all[b, i]     = 1.0 / vars_z_all[b, i]             (when i < num_ineq)
        w_delta_inv_all[b, i] = 1.0 / (s * z_inv + delta_in[b])    (when i < num_ineq)

        Launch dim is ``(B, max(num_ineq, 1))`` so that the ``i == 0`` thread
        always exists to copy rho/delta even when there are no inequalities.
    """
    @wp.kernel
    def update_regularizations_step_1_kernel(
        vars_s_all: wp.array2d(dtype=dtype),        # type: ignore
        vars_z_all: wp.array2d(dtype=dtype),        # type: ignore
        finite_mask_all: wp.array2d(dtype=dtype),   # type: ignore
        m_s_all: wp.array2d(dtype=dtype),           # type: ignore
        m_z_inv_all: wp.array2d(dtype=dtype),       # type: ignore
        w_delta_inv_all: wp.array2d(dtype=dtype),   # type: ignore
        rho_in: wp.array(dtype=dtype),              # type: ignore
        delta_in: wp.array(dtype=dtype),            # type: ignore
        rho_out: wp.array(dtype=dtype),             # type: ignore
        delta_out: wp.array(dtype=dtype),           # type: ignore
    ):
        b, i = wp.tid()
        num_ineq_static = wp.static(num_ineq)
        if i == 0:
            rho_out[b] = rho_in[b]
            delta_out[b] = delta_in[b]
        if i < num_ineq_static:
            if finite_mask_all[b, i] > dtype(0.5):
                s = vars_s_all[b, i]
                z_inv = dtype(1.0) / vars_z_all[b, i]
                m_s_all[b, i] = s
                m_z_inv_all[b, i] = z_inv
                w_delta_inv_all[b, i] = dtype(1.0) / (s * z_inv + delta_in[b])
            else:
                m_s_all[b, i] = dtype(0.0)
                m_z_inv_all[b, i] = dtype(0.0)
                w_delta_inv_all[b, i] = dtype(0.0)
    return update_regularizations_step_1_kernel


def create_update_regularizations_step_2_kernel(nx: int, nz: int, has_x_l: bool, has_x_u: bool, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Create kernel specialized for the condensed KKT regularization terms.

    The full-length cone layout stores all four bound classes without compacting
    finite entries. ``w_*_delta_inv`` is already zero on inactive entries, so the
    full-index formulas are simple and do not need inverse gather maps::

        x_reg[:] = rho
        x_reg += x_b_scaling**2 * (w_bu_delta_inv + w_bl_delta_inv)

        z_reg[:] = 1. / (w_u_delta_inv + w_l_delta_inv)

    An omitted box block (``has_x_l`` / ``has_x_u`` False) contributes nothing to
    ``x_reg`` and its ``w_b*_delta_inv`` array is empty ``(B, 0)``, so the box
    reads are guarded by the static presence flags.

    NOTE: if there are inactive rows in G s.t. -inf <= G[i] * x <= +inf, the
    corresponding z_reg[i] will be 0.

    Each thread writes only to its own unique slot: one variable regularization
    entry ``x_reg[b, t]`` or one inequality row weight ``z_reg[b, tz]``.
    """
    @wp.kernel
    def update_regularizations_step_2_kernel(
        w_bu_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        w_bl_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        x_b_scaling: wp.array2d(dtype=dtype),  # type: ignore
        rho: wp.array(dtype=dtype),  # type: ignore
        x_reg: wp.array2d(dtype=dtype),  # type: ignore
        w_u_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        w_l_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        z_reg: wp.array2d(dtype=dtype),  # type: ignore
        z_reg_inv: wp.array2d(dtype=dtype), # type: ignore
    ):
        b, t = wp.tid()
        nx_static = wp.static(nx)
        nz_static = wp.static(nz)

        if t < nx_static:
            xb_scaling = x_b_scaling[b, t]
            box_weight = dtype(0.0)
            if wp.static(has_x_u):
                box_weight += w_bu_delta_inv[b, t]
            if wp.static(has_x_l):
                box_weight += w_bl_delta_inv[b, t]
            x_reg[b, t] = rho[b] + xb_scaling * xb_scaling * box_weight
        elif t < nx_static + nz_static:
            tz = t - nx_static
            tmp = w_u_delta_inv[b, tz] + w_l_delta_inv[b, tz]
            if tmp > dtype(0.0):  # means this row of G is active
                z_reg[b, tz] = dtype(1.0) / tmp
                z_reg_inv[b, tz] = tmp
            else:
                z_reg[b, tz] = dtype(0.0)
                z_reg_inv[b, tz] = dtype(0.0)

    return update_regularizations_step_2_kernel

def create_eliminate_duals_kernel(nx: int, nz: int, has_x_l: bool, has_x_u: bool, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Create kernel specialized for eliminating dual rows after slack elimination.

    The full-length layout means the four bound blocks have shapes ``(B, m)``,
    ``(B, m)``, ``(B, n)``, and ``(B, n)``. Inactive entries are represented by
    zero ``w_*_delta_inv`` weights, so they are inert without any compact
    ``idx_*`` or ``inv_idx_*`` gather/scatter.

    Forward and transposed KKT solves share this dual-elimination step. The
    preceding slack-elimination kernel has already built ``updated_rhs_z_*`` as
    either ``rhs.z - inv(Z) * rhs.s`` for the forward system or
    ``rhs.z - (S/Z) * rhs.s`` for the transposed system.

    Equivalent full-length operations::

        rhs_x_updated[:] = rhs_x
        rhs_x_updated += x_b_scaling * w_bu_delta_inv * updated_rhs_z_bu
        rhs_x_updated -= x_b_scaling * w_bl_delta_inv * updated_rhs_z_bl

        tmp_z = w_u_delta_inv * updated_rhs_z_u
              - w_l_delta_inv * updated_rhs_z_l
        rhs_z_updated = tmp_z * z_reg          # z_reg = 1 / (w_u_delta_inv + w_l_delta_inv)

    ``z_reg`` is the explicit augmented inequality diagonal magnitude
    (``1 / weight``), so multiplying by it is the condensed division by the row
    weight. It is zero on inactive inequality rows (both bounds infinite), so
    those rows take the ``else`` branch and the condensed RHS is set to zero.

    Each thread writes only to its own unique slot: one variable RHS entry
    ``rhs_x_updated[b, t]`` or one inequality RHS entry ``rhs_z_updated[b, tz]``.
    """
    @wp.kernel
    def eliminate_duals_kernel(
        rhs_x: wp.array2d(dtype=dtype),  # type: ignore
        w_bu_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        w_bl_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        x_b_scaling: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_bu: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_bl: wp.array2d(dtype=dtype),  # type: ignore
        rhs_x_updated: wp.array2d(dtype=dtype),  # type: ignore
        w_u_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        w_l_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_u: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_l: wp.array2d(dtype=dtype),  # type: ignore
        z_reg: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_updated: wp.array2d(dtype=dtype),  # type: ignore
    ):
        b, t = wp.tid()
        nx_static = wp.static(nx)
        nz_static = wp.static(nz)

        if t < nx_static:
            xb_scaling = x_b_scaling[b, t]
            box_term = dtype(0.0)
            if wp.static(has_x_u):
                box_term += w_bu_delta_inv[b, t] * rhs_z_bu[b, t]
            if wp.static(has_x_l):
                box_term -= w_bl_delta_inv[b, t] * rhs_z_bl[b, t]
            rhs_x_updated[b, t] = rhs_x[b, t] + xb_scaling * box_term
        elif t < nx_static + nz_static:
            tz = t - nx_static
            val = w_u_delta_inv[b, tz] * rhs_z_u[b, tz] - w_l_delta_inv[b, tz] * rhs_z_l[b, tz]
            zr = z_reg[b, tz]  # explicit augmented diagonal magnitude = 1 / weight
            if zr > dtype(0.0):
                rhs_z_updated[b, tz] = val * zr
            else:
                rhs_z_updated[b, tz] = dtype(0.0)

    return eliminate_duals_kernel

def create_eliminate_slacks_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Batched element-wise kernel for eliminating slacks for inequalities.

        updated_rhs_z_all[b, i] = rhs_z_all[b, i] - m_z_inv_all[b, i] * rhs_s_all[b, i]
    """
    @wp.kernel
    def eliminate_slacks_kernel(
        rhs_z_all: wp.array2d(dtype=dtype),          # type: ignore
        rhs_s_all: wp.array2d(dtype=dtype),          # type: ignore
        m_z_inv_all: wp.array2d(dtype=dtype),        # type: ignore
        updated_rhs_z_all: wp.array2d(dtype=dtype),  # type: ignore
    ):
        b, i = wp.tid()
        updated_rhs_z_all[b, i] = -m_z_inv_all[b, i] * rhs_s_all[b, i] + rhs_z_all[b, i]

    return eliminate_slacks_kernel


def create_eliminate_slacks_transposed_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Transposed (K^T) variant of eliminate_slacks. Scales rhs_s by W = S/Z instead
    of 1/Z, because row 7..10 of K^T have S in the off-diagonal (vs. I in K).

        updated_rhs_z_all[b, i] = rhs_z_all[b, i] - m_s_all[b, i] * m_z_inv_all[b, i] * rhs_s_all[b, i]
    """
    @wp.kernel
    def eliminate_slacks_transposed_kernel(
        rhs_z_all: wp.array2d(dtype=dtype),          # type: ignore
        rhs_s_all: wp.array2d(dtype=dtype),          # type: ignore
        m_s_all: wp.array2d(dtype=dtype),            # type: ignore
        m_z_inv_all: wp.array2d(dtype=dtype),        # type: ignore
        updated_rhs_z_all: wp.array2d(dtype=dtype),  # type: ignore
    ):
        b, i = wp.tid()
        w = m_s_all[b, i] * m_z_inv_all[b, i]  # W = S / Z
        updated_rhs_z_all[b, i] = -w * rhs_s_all[b, i] + rhs_z_all[b, i]

    return eliminate_slacks_transposed_kernel


def create_recover_duals_kernel(num_hu: int, num_hl: int, num_xu: int, num_xl: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Create kernel specialized for recovering duals.

    Full-length layout (``num_hu == num_hl == m``, ``num_xu == num_xl == n``,
    identity index maps): the row index equals the bound index, so no gather is
    needed. Inactive bounds carry ``w_*_delta_inv == 0`` and recover to 0::

        lhs.z_u  =  w_u_delta_inv  * ( G_dx                 - updated_rhs_z_u)
        lhs.z_l  =  w_l_delta_inv  * (-G_dx                 - updated_rhs_z_l)
        lhs.z_bu =  w_bu_delta_inv * ( xbs * lhs.x - rhs.z_bu + m_z_bu_inv * rhs.s_bu)
        lhs.z_bl = -w_bl_delta_inv * ( xbs * lhs.x + rhs.z_bl - m_z_bl_inv * rhs.s_bl)
    """
    @wp.kernel
    def recover_duals_kernel(
        G_dx: wp.array2d(dtype=dtype),  # type: ignore
        lhs_x: wp.array2d(dtype=dtype),  # type: ignore
        # h_u
        w_u_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_u: wp.array2d(dtype=dtype),  # type: ignore
        lhs_z_u: wp.array2d(dtype=dtype),  # type: ignore
        # h_l
        w_l_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_l: wp.array2d(dtype=dtype),  # type: ignore
        lhs_z_l: wp.array2d(dtype=dtype),  # type: ignore
        # x_u
        w_bu_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        m_z_bu_inv: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_bu: wp.array2d(dtype=dtype),  # type: ignore
        rhs_s_bu: wp.array2d(dtype=dtype),  # type: ignore
        lhs_z_bu: wp.array2d(dtype=dtype),  # type: ignore
        # x_l
        w_bl_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        m_z_bl_inv: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_bl: wp.array2d(dtype=dtype),  # type: ignore
        rhs_s_bl: wp.array2d(dtype=dtype),  # type: ignore
        lhs_z_bl: wp.array2d(dtype=dtype),  # type: ignore
        # x_b_scaling
        x_b_scaling: wp.array2d(dtype=dtype),  # type: ignore
    ):
        b, t = wp.tid()
        num_hu_static = wp.static(num_hu)
        num_hl_static = wp.static(num_hl)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)

        if t < num_hu_static:
            lhs_z_u[b, t] = (G_dx[b, t] - rhs_z_u[b, t]) * w_u_delta_inv[b, t]
        elif t < num_hu_static + num_hl_static:
            j = t - num_hu_static
            lhs_z_l[b, j] = (-G_dx[b, j] - rhs_z_l[b, j]) * w_l_delta_inv[b, j]
        elif t < num_hu_static + num_hl_static + num_xu_static:
            j = t - num_hu_static - num_hl_static
            lhs_z_bu[b, j] = (x_b_scaling[b, j] * lhs_x[b, j] - rhs_z_bu[b, j] + m_z_bu_inv[b, j] * rhs_s_bu[b, j]) * w_bu_delta_inv[b, j]
        elif t < num_hu_static + num_hl_static + num_xu_static + num_xl_static:
            j = t - num_hu_static - num_hl_static - num_xu_static
            lhs_z_bl[b, j] = -(x_b_scaling[b, j] * lhs_x[b, j] + rhs_z_bl[b, j] - m_z_bl_inv[b, j] * rhs_s_bl[b, j]) * w_bl_delta_inv[b, j]
        else:
            return

    return recover_duals_kernel


def create_recover_slacks_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Create kernel specialized for eliminating slacks. Performs the operation:

        updated_lhs_z_u = inv(Z_u) (r_s_u - S_u lhs_z_u)
        updated_lhs_s_l = inv(Z_l) (r_s_l - S_l lhs_z_l)
        updated_lhs_s_bu = inv(Z_bu) (r_s_bu - S_bu lhs_z_bu)
        updated_lhs_s_bl = inv(Z_bl) (r_s_bl - S_bl lhs_z_bl)

        Since s and z are stored contiguously, it becomes:
        lhs_s_all[t] = m_z_inv_all[t] * (-m_s_all[t] * lhs_z_all[t] + rhs_s_all[t])

        The expression is written as (-m_s) * lhs_z + rhs_s to trigger FMA on GPU.
    """
    @wp.kernel
    def recover_slacks_kernel(
        rhs_s_all: wp.array2d(dtype=dtype),    # type: ignore
        lhs_z_all: wp.array2d(dtype=dtype),    # type: ignore
        m_s_all: wp.array2d(dtype=dtype),      # type: ignore
        m_z_inv_all: wp.array2d(dtype=dtype),  # type: ignore
        lhs_s_all: wp.array2d(dtype=dtype),    # type: ignore
    ):
        b, i = wp.tid()
        lhs_s_all[b, i] = m_z_inv_all[b, i] * ((-m_s_all[b, i]) * lhs_z_all[b, i] + rhs_s_all[b, i])

    return recover_slacks_kernel


def create_recover_slacks_transposed_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Transposed (K^T) variant of recover_slacks. The slack rows in K^T read
    ``I lhs_z + Z lhs_s = rhs_s``, so ``lhs_s = inv(Z) (rhs_s - lhs_z)``.

        lhs_s_all[b, i] = m_z_inv_all[b, i] * (-lhs_z_all[b, i] + rhs_s_all[b, i])
    """
    @wp.kernel
    def recover_slacks_transposed_kernel(
        rhs_s_all: wp.array2d(dtype=dtype),    # type: ignore
        lhs_z_all: wp.array2d(dtype=dtype),    # type: ignore
        m_z_inv_all: wp.array2d(dtype=dtype),  # type: ignore
        lhs_s_all: wp.array2d(dtype=dtype),    # type: ignore
    ):
        b, i = wp.tid()
        lhs_s_all[b, i] = m_z_inv_all[b, i] * (-lhs_z_all[b, i] + rhs_s_all[b, i])

    return recover_slacks_transposed_kernel