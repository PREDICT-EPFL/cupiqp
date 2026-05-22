import warp as wp
from .utils import to_warp_dtype


def create_build_inverse_index_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Create a kernel that builds an inverse index map: inv_idx[idx[t]] = t."""
    @wp.kernel
    def build_inverse_index_kernel(
        idx: wp.array(dtype=wp.int32),       # type: ignore
        inv_idx: wp.array(dtype=wp.int32),   # type: ignore
    ):
        t = wp.tid()
        inv_idx[idx[t]] = t
    return build_inverse_index_kernel


def create_update_regularizations_step_1_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Create kernel operating on contiguous s_all/z_all buffers. Performs:

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

        m_s_all[b, i]         = vars_s_all[b, i]
        m_z_inv_all[b, i]     = 1.0 / vars_z_all[b, i]
        w_delta_inv_all[b, i] = 1.0 / (s * z_inv + delta[b])
    """
    @wp.kernel
    def update_regularizations_step_1_kernel(
        vars_s_all: wp.array2d(dtype=dtype),       # type: ignore
        vars_z_all: wp.array2d(dtype=dtype),       # type: ignore
        m_s_all: wp.array2d(dtype=dtype),           # type: ignore
        m_z_inv_all: wp.array2d(dtype=dtype),       # type: ignore
        w_delta_inv_all: wp.array2d(dtype=dtype),   # type: ignore
        delta: wp.array(dtype=dtype),               # type: ignore
    ):
        b, i = wp.tid()
        s = vars_s_all[b, i]
        z_inv = dtype(1.0) / vars_z_all[b, i]
        m_s_all[b, i] = s
        m_z_inv_all[b, i] = z_inv
        w_delta_inv_all[b, i] = dtype(1.0) / (s * z_inv + delta[b])
    return update_regularizations_step_1_kernel


def create_update_regularizations_step_2_kernel(nx: int, nz: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Create kernel specialized for computing the regularization terms for x and z
    using a gather pattern.

    Equivalent to:
        x_reg[:] = rho
        x_reg[idx_xu] += w_bu_delta_inv
        x_reg[idx_xl] += w_bl_delta_inv

        z_reg[:] = 0.
        z_reg[idx_hu] += w_u_delta_inv
        z_reg[idx_hl] += w_l_delta_inv
        z_reg[:] = 1. / z_reg

    Each thread writes only to its own unique slot (x_reg[b, t] or z_reg[b, tz]),
    using inverse index maps to gather contributions.
    """
    @wp.kernel
    def update_regularizations_step_2_kernel(
        inv_idx_xu: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_xl: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_hu: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_hl: wp.array(dtype=wp.int32),  # type: ignore
        w_bu_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        w_bl_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        x_b_scaling: wp.array2d(dtype=dtype),  # type: ignore
        rho: wp.array(dtype=dtype),  # type: ignore
        x_reg: wp.array2d(dtype=dtype),  # type: ignore
        w_u_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        w_l_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        z_reg: wp.array2d(dtype=dtype),  # type: ignore
    ):
        b, t = wp.tid()
        nx_static = wp.static(nx)
        nz_static = wp.static(nz)

        if t < nx_static:
            val = rho[b]
            xb_scaling = x_b_scaling[b, t]
            xb_scaling_squared = xb_scaling * xb_scaling
            ixu = inv_idx_xu[t]
            ixl = inv_idx_xl[t]
            if ixu >= 0:
                val = val + xb_scaling_squared * w_bu_delta_inv[b, ixu]
            if ixl >= 0:
                val = val + xb_scaling_squared * w_bl_delta_inv[b, ixl]
            x_reg[b, t] = val
        elif t < nx_static + nz_static:
            tz = t - nx_static
            val = dtype(0.)
            ihu = inv_idx_hu[tz]
            ihl = inv_idx_hl[tz]
            if ihu >= 0:
                val = val + w_u_delta_inv[b, ihu]
            if ihl >= 0:
                val = val + w_l_delta_inv[b, ihl]
            z_reg[b, tz] = dtype(1.0) / val

    return update_regularizations_step_2_kernel


def create_eliminate_duals_kernel(nx: int, nz: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Create kernel specialized for eliminating duals using a gather pattern.

    Equivalent to:
        rhs_x_updated[:] = rhs_x
        rhs_x_updated[idx_xu] += w_bu_delta_inv * rhs_z_bu
        rhs_x_updated[idx_xl] -= w_bl_delta_inv * rhs_z_bl

        rhs_z_updated[:] = 0.
        rhs_z_updated[idx_hu] += w_u_delta_inv * rhs_z_u
        rhs_z_updated[idx_hl] -= w_l_delta_inv * rhs_z_l
        rhs_z_updated[:] *= z_reg

    Each thread writes only to its own unique slot (rhs_x_updated[t] or
    rhs_z_updated[tz]), using inverse index maps to gather contributions.
    """
    @wp.kernel
    def eliminate_duals_kernel(
        # inverse index maps (gather lookups)
        inv_idx_xu: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_xl: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_hu: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_hl: wp.array(dtype=wp.int32),  # type: ignore
        # prepare new rhs_x
        rhs_x: wp.array2d(dtype=dtype),  # type: ignore
        w_bu_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        w_bl_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        x_b_scaling: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_bu: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_bl: wp.array2d(dtype=dtype),  # type: ignore
        rhs_x_updated: wp.array2d(dtype=dtype),  # type: ignore
        # prepare new rhs_z
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
            val = rhs_x[b, t]
            xb_scaling = x_b_scaling[b, t]
            ixu = inv_idx_xu[t]
            ixl = inv_idx_xl[t]
            if ixu >= 0:
                val = val + xb_scaling * w_bu_delta_inv[b, ixu] * rhs_z_bu[b, ixu]
            if ixl >= 0:
                val = val - xb_scaling * w_bl_delta_inv[b, ixl] * rhs_z_bl[b, ixl]
            rhs_x_updated[b, t] = val

        elif t < nx_static + nz_static:
            tz = t - nx_static
            val = dtype(0.)
            ihu = inv_idx_hu[tz]
            ihl = inv_idx_hl[tz]
            if ihu >= 0:
                val = val + w_u_delta_inv[b, ihu] * rhs_z_u[b, ihu]
            if ihl >= 0:
                val = val - w_l_delta_inv[b, ihl] * rhs_z_l[b, ihl]
            rhs_z_updated[b, tz] = val * z_reg[b, tz]

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
    """Create kernel specialized for recovering duals. Performs the operation:

    Performs the operation:
        lhs.z_u = self._w_u_delta_inv * (G_dx[:, data.idx_hu] - self._updated_rhs_z_u)
        lhs.z_l = self._w_l_delta_inv * (-G_dx[:, data.idx_hl] - self._updated_rhs_z_l)
        lhs.z_bu = self._w_bu_delta_inv * (xbs * lhs.x[:, data.idx_xu] - rhs.z_bu + self._m_z_bu_inv * rhs.s_bu)
        lhs.z_bl = -self._w_bl_delta_inv * (xbs * lhs.x[:, data.idx_xl] + rhs.z_bl - self._m_z_bl_inv * rhs.s_bl)
    """
    @wp.kernel
    def recover_duals_kernel(
        G_dx: wp.array2d(dtype=dtype),  # type: ignore
        lhs_x: wp.array2d(dtype=dtype),  # type: ignore
        # h_u
        idx_hu: wp.array(dtype=wp.int32),  # type: ignore
        w_u_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_u: wp.array2d(dtype=dtype),  # type: ignore
        lhs_z_u: wp.array2d(dtype=dtype),  # type: ignore
        # h_l
        idx_hl: wp.array(dtype=wp.int32),  # type: ignore
        w_l_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_l: wp.array2d(dtype=dtype),  # type: ignore
        lhs_z_l: wp.array2d(dtype=dtype),  # type: ignore
        # x_u
        idx_xu: wp.array(dtype=wp.int32),  # type: ignore
        w_bu_delta_inv: wp.array2d(dtype=dtype),  # type: ignore
        m_z_bu_inv: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_bu: wp.array2d(dtype=dtype),  # type: ignore
        rhs_s_bu: wp.array2d(dtype=dtype),  # type: ignore
        lhs_z_bu: wp.array2d(dtype=dtype),  # type: ignore
        # x_l
        idx_xl: wp.array(dtype=wp.int32),  # type: ignore
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
            lhs_z_u[b, t] = (G_dx[b, idx_hu[t]] - rhs_z_u[b, t]) * w_u_delta_inv[b, t]
        elif t < num_hu_static + num_hl_static:
            j = t - num_hu_static
            lhs_z_l[b, j] = (-G_dx[b, idx_hl[j]] - rhs_z_l[b, j]) * w_l_delta_inv[b, j]
        elif t < num_hu_static + num_hl_static + num_xu_static:
            j = t - num_hu_static - num_hl_static
            idx = idx_xu[j]
            lhs_z_bu[b, j] = (x_b_scaling[b, idx] * lhs_x[b, idx] - rhs_z_bu[b, j] + m_z_bu_inv[b, j] * rhs_s_bu[b, j]) * w_bu_delta_inv[b, j]
        elif t < num_hu_static + num_hl_static + num_xu_static + num_xl_static:
            j = t - num_hu_static - num_hl_static - num_xu_static
            idx = idx_xl[j]
            lhs_z_bl[b, j] = -(x_b_scaling[b, idx] * lhs_x[b, idx] + rhs_z_bl[b, j] - m_z_bl_inv[b, j] * rhs_s_bl[b, j]) * w_bl_delta_inv[b, j]
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