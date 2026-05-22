import warp as wp
from ..utils import to_warp_dtype


def create_update_kkt_diag_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused scatter into the diagonal of the KKT matrix in sparse kkt solver. Performs:
    
        kkt_data[:, idx_x] = P_diag + x_reg
        kkt_data[:, idx_y] = -delta
        kkt_data[:, idx_z] = -z_reg
    """
    @wp.kernel
    def update_kkt_diag_kernel(
        P_diag:         wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        x_reg:          wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        delta:          wp.array(dtype=dtype),    # type: ignore  (B,)
        z_reg:          wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        diag_x_indices: wp.array(dtype=wp.int32),      # type: ignore  (n,)
        diag_y_indices: wp.array(dtype=wp.int32),      # type: ignore  (p,)
        diag_z_indices: wp.array(dtype=wp.int32),      # type: ignore  (m,)
        kkt_data:       wp.array2d(dtype=dtype),  # type: ignore  (B, kkt_nnz)
    ):
        b, t = wp.tid()
        n_static = wp.static(n)
        p_static = wp.static(p)
        m_static = wp.static(m)

        if t < n_static:
            kkt_data[b, diag_x_indices[t]] = P_diag[b, t] + x_reg[b, t]
        elif t < n_static + p_static:
            i = t - n_static
            kkt_data[b, diag_y_indices[i]] = -delta[b]
        elif t < n_static + p_static + m_static:
            k = t - n_static - p_static
            kkt_data[b, diag_z_indices[k]] = -z_reg[b, k]

    return update_kkt_diag_kernel
