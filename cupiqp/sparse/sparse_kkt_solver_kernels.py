import warp as wp
from ..utils import to_warp_dtype


def create_update_kkt_diag_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused scatter into the diagonal of the sparse KKT matrix:

        kkt_data[:, idx_x] = P_diag + x_reg
        kkt_data[:, idx_y] = -delta
        kkt_data[:, idx_z] = -z_reg          (active row,   z_reg > 0)
        kkt_data[:, idx_z] = -1              (inactive row, z_reg == 0)

    An inactive inequality row (both bounds infinite, z_reg == 0) is given a
    benign -1 diagonal and its G coupling is zeroed elsewhere so that this row
    contributes nothing to the solution of kkt solution.
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
            zr = z_reg[b, k]
            # store the diagonal value for inactive G rows to be -1
            kkt_data[b, diag_z_indices[k]] = wp.where(zr > dtype(0.0), -zr, -dtype(1.0))

    return update_kkt_diag_kernel


def create_scatter_masked_G_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Scatter G values into the KKT data buffer, zeroing the
    contribution of inactive inequality rows.

    Replaces the two-allocation cupy chain

        G_mask = active_G_row[:, G_row_idx]            # (B, nnz_G) gather
        kkt_data[:, G_indices] = G.data * G_mask       # (B, nnz_G) temporary

    with a single launch over ``(B, nnz_G)``. For each batch ``b`` and each
    stored non-zero ``k`` of ``G``::

        kkt_data[b, G_indices[k]] = G_data[b, k] * active_G_row[b, G_row_idx[k]]

    where ``active_G_row[b, row]`` is 1.0 for an active inequality row and 0.0
    for an inactive one (both bounds infinite), so the inactive rows' G / G^T
    coupling is zeroed in place.
    """
    @wp.kernel
    def scatter_masked_G_kernel(
        G_data:       wp.array2d(dtype=dtype),    # type: ignore  (B, nnz_G)
        active_G_row: wp.array2d(dtype=dtype),    # type: ignore  (B, m)
        G_row_idx:    wp.array(dtype=wp.int32),   # type: ignore  (nnz_G,) row of each G nnz
        G_indices:    wp.array(dtype=wp.int32),   # type: ignore  (nnz_G,) slot in kkt_data
        kkt_data:     wp.array2d(dtype=dtype),    # type: ignore  (B, kkt_nnz) in/out
    ):
        b, k = wp.tid()
        row = G_row_idx[k]
        kkt_data[b, G_indices[k]] = G_data[b, k] * active_G_row[b, row]

    return scatter_masked_G_kernel
