import warp as wp


def create_block_tridiag_diaad_kernel(block_size: int, dtype=wp.float64):
    """``diag(A[b]) += x[b]`` for each batch b. A is block-tridiagonal.

    Launch with ``dim=(B, num_blocks * block_size)``.
    """
    @wp.kernel
    def _block_tridiag_diaad_kernel(
        x: wp.array2d(dtype=dtype),            # type: ignore   (B, N*d)
        diag_blocks: wp.array4d(dtype=dtype),  # type: ignore   (B, N, d, d)
    ):
        b, row = wp.tid()
        block_size_static = wp.static(block_size)
        br = row // block_size_static
        lr = row - br * block_size_static
        diag_blocks[b, br, lr, lr] += x[b, row]

    return _block_tridiag_diaad_kernel


def create_block_tridiag_gead_kernel(num_blocks: int, block_size: int, dtype=wp.float64):
    """``B[b] += alpha[b] * A[b]`` for each batch b. Both A and B are block-tridiag.

    ``alpha`` is a per-batch scaling array of shape ``(B,)``.
    Launch with ``dim=(B, N, d, d)``.
    """
    @wp.kernel
    def _block_tridiag_gead_kernel(
        alpha: wp.array(dtype=dtype),             # type: ignore   (B,)
        A_diag: wp.array4d(dtype=dtype),          # type: ignore   (B, N, d, d)
        A_offdiag: wp.array4d(dtype=dtype),       # type: ignore   (B, N-1, d, d)
        B_diag: wp.array4d(dtype=dtype),          # type: ignore   (B, N, d, d)
        B_offdiag: wp.array4d(dtype=dtype),       # type: ignore   (B, N-1, d, d)
    ):
        b, k, i, j = wp.tid()
        N = wp.static(num_blocks)
        a = alpha[b]
        B_diag[b, k, i, j] += a * A_diag[b, k, i, j]
        if k < N - 1:
            B_offdiag[b, k, i, j] += a * A_offdiag[b, k, i, j]

    return _block_tridiag_gead_kernel


def create_block_bidiag_gemv_n_kernel(num_blocks: int, rows_of_blocks: int, cols_of_blocks: int, dtype=wp.float64):
    """``y[b] = alpha * A[b] * x[b] + beta * y[b]``, A block lower bidiagonal.

    A has N+1 block rows, N block columns.
    Launch with ``dim=(B, num_blocks + 1, rows_of_blocks)``.
    """
    @wp.kernel
    def _block_bidiag_gemv_n_kernel(
        alpha: dtype,                             # type: ignore
        A_D: wp.array4d(dtype=dtype),             # type: ignore   (B, N, r, c)
        A_E: wp.array4d(dtype=dtype),             # type: ignore   (B, N, r, c)
        x: wp.array2d(dtype=dtype),               # type: ignore   (B, N*c)
        beta: dtype,                              # type: ignore
        y: wp.array2d(dtype=dtype),               # type: ignore   (B, (N+1)*r)
    ):
        b, block_row, local_row = wp.tid()
        N = wp.static(num_blocks)
        r = wp.static(rows_of_blocks)
        c = wp.static(cols_of_blocks)

        if block_row > N:
            return

        acc = wp.float64(0.0)

        if block_row < N:
            for j in range(c):
                acc += A_D[b, block_row, local_row, j] * x[b, block_row * c + j]

        if block_row > 0:
            for j in range(c):
                acc += A_E[b, block_row - 1, local_row, j] * x[b, (block_row - 1) * c + j]

        idx = block_row * r + local_row
        y[b, idx] = alpha * acc + beta * y[b, idx]

    return _block_bidiag_gemv_n_kernel


def create_block_bidiag_gemv_t_kernel(num_blocks: int, rows_of_blocks: int, cols_of_blocks: int, dtype=wp.float64):
    """``z[b] = alpha * A[b]^T * y[b] + beta * z[b]``, A block lower bidiagonal.

    A^T has N block rows (cols of A), N+1 block columns (rows of A).
    Launch with ``dim=(B, num_blocks, cols_of_blocks)``.
    """
    @wp.kernel
    def _block_bidiag_gemv_t_kernel(
        alpha: dtype,                             # type: ignore
        A_D: wp.array4d(dtype=dtype),             # type: ignore   (B, N, r, c)
        A_E: wp.array4d(dtype=dtype),             # type: ignore   (B, N, r, c)
        y: wp.array2d(dtype=dtype),               # type: ignore   (B, (N+1)*r)
        beta: dtype,                              # type: ignore
        z: wp.array2d(dtype=dtype),               # type: ignore   (B, N*c)
    ):
        b, k, local_col = wp.tid()
        N = wp.static(num_blocks)
        r = wp.static(rows_of_blocks)
        c = wp.static(cols_of_blocks)

        if k >= N:
            return

        acc = wp.float64(0.0)

        # D_k^T * y[k]
        for p in range(r):
            acc += A_D[b, k, p, local_col] * y[b, k * r + p]

        # E_k^T * y[k+1]
        for p in range(r):
            acc += A_E[b, k, p, local_col] * y[b, (k + 1) * r + p]

        idx = k * c + local_col
        z[b, idx] = alpha * acc + beta * z[b, idx]

    return _block_bidiag_gemv_t_kernel


def create_block_tridiag_gemv_kernel(num_blocks: int, block_size: int, dtype=wp.float64):
    """``z[b] = alpha * P[b] * x[b] + beta * z[b]``, P symmetric block-tridiagonal.

    Launch with ``dim=(B, num_blocks, block_size)``.
    """
    @wp.kernel
    def _block_tridiag_gemv_kernel(
        alpha: dtype,                             # type: ignore
        P_D: wp.array4d(dtype=dtype),             # type: ignore   (B, N, d, d)
        P_E: wp.array4d(dtype=dtype),             # type: ignore   (B, N-1, d, d)
        x: wp.array2d(dtype=dtype),               # type: ignore   (B, N*d)
        beta: dtype,                              # type: ignore
        z: wp.array2d(dtype=dtype),               # type: ignore   (B, N*d)
    ):
        b, k, local_row = wp.tid()
        N = wp.static(num_blocks)
        d = wp.static(block_size)

        if k >= N:
            return

        acc = wp.float64(0.0)

        # P_D[k] * x[k]
        for j in range(d):
            acc += P_D[b, k, local_row, j] * x[b, k * d + j]

        # P_E[k-1] * x[k-1]
        if k > 0:
            for j in range(d):
                acc += P_E[b, k - 1, local_row, j] * x[b, (k - 1) * d + j]

        # P_E[k]^T * x[k+1]
        if k < N - 1:
            for j in range(d):
                acc += P_E[b, k, j, local_row] * x[b, (k + 1) * d + j]

        idx = k * d + local_row
        z[b, idx] = alpha * acc + beta * z[b, idx]

    return _block_tridiag_gemv_kernel


class DenseBlocks:
    """Contiguous storage for a list of dense blocks with leading batch axis.

    ``data`` shape: ``(batch_size, num_blocks, rows, cols)``.
    Default ``batch_size=1`` keeps the single-QP API ergonomic.
    """
    def __init__(self, num_blocks: int, rows: int, cols: int,
                 dtype=wp.float64, device="cuda", batch_size: int = 1):
        self._device = device
        self._dtype = dtype
        self.data = wp.zeros((batch_size, num_blocks, rows, cols), dtype=dtype, device=device)


def create_block_syrk_kernel(num_blocks: int, rows_of_blocks: int, cols_of_blocks: int):
    """Batched block-tridiagonal SYRK: ``C[b] = alpha * A[b]^T A[b] + beta * C[b]``.

    A is block lower bidiagonal::

        row 0:      D0
        row 1:      E0, D1
        ...
        row N-1:                E_{N-2}, D_{N-1}
        row N:                           E_{N-1}

    Producing::

        C_D[b, k] = alpha * (D_k^T D_k + E_k^T E_k) + beta * C_D[b, k]   for k = 0..N-1
        C_E[b, k] = alpha * (D_{k+1}^T E_k)         + beta * C_E[b, k]   for k = 0..N-2

    Launch with ``dim=(B, num_blocks, cols_of_blocks, cols_of_blocks)``.
    """
    @wp.kernel
    def block_syrk_kernel(
        alpha: wp.float64,
        A_D: wp.array4d(dtype=wp.float64),        # type: ignore   (B, N, r, c)
        A_E: wp.array4d(dtype=wp.float64),        # type: ignore   (B, N, r, c)
        beta: wp.float64,
        C_D: wp.array4d(dtype=wp.float64),        # type: ignore   (B, N, c, c)
        C_E: wp.array4d(dtype=wp.float64),        # type: ignore   (B, N-1, c, c)
    ):
        b, k, i, j = wp.tid()
        N = wp.static(num_blocks)
        r = wp.static(rows_of_blocks)

        # Diagonal: alpha*(D^T D + E^T E)
        acc_diag = wp.float64(0.0)
        for p in range(r):
            acc_diag += A_D[b, k, p, i] * A_D[b, k, p, j]
            acc_diag += A_E[b, k, p, i] * A_E[b, k, p, j]
        C_D[b, k, i, j] = alpha * acc_diag + beta * C_D[b, k, i, j]

        # Lower off-diagonal: alpha * D_{k+1}^T E_k
        if k < N - 1:
            acc_off = wp.float64(0.0)
            for p in range(r):
                acc_off += A_D[b, k + 1, p, i] * A_E[b, k, p, j]
            C_E[b, k, i, j] = alpha * acc_off + beta * C_E[b, k, i, j]

    return block_syrk_kernel


def create_weighted_block_syrk_kernel(num_blocks: int, rows_of_blocks: int, cols_of_blocks: int):
    """Batched weighted block SYRK: ``C[b] = alpha * A[b]^T diag(w[b]) A[b] + beta * C[b]``.

    Same block-bidiagonal structure as ``create_block_syrk_kernel``, but each
    row of A[b] is scaled by ``w[b]``. ``w`` is shape ``(B, (N+1)*r)``.

    Launch with ``dim=(B, num_blocks, cols_of_blocks, cols_of_blocks)``.
    """
    @wp.kernel
    def weighted_block_syrk_kernel(
        alpha: wp.float64,
        A_D: wp.array4d(dtype=wp.float64),        # type: ignore   (B, N, r, c)
        A_E: wp.array4d(dtype=wp.float64),        # type: ignore   (B, N, r, c)
        w: wp.array2d(dtype=wp.float64),          # type: ignore   (B, (N+1)*r)
        beta: wp.float64,
        C_D: wp.array4d(dtype=wp.float64),        # type: ignore   (B, N, c, c)
        C_E: wp.array4d(dtype=wp.float64),        # type: ignore   (B, N-1, c, c)
    ):
        b, k, i, j = wp.tid()
        N = wp.static(num_blocks)
        r = wp.static(rows_of_blocks)

        acc_diag = wp.float64(0.0)
        for p in range(r):
            w_dk = w[b, k * r + p]
            w_ek = w[b, (k + 1) * r + p]
            acc_diag += w_dk * A_D[b, k, p, i] * A_D[b, k, p, j]
            acc_diag += w_ek * A_E[b, k, p, i] * A_E[b, k, p, j]
        C_D[b, k, i, j] = alpha * acc_diag + beta * C_D[b, k, i, j]

        if k < N - 1:
            acc_off = wp.float64(0.0)
            for p in range(r):
                w_kp1 = w[b, (k + 1) * r + p]
                acc_off += w_kp1 * A_D[b, k + 1, p, i] * A_E[b, k, p, j]
            C_E[b, k, i, j] = alpha * acc_off + beta * C_E[b, k, i, j]

    return weighted_block_syrk_kernel


# ----------------------------------------------------------------------
# Block container classes — always carry a leading batch axis. ``batch_size``
# defaults to 1 so existing single-QP callers see shape (1, ...) under the
# hood with no API change beyond a leading axis on the underlying ``.data``.
# ----------------------------------------------------------------------


class BlockTridiagMat:
    """Symmetric block-tridiagonal matrix, batched.

    Layout::

        diag_blocks.data           shape (B, N, d, d)     — diagonal blocks
        off_diag_blocks_lower.data shape (B, N-1, d, d)   — sub-diagonal blocks

    The matrix is symmetric; the upper off-diagonal is implicitly the
    transpose of the lower one. ``batch_size`` defaults to 1.
    """
    def __init__(self, num_diag_blocks: int, block_size: int,
                 dtype=wp.float64, device="cuda", batch_size: int = 1):
        self.block_size = block_size
        self.diag_blocks = DenseBlocks(
            num_blocks=num_diag_blocks, rows=block_size, cols=block_size,
            dtype=dtype, device=device, batch_size=batch_size,
        )
        self.off_diag_blocks_lower = DenseBlocks(
            num_blocks=num_diag_blocks - 1, rows=block_size, cols=block_size,
            dtype=dtype, device=device, batch_size=batch_size,
        )

    @property
    def batch_size(self):
        return self.diag_blocks.data.shape[0]

    @property
    def num_diag_blocks(self):
        return self.diag_blocks.data.shape[1]

    @property
    def rows(self):
        return self.num_diag_blocks * self.diag_blocks.data.shape[2]

    @property
    def cols(self):
        return self.num_diag_blocks * self.diag_blocks.data.shape[3]


class BlockBidiagMat:
    """Block lower-bidiagonal matrix, batched.

    Stores A or G of the multistage problem with shape::

        D shape (B, N, rows_of_blocks, cols_of_blocks)
        E shape (B, N, rows_of_blocks, cols_of_blocks)

    Logical structure (per batch)::

        A =
        [ D0
          E0  D1
              E1  D2
                  ...
                  E_{N-2} D_{N-1}
                          E_{N-1} ]
    """
    def __init__(self, rows_of_blocks: int, cols_of_blocks: int, N: int,
                 dtype=wp.float64, device="cuda", batch_size: int = 1):
        self.N = N
        self.cols_of_blocks = cols_of_blocks
        self.rows_of_blocks = rows_of_blocks
        self.D = wp.zeros((batch_size, N, rows_of_blocks, cols_of_blocks), dtype=dtype, device=device)
        self.E = wp.zeros((batch_size, N, rows_of_blocks, cols_of_blocks), dtype=dtype, device=device)

    @property
    def batch_size(self):
        return self.D.shape[0]


class BlockVec:
    """Block vector, batched. ``data`` shape ``(B, num_blocks, rows)``."""
    def __init__(self, num_blocks: int, rows: int,
                 dtype=wp.float64, device="cuda", batch_size: int = 1):
        self.num_blocks = num_blocks
        self.rows = rows
        self.data = wp.zeros((batch_size, num_blocks, rows), dtype=dtype, device=device)

    @property
    def batch_size(self):
        return self.data.shape[0]
