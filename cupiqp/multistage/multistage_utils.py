import warp as wp
import cupyx.scipy.sparse as cpsp


def create_csr_add_btd_kernel(num_blocks: int, block_size: int, dtype=wp.float64):
    @wp.kernel
    def _csr_add_btd_kernel(
        alpha: dtype,                               # type: ignore
        row_ptr: wp.array(dtype=wp.int32),          # type: ignore
        col_idx: wp.array(dtype=wp.int32),          # type: ignore
        vals: wp.array(dtype=dtype),                # type: ignore
        diag_data: wp.array3d(dtype=dtype),         # type: ignore
        off_lower_data: wp.array3d(dtype=dtype),    # type: ignore
    ):
        """
        Perform Y = alpha*X + Y, where X is in CSR format and Y is in block-tridiagonal (BTD) format.
        Only adds to diagonal and lower blocks.
        """
        num_blocks_static = wp.static(num_blocks)
        block_size_static = wp.static(block_size)
        row = wp.tid()  # each thread processes one row

        if row >= num_blocks_static * block_size_static:
            return
        
        start = row_ptr[row]
        end = row_ptr[row + 1]
        
        br = row // block_size_static       # block row index
        lr = row - br * block_size_static   # local row index within the block
        
        for p in range(start, end):
            c = col_idx[p]
            v = vals[p]
            
            bc = c // block_size_static       # block column index
            lc = c - bc * block_size_static   # local column index within the block

            # diagonal block
            if br == bc:
                diag_data[br, lr, lc] += alpha * v
            # lower off-diagonal
            elif br == bc + 1:                
                off_lower_data[br - 1, lr, lc] += alpha * v
            else:
                pass
        
    return _csr_add_btd_kernel


def create_block_tridiag_diaad_kernel(block_size: int, dtype=wp.float64):
    """diag(A) += alpha*x, where A is block-tridiagonal and x is a vector."""
    @wp.kernel
    def _block_tridiag_diaad_kernel(
        x: wp.array(dtype=dtype),              # type: ignore
        diag_blocks: wp.array3d(dtype=dtype),  # type: ignore
    ):
        block_size_static = wp.static(block_size)
        row = wp.tid()
        br = row // block_size_static
        lr = row - br * block_size_static
        diag_blocks[br, lr, lr] += x[row]
        
    return _block_tridiag_diaad_kernel


def create_block_tridiag_gead_kernel(num_blocks: int, block_size: int, dtype=wp.float64):
    """B += alpha*A, where A and B are block-tridiagonal matrices."""
    @wp.kernel
    def _block_tridiag_gead_kernel(
        alpha: wp.array(dtype=dtype),  # type: ignore
        A_diag: wp.array3d(dtype=dtype),  # type: ignore
        A_offdiag: wp.array3d(dtype=dtype),  # type: ignore
        B_diag: wp.array3d(dtype=dtype),  # type: ignore
        B_offdiag: wp.array3d(dtype=dtype),  # type: ignore
    ):
        k, i, j = wp.tid()
        N = wp.static(num_blocks)
        B_diag[k, i, j] += alpha[0] * A_diag[k, i, j]
        if k < N - 1:
            B_offdiag[k, i, j] += alpha[0] * A_offdiag[k, i, j]
    return _block_tridiag_gead_kernel


def create_block_bidiag_gemv_n_kernel(num_blocks: int, rows_of_blocks: int, cols_of_blocks: int, dtype=wp.float64):
    """y = alpha * A * x + beta * y, where A is block lower bidiagonal.

    A has N+1 block rows, N block columns.
    Launch with dim=(num_blocks + 1, rows_of_blocks).
    """
    @wp.kernel
    def _block_bidiag_gemv_n_kernel(
        alpha: dtype,  # type: ignore
        A_D: wp.array3d(dtype=dtype),  # type: ignore   Shape (N, r, c)
        A_E: wp.array3d(dtype=dtype),  # type: ignore   Shape (N, r, c)
        x: wp.array(dtype=dtype),      # type: ignore   Shape (N*c,)
        beta: dtype,  # type: ignore
        y: wp.array(dtype=dtype),      # type: ignore   Shape ((N+1)*r,)
    ):
        block_row, local_row = wp.tid()
        N = wp.static(num_blocks)
        r = wp.static(rows_of_blocks)
        c = wp.static(cols_of_blocks)

        if block_row > N:
            return

        acc = wp.float64(0.0)

        # D_{block_row} contribution (block_row = 0..N-1)
        if block_row < N:
            for j in range(c):
                acc += A_D[block_row, local_row, j] * x[block_row * c + j]

        # E_{block_row-1} contribution (block_row = 1..N)
        if block_row > 0:
            for j in range(c):
                acc += A_E[block_row - 1, local_row, j] * x[(block_row - 1) * c + j]

        idx = block_row * r + local_row
        y[idx] = alpha * acc + beta * y[idx]

    return _block_bidiag_gemv_n_kernel


def create_block_bidiag_gemv_t_kernel(num_blocks: int, rows_of_blocks: int, cols_of_blocks: int, dtype=wp.float64):
    """z = alpha * A^T * y + beta * z, where A is block lower bidiagonal.

    A^T has N block rows (cols of A), N+1 block columns (rows of A).
    z[k] = alpha * (D_k^T y[k] + E_k^T y[k+1]) + beta * z[k]   for k=0..N-1
    Launch with dim=(num_blocks, cols_of_blocks).
    """
    @wp.kernel
    def _block_bidiag_gemv_t_kernel(
        alpha: dtype,                  # type: ignore
        A_D: wp.array3d(dtype=dtype),  # type: ignore   Shape (N, r, c)
        A_E: wp.array3d(dtype=dtype),  # type: ignore   Shape (N, r, c)
        y: wp.array(dtype=dtype),      # type: ignore   Shape ((N+1)*r,)
        beta: dtype,                   # type: ignore
        z: wp.array(dtype=dtype),      # type: ignore   Shape (N*c,)
    ):
        k, local_col = wp.tid()
        N = wp.static(num_blocks)
        r = wp.static(rows_of_blocks)
        c = wp.static(cols_of_blocks)

        if k >= N:
            return

        acc = wp.float64(0.0)

        # D_k^T * y[k]
        for p in range(r):
            acc += A_D[k, p, local_col] * y[k * r + p]

        # E_k^T * y[k+1]
        for p in range(r):
            acc += A_E[k, p, local_col] * y[(k + 1) * r + p]

        idx = k * c + local_col
        z[idx] = alpha * acc + beta * z[idx]

    return _block_bidiag_gemv_t_kernel


def create_block_tridiag_gemv_kernel(num_blocks: int, block_size: int, dtype=wp.float64):
    """z = alpha * P * x + beta * z, where P is symmetric block-tridiagonal.

    P_D: (N, d, d) diagonal blocks, P_E: (N-1, d, d) lower off-diagonal.
    Launch with dim=(num_blocks, block_size).
    """
    @wp.kernel
    def _block_tridiag_gemv_kernel(
        alpha: dtype,                  # type: ignore
        P_D: wp.array3d(dtype=dtype),  # type: ignore   Shape (N, d, d)
        P_E: wp.array3d(dtype=dtype),  # type: ignore   Shape (N-1, d, d)
        x: wp.array(dtype=dtype),      # type: ignore   Shape (N*d,)
        beta: dtype,                   # type: ignore
        z: wp.array(dtype=dtype),      # type: ignore   Shape (N*d,)
    ):
        k, local_row = wp.tid()
        N = wp.static(num_blocks)
        d = wp.static(block_size)

        if k >= N:
            return

        acc = wp.float64(0.0)

        # P_D[k] * x[k]
        for j in range(d):
            acc += P_D[k, local_row, j] * x[k * d + j]

        # P_E[k-1] * x[k-1]  (lower off-diagonal, k >= 1)
        if k > 0:
            for j in range(d):
                acc += P_E[k - 1, local_row, j] * x[(k - 1) * d + j]

        # P_E[k]^T * x[k+1]  (upper = transpose of lower, k < N-1)
        if k < N - 1:
            for j in range(d):
                acc += P_E[k, j, local_row] * x[(k + 1) * d + j]

        idx = k * d + local_row
        z[idx] = alpha * acc + beta * z[idx]

    return _block_tridiag_gemv_kernel


class DenseBlocks:
    """
    Contiguous storage for a list of dense blocks with metadata.

    Only supports uniform block sizes for now, but can be extended to variable block sizes.
    """
    def __init__(self, num_blocks: int, rows: int, cols: int, dtype=wp.float64, device="cuda"):
        self._device = device
        self._dtype = dtype
        self.data = wp.zeros((num_blocks, rows, cols), dtype=dtype, device=device)


class BlockBidiagMat:
    """
    Used to store the A and G matrices in the multistage problem, which have a block lower bidiagonal structure.

    A = 
    [
    D0                                   
    E0  D1                               
        E1  D2                           
            E2  D3                       
                    ...                  
                    E_{N-2} D_{N-1}      
                            E_{N-1}
    ]
    """
    def __init__(self, rows_of_blocks: int, cols_of_blocks: int, N: int):
        self.N = N
        self.cols_of_blocks = cols_of_blocks
        self.rows_of_blocks = rows_of_blocks
        self.D = wp.zeros((N, self.rows_of_blocks, self.cols_of_blocks), dtype=wp.float64, device="cuda")
        self.E = wp.zeros((N, self.rows_of_blocks, self.cols_of_blocks), dtype=wp.float64, device="cuda")



def create_block_syrk_kernel(
    num_blocks: int,
    rows_of_blocks: int,
    cols_of_blocks: int,
):
    """
    Create a Warp kernel that computes the block-tridiagonal symmetric rank-k update:

        C = alpha * A^T A + beta * C

    where A is block lower bidiagonal with structure:

        row 0:      D0
        row 1:      E0, D1
        row 2:          E1, D2
        ...
        row N-1:                E_{N-2}, D_{N-1}
        row N:                           E_{N-1}

    Producing:
        C_D[k] = alpha * (D_k^T D_k + E_k^T E_k) + beta * C_D[k]   for k = 0..N-1
        C_E[k] = alpha * (D_{k+1}^T E_k)          + beta * C_E[k]   for k = 0..N-2

    Parameters
    ----------
    num_blocks : int
        Number of diagonal blocks N.
    rows_of_blocks : int
        Row dimension of each D_k / E_k block.
    cols_of_blocks : int
        Column dimension of each D_k / E_k block (= size of C blocks).

    Returns
    -------
    block_syrk_kernel : wp.Kernel
        Launch with ``wp.launch(kernel, dim=(num_blocks, cols_of_blocks, cols_of_blocks), ...)``.
    """

    @wp.kernel
    def block_syrk_kernel(
        alpha: wp.float64,
        A_D: wp.array3d(dtype=wp.float64),  # type: ignore   Shape (num_blocks, rows_of_blocks, cols_of_blocks)
        A_E: wp.array3d(dtype=wp.float64),  # type: ignore   Shape (num_blocks, rows_of_blocks, cols_of_blocks)
        beta: wp.float64,
        C_D: wp.array3d(dtype=wp.float64),  # type: ignore   Shape (num_blocks, cols_of_blocks, cols_of_blocks)
        C_E: wp.array3d(dtype=wp.float64),  # type: ignore   Shape (num_blocks - 1, cols_of_blocks, cols_of_blocks)
    ):
        k, i, j = wp.tid()

        N = wp.static(num_blocks)
        r = wp.static(rows_of_blocks)

        # ----- Diagonal block C_D[k] = alpha * (D_k^T D_k + E_k^T E_k) + beta * C_D[k] -----
        acc_diag = wp.float64(0.0)
        for p in range(r):
            acc_diag += A_D[k, p, i] * A_D[k, p, j]
            acc_diag += A_E[k, p, i] * A_E[k, p, j]

        C_D[k, i, j] = alpha * acc_diag + beta * C_D[k, i, j]

        # ----- Lower off-diagonal C_E[k] = alpha * D_{k+1}^T E_k + beta * C_E[k] -----
        if k < N - 1:
            acc_off = wp.float64(0.0)
            for p in range(r):
                acc_off += A_D[k + 1, p, i] * A_E[k, p, j]

            C_E[k, i, j] = alpha * acc_off + beta * C_E[k, i, j]

    return block_syrk_kernel


def create_weighted_block_syrk_kernel(
    num_blocks: int,
    rows_of_blocks: int,
    cols_of_blocks: int,
):
    """
    Weighted block SYRK: C = alpha * A^T diag(w) A + beta * C

    Same block bidiagonal structure as ``create_block_syrk_kernel``, but each
    row of A is scaled by the corresponding weight in *w*.

    Parameters
    ----------
    num_blocks, rows_of_blocks, cols_of_blocks : int
        Same meaning as in ``create_block_syrk_kernel``.

    Returns
    -------
    weighted_block_syrk_kernel : wp.Kernel
        Launch with ``dim=(num_blocks, cols_of_blocks, cols_of_blocks)``.
        *w* is a flat vector of length ``(num_blocks + 1) * rows_of_blocks``.
    """

    @wp.kernel
    def weighted_block_syrk_kernel(
        alpha: wp.float64,
        A_D: wp.array3d(dtype=wp.float64),  # type: ignore   Shape (num_blocks, rows_of_blocks, cols_of_blocks)
        A_E: wp.array3d(dtype=wp.float64),  # type: ignore   Shape (num_blocks, rows_of_blocks, cols_of_blocks)
        w: wp.array(dtype=wp.float64),      # type: ignore   Shape ((num_blocks + 1) * rows_of_blocks,)
        beta: wp.float64,
        C_D: wp.array3d(dtype=wp.float64),  # type: ignore   Shape (num_blocks, cols_of_blocks, cols_of_blocks)
        C_E: wp.array3d(dtype=wp.float64),  # type: ignore   Shape (num_blocks - 1, cols_of_blocks, cols_of_blocks)
    ):
        k, i, j = wp.tid()

        N = wp.static(num_blocks)
        r = wp.static(rows_of_blocks)

        # ----- Diagonal: D_k^T diag(w_k) D_k + E_k^T diag(w_{k+1}) E_k -----
        acc_diag = wp.float64(0.0)
        for p in range(r):
            w_dk = w[k * r + p]
            w_ek = w[(k + 1) * r + p]
            acc_diag += w_dk * A_D[k, p, i] * A_D[k, p, j]
            acc_diag += w_ek * A_E[k, p, i] * A_E[k, p, j]

        C_D[k, i, j] = alpha * acc_diag + beta * C_D[k, i, j]

        # ----- Off-diagonal: D_{k+1}^T diag(w_{k+1}) E_k -----
        if k < N - 1:
            acc_off = wp.float64(0.0)
            for p in range(r):
                w_kp1 = w[(k + 1) * r + p]
                acc_off += w_kp1 * A_D[k + 1, p, i] * A_E[k, p, j]

            C_E[k, i, j] = alpha * acc_off + beta * C_E[k, i, j]

    return weighted_block_syrk_kernel


class BlockTridiagMat:
    """
    Stores a symmetric block-tridiagonal matrix as 2 DenseBlocks:
      - diag_blocks: square blocks on the diagonal
      - off_diag_blocks_lower: rectangular blocks below the diagonal
    """
    def __init__(self, num_diag_blocks: int, block_size: int, dtype=wp.float64, device="cuda"):
        self.block_size = block_size
        self.diag_blocks = DenseBlocks(num_blocks=num_diag_blocks, rows=block_size, cols=block_size, dtype=dtype, device=device)
        self.off_diag_blocks_lower = DenseBlocks(num_blocks=num_diag_blocks-1, rows=block_size, cols=block_size, dtype=dtype, device=device)
        
        self._csr_add_to_btd_kernel = create_csr_add_btd_kernel(num_diag_blocks, block_size, dtype)

    @property
    def num_diag_blocks(self):
        return self.diag_blocks.data.shape[0]
    
    @property
    def rows(self):
        return self.num_diag_blocks * self.diag_blocks.data.shape[1]
    
    @property
    def cols(self):
        return self.num_diag_blocks * self.diag_blocks.data.shape[2]

    @classmethod 
    def from_csr(cls, A_csr: cpsp.csr_matrix, block_size: int, dtype=wp.float64, device="cuda"):
        if A_csr.shape[0] != A_csr.shape[1]:
            raise ValueError("The CSR matrix must be square")
        
        # create BlockTridiagMat instance
        num_diag_blocks = A_csr.shape[0] // block_size
        A_blk_tridiag = cls(num_diag_blocks=num_diag_blocks, block_size=block_size, dtype=dtype, device=device)        
        
        
        wp.launch(
            kernel=A_blk_tridiag._csr_add_to_btd_kernel,
            dim=A_csr.shape[0],
            inputs=[
                1.0,
                A_csr.indptr, A_csr.indices, A_csr.data,
                0.0,
                A_blk_tridiag.diag_blocks.data,
                A_blk_tridiag.off_diag_blocks_lower.data,
            ],
            device=device
        )
        return A_blk_tridiag
    
    def copy_from_csr(self, A_csr: cpsp.csr_matrix):
        if A_csr.shape[0] != A_csr.shape[1]:
            raise ValueError("The CSR matrix must be square")
        if A_csr.shape[0] != self.rows:
            raise ValueError("The CSR matrix size must match the block matrix size")
        
        wp.launch(
            kernel=self._csr_add_to_btd_kernel,
            dim=A_csr.shape[0],
            inputs=[
                1.0,
                A_csr.indptr, A_csr.indices, A_csr.data,
                0.0,
                self.diag_blocks.data,
                self.off_diag_blocks_lower.data,
            ],
            device=self.diag_blocks.data.device
        )

    def add_with_csr(self, alpha: float, A_csr: cpsp.csr_matrix, beta: float = 1.0):
        """self += alpha * A_csr"""
        if A_csr.shape[0] != A_csr.shape[1]:
            raise ValueError("The CSR matrix must be square")
        if A_csr.shape[0] != self.rows:
            raise ValueError("The CSR matrix size must match the block matrix size")
        
        wp.launch(
            kernel=self._csr_add_to_btd_kernel,
            dim=A_csr.shape[0],
            inputs=[
                alpha,
                A_csr.indptr, A_csr.indices, A_csr.data,
                beta,
                self.diag_blocks.data,
                self.off_diag_blocks_lower.data,
            ],
            device=self.diag_blocks.data.device
        )

    def add_on_diag(self, x: wp.array):
        """self += diag(x)"""
        block_size = self.diag_blocks.data.shape[1]
        num_blocks = self.num_diag_blocks
        
        if x.shape[0] != self.rows:
            raise ValueError("Dimension mismatch in add_on_diag")
            
        wp.launch(
            kernel=self._add_on_diag_kernel,
            dim=self.rows,
            inputs=[x, self.diag_blocks.data],
            device=self.diag_blocks.data.device
        )


class BlockVec:
    """
    Stores a block vector as a DenseBlocks with metadata.
    """
    def __init__(self, num_blocks: int, rows: int, dtype=wp.float64, device="cuda"):
        self.num_blocks = num_blocks
        self.rows = rows
        self.data = wp.zeros((num_blocks, rows), dtype=dtype, device=device)
