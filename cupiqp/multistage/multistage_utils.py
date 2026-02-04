import warp as wp
import cupyx.scipy.sparse as cpsp


def create_csr_add_btd_kernel(num_blocks: int, block_size: int, dtype=wp.float64):
    @wp.kernel
    def _csr_add_btd_kernel(
        alpha: dtype,
        row_ptr: wp.array(dtype=wp.int32),
        col_idx: wp.array(dtype=wp.int32),
        vals: wp.array(dtype=dtype),
        diag_data: wp.array3d(dtype=dtype),
        off_lower_data: wp.array3d(dtype=dtype),
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


def create_add_on_diag_kernel(block_size: int, dtype=wp.float64):
    @wp.kernel
    def _add_on_diag_kernel(
        x: wp.array(dtype=dtype),
        diag_data: wp.array3d(dtype=dtype),
    ):
        """
        Add a vector x to the diagonal of a block-tridiagonal matrix.
        """
        block_size_static = wp.static(block_size)
        row = wp.tid()
        br = row // block_size_static
        lr = row - br * block_size_static
        diag_data[br, lr, lr] += x[row]
        
    return _add_on_diag_kernel


class DenseBlocks:
    """
    Contiguous storage for a list of dense blocks with metadata.

    Only supports uniform block sizes for now, but can be extended to variable block sizes.
    """
    def __init__(self, num_blocks: int, rows: int, cols: int, dtype=wp.float64, device="cuda"):
        self._device = device
        self._dtype = dtype
        self.data = wp.zeros((num_blocks, rows, cols), dtype=dtype, device=device)
        

class BlockTridiagMat:
    """
    Stores a symmetric block-tridiagonal matrix as 2 DenseBlocks:
      - diag_blocks: square blocks on the diagonal
      - off_diag_blocks_lower: rectangular blocks below the diagonal
    """
    def __init__(self, num_diag_blocks: int, block_size: int, dtype=wp.float64, device="cuda"):
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
