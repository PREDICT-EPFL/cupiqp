import numpy as np
import warp as wp
import cupy as cp
import cupyx.scipy.sparse as cpsp
import sys
sys.path.append('./')
sys.path.append('../')
import unittest

from cupiqp.multistage.multistage_utils import BlockTridiagMat


class TestMultistageUtils(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.block_size = 5
        self.num_blocks = 4

        rng = np.random.default_rng(42)
        
        # Create numpy reference data
        self.D_np = rng.standard_normal((self.num_blocks, self.block_size, self.block_size))
        self.L_np = rng.standard_normal((self.num_blocks - 1, self.block_size, self.block_size))

        self.A_block = BlockTridiagMat(num_diag_blocks=self.num_blocks, block_size=self.block_size, device="cuda")
        wp.copy(self.A_block.diag_blocks.data, wp.array(self.D_np, dtype=wp.float64, device="cuda"))
        wp.copy(self.A_block.off_diag_blocks_lower.data, wp.array(self.L_np, dtype=wp.float64, device="cuda"))
        
        # Reconstruction logic: BlockTridiag -> CSR
        D_cp = cp.array(self.D_np)
        L_cp = cp.array(self.L_np)
        
        blocks = [[None for _ in range(self.num_blocks)] for _ in range(self.num_blocks)]
        for i in range(self.num_blocks):
            blocks[i][i] = cpsp.csr_matrix(D_cp[i]) # Diag
            if i < self.num_blocks - 1:
                blocks[i+1][i] = cpsp.csr_matrix(L_cp[i]) # Lower at (i+1, i)
                
        self.A_csr = cpsp.bmat(blocks, format='csr', dtype=cp.float64)

    def test_csr_to_block_tridiag(self):
        A_blk_tridiag = BlockTridiagMat.from_csr(self.A_csr, block_size=self.block_size)
        
        D_reconstructed = A_blk_tridiag.diag_blocks.data.numpy()
        L_reconstructed = A_blk_tridiag.off_diag_blocks_lower.data.numpy()
        
        self.assertTrue(np.allclose(np.tril(D_reconstructed), np.tril(self.D_np), atol=1e-8))
        self.assertTrue(np.allclose(L_reconstructed, self.L_np, atol=1e-8))


if __name__ == "__main__":
    unittest.main()