from matplotlib.pylab import beta
import numpy as np
import warp as wp
import cupy as cp
import cupyx.scipy.sparse as cpsp
import sys
sys.path.append('./')
sys.path.append('../')
import unittest

from cupiqp.multistage.multistage_utils import create_csr_add_btd_kernel


class TestMultistageUtils(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.block_size = 20
        self.num_blocks = 40

        rng = np.random.default_rng(42)
        
        # Create numpy reference data
        self.D_np = rng.standard_normal((self.num_blocks, self.block_size, self.block_size))
        self.L_np = rng.standard_normal((self.num_blocks - 1, self.block_size, self.block_size))

        self._csr_add_to_btd_kernel = create_csr_add_btd_kernel(self.num_blocks, self.block_size, dtype=wp.float64)
        
        # Reconstruction logic: BlockTridiag -> CSR
        D_cp = cp.array(self.D_np)
        L_cp = cp.array(self.L_np)
        
        blocks = [[None for _ in range(self.num_blocks)] for _ in range(self.num_blocks)]
        for i in range(self.num_blocks):
            blocks[i][i] = cpsp.csr_matrix(D_cp[i]) # Diag
            if i < self.num_blocks - 1:
                blocks[i+1][i] = cpsp.csr_matrix(L_cp[i]) # Lower at (i+1, i)
                
        self.A_csr = cpsp.bmat(blocks, format='csr', dtype=cp.float64)


    def test_csr_add_btd(self):
        diag_blocks = wp.from_numpy(np.random.randn(self.num_blocks, self.block_size, self.block_size), dtype=wp.float64, device="cuda")
        offdiag_blocks = wp.from_numpy(np.random.randn(self.num_blocks - 1, self.block_size, self.block_size), dtype=wp.float64, device="cuda")
        alpha = np.random.randn()

        D_benchmark = diag_blocks.numpy() + alpha * self.D_np
        L_benchmark = offdiag_blocks.numpy() + alpha * self.L_np
        wp.launch(
            kernel=self._csr_add_to_btd_kernel,
            dim=self.A_csr.shape[0],
            inputs=[
                alpha,
                self.A_csr.indptr, self.A_csr.indices, self.A_csr.data,
                diag_blocks,
                offdiag_blocks,
            ],
            device="cuda"
        )
        D_reconstructed = diag_blocks.numpy()
        L_reconstructed = offdiag_blocks.numpy()
        
        self.assertTrue(np.allclose(D_reconstructed, D_benchmark, atol=1e-8))
        self.assertTrue(np.allclose(L_reconstructed, L_benchmark, atol=1e-8))


if __name__ == "__main__":
    unittest.main()