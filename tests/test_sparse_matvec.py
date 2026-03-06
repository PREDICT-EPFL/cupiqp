import unittest
import cupy as cp
import cupyx.scipy.sparse as sparse
import numpy as np

from cupiqp.sparse.sparse_matvec import SparseMatVecProduct


class TestSparseMatVecProduct(unittest.TestCase):
    """Unit tests for SparseMatVecProduct."""

    def _random_csr(self, m, n, density=0.3):
        rng = np.random.default_rng(42)
        A_dense = rng.standard_normal((m, n))
        mask = rng.random((m, n)) < density
        A_dense[~mask] = 0.0
        return sparse.csr_matrix(cp.asarray(A_dense, dtype=cp.float64))

    # ------------------------------------------------------------------
    # basic y = A @ x  (alpha=1, beta=0)
    # ------------------------------------------------------------------
    def test_basic_spmv(self):
        A = self._random_csr(4, 5)
        x = cp.ones(5, dtype=cp.float64)
        y = cp.zeros(4, dtype=cp.float64)

        op = SparseMatVecProduct(A)
        op(x, y, alpha=1.0, beta=0.0)
        cp.cuda.get_current_stream().synchronize()

        expected = A.toarray() @ x
        cp.testing.assert_allclose(y, expected, atol=1e-12)

    # ------------------------------------------------------------------
    # alpha / beta scalars
    # ------------------------------------------------------------------
    def test_alpha_beta(self):
        A = self._random_csr(3, 3)
        x = cp.array([1.0, 2.0, 3.0])
        y = cp.array([10.0, 20.0, 30.0])

        alpha, beta = 2.0, 0.5
        op = SparseMatVecProduct(A)

        y_before = y.copy()
        op(x, y, alpha=alpha, beta=beta)
        cp.cuda.get_current_stream().synchronize()

        expected = alpha * (A.toarray() @ x) + beta * y_before
        cp.testing.assert_allclose(y, expected, atol=1e-12)

    # ------------------------------------------------------------------
    # transpose: y = alpha * A^T @ x + beta * y
    # ------------------------------------------------------------------
    def test_transpose(self):
        A = self._random_csr(4, 5)
        x = cp.ones(4, dtype=cp.float64)
        y = cp.zeros(5, dtype=cp.float64)

        op = SparseMatVecProduct(A, transa=True)
        op(x, y, alpha=1.0, beta=0.0)
        cp.cuda.get_current_stream().synchronize()

        expected = A.toarray().T @ x
        cp.testing.assert_allclose(y, expected, atol=1e-12)

    # ------------------------------------------------------------------
    # in-place x/y update (same buffer, new values)
    # ------------------------------------------------------------------
    def test_inplace_buffer_update(self):
        A = self._random_csr(3, 3)
        x = cp.ones(3, dtype=cp.float64)
        y = cp.zeros(3, dtype=cp.float64)

        op = SparseMatVecProduct(A)

        # change x contents in-place
        x[:] = cp.array([4.0, 5.0, 6.0])
        y[:] = 0.0
        op(x, y, alpha=1.0, beta=0.0)
        cp.cuda.get_current_stream().synchronize()

        expected = A.toarray() @ x
        cp.testing.assert_allclose(y, expected, atol=1e-12)

    # ------------------------------------------------------------------
    # different x/y buffers across calls
    # ------------------------------------------------------------------
    def test_different_buffers(self):
        A = self._random_csr(3, 4)
        op = SparseMatVecProduct(A)

        x1 = cp.ones(4, dtype=cp.float64)
        y1 = cp.zeros(3, dtype=cp.float64)
        op(x1, y1)
        cp.cuda.get_current_stream().synchronize()
        cp.testing.assert_allclose(y1, A.toarray() @ x1, atol=1e-12)

        x2 = cp.arange(4, dtype=cp.float64)
        y2 = cp.zeros(3, dtype=cp.float64)
        op(x2, y2)
        cp.cuda.get_current_stream().synchronize()
        cp.testing.assert_allclose(y2, A.toarray() @ x2, atol=1e-12)

    # ------------------------------------------------------------------
    # multiple calls with different scalars
    # ------------------------------------------------------------------
    def test_reuse_different_scalars(self):
        A = self._random_csr(4, 4)
        x = cp.arange(4, dtype=cp.float64)
        y = cp.zeros(4, dtype=cp.float64)

        op = SparseMatVecProduct(A)

        # first call: alpha=1, beta=0
        op(x, y, alpha=1.0, beta=0.0)
        cp.cuda.get_current_stream().synchronize()
        y1 = y.copy()
        cp.testing.assert_allclose(y1, A.toarray() @ x, atol=1e-12)

        # second call: alpha=3, beta=1  (y already holds y1)
        op(x, y, alpha=3.0, beta=1.0)
        cp.cuda.get_current_stream().synchronize()
        expected = 3.0 * (A.toarray() @ x) + y1
        cp.testing.assert_allclose(y, expected, atol=1e-12)

    # ------------------------------------------------------------------
    # identity matrix
    # ------------------------------------------------------------------
    def test_identity(self):
        n = 6
        A = sparse.eye(n, dtype=cp.float64, format="csr")
        x = cp.arange(n, dtype=cp.float64)
        y = cp.zeros(n, dtype=cp.float64)

        op = SparseMatVecProduct(A)
        op(x, y)
        cp.cuda.get_current_stream().synchronize()

        cp.testing.assert_allclose(y, x, atol=1e-14)

    # ------------------------------------------------------------------
    # graph capture round-trip
    # ------------------------------------------------------------------
    def test_cuda_graph_capture(self):
        A = self._random_csr(4, 4)
        x = cp.ones(4, dtype=cp.float64)
        y = cp.zeros(4, dtype=cp.float64)

        op = SparseMatVecProduct(A)

        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            op(x, y, alpha=1.0, beta=0.0)
            graph = stream.end_capture()

        graph.launch(stream)
        stream.synchronize()

        expected = A.toarray() @ x
        cp.testing.assert_allclose(y, expected, atol=1e-12)

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------
    def test_destructor_no_error(self):
        A = self._random_csr(2, 2)
        x = cp.ones(2, dtype=cp.float64)
        y = cp.zeros(2, dtype=cp.float64)
        op = SparseMatVecProduct(A)
        del op  # should not raise


if __name__ == "__main__":
    unittest.main()
