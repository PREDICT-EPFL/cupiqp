"""Tests for CholeskyInplaceSolver in both single and batched modes."""
import pytest
import numpy as np
import cupy as cp

from cupiqp.dense.dense_cholesky import CholeskyInplaceSolver, BatchedCholeskyInplaceSolver


def _make_spd(n, seed=42):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def _make_spd_batch(batch_size, n, seed=42):
    rng = np.random.default_rng(seed)
    As = []
    for _ in range(batch_size):
        M = rng.standard_normal((n, n))
        As.append(M @ M.T + n * np.eye(n))
    return np.stack(As)


# ======================================================================
# Single mode
# ======================================================================
class TestSingleMode:

    def test_factorize_succeeds(self):
        n = 6
        A = cp.array(_make_spd(n))
        solver = CholeskyInplaceSolver(n)
        assert solver.factorize(A)

    def test_factorize_and_solve(self):
        n = 8
        rng = np.random.default_rng(77)
        A_orig = _make_spd(n, seed=77)
        b = rng.standard_normal(n)

        A_work = cp.array(A_orig.copy())
        x = cp.array(b.copy())

        solver = CholeskyInplaceSolver(n)
        assert solver.factorize(A_work)
        solver.solve(x)

        Ax = A_orig @ cp.asnumpy(x)
        np.testing.assert_allclose(Ax, b, atol=1e-10)

    def test_multiple_rhs(self):
        n = 5
        rng = np.random.default_rng(42)
        A_orig = _make_spd(n)
        A_work = cp.array(A_orig.copy())

        solver = CholeskyInplaceSolver(n)
        solver.factorize(A_work)

        for _ in range(3):
            b = rng.standard_normal(n)
            x = cp.array(b.copy())
            solver.solve(x)
            np.testing.assert_allclose(A_orig @ cp.asnumpy(x), b, atol=1e-10)

    def test_various_sizes(self):
        for n in [1, 2, 4, 8, 16, 32]:
            rng = np.random.default_rng(n)
            A_orig = _make_spd(n, seed=n)
            b = rng.standard_normal(n)
            A_work = cp.array(A_orig.copy())
            x = cp.array(b.copy())

            solver = CholeskyInplaceSolver(n)
            solver.factorize(A_work)
            solver.solve(x)
            np.testing.assert_allclose(A_orig @ cp.asnumpy(x), b, atol=1e-9,
                                       err_msg=f"n={n}")


# ======================================================================
# Batched mode
# ======================================================================
class TestBatchedMode:

    def test_factorize_succeeds(self):
        B, n = 4, 6
        A = cp.array(_make_spd_batch(B, n))
        solver = BatchedCholeskyInplaceSolver(n, B)
        assert solver.factorize(A)

    def test_factorize_and_solve(self):
        B, n = 5, 8
        rng = np.random.default_rng(77)
        A_orig = _make_spd_batch(B, n, seed=77)
        b = rng.standard_normal((B, n))

        A_work = cp.array(A_orig.copy())
        x = cp.array(b.copy())

        solver = BatchedCholeskyInplaceSolver(n, B)
        assert solver.factorize(A_work)
        solver.solve(x)

        for i in range(B):
            Ax = A_orig[i] @ cp.asnumpy(x[i])
            np.testing.assert_allclose(Ax, b[i], atol=1e-10,
                                       err_msg=f"batch {i}")

    def test_multiple_solves_same_factorization(self):
        B, n = 3, 5
        rng = np.random.default_rng(42)
        A_orig = _make_spd_batch(B, n)
        A_work = cp.array(A_orig.copy())

        solver = BatchedCholeskyInplaceSolver(n, B)
        solver.factorize(A_work)

        for _ in range(3):
            b = rng.standard_normal((B, n))
            x = cp.array(b.copy())
            solver.solve(x)
            for i in range(B):
                np.testing.assert_allclose(
                    A_orig[i] @ cp.asnumpy(x[i]), b[i], atol=1e-10)

    def test_batch_size_one(self):
        B, n = 1, 4
        rng = np.random.default_rng(0)
        A_orig = _make_spd_batch(B, n, seed=0)
        b = rng.standard_normal((B, n))
        A_work = cp.array(A_orig.copy())
        x = cp.array(b.copy())

        solver = BatchedCholeskyInplaceSolver(n, B)
        solver.factorize(A_work)
        solver.solve(x)

        np.testing.assert_allclose(A_orig[0] @ cp.asnumpy(x[0]), b[0], atol=1e-10)

    def test_various_sizes(self):
        for n in [1, 2, 4, 8, 12, 16, 24, 32]:
            B = 8
            rng = np.random.default_rng(n)
            A_orig = _make_spd_batch(B, n, seed=n)
            b = rng.standard_normal((B, n))
            A_work = cp.array(A_orig.copy())
            x = cp.array(b.copy())

            solver = BatchedCholeskyInplaceSolver(n, B)
            solver.factorize(A_work)
            solver.solve(x)

            for i in range(B):
                np.testing.assert_allclose(
                    A_orig[i] @ cp.asnumpy(x[i]), b[i], atol=1e-9,
                    err_msg=f"n={n}, batch={i}")

    def test_large_batch(self):
        B, n = 128, 6
        rng = np.random.default_rng(42)
        A_orig = _make_spd_batch(B, n)
        b = rng.standard_normal((B, n))
        A_work = cp.array(A_orig.copy())
        x = cp.array(b.copy())

        solver = BatchedCholeskyInplaceSolver(n, B)
        solver.factorize(A_work)
        solver.solve(x)

        for i in range(B):
            np.testing.assert_allclose(
                A_orig[i] @ cp.asnumpy(x[i]), b[i], atol=1e-10)

    def test_non_contiguous_rhs(self):
        """Batched solve with non-contiguous RHS (view of larger buffer)."""
        B, n = 3, 4
        rng = np.random.default_rng(42)
        A_orig = _make_spd_batch(B, n)
        b = rng.standard_normal((B, n))
        A_work = cp.array(A_orig.copy())

        # Create non-contiguous x as a view of a larger buffer
        buf = cp.zeros((B, n + 5), dtype=cp.float64)
        buf[:, :n] = cp.array(b)
        x_view = buf[:, :n]  # non-contiguous: stride[0] = (n+5)*8
        assert not x_view.flags['C_CONTIGUOUS']

        solver = BatchedCholeskyInplaceSolver(n, B)
        solver.factorize(A_work)
        solver.solve(x_view)

        for i in range(B):
            np.testing.assert_allclose(
                A_orig[i] @ cp.asnumpy(x_view[i]), b[i], atol=1e-10,
                err_msg=f"batch {i}")


# ======================================================================
# Cross-mode: batched B=1 vs single should give same result
# ======================================================================
class TestCrossMode:

    def test_single_vs_batched_b1(self):
        n = 6
        rng = np.random.default_rng(42)
        A_np = _make_spd(n)
        b_np = rng.standard_normal(n)

        # Single mode
        A1 = cp.array(A_np.copy())
        x1 = cp.array(b_np.copy())
        s1 = CholeskyInplaceSolver(n)
        s1.factorize(A1)
        s1.solve(x1)

        # Batched mode B=1
        A2 = cp.array(A_np.copy().reshape(1, n, n))
        x2 = cp.array(b_np.copy().reshape(1, n))
        s2 = BatchedCholeskyInplaceSolver(n, 1)
        s2.factorize(A2)
        s2.solve(x2)

        np.testing.assert_allclose(cp.asnumpy(x1), cp.asnumpy(x2[0]), atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
