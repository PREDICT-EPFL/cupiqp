"""Tests for CholeskyInplaceSolver in both single and batched modes.

Every numerical test is parameterized over dtype so the float64 and
float32 code paths (cuSOLVER ``d*potrf/potrs`` vs ``s*potrf/potrs``)
get identical shape, batch, and contiguity coverage.
"""
import pytest
import numpy as np
import cupy as cp

from cupiqp.dense.dense_cholesky import CholeskyInplaceSolver, BatchedCholeskyInplaceSolver


# ---------------------------------------------------------------------------
# dtype dispatch and tolerances
# ---------------------------------------------------------------------------
DTYPES = [cp.float64, cp.float32]


def _atol(dtype):
    return 1e-10 if dtype == cp.float64 else 1e-4


def _opposite_dtype(dtype):
    return cp.float32 if dtype == cp.float64 else cp.float64


def _make_spd(n, seed=42, dtype=np.float64):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M @ M.T + n * np.eye(n)).astype(dtype)


def _make_spd_batch(batch_size, n, seed=42, dtype=np.float64):
    rng = np.random.default_rng(seed)
    As = []
    for _ in range(batch_size):
        M = rng.standard_normal((n, n))
        As.append(M @ M.T + n * np.eye(n))
    return np.stack(As).astype(dtype)


# ======================================================================
# Single mode
# ======================================================================
class TestSingleMode:

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_factorize_succeeds(self, dtype):
        n = 6
        A = cp.asarray(_make_spd(n, dtype=dtype))
        solver = CholeskyInplaceSolver(n, dtype=dtype)
        assert solver.factorize(A)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 256])
    def test_factorize_and_solve(self, n, dtype):
        rng = np.random.default_rng(n)
        A_orig = _make_spd(n, seed=n, dtype=dtype)
        b = rng.standard_normal(n).astype(dtype)
        A_work = cp.asarray(A_orig.copy())
        x = cp.asarray(b.copy())

        solver = CholeskyInplaceSolver(n, dtype=dtype)
        assert solver.factorize(A_work)
        solver.solve(x)
        np.testing.assert_allclose(
            A_orig @ cp.asnumpy(x), b, atol=_atol(dtype), err_msg=f"n={n}",
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_multiple_rhs(self, dtype):
        n = 5
        rng = np.random.default_rng(42)
        A_orig = _make_spd(n, dtype=dtype)
        A_work = cp.asarray(A_orig.copy())

        solver = CholeskyInplaceSolver(n, dtype=dtype)
        solver.factorize(A_work)

        for _ in range(3):
            b = rng.standard_normal(n).astype(dtype)
            x = cp.asarray(b.copy())
            solver.solve(x)
            np.testing.assert_allclose(A_orig @ cp.asnumpy(x), b, atol=_atol(dtype))


# ======================================================================
# Batched mode
# ======================================================================
class TestBatchedMode:

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_factorize_succeeds(self, dtype):
        B, n = 4, 6
        A = cp.asarray(_make_spd_batch(B, n, dtype=dtype))
        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        assert solver.factorize(A)

    # (B, n) matrix exercises: batch=1 (smallest batched case), small
    # batches at various n, and a large-batch case. Fused from the former
    # test_factorize_and_solve / test_batch_size_one / test_various_sizes
    # / test_large_batch — they were all the same "build batched SPD,
    # factor, solve, check per batch" with different (B, n).
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("B,n", [
        (1, 4),
        (5, 8),
        (8, 1), (8, 2), (8, 4), (8, 8), (8, 12), (8, 16), (8, 24), (8, 32),
        (128, 6),
    ])
    def test_factorize_and_solve(self, B, n, dtype):
        rng = np.random.default_rng(B * 100 + n)
        A_orig = _make_spd_batch(B, n, seed=B * 100 + n, dtype=dtype)
        b = rng.standard_normal((B, n)).astype(dtype)
        A_work = cp.asarray(A_orig.copy())
        x = cp.asarray(b.copy())

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        assert solver.factorize(A_work)
        solver.solve(x)

        for i in range(B):
            np.testing.assert_allclose(
                A_orig[i] @ cp.asnumpy(x[i]), b[i], atol=_atol(dtype),
                err_msg=f"B={B}, n={n}, batch={i}")

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_multiple_solves_same_factorization(self, dtype):
        B, n = 3, 5
        rng = np.random.default_rng(42)
        A_orig = _make_spd_batch(B, n, dtype=dtype)
        A_work = cp.asarray(A_orig.copy())

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        solver.factorize(A_work)

        for _ in range(3):
            b = rng.standard_normal((B, n)).astype(dtype)
            x = cp.asarray(b.copy())
            solver.solve(x)
            for i in range(B):
                np.testing.assert_allclose(
                    A_orig[i] @ cp.asnumpy(x[i]), b[i], atol=_atol(dtype))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_non_contiguous_rhs(self, dtype):
        """Batched solve with non-contiguous RHS (view of larger buffer)."""
        B, n = 3, 4
        rng = np.random.default_rng(42)
        A_orig = _make_spd_batch(B, n, dtype=dtype)
        b = rng.standard_normal((B, n)).astype(dtype)
        A_work = cp.asarray(A_orig.copy())

        buf = cp.zeros((B, n + 5), dtype=dtype)
        buf[:, :n] = cp.asarray(b)
        x_view = buf[:, :n]  # non-contiguous outer stride
        assert not x_view.flags['C_CONTIGUOUS']

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        solver.factorize(A_work)
        solver.solve(x_view)

        for i in range(B):
            np.testing.assert_allclose(
                A_orig[i] @ cp.asnumpy(x_view[i]), b[i], atol=_atol(dtype),
                err_msg=f"batch {i}")


# ======================================================================
# Cross-mode: batched B=1 vs single should give same result
# ======================================================================
class TestCrossMode:

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_single_vs_batched_b1(self, dtype):
        n = 6
        rng = np.random.default_rng(42)
        A_np = _make_spd(n, dtype=dtype)
        b_np = rng.standard_normal(n).astype(dtype)

        # Single mode
        A1 = cp.asarray(A_np.copy())
        x1 = cp.asarray(b_np.copy())
        s1 = CholeskyInplaceSolver(n, dtype=dtype)
        s1.factorize(A1)
        s1.solve(x1)

        # Batched mode B=1
        A2 = cp.asarray(A_np.copy().reshape(1, n, n))
        x2 = cp.asarray(b_np.copy().reshape(1, n))
        s2 = BatchedCholeskyInplaceSolver(n, 1, dtype=dtype)
        s2.factorize(A2)
        s2.solve(x2)

        np.testing.assert_allclose(cp.asnumpy(x1), cp.asnumpy(x2[0]), atol=_atol(dtype))


class TestFFIBoundaryValidation:
    """Input validation at the cuSOLVER FFI boundary.

    Internal cuPIQP callers always pass correctly-shaped, correctly-typed,
    contiguous buffers. These tests cover direct external use, where a
    short / mismatched-dtype / non-contiguous buffer would otherwise reach
    ``potrs[Batched]`` and cause device OOB or silent garbage.
    """

    # -- Single mode ------------------------------------------------------

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_single_solve_rejects_short_1d_rhs(self, dtype):
        n = 6
        solver = CholeskyInplaceSolver(n, dtype=dtype)
        solver.factorize(cp.asarray(_make_spd(n, dtype=dtype)))
        with pytest.raises(ValueError, match="1-D RHS length mismatch"):
            solver.solve(cp.zeros(n - 1, dtype=dtype))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_single_solve_rejects_wrong_dtype_rhs(self, dtype):
        n = 6
        solver = CholeskyInplaceSolver(n, dtype=dtype)
        solver.factorize(cp.asarray(_make_spd(n, dtype=dtype)))
        with pytest.raises(TypeError, match="dtype"):
            solver.solve(cp.zeros(n, dtype=_opposite_dtype(dtype)))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_single_factorize_rejects_wrong_dtype(self, dtype):
        n = 6
        solver = CholeskyInplaceSolver(n, dtype=dtype)
        A_bad = cp.asarray(_make_spd(n, dtype=_opposite_dtype(dtype)))
        with pytest.raises(TypeError, match="dtype"):
            solver.factorize(A_bad)

    # -- Batched mode -----------------------------------------------------

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_batched_factorize_rejects_wrong_shape(self, dtype):
        n, B = 4, 3
        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        bad = cp.zeros((B, n, n + 1), dtype=dtype)
        with pytest.raises(ValueError, match="A shape mismatch"):
            solver.factorize(bad)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_batched_factorize_rejects_wrong_dtype(self, dtype):
        n, B = 4, 3
        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        bad = cp.zeros((B, n, n), dtype=_opposite_dtype(dtype))
        with pytest.raises(TypeError, match="dtype"):
            solver.factorize(bad)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_batched_factorize_rejects_inner_non_contiguous(self, dtype):
        """``A[:, :n, :n]`` view from a larger ``(B, 2n, 2n)`` buffer
        leaves each inner ``(n, n)`` block non-contiguous (row stride
        is ``2n*itemsize``, not ``n*itemsize``). cuSOLVER would read
        garbage; we reject."""
        n, B = 4, 3
        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        big = cp.asarray(_make_spd_batch(B, 2 * n, dtype=dtype))
        view = big[:, :n, :n]
        assert not view[0].flags.c_contiguous
        with pytest.raises(ValueError, match="must be C-contiguous"):
            solver.factorize(view)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_batched_factorize_accepts_outer_strided(self, dtype):
        """``A`` whose outer batch stride exceeds ``n*n*itemsize`` is
        still valid as long as each inner matrix is C-contiguous —
        ``_ensure_ptrs`` uses ``strides[0]`` directly."""
        n, B = 4, 3
        spds = _make_spd_batch(B, n, dtype=dtype)
        padded = cp.zeros((B + 2, n, n), dtype=dtype)
        padded[:B] = cp.asarray(spds)
        # A view that picks every other batch — outer stride = 2 * n*n*itemsize.
        view = padded[::2][:B]
        assert view.shape == (B, n, n)
        assert view[0].flags.c_contiguous
        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        # No ValueError from the contiguity check (factorize may still
        # report failure on the zero block; we don't assert that here).
        solver.factorize(view)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_batched_solve_rejects_wrong_shape(self, dtype):
        n, B = 4, 3
        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        solver.factorize(cp.asarray(_make_spd_batch(B, n, dtype=dtype)))
        with pytest.raises(ValueError, match="B shape mismatch"):
            solver.solve(cp.zeros((B, n - 1), dtype=dtype))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_batched_solve_rejects_wrong_dtype(self, dtype):
        n, B = 4, 3
        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        solver.factorize(cp.asarray(_make_spd_batch(B, n, dtype=dtype)))
        with pytest.raises(TypeError, match="dtype"):
            solver.solve(cp.zeros((B, n), dtype=_opposite_dtype(dtype)))
