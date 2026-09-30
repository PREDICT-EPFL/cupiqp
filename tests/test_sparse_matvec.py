"""Tests for SparseMatVecProduct, with B = 1 and B > 1.

The operators take Warp arrays; the reference values are computed with
cupy / scipy. ``wp.synchronize()`` orders the Warp stream the operators run
on with the cupy reads that follow.
"""
import cupy as cp
import cupyx.scipy.sparse as sparse
import numpy as np
import pytest
import scipy.sparse as sp_cpu
import warp as wp

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
from cupiqp.sparse.sparse_matvec import SparseMatVecProduct


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def _w(a) -> wp.array:
    """Zero-copy Warp view of a cupy array (the operators take Warp arrays only)."""
    return wp.array(a, copy=False)


def _random_csr(
    m: int, n: int, density: float = 0.3, seed: int = 42,
) -> tuple[UniformBatchedCsrMatrix, sparse.csr_matrix]:
    """Random single-matrix batch plus the cupy CSR it was built from."""
    rng = np.random.default_rng(seed)
    A_dense = rng.standard_normal((m, n))
    A_dense[rng.random((m, n)) >= density] = 0.0
    A = sparse.csr_matrix(cp.asarray(A_dense, dtype=cp.float64))
    return UniformBatchedCsrMatrix.from_cupy_csr_matrix(A), A


def _random_batched_csr(
    B: int, m: int, n: int, density: float = 0.5, seed: int = 42,
) -> tuple[UniformBatchedCsrMatrix, sparse.csr_matrix, cp.ndarray]:
    """Random ``UniformBatchedCsrMatrix`` and the (B, nnz) values backing it."""
    rng = np.random.default_rng(seed)
    A_cpu = sp_cpu.random(m, n, density=density, format='csr',
                          random_state=rng, dtype=np.float64)
    template = sparse.csr_matrix(A_cpu)
    values = cp.asarray(rng.standard_normal((B, template.nnz)))
    return (
        UniformBatchedCsrMatrix(
            batch_size=B,
            indptr=template.indptr,
            indices=template.indices,
            data=values,
            shape=template.shape,
        ),
        template,
        values,
    )


def _batched_ref(
    template: sparse.csr_matrix,
    values: cp.ndarray,
    x: cp.ndarray,
    transpose: bool = False,
) -> np.ndarray:
    """Per-batch CPU reference: ``y[b] = A[b] @ x[b]`` (or ``A[b].T @ x[b]``)."""
    tpl_cpu = template.get()
    B = values.shape[0]
    out_dim = template.shape[1] if transpose else template.shape[0]
    y = np.empty((B, out_dim), dtype=np.float64)
    for b in range(B):
        Ab = tpl_cpu.copy()
        Ab.data[:] = cp.asnumpy(values[b])
        x_b = cp.asnumpy(x[b])
        y[b] = (Ab.T @ x_b) if transpose else (Ab @ x_b)
    return y


# Mixed shape coverage: tiny / square / tall / wide / asymmetric / large.
SHAPES: list[tuple[int, int]] = [
    (1, 1),
    (1, 8),
    (8, 1),
    (3, 3),
    (4, 5),
    (5, 4),
    (16, 16),
    (128, 128),
    (200, 32),     # tall
    (32, 200),     # wide
    (37, 53),      # non-power-of-two
    (512, 512),    # larger square
    (1024, 256),   # large tall
]

# A smaller grid for batched correctness - the batched path stacks B copies
# block-diagonally so the dimensions multiply by B and "large square" hurts.
BATCHED_SHAPES: list[tuple[int, int]] = [
    (1, 1),
    (3, 3),
    (4, 5),
    (5, 4),
    (16, 16),
    (37, 53),
    (200, 32),
    (32, 200),
]


# ===========================================================================
# One-matrix batches (B = 1)
# ===========================================================================

@pytest.mark.parametrize("density", [0.05, 0.3, 0.9])
@pytest.mark.parametrize("m,n", SHAPES)
def test_basic_spmv(m: int, n: int, density: float) -> None:
    mat, A = _random_csr(m, n, density=density, seed=m * 1000 + n)
    rng = np.random.default_rng(m * 7 + n)
    x = cp.asarray(rng.standard_normal(n), dtype=cp.float64)
    y = cp.zeros(m, dtype=cp.float64)

    op = SparseMatVecProduct(mat)
    op(_w(x), _w(y), alpha=1.0, beta=0.0)
    wp.synchronize()
    cp.testing.assert_allclose(y, A.toarray() @ x, atol=1e-12)


@pytest.mark.parametrize("density", [0.05, 0.3, 0.9])
@pytest.mark.parametrize("m,n", SHAPES)
def test_transpose(m: int, n: int, density: float) -> None:
    mat, A = _random_csr(m, n, density=density, seed=m * 31 + n)
    rng = np.random.default_rng(m * 13 + n + 1)
    x = cp.asarray(rng.standard_normal(m), dtype=cp.float64)
    y = cp.zeros(n, dtype=cp.float64)

    op = SparseMatVecProduct(mat, transa=True)
    op(_w(x), _w(y), alpha=1.0, beta=0.0)
    wp.synchronize()
    cp.testing.assert_allclose(y, A.toarray().T @ x, atol=1e-12)


@pytest.mark.parametrize("m,n", SHAPES)
def test_alpha_beta(m: int, n: int) -> None:
    mat, A = _random_csr(m, n, seed=m * 17 + n + 2)
    rng = np.random.default_rng(m * 19 + n + 3)
    x = cp.asarray(rng.standard_normal(n), dtype=cp.float64)
    y = cp.asarray(rng.standard_normal(m), dtype=cp.float64)
    y_before = y.copy()

    op = SparseMatVecProduct(mat)
    op(_w(x), _w(y), alpha=2.0, beta=0.5)
    wp.synchronize()
    cp.testing.assert_allclose(
        y, 2.0 * (A.toarray() @ x) + 0.5 * y_before, atol=1e-12,
    )


@pytest.mark.parametrize("m,n", SHAPES)
def test_alpha_beta_transpose(m: int, n: int) -> None:
    mat, A = _random_csr(m, n, seed=m * 23 + n + 4)
    rng = np.random.default_rng(m * 29 + n + 5)
    x = cp.asarray(rng.standard_normal(m), dtype=cp.float64)
    y = cp.asarray(rng.standard_normal(n), dtype=cp.float64)
    y_before = y.copy()

    op = SparseMatVecProduct(mat, transa=True)
    op(_w(x), _w(y), alpha=-1.5, beta=2.0)
    wp.synchronize()
    cp.testing.assert_allclose(
        y, -1.5 * (A.toarray().T @ x) + 2.0 * y_before, atol=1e-12,
    )


def test_one_row_view_accepted() -> None:
    """A ``(1, k)`` row view of a wider buffer is accepted for x and y."""
    mat, A = _random_csr(4, 5, seed=3)
    big_x = cp.asarray(np.random.default_rng(4).standard_normal((1, 9)))
    big_y = cp.zeros((1, 7), dtype=cp.float64)
    x_view, y_view = big_x[:, 2:7], big_y[:, 1:5]

    op = SparseMatVecProduct(mat)
    op(_w(x_view), _w(y_view))
    wp.synchronize()
    cp.testing.assert_allclose(y_view[0], A.toarray() @ x_view[0], atol=1e-12)
    cp.testing.assert_allclose(big_y[0, 0], 0.0)
    cp.testing.assert_allclose(big_y[0, 5:], 0.0)


def test_different_buffers() -> None:
    mat, A = _random_csr(3, 4)
    op = SparseMatVecProduct(mat)
    for x in (cp.ones(4, dtype=cp.float64), cp.arange(4, dtype=cp.float64)):
        y = cp.zeros(3, dtype=cp.float64)
        op(_w(x), _w(y))
        wp.synchronize()
        cp.testing.assert_allclose(y, A.toarray() @ x, atol=1e-12)


def test_reuse_different_scalars() -> None:
    mat, A = _random_csr(4, 4)
    x = cp.arange(4, dtype=cp.float64)
    y = cp.zeros(4, dtype=cp.float64)

    op = SparseMatVecProduct(mat)
    op(_w(x), _w(y), alpha=1.0, beta=0.0)
    wp.synchronize()
    y1 = y.copy()
    cp.testing.assert_allclose(y1, A.toarray() @ x, atol=1e-12)

    op(_w(x), _w(y), alpha=3.0, beta=1.0)
    wp.synchronize()
    cp.testing.assert_allclose(y, 3.0 * (A.toarray() @ x) + y1, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 8, 64, 256])
def test_identity(n: int) -> None:
    A = sparse.eye(n, dtype=cp.float64, format="csr")
    mat = UniformBatchedCsrMatrix.from_cupy_csr_matrix(A)
    x = cp.arange(n, dtype=cp.float64)
    y = cp.zeros(n, dtype=cp.float64)

    op = SparseMatVecProduct(mat)
    op(_w(x), _w(y))
    wp.synchronize()
    cp.testing.assert_allclose(y, x, atol=1e-14)


def test_in_place_value_update_is_seen() -> None:
    """The operator reads ``mat.data`` at every call."""
    mat, A = _random_csr(4, 5, seed=11)
    x = cp.asarray(np.random.default_rng(12).standard_normal(5))
    y = cp.zeros(4, dtype=cp.float64)
    op = SparseMatVecProduct(mat)

    new_vals = cp.asarray(np.random.default_rng(13).standard_normal(mat.nnz))
    wp.copy(mat.data, _w(new_vals.reshape(1, -1)))
    op(_w(x), _w(y))
    wp.synchronize()
    A2 = A.copy()
    A2.data[:] = new_vals
    cp.testing.assert_allclose(y, A2.toarray() @ x, atol=1e-12)


def test_cuda_graph_capture() -> None:
    mat, A = _random_csr(4, 4)
    x = cp.ones(4, dtype=cp.float64)
    y = cp.zeros(4, dtype=cp.float64)
    xw, yw = _w(x), _w(y)

    op = SparseMatVecProduct(mat)
    stream = wp.Stream("cuda")
    with wp.ScopedStream(stream):
        with wp.ScopedCapture(stream=stream) as capture:
            op(xw, yw, alpha=1.0, beta=0.0)
        wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    cp.testing.assert_allclose(y, A.toarray() @ x, atol=1e-12)


def test_destructor_no_error() -> None:
    mat, _ = _random_csr(2, 2)
    op = SparseMatVecProduct(mat)
    del op  # must not raise


# ===========================================================================
# Batches of several (B > 1)
# ===========================================================================

@pytest.mark.parametrize("B", [1, 2, 5, 16])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
def test_batched_basic_contiguous(B: int, m: int, n: int) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, density=0.4, seed=B * 100 + m * 10 + n)
    op = SparseMatVecProduct(bmat, transa=False)

    x = cp.asarray(np.random.default_rng(B + m + n).standard_normal((B, n)))
    out = cp.zeros((B, m), dtype=cp.float64)
    op(_w(x), _w(out))
    wp.synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, vals, x), atol=1e-12,
    )


@pytest.mark.parametrize("B", [1, 2, 5, 16])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
def test_batched_transpose(B: int, m: int, n: int) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, density=0.5, seed=B * 200 + m * 7 + n)
    op = SparseMatVecProduct(bmat, transa=True)

    x = cp.asarray(np.random.default_rng(B * 3 + m + n).standard_normal((B, m)))
    out = cp.zeros((B, n), dtype=cp.float64)
    op(_w(x), _w(out))
    wp.synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, vals, x, transpose=True), atol=1e-12,
    )


@pytest.mark.parametrize("B", [1, 2, 5, 16])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
def test_batched_alpha_beta(B: int, m: int, n: int) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, density=0.5, seed=B * 300 + m * 11 + n)
    op = SparseMatVecProduct(bmat)

    rng = np.random.default_rng(B * 5 + m + n + 1)
    x = cp.asarray(rng.standard_normal((B, n)))
    y = cp.asarray(rng.standard_normal((B, m)))
    y_init = cp.asnumpy(y).copy()

    alpha, beta = 2.0, 3.0
    op(_w(x), _w(y), alpha=alpha, beta=beta)
    wp.synchronize()
    expected = alpha * _batched_ref(tpl, vals, x) + beta * y_init
    np.testing.assert_allclose(cp.asnumpy(y), expected, atol=1e-12)


def test_batched_non_contiguous_x() -> None:
    B, m, n = 4, 5, 6
    K = 10
    col_start = 2
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=60)
    op = SparseMatVecProduct(bmat)

    big = cp.asarray(np.random.default_rng(62).standard_normal((B, K)))
    x_view = big[:, col_start:col_start + n]
    assert not x_view.flags['C_CONTIGUOUS']

    out = cp.zeros((B, m), dtype=cp.float64)
    op(_w(x_view), _w(out))
    wp.synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, vals, x_view), atol=1e-12,
    )


def test_batched_non_contiguous_out() -> None:
    B, m, n = 3, 4, 5
    K = 9
    col_start = 3
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=63)
    op = SparseMatVecProduct(bmat)

    x = cp.asarray(np.random.default_rng(65).standard_normal((B, n)))
    big_out = cp.zeros((B, K), dtype=cp.float64) - 7.0  # sentinel outside the window
    out_view = big_out[:, col_start:col_start + m]
    assert not out_view.flags['C_CONTIGUOUS']

    op(_w(x), _w(out_view))
    wp.synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out_view), _batched_ref(tpl, vals, x), atol=1e-12,
    )
    # Columns outside the target window are untouched.
    np.testing.assert_allclose(cp.asnumpy(big_out[:, :col_start]), -7.0)
    np.testing.assert_allclose(cp.asnumpy(big_out[:, col_start + m:]), -7.0)


def test_batched_non_contiguous_out_with_beta() -> None:
    """With ``beta != 0`` the current values of a strided y are read first."""
    B, m, n = 3, 4, 5
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=66)
    op = SparseMatVecProduct(bmat)

    rng = np.random.default_rng(67)
    x = cp.asarray(rng.standard_normal((B, n)))
    big_out = cp.asarray(rng.standard_normal((B, 9)))
    out_view = big_out[:, 2:2 + m]
    y_init = cp.asnumpy(out_view).copy()

    op(_w(x), _w(out_view), alpha=1.0, beta=0.5)
    wp.synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out_view), _batched_ref(tpl, vals, x) + 0.5 * y_init, atol=1e-12,
    )


def test_batched_multiple_calls_different_x() -> None:
    B, m, n = 3, 4, 5
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=70)
    op = SparseMatVecProduct(bmat)

    x1 = cp.asarray(np.random.default_rng(72).standard_normal((B, n)))
    x2 = cp.asarray(np.random.default_rng(73).standard_normal((B, n)))
    out = cp.empty((B, m), dtype=cp.float64)

    op(_w(x1), _w(out))
    wp.synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, vals, x1), atol=1e-12,
    )
    op(_w(x2), _w(out))
    wp.synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, vals, x2), atol=1e-12,
    )


def test_batched_tracks_in_place_value_update() -> None:
    """After copying new values into ``bmat.data``, the SpMV uses them."""
    B, m, n = 3, 4, 5
    bmat, tpl, _ = _random_batched_csr(B, m, n, seed=74)
    op = SparseMatVecProduct(bmat)

    x = cp.asarray(np.random.default_rng(76).standard_normal((B, n)))
    out = cp.empty((B, m), dtype=cp.float64)
    op(_w(x), _w(out))

    new_vals = cp.asarray(np.random.default_rng(77).standard_normal((B, tpl.nnz)))
    wp.copy(bmat.data, _w(new_vals))
    op(_w(x), _w(out))
    wp.synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, new_vals, x), atol=1e-12,
    )
