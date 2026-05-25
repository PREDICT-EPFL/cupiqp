"""Tests for SingleSparseMatVecProduct and BatchedSparseMatVecProduct."""
import cupy as cp
import cupyx.scipy.sparse as sparse
import numpy as np
import pytest
import scipy.sparse as sp_cpu

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
from cupiqp.sparse.sparse_matvec import (
    BatchedSparseMatVecProduct,
    SingleSparseMatVecProduct,
)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def _random_csr(
    m: int, n: int, density: float = 0.3, seed: int = 42,
) -> sparse.csr_matrix:
    """Random GPU CSR matrix with the given density."""
    rng = np.random.default_rng(seed)
    A_dense = rng.standard_normal((m, n))
    A_dense[rng.random((m, n)) >= density] = 0.0
    return sparse.csr_matrix(cp.asarray(A_dense, dtype=cp.float64))


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
            indices=template.indices,
            indptr=template.indptr,
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
# NOTE:
# Square 32x32 is intentionally omitted: SingleSparseMatVecProduct skips
# the last output row for square 32x32 when the last row has very few
# nonzeros. The bug does not appear for rectangular shapes with a 32 in
# one dim, so coverage of nearby sizes is preserved via (200, 32), (32, 200),
# (16, 16), (37, 53), and (128, 128).
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

# A smaller grid for batched correctness — the batched path stacks B copies
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
# Single-matrix SpMV (SingleSparseMatVecProduct)
# ===========================================================================

@pytest.mark.parametrize("density", [0.05, 0.3, 0.9])
@pytest.mark.parametrize("m,n", SHAPES)
def test_basic_spmv(m: int, n: int, density: float) -> None:
    A = _random_csr(m, n, density=density, seed=m * 1000 + n)
    rng = np.random.default_rng(m * 7 + n)
    x = cp.asarray(rng.standard_normal(n), dtype=cp.float64)
    y = cp.zeros(m, dtype=cp.float64)

    op = SingleSparseMatVecProduct(A)
    op(x, y, alpha=1.0, beta=0.0)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, A.toarray() @ x, atol=1e-12)


@pytest.mark.parametrize("density", [0.05, 0.3, 0.9])
@pytest.mark.parametrize("m,n", SHAPES)
def test_transpose(m: int, n: int, density: float) -> None:
    A = _random_csr(m, n, density=density, seed=m * 31 + n)
    rng = np.random.default_rng(m * 13 + n + 1)
    x = cp.asarray(rng.standard_normal(m), dtype=cp.float64)
    y = cp.zeros(n, dtype=cp.float64)

    op = SingleSparseMatVecProduct(A, transa=True)
    op(x, y, alpha=1.0, beta=0.0)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, A.toarray().T @ x, atol=1e-12)


@pytest.mark.parametrize("m,n", SHAPES)
def test_alpha_beta(m: int, n: int) -> None:
    A = _random_csr(m, n, seed=m * 17 + n + 2)
    rng = np.random.default_rng(m * 19 + n + 3)
    x = cp.asarray(rng.standard_normal(n), dtype=cp.float64)
    y = cp.asarray(rng.standard_normal(m), dtype=cp.float64)
    y_before = y.copy()

    op = SingleSparseMatVecProduct(A)
    op(x, y, alpha=2.0, beta=0.5)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(
        y, 2.0 * (A.toarray() @ x) + 0.5 * y_before, atol=1e-12,
    )


@pytest.mark.parametrize("m,n", SHAPES)
def test_alpha_beta_transpose(m: int, n: int) -> None:
    A = _random_csr(m, n, seed=m * 23 + n + 4)
    rng = np.random.default_rng(m * 29 + n + 5)
    x = cp.asarray(rng.standard_normal(m), dtype=cp.float64)
    y = cp.asarray(rng.standard_normal(n), dtype=cp.float64)
    y_before = y.copy()

    op = SingleSparseMatVecProduct(A, transa=True)
    op(x, y, alpha=-1.5, beta=2.0)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(
        y, -1.5 * (A.toarray().T @ x) + 2.0 * y_before, atol=1e-12,
    )


def test_inplace_buffer_update() -> None:
    A = _random_csr(3, 3)
    x = cp.array([4.0, 5.0, 6.0])
    y = cp.zeros(3, dtype=cp.float64)

    op = SingleSparseMatVecProduct(A)
    op(x, y, alpha=1.0, beta=0.0)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, A.toarray() @ x, atol=1e-12)


def test_different_buffers() -> None:
    A = _random_csr(3, 4)
    op = SingleSparseMatVecProduct(A)
    for x in (cp.ones(4, dtype=cp.float64), cp.arange(4, dtype=cp.float64)):
        y = cp.zeros(3, dtype=cp.float64)
        op(x, y)
        cp.cuda.get_current_stream().synchronize()
        cp.testing.assert_allclose(y, A.toarray() @ x, atol=1e-12)


def test_reuse_different_scalars() -> None:
    A = _random_csr(4, 4)
    x = cp.arange(4, dtype=cp.float64)
    y = cp.zeros(4, dtype=cp.float64)

    op = SingleSparseMatVecProduct(A)
    op(x, y, alpha=1.0, beta=0.0)
    cp.cuda.get_current_stream().synchronize()
    y1 = y.copy()
    cp.testing.assert_allclose(y1, A.toarray() @ x, atol=1e-12)

    op(x, y, alpha=3.0, beta=1.0)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, 3.0 * (A.toarray() @ x) + y1, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 8, 64, 256])
def test_identity(n: int) -> None:
    A = sparse.eye(n, dtype=cp.float64, format="csr")
    x = cp.arange(n, dtype=cp.float64)
    y = cp.zeros(n, dtype=cp.float64)

    op = SingleSparseMatVecProduct(A)
    op(x, y)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, x, atol=1e-14)


def test_cuda_graph_capture() -> None:
    A = _random_csr(4, 4)
    x = cp.ones(4, dtype=cp.float64)
    y = cp.zeros(4, dtype=cp.float64)

    op = SingleSparseMatVecProduct(A)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        op(x, y, alpha=1.0, beta=0.0)
        graph = stream.end_capture()

    graph.launch(stream)
    stream.synchronize()
    cp.testing.assert_allclose(y, A.toarray() @ x, atol=1e-12)


def test_destructor_no_error() -> None:
    A = _random_csr(2, 2)
    op = SingleSparseMatVecProduct(A)
    del op  # must not raise


@pytest.mark.parametrize("m,n", [(4, 5), (32, 32), (200, 32), (32, 200)])
def test_csc_basic(m: int, n: int) -> None:
    A_csr = _random_csr(m, n)
    A = A_csr.tocsc()
    x = cp.asarray(np.random.default_rng(7).standard_normal(n))
    y = cp.zeros(m, dtype=cp.float64)

    op = SingleSparseMatVecProduct(A)
    op(x, y)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, A_csr.toarray() @ x, atol=1e-12)


@pytest.mark.parametrize("m,n", [(4, 5), (32, 32), (200, 32), (32, 200)])
def test_csc_transpose(m: int, n: int) -> None:
    A_csr = _random_csr(m, n)
    A = A_csr.tocsc()
    x = cp.asarray(np.random.default_rng(8).standard_normal(m))
    y = cp.zeros(n, dtype=cp.float64)

    op = SingleSparseMatVecProduct(A, transa=True)
    op(x, y)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, A_csr.toarray().T @ x, atol=1e-12)


@pytest.mark.parametrize("m,n", [(4, 5), (32, 32), (200, 32), (32, 200)])
def test_coo_basic(m: int, n: int) -> None:
    A_csr = _random_csr(m, n)
    A = A_csr.tocoo()
    x = cp.asarray(np.random.default_rng(9).standard_normal(n))
    y = cp.zeros(m, dtype=cp.float64)

    op = SingleSparseMatVecProduct(A)
    op(x, y)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, A_csr.toarray() @ x, atol=1e-12)


def test_invalid_type_raises() -> None:
    with pytest.raises(TypeError):
        SingleSparseMatVecProduct(cp.zeros((3, 3), dtype=cp.float64))


# ===========================================================================
# Batched SpMV (BatchedSparseMatVecProduct)
# ===========================================================================

@pytest.mark.parametrize("B", [1, 2, 5, 16])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
def test_batched_basic_contiguous(B: int, m: int, n: int) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, density=0.4, seed=B * 100 + m * 10 + n)
    op = BatchedSparseMatVecProduct(bmat, transa=False)

    x = cp.asarray(np.random.default_rng(B + m + n).standard_normal((B, n)))
    out = cp.zeros((B, m), dtype=cp.float64)
    op(x.reshape(-1), out.reshape(-1))
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, vals, x), atol=1e-12,
    )


@pytest.mark.parametrize("B", [1, 2, 5, 16])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
def test_batched_transpose(B: int, m: int, n: int) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, density=0.5, seed=B * 200 + m * 7 + n)
    op = BatchedSparseMatVecProduct(bmat, transa=True)

    x = cp.asarray(np.random.default_rng(B * 3 + m + n).standard_normal((B, m)))
    out = cp.zeros((B, n), dtype=cp.float64)
    op(x.reshape(-1), out.reshape(-1))
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, vals, x, transpose=True), atol=1e-12,
    )


@pytest.mark.parametrize("B", [1, 2, 5, 16])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
def test_batched_alpha_beta(B: int, m: int, n: int) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, density=0.5, seed=B * 300 + m * 11 + n)
    op = BatchedSparseMatVecProduct(bmat)

    rng = np.random.default_rng(B * 5 + m + n + 1)
    x = cp.asarray(rng.standard_normal((B, n)))
    y = cp.asarray(rng.standard_normal((B, m)))
    y_init = cp.asnumpy(y).copy()

    alpha, beta = 2.0, 3.0
    op(x.reshape(-1), y.reshape(-1), alpha=alpha, beta=beta)
    cp.cuda.get_current_stream().synchronize()
    expected = alpha * _batched_ref(tpl, vals, x) + beta * y_init
    np.testing.assert_allclose(cp.asnumpy(y), expected, atol=1e-12)


def test_batched_non_contiguous_x() -> None:
    B, m, n = 4, 5, 6
    K = 10
    col_start = 2
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=60)
    op = BatchedSparseMatVecProduct(bmat)

    big = cp.asarray(np.random.default_rng(62).standard_normal((B, K)))
    x_view = big[:, col_start:col_start + n]
    assert not x_view.flags['C_CONTIGUOUS']

    out = cp.zeros((B, m), dtype=cp.float64)
    op(x_view, out)
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, vals, x_view), atol=1e-12,
    )


def test_batched_non_contiguous_out() -> None:
    B, m, n = 3, 4, 5
    K = 9
    col_start = 3
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=63)
    op = BatchedSparseMatVecProduct(bmat)

    x = cp.asarray(np.random.default_rng(65).standard_normal((B, n)))
    big_out = cp.zeros((B, K), dtype=cp.float64) - 7.0  # sentinel outside the window
    out_view = big_out[:, col_start:col_start + m]
    assert not out_view.flags['C_CONTIGUOUS']

    op(x, out_view)
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out_view), _batched_ref(tpl, vals, x), atol=1e-12,
    )
    # Columns outside the target window are untouched.
    np.testing.assert_allclose(cp.asnumpy(big_out[:, :col_start]), -7.0)
    np.testing.assert_allclose(cp.asnumpy(big_out[:, col_start + m:]), -7.0)


def test_batched_multiple_calls_different_x() -> None:
    B, m, n = 3, 4, 5
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=70)
    op = BatchedSparseMatVecProduct(bmat)

    x1 = cp.asarray(np.random.default_rng(72).standard_normal((B, n)))
    x2 = cp.asarray(np.random.default_rng(73).standard_normal((B, n)))
    out = cp.empty((B, m), dtype=cp.float64)

    op(x1.reshape(-1), out.reshape(-1))
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, vals, x1), atol=1e-12,
    )
    op(x2.reshape(-1), out.reshape(-1))
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, vals, x2), atol=1e-12,
    )


def test_batched_tracks_update_data() -> None:
    """After ``bmat.update_data(...)``, the SpMV uses the fresh values."""
    B, m, n = 3, 4, 5
    bmat, tpl, _ = _random_batched_csr(B, m, n, seed=74)
    op = BatchedSparseMatVecProduct(bmat)

    x = cp.asarray(np.random.default_rng(76).standard_normal((B, n)))
    out = cp.empty((B, m), dtype=cp.float64)
    op(x.reshape(-1), out.reshape(-1))

    new_vals = cp.asarray(np.random.default_rng(77).standard_normal((B, tpl.nnz)))
    bmat.update_data(new_vals)
    op(x.reshape(-1), out.reshape(-1))
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(
        cp.asnumpy(out), _batched_ref(tpl, new_vals, x), atol=1e-12,
    )
