import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp_cpu
import warp as wp
from cupyx.scipy.sparse import csr_matrix

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_csr_template(
    m: int, n: int, density: float = 0.5, seed: int = 42,
) -> csr_matrix:
    """A GPU ``csr_matrix`` with random nonzeros at density ``density``."""
    rng = np.random.default_rng(seed)
    A_cpu = sp_cpu.random(
        m, n, density=density, format='csr', random_state=rng, dtype=np.float64,
    )
    return csr_matrix(A_cpu)


def random_values(B: int, nnz: int, seed: int = 0) -> cp.ndarray:
    """A random ``(B, nnz)`` cupy float64 array of values."""
    rng = np.random.default_rng(seed)
    return cp.asarray(rng.standard_normal((B, nnz)))


def build_from_template(
    template: csr_matrix, B: int, seed: int = 0,
) -> tuple[UniformBatchedCsrMatrix, cp.ndarray]:
    """Construct a uniform batched CSR sharing the template's sparsity."""
    values = random_values(B, template.nnz, seed=seed)
    mat = UniformBatchedCsrMatrix(
        batch_size=B,
        indptr=template.indptr,
        indices=template.indices,
        data=values,
        shape=template.shape,
    )
    return mat, values


# ===========================================================================
# Construction and storage
# ===========================================================================

def test_storage_is_warp() -> None:
    """indptr / indices are int32 Warp arrays, data is a (B, nnz) Warp array."""
    tpl = random_csr_template(4, 5, seed=1)
    mat, values = build_from_template(tpl, B=3, seed=2)
    assert isinstance(mat.indptr, wp.array) and mat.indptr.dtype is wp.int32
    assert isinstance(mat.indices, wp.array) and mat.indices.dtype is wp.int32
    assert isinstance(mat.data, wp.array) and mat.data.dtype is wp.float64
    assert mat.data.shape == (3, tpl.nnz)
    np.testing.assert_array_equal(mat.indptr.numpy(), tpl.get().indptr)
    np.testing.assert_array_equal(mat.indices.numpy(), tpl.get().indices)
    np.testing.assert_allclose(mat.data.numpy(), cp.asnumpy(values))


def test_construction_copies_values() -> None:
    """The values are copied: mutating the source afterwards has no effect."""
    tpl = random_csr_template(3, 4, seed=2)
    mat, values = build_from_template(tpl, B=2, seed=3)
    values[:] = 0.0
    assert not bool(np.allclose(mat.data.numpy(), 0.0))


def test_construction_broadcasts_shared_values() -> None:
    """``(nnz,)`` values are replicated into every matrix of the batch."""
    tpl = random_csr_template(3, 4, seed=5)
    mat = UniformBatchedCsrMatrix(3, tpl.indptr, tpl.indices, tpl.data, shape=tpl.shape)
    np.testing.assert_allclose(mat.data.numpy(), np.broadcast_to(cp.asnumpy(tpl.data), (3, tpl.nnz)))


def test_construction_wrong_data_shape_raises() -> None:
    B, m, n = 3, 4, 5
    tpl = random_csr_template(m, n, seed=3)
    bad = cp.zeros((B, tpl.nnz + 1), dtype=cp.float64)
    with pytest.raises(ValueError):
        UniformBatchedCsrMatrix(B, tpl.indptr, tpl.indices, bad, shape=tpl.shape)


def test_construction_zero_batch_size_raises() -> None:
    tpl = random_csr_template(3, 4, seed=4)
    with pytest.raises(ValueError, match="batch_size"):
        UniformBatchedCsrMatrix(0, tpl.indptr, tpl.indices, cp.zeros((0, tpl.nnz)), shape=tpl.shape)


def test_construction_wrong_indptr_length_raises() -> None:
    tpl = random_csr_template(3, 4, seed=4)
    with pytest.raises(ValueError, match="indptr"):
        UniformBatchedCsrMatrix(1, tpl.indptr, tpl.indices, tpl.data, shape=(5, 4))


@pytest.mark.parametrize("B,m,n", [(1, 3, 4), (3, 5, 7), (8, 6, 6)])
def test_row_indices(B: int, m: int, n: int) -> None:
    """``row_indices`` is the COO row array of the shared pattern."""
    tpl = random_csr_template(m, n, seed=B * 3 + n)
    mat, _ = build_from_template(tpl, B, seed=1)
    expected = tpl.get().tocoo().row
    assert mat.row_indices.dtype is wp.int32
    np.testing.assert_array_equal(mat.row_indices.numpy(), expected)


# ===========================================================================
# pattern(): host scipy pattern for setup-time algebra
# ===========================================================================

def test_pattern_keeps_explicit_zeros() -> None:
    """``pattern()`` has a one on every stored entry, also where the values are zero."""
    tpl = random_csr_template(4, 5, seed=8)
    values = cp.zeros((2, tpl.nnz))
    mat = UniformBatchedCsrMatrix(2, tpl.indptr, tpl.indices, values, shape=tpl.shape)

    pat = mat.pattern()
    assert isinstance(pat, sp_cpu.csr_matrix)
    assert pat.nnz == tpl.nnz
    np.testing.assert_array_equal(pat.data, np.ones(tpl.nnz))
    np.testing.assert_array_equal(pat.indptr, cp.asnumpy(tpl.indptr))
    np.testing.assert_array_equal(pat.indices, cp.asnumpy(tpl.indices))


# ===========================================================================
# empty()
# ===========================================================================

def test_empty_classmethod() -> None:
    """``empty(B, rows, cols)`` returns a zero-nnz batched CSR with the declared shape."""
    mat = UniformBatchedCsrMatrix.empty(batch_size=4, rows=5, cols=7)
    assert mat.batch_size == 4
    assert mat.nnz == 0
    assert mat.shape == (4, 5, 7)
    assert mat.data.shape == (4, 0)
    assert mat.indptr.shape == (6,)
    assert mat.row_indices.shape == (0,)
