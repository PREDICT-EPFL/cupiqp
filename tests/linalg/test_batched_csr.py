import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
from tests.helpers import DTYPES, csr_triple, to_wp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_csr_template(
    m: int, n: int, density: float = 0.5, seed: int = 42,
) -> sp.csr_matrix:
    """A host ``sp.csr_matrix`` with random nonzeros at density ``density``."""
    rng = np.random.default_rng(seed)
    A_cpu = sp.random(
        m, n, density=density, format='csr', random_state=rng, dtype=np.float64,
    )
    return sp.csr_matrix(A_cpu)


def random_values(B: int, nnz: int, seed: int = 0, dtype=wp.float64) -> wp.array:
    """A random ``(B, nnz)`` Warp array of values."""
    rng = np.random.default_rng(seed)
    return to_wp(rng.standard_normal((B, nnz)), dtype)


def build_from_template(
    template: sp.csr_matrix, B: int, seed: int = 0, dtype=wp.float64,
) -> tuple[UniformBatchedCsrMatrix, wp.array]:
    """Construct a uniform batched CSR sharing the template's sparsity."""
    indptr, indices, _ = csr_triple(template)
    values = random_values(B, template.nnz, seed=seed, dtype=dtype)
    mat = UniformBatchedCsrMatrix(
        batch_size=B,
        indptr=indptr,
        indices=indices,
        data=values,
        shape=template.shape,
        dtype=dtype,
    )
    return mat, values


# ===========================================================================
# Construction and storage
# ===========================================================================

@pytest.mark.parametrize("dtype", DTYPES)
def test_storage_is_warp(dtype) -> None:
    """indptr / indices are int32 Warp arrays, data is a (B, nnz) Warp array
    of the matrix dtype."""
    tpl = random_csr_template(4, 5, seed=1)
    mat, values = build_from_template(tpl, B=3, seed=2, dtype=dtype)
    assert isinstance(mat.indptr, wp.array) and mat.indptr.dtype is wp.int32
    assert isinstance(mat.indices, wp.array) and mat.indices.dtype is wp.int32
    assert isinstance(mat.data, wp.array) and mat.data.dtype is dtype
    assert mat.data.shape == (3, tpl.nnz)
    np.testing.assert_array_equal(mat.indptr.numpy(), tpl.indptr)
    np.testing.assert_array_equal(mat.indices.numpy(), tpl.indices)
    np.testing.assert_allclose(mat.data.numpy(), values.numpy())


@pytest.mark.parametrize("dtype", DTYPES)
def test_construction_copies_values(dtype) -> None:
    """The values are copied: mutating the source afterwards has no effect."""
    tpl = random_csr_template(3, 4, seed=2)
    mat, values = build_from_template(tpl, B=2, seed=3, dtype=dtype)
    values.zero_()
    assert not bool(np.allclose(mat.data.numpy(), 0.0))


@pytest.mark.parametrize("dtype", DTYPES)
def test_construction_broadcasts_shared_values(dtype) -> None:
    """``(nnz,)`` values are replicated into every matrix of the batch."""
    tpl = random_csr_template(3, 4, seed=5)
    triple = csr_triple(tpl, wp.dtype_to_numpy(dtype))
    mat = UniformBatchedCsrMatrix(3, *triple, shape=tpl.shape, dtype=dtype)
    np.testing.assert_array_equal(mat.data.numpy(), np.broadcast_to(triple[2].numpy(), (3, tpl.nnz)))


def test_construction_wrong_data_shape_raises() -> None:
    B, m, n = 3, 4, 5
    tpl = random_csr_template(m, n, seed=3)
    indptr, indices, _ = csr_triple(tpl)
    bad = to_wp(np.zeros((B, tpl.nnz + 1)))
    with pytest.raises(ValueError):
        UniformBatchedCsrMatrix(B, indptr, indices, bad, shape=tpl.shape)


def test_construction_zero_batch_size_raises() -> None:
    tpl = random_csr_template(3, 4, seed=4)
    with pytest.raises(ValueError, match="batch_size"):
        indptr, indices, _ = csr_triple(tpl)
        UniformBatchedCsrMatrix(0, indptr, indices, to_wp(np.zeros((0, tpl.nnz))), shape=tpl.shape)


def test_construction_wrong_indptr_length_raises() -> None:
    tpl = random_csr_template(3, 4, seed=4)
    with pytest.raises(ValueError, match="indptr"):
        UniformBatchedCsrMatrix(1, *csr_triple(tpl), shape=(5, 4))


@pytest.mark.parametrize("B,m,n", [(1, 3, 4), (3, 5, 7), (8, 6, 6)])
def test_row_indices(B: int, m: int, n: int) -> None:
    """``row_indices`` is the COO row array of the shared pattern."""
    tpl = random_csr_template(m, n, seed=B * 3 + n)
    mat, _ = build_from_template(tpl, B, seed=1)
    expected = tpl.tocoo().row
    assert mat.row_indices.dtype is wp.int32
    np.testing.assert_array_equal(mat.row_indices.numpy(), expected)


# ===========================================================================
# pattern(): host scipy pattern for setup-time algebra
# ===========================================================================

def test_pattern_keeps_explicit_zeros() -> None:
    """``pattern()`` has a one on every stored entry, also where the values are zero."""
    tpl = random_csr_template(4, 5, seed=8)
    indptr, indices, _ = csr_triple(tpl)
    values = to_wp(np.zeros((2, tpl.nnz)))
    mat = UniformBatchedCsrMatrix(2, indptr, indices, values, shape=tpl.shape)

    pat = mat.pattern()
    assert isinstance(pat, sp.csr_matrix)
    assert pat.nnz == tpl.nnz
    np.testing.assert_array_equal(pat.data, np.ones(tpl.nnz))
    np.testing.assert_array_equal(pat.indptr, tpl.indptr)
    np.testing.assert_array_equal(pat.indices, tpl.indices)


# ===========================================================================
# empty()
# ===========================================================================

@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_classmethod(dtype) -> None:
    """``empty(B, rows, cols)`` returns a zero-nnz batched CSR with the declared shape."""
    mat = UniformBatchedCsrMatrix.empty(batch_size=4, rows=5, cols=7, dtype=dtype)
    assert mat.dtype is dtype
    assert mat.batch_size == 4
    assert mat.nnz == 0
    assert mat.shape == (4, 5, 7)
    assert mat.data.shape == (4, 0)
    assert mat.indptr.shape == (6,)
    assert mat.row_indices.shape == (0,)
