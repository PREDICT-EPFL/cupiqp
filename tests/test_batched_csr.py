import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp_cpu
import warp as wp
from cupyx.scipy.sparse import csr_matrix

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix

try:
    import torch
    _TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    _TORCH_CUDA = False


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
# __getitem__: cupy CSR views for setup-time algebra
# ===========================================================================

@pytest.mark.parametrize("B,m,n", [(1, 3, 4), (3, 5, 7), (8, 6, 6)])
def test_getitem_returns_correct_csr(B: int, m: int, n: int) -> None:
    tpl = random_csr_template(m, n, seed=4)
    mat, values = build_from_template(tpl, B, seed=5)

    tpl_cpu = tpl.get()
    for i in range(B):
        got = mat[i]
        assert isinstance(got, csr_matrix)
        assert got.shape == (m, n)
        np.testing.assert_allclose(cp.asnumpy(got.data), cp.asnumpy(values[i]))

        ref = tpl_cpu.copy()
        ref.data[:] = cp.asnumpy(values[i])
        np.testing.assert_allclose(got.toarray().get(), ref.toarray())


def test_getitem_returns_view() -> None:
    """Mutating the returned csr_matrix's data writes through to ``mat.data``."""
    tpl = random_csr_template(3, 4, seed=6)
    mat, _ = build_from_template(tpl, B=2, seed=7)

    got = mat[0]
    got.data[0] = 999.0
    assert float(mat.data.numpy()[0, 0]) == 999.0


# ===========================================================================
# Classmethod constructors
# ===========================================================================

def test_from_cupy_csr_matrix_single() -> None:
    """``from_cupy_csr_matrix`` wraps a single cupy CSR as a B=1 batched matrix."""
    tpl = random_csr_template(5, 7, seed=30)
    mat = UniformBatchedCsrMatrix.from_cupy_csr_matrix(tpl)

    assert mat.batch_size == 1
    assert mat.nnz == tpl.nnz
    assert mat.shape == (1, 5, 7)
    np.testing.assert_allclose(mat.data.numpy()[0], cp.asnumpy(tpl.data))


def test_from_cupy_csr_matrix_tiled() -> None:
    """``batch_size > 1`` replicates the values into every matrix."""
    tpl = random_csr_template(5, 7, seed=30)
    mat = UniformBatchedCsrMatrix.from_cupy_csr_matrix(tpl, batch_size=4)
    assert mat.shape == (4, 5, 7)
    np.testing.assert_allclose(mat.data.numpy(), np.broadcast_to(cp.asnumpy(tpl.data), (4, tpl.nnz)))


def test_from_cupy_csr_matrix_float32() -> None:
    tpl = random_csr_template(5, 7, seed=30)
    mat = UniformBatchedCsrMatrix.from_cupy_csr_matrix(tpl, dtype=wp.float32)
    assert mat.dtype is wp.float32 and mat.data.dtype is wp.float32
    np.testing.assert_allclose(mat.data.numpy()[0], cp.asnumpy(tpl.data).astype(np.float32))


def test_from_cupy_csr_matrix_sequence_uniform() -> None:
    """A list of CSR matrices that share sparsity is packed into one batched matrix."""
    tpl = random_csr_template(4, 5, seed=31)
    B = 3
    rng = np.random.default_rng(32)
    matrices = [
        csr_matrix(
            (cp.asarray(rng.standard_normal(tpl.nnz)), tpl.indices, tpl.indptr),
            shape=tpl.shape,
        )
        for _ in range(B)
    ]
    mat = UniformBatchedCsrMatrix.from_cupy_csr_matrix_sequence(matrices)

    assert mat.batch_size == B
    assert mat.shape == (B, 4, 5)
    for i in range(B):
        np.testing.assert_allclose(mat.data.numpy()[i], cp.asnumpy(matrices[i].data))


def test_from_cupy_csr_matrix_sequence_rejects_mismatched_pattern() -> None:
    """Two CSRs with the same shape but different patterns must be rejected."""
    tpl_a = random_csr_template(3, 4, density=0.4, seed=33)
    tpl_b = random_csr_template(3, 4, density=0.9, seed=34)
    if tpl_a.nnz == tpl_b.nnz and bool(cp.array_equal(tpl_a.indices, tpl_b.indices)):
        pytest.skip("random templates happened to coincide")
    with pytest.raises(ValueError, match="sparsity pattern"):
        UniformBatchedCsrMatrix.from_cupy_csr_matrix_sequence([tpl_a, tpl_b])


def test_from_cupy_csr_matrix_sequence_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        UniformBatchedCsrMatrix.from_cupy_csr_matrix_sequence([])


def test_from_input_dispatches_each_form() -> None:
    """``from_input`` accepts: single cupy CSR, list of CSRs, and existing
    ``UniformBatchedCsrMatrix`` (returns an independent copy)."""
    tpl = random_csr_template(4, 6, seed=35)

    single = UniformBatchedCsrMatrix.from_input(tpl)
    assert single.batch_size == 1 and single.shape == (1, 4, 6)

    listed = UniformBatchedCsrMatrix.from_input([tpl, tpl])
    assert listed.batch_size == 2 and listed.shape == (2, 4, 6)

    existing, _ = build_from_template(tpl, B=3, seed=36)
    copied = UniformBatchedCsrMatrix.from_input(existing)
    assert copied.batch_size == 3 and copied.shape == (3, 4, 6)
    np.testing.assert_allclose(copied.data.numpy(), existing.data.numpy())
    # Independence: mutating the copy must not bleed into the original.
    copied.data.zero_()
    assert not bool(np.allclose(existing.data.numpy(), 0.0))


def test_empty_classmethod() -> None:
    """``empty(B, rows, cols)`` returns a zero-nnz batched CSR with the declared shape."""
    mat = UniformBatchedCsrMatrix.empty(batch_size=4, rows=5, cols=7)
    assert mat.batch_size == 4
    assert mat.nnz == 0
    assert mat.shape == (4, 5, 7)
    assert mat.data.shape == (4, 0)
    assert mat.indptr.shape == (6,)
    assert mat.row_indices.shape == (0,)


# ===========================================================================
# Torch interop (skipped when CUDA torch isn't available)
# ===========================================================================

@pytest.mark.skipif(not _TORCH_CUDA, reason="torch with CUDA not available")
class TestFromTorchSparseCsrTensor:
    """Round-trip and rejection tests for ``from_torch_sparse_csr_tensor``."""

    @staticmethod
    def _build_torch_batched_csr(template: csr_matrix, B: int, seed: int):
        """Build a 3-D ``torch.sparse_csr_tensor`` sharing sparsity across B batches."""
        tpl_cpu = template.get()
        crow = torch.from_numpy(tpl_cpu.indptr.astype(np.int32)).cuda()
        col = torch.from_numpy(tpl_cpu.indices.astype(np.int32)).cuda()
        vals_np = (
            np.random.default_rng(seed)
            .standard_normal((B, tpl_cpu.nnz))
            .astype(np.float64)
        )
        vals = torch.from_numpy(vals_np).cuda()
        crow_b = crow.unsqueeze(0).expand(B, -1).contiguous()
        col_b = col.unsqueeze(0).expand(B, -1).contiguous()
        return (
            torch.sparse_csr_tensor(
                crow_b, col_b, vals,
                size=(B, tpl_cpu.shape[0], tpl_cpu.shape[1]),
            ),
            vals_np,
        )

    def test_basic_roundtrip(self) -> None:
        B, m, n = 4, 5, 7
        tpl = random_csr_template(m, n, density=0.4, seed=101)
        tensor, vals_np = self._build_torch_batched_csr(tpl, B, seed=102)

        mat = UniformBatchedCsrMatrix.from_torch_sparse_csr_tensor(tensor)

        assert mat.batch_size == B
        assert mat.nnz == tpl.nnz
        assert mat.shape == (B, m, n)
        np.testing.assert_allclose(mat.data.numpy(), vals_np)
        np.testing.assert_array_equal(mat.indptr.numpy(), tpl.get().indptr)
        np.testing.assert_array_equal(mat.indices.numpy(), tpl.get().indices)

    def test_reconstructed_matrix_equals_torch_dense(self) -> None:
        B, m, n = 3, 4, 6
        tpl = random_csr_template(m, n, density=0.5, seed=103)
        tensor, _ = self._build_torch_batched_csr(tpl, B, seed=104)

        mat = UniformBatchedCsrMatrix.from_torch_sparse_csr_tensor(tensor)
        dense_torch = tensor.to_dense().cpu().numpy()
        for b in range(B):
            np.testing.assert_allclose(
                mat[b].toarray().get(), dense_torch[b], atol=1e-12,
            )

    def test_rejects_non_cuda_tensor(self) -> None:
        tpl = random_csr_template(3, 4, seed=105)
        tensor, _ = self._build_torch_batched_csr(tpl, B=2, seed=106)
        with pytest.raises(ValueError, match="CUDA"):
            UniformBatchedCsrMatrix.from_torch_sparse_csr_tensor(tensor.cpu())

    def test_accepts_2d_tensor(self) -> None:
        """2-D ``(M, N)`` torch CSR is supported as a single-batch matrix, or
        tiled into a batch with ``batch_size``."""
        tpl_cpu = random_csr_template(3, 4, seed=107).get()
        crow = torch.from_numpy(tpl_cpu.indptr.astype(np.int32)).cuda()
        col = torch.from_numpy(tpl_cpu.indices.astype(np.int32)).cuda()
        vals = torch.from_numpy(tpl_cpu.data.astype(np.float64)).cuda()
        tensor = torch.sparse_csr_tensor(crow, col, vals, size=tpl_cpu.shape)

        mat = UniformBatchedCsrMatrix.from_torch_sparse_csr_tensor(tensor)
        assert mat.batch_size == 1
        assert mat.shape == (1, *tpl_cpu.shape)

        tiled = UniformBatchedCsrMatrix.from_torch_sparse_csr_tensor(tensor, batch_size=3)
        assert tiled.shape == (3, *tpl_cpu.shape)
        np.testing.assert_allclose(tiled.data.numpy(), np.broadcast_to(tpl_cpu.data, (3, tpl_cpu.nnz)))

    def test_rejects_wrong_layout(self) -> None:
        dense = torch.randn(2, 3, 4, device='cuda')
        with pytest.raises(ValueError, match="sparse_csr"):
            UniformBatchedCsrMatrix.from_torch_sparse_csr_tensor(dense)

    def test_rejects_non_tensor(self) -> None:
        with pytest.raises(TypeError):
            UniformBatchedCsrMatrix.from_torch_sparse_csr_tensor("not a tensor")
