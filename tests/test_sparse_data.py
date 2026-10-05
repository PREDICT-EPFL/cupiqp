"""Tests for SparseData (CSR triples, the values-only matrix setters) and the
SparseSolver setup contract: CSR triples ``(indptr, indices, values)``, each
value array shared or batched, the batch size read from the batched arrays.
"""
import numpy as np
import pytest
import cupy as cp
import warp as wp
import scipy.sparse as sp_cpu
from cupyx.scipy.sparse import csr_matrix

from cupiqp import SparseSolver
from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
from cupiqp.sparse.sparse_data import SparseData

try:
    import torch
    _TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    _TORCH_CUDA = False


def _w(a):
    """Zero-copy Warp view: Data objects take Warp arrays only (the solver converts user arrays)."""
    return wp.array(a, copy=False)


def _triple(M, values=None):
    """CSR triple of a cupy csr_matrix, optionally with other values."""
    return (M.indptr, M.indices, M.data if values is None else values)


def _w_triple(M, values=None):
    return tuple(_w(a) for a in _triple(M, values))


def _sparse_data(**kw):
    """Build SparseData; matrices are given as cupy csr_matrix (or a triple),
    vectors as cupy arrays."""
    d = SparseData()
    args = {}
    for k, v in kw.items():
        if k in ("P", "A", "G"):
            args[k] = _w_triple(v) if isinstance(v, csr_matrix) else tuple(_w(a) for a in v)
        else:
            args[k] = _w(v)
    d.init(**args)
    return d


def _make_spd_csr(n, density=0.4, seed=42):
    rng = np.random.default_rng(seed)
    M = sp_cpu.random(n, n, density=density, format='csr', random_state=rng)
    P = ((M @ M.T + n * sp_cpu.eye(n)) / 2).tocsr()
    P = (P + P.T).tocsr()
    P.sort_indices()
    return csr_matrix(P)


def _make_rect_csr(rows, cols, density=0.5, seed=42):
    rng = np.random.default_rng(seed)
    A = sp_cpu.random(rows, cols, density=density, format='csr', random_state=rng)
    A.sort_indices()
    return csr_matrix(A)


# ===========================================================================
# SparseData
# ===========================================================================

class TestSparseDataInit:

    def test_shared_values(self):
        n = 6
        P = _make_spd_csr(n, seed=1)
        data = _sparse_data(P=P, c=cp.ones(n))
        assert isinstance(data.P, UniformBatchedCsrMatrix)
        assert data.batch_size == 1
        assert (data.n, data.p, data.m) == (n, 0, 0)
        assert data.P.shape == (1, n, n)
        np.testing.assert_allclose(data.P.data.numpy()[0], cp.asnumpy(P.data))

    def test_batched_values(self):
        B, n = 4, 5
        P = _make_spd_csr(n, seed=10)
        vals = cp.asarray(np.random.default_rng(11).standard_normal((B, P.nnz)))
        data = _sparse_data(P=_triple(P, vals), c=cp.zeros(n))
        assert data.batch_size == B
        assert data.P.shape == (B, n, n)
        np.testing.assert_allclose(data.P.data.numpy(), cp.asnumpy(vals))

    def test_batch_size_from_a_vector(self):
        """Shared matrix values with a batched vector: B comes from the vector
        and the values are copied into every problem."""
        B, n = 3, 5
        P = _make_spd_csr(n, seed=12)
        data = _sparse_data(P=P, c=cp.zeros((B, n)))
        assert data.batch_size == B
        np.testing.assert_allclose(data.P.data.numpy(), np.broadcast_to(cp.asnumpy(P.data), (B, P.nnz)))

    def test_values_are_copied(self):
        n = 6
        P = _make_spd_csr(n, seed=20)
        vals = P.data.copy()
        data = _sparse_data(P=_triple(P, vals), c=cp.ones(n))
        vals[:] = 0.0
        assert not np.allclose(data.P.data.numpy(), 0.0)

    def test_all_blocks(self):
        B, n, p, m = 2, 5, 3, 4
        P = _make_spd_csr(n, seed=40)
        A = _make_rect_csr(p, n, seed=41)
        G = _make_rect_csr(m, n, seed=42)
        data = _sparse_data(
            P=P, c=cp.zeros((B, n)),
            A=A, b=cp.zeros((B, p)),
            G=G, h_u=cp.ones((B, m)), h_l=-cp.ones((B, m)),
        )
        assert data.batch_size == B
        assert (data.n, data.p, data.m) == (n, p, m)
        assert data.A.shape == (B, p, n) and data.G.shape == (B, m, n)

    def test_empty_A_and_G(self):
        """Omitting A and G yields empty batched CSR placeholders."""
        B, n = 2, 4
        P = _make_spd_csr(n, seed=50)
        data = _sparse_data(P=P, c=cp.zeros((B, n)))
        assert data.p == 0 and data.m == 0
        assert data.A.shape == (B, 0, n) and data.G.shape == (B, 0, n)
        assert data.A.nnz == 0 and data.G.nnz == 0

    def test_batch_mismatch_raises(self):
        n = 4
        P = _make_spd_csr(n, seed=51)
        vals = cp.ones((3, P.nnz))
        with pytest.raises(ValueError, match="disagree on the batch size"):
            _sparse_data(P=_triple(P, vals), c=cp.zeros((2, n)))


class TestSetters:
    """``set_P`` / ``set_A`` / ``set_G`` take only the stored values, as a
    ``(B, nnz)`` array (or ``(nnz,)`` broadcast), in the setup CSR order."""

    def test_set_P_batched_values(self):
        B, n = 3, 5
        P = _make_spd_csr(n, seed=60)
        data = _sparse_data(P=P, c=cp.zeros((B, n)))
        vals = cp.asarray(np.random.default_rng(61).standard_normal((B, P.nnz)))
        data.set_P(_w(vals))
        np.testing.assert_allclose(data.P.data.numpy(), cp.asnumpy(vals))

    def test_set_P_broadcast_values(self):
        B, n = 3, 5
        P = _make_spd_csr(n, seed=70)
        data = _sparse_data(P=P, c=cp.zeros((B, n)))
        data.set_P(_w(cp.ones(P.nnz)))
        np.testing.assert_allclose(data.P.data.numpy(), np.ones((B, P.nnz)))

    def test_set_P_keeps_buffer_address(self):
        n = 5
        P = _make_spd_csr(n, seed=71)
        data = _sparse_data(P=P, c=cp.zeros(n))
        ptr = data.P.data.ptr
        data.set_P(_w(cp.ones(P.nnz)))
        assert data.P.data.ptr == ptr

    def test_set_P_rejects_wrong_nnz(self):
        n = 5
        P = _make_spd_csr(n, seed=80)
        data = _sparse_data(P=P, c=cp.zeros(n))
        with pytest.raises(ValueError, match="shape mismatch"):
            data.set_P(_w(cp.ones(P.nnz + 1)))

    def test_set_A_and_G_values(self):
        n = 4
        P = _make_spd_csr(n, seed=83)
        A = csr_matrix(cp.asarray(np.array([[1.0, 0.0, 2.0, 0.0]])))
        G = csr_matrix(cp.eye(n))
        data = _sparse_data(P=P, c=cp.zeros(n), A=A, b=cp.zeros(1),
                            G=G, h_u=cp.ones(n))
        data.set_A(_w(cp.asarray([3.0, 4.0])))
        data.set_G(_w(2.0 * cp.ones(n)))
        np.testing.assert_allclose(data.A.data.numpy(), [[3.0, 4.0]])
        np.testing.assert_allclose(data.G.data.numpy(), 2.0 * np.ones((1, n)))

    def test_inplace_mutation_visible_via_getitem(self):
        """``data.P[b]`` is a view of the stored values."""
        B, n = 2, 4
        P = _make_spd_csr(n, seed=90)
        data = _sparse_data(P=P, c=cp.zeros((B, n)))
        cp.asarray(data.P.data)[:] *= 3.0
        np.testing.assert_allclose(data.P[0].data.get(), 3.0 * P.data.get())


# ===========================================================================
# SparseSolver.setup
# ===========================================================================

def _solver():
    s = SparseSolver()
    s.settings.verbose = False
    return s


class TestSolverSetup:

    def test_all_shared_is_one_problem(self):
        n = 4
        P = _make_spd_csr(n, seed=90)
        s = _solver()
        s.setup(P=_triple(P), c=cp.arange(n, dtype=cp.float64))
        assert s.data.batch_size == 1
        s.solve()

    def test_batched_setup_matches_individual_solves(self):
        """Per-problem values given at setup solve each problem."""
        B, n = 3, 4
        P = _make_spd_csr(n, seed=91)
        rng = np.random.default_rng(92)
        P_vals = cp.asarray(1.0 + rng.random((B, 1))) * P.data[None, :]
        c = cp.asarray(rng.standard_normal((B, n)))

        s = _solver()
        s.setup(P=_triple(P, P_vals), c=c)
        assert s.data.batch_size == B
        s.solve()
        for k in range(B):
            ref = _solver()
            ref.setup(P=_triple(P, P_vals[k]), c=c[k])
            ref.solve()
            np.testing.assert_allclose(
                cp.asnumpy(s.result.x)[k], cp.asnumpy(ref.result.x)[0], atol=1e-8,
            )

    def test_broadcast_setup_then_update_matches_batched_setup(self):
        """Equal problems at setup (a zero-stride batched view) then per-problem
        values through update give the same result as a batched setup."""
        B, n = 3, 4
        P = _make_spd_csr(n, seed=93)
        rng = np.random.default_rng(94)
        P_vals = cp.asarray(1.0 + rng.random((B, 1))) * P.data[None, :]
        c = cp.asarray(rng.standard_normal((B, n)))

        direct = _solver()
        direct.setup(P=_triple(P, P_vals), c=c)
        direct.solve()

        staged = _solver()
        staged.setup(P=_triple(P), c=cp.broadcast_to(c[0], (B, n)))
        staged.update(P=P_vals, c=c)
        staged.solve()
        np.testing.assert_allclose(cp.asnumpy(staged.result.x), cp.asnumpy(direct.result.x), atol=1e-10)

    def test_int64_indices(self):
        n = 5
        P = _make_spd_csr(n, seed=95)
        s = _solver()
        s.setup(P=(P.indptr.astype(cp.int64), P.indices.astype(cp.int64), P.data), c=cp.ones(n))
        s.solve()

    @pytest.mark.skipif(not _TORCH_CUDA, reason="torch+CUDA required")
    def test_torch_csr_tensor_triple(self):
        n = 5
        P = _make_spd_csr(n, seed=96)
        t = torch.sparse_csr_tensor(
            torch.from_dlpack(P.indptr), torch.from_dlpack(P.indices), torch.from_dlpack(P.data),
            size=P.shape,
        )
        s = _solver()
        s.setup(P=(t.crow_indices(), t.col_indices(), t.values()), c=cp.ones(n))
        s.solve()
        ref = _solver()
        ref.setup(P=_triple(P), c=cp.ones(n))
        ref.solve()
        np.testing.assert_allclose(cp.asnumpy(s.result.x), cp.asnumpy(ref.result.x), atol=1e-12)


class TestSolverSetupRejects:

    def test_csr_object(self):
        n = 4
        P = _make_spd_csr(n, seed=100)
        with pytest.raises(TypeError, match="CSR triple"):
            _solver().setup(P=P, c=cp.zeros(n))

    def test_host_arrays(self):
        n = 4
        P = _make_spd_csr(n, seed=101)
        with pytest.raises(TypeError, match="GPU array"):
            _solver().setup(P=(P.indptr.get(), P.indices.get(), P.data.get()), c=cp.zeros(n))

    def test_float_indices(self):
        n = 4
        P = _make_spd_csr(n, seed=102)
        with pytest.raises(TypeError, match="int32 or int64"):
            _solver().setup(P=(P.indptr.astype(cp.float64), P.indices, P.data), c=cp.zeros(n))

    def test_unsorted_indices(self):
        indptr = cp.asarray([0, 2, 3], dtype=cp.int32)
        indices = cp.asarray([1, 0, 1], dtype=cp.int32)
        with pytest.raises(ValueError, match="strictly increasing"):
            _solver().setup(P=(indptr, indices, cp.ones(3)), c=cp.zeros(2))

    def test_bad_indptr(self):
        n = 4
        P = _make_spd_csr(n, seed=103)
        bad = P.indptr.copy()
        bad[-1] += 1
        with pytest.raises(ValueError, match="indptr"):
            _solver().setup(P=(bad, P.indices, P.data), c=cp.zeros(n))

    def test_column_out_of_range(self):
        n = 3
        P = _make_spd_csr(n, seed=104)
        A = (cp.asarray([0, 1], dtype=cp.int32), cp.asarray([n], dtype=cp.int32), cp.ones(1))
        with pytest.raises(ValueError, match="column indices"):
            _solver().setup(P=_triple(P), c=cp.zeros(n), A=A, b=cp.zeros(1))

    def test_values_shape(self):
        n = 4
        P = _make_spd_csr(n, seed=105)
        with pytest.raises(ValueError, match="values must have shape"):
            _solver().setup(P=_triple(P, cp.ones(P.nnz + 1)), c=cp.zeros(n))

    def test_batch_mismatch(self):
        n = 4
        P = _make_spd_csr(n, seed=106)
        with pytest.raises(ValueError, match="disagree on the batch size"):
            _solver().setup(P=_triple(P, cp.ones((2, P.nnz))), c=cp.zeros((3, n)))

    def test_update_rejects_sparse_matrix(self):
        """The pattern is fixed at setup: update takes values, not matrices."""
        n = 4
        P = _make_spd_csr(n, seed=107)
        s = _solver()
        s.setup(P=_triple(P), c=cp.zeros(n))
        with pytest.raises(TypeError, match="nonzero values"):
            s.update(P=P)
