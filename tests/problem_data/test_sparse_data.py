"""Tests for SparseData (CSR triples, the values-only matrix setters) and the
SparseSolver setup contract: CSR triples ``(indptr, indices, values)``, each
value array shared or batched, the batch size read from the batched arrays.
"""
import numpy as np
import pytest
import warp as wp
import scipy.sparse as sp_cpu

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
from cupiqp.sparse.sparse_data import SparseData
from cupiqp import SparseSolver
from tests.helpers import broadcast, csr_triple, quiet, to_wp


def _triple(M, values=None):
    """GPU CSR triple of a scipy csr_matrix, optionally with other values
    (a host array)."""
    indptr, indices, data = csr_triple(M)
    return (indptr, indices, data if values is None else to_wp(values))


def _sparse_data(**kw):
    """Build SparseData; matrices are given as scipy csr_matrix (or a GPU
    triple), vectors as host arrays. Data objects take Warp arrays only (the
    solver converts user arrays)."""
    d = SparseData()
    args = {}
    for k, v in kw.items():
        if k in ("P", "A", "G"):
            args[k] = _triple(v) if sp_cpu.issparse(v) else v
        else:
            args[k] = to_wp(v)
    d.init(**args)
    return d


def _make_spd_csr(n, density=0.4, seed=42):
    rng = np.random.default_rng(seed)
    M = sp_cpu.random(n, n, density=density, format='csr', random_state=rng)
    P = ((M @ M.T + n * sp_cpu.eye(n)) / 2).tocsr()
    P = (P + P.T).tocsr()
    P.sort_indices()
    return P


def _make_rect_csr(rows, cols, density=0.5, seed=42):
    rng = np.random.default_rng(seed)
    A = sp_cpu.random(rows, cols, density=density, format='csr', random_state=rng)
    A.sort_indices()
    return A


# ===========================================================================
# SparseData
# ===========================================================================

class TestSparseDataInit:

    def test_shared_values(self):
        n = 6
        P = _make_spd_csr(n, seed=1)
        data = _sparse_data(P=P, c=np.ones(n))
        assert isinstance(data.P, UniformBatchedCsrMatrix)
        assert data.batch_size == 1
        assert (data.n, data.p, data.m) == (n, 0, 0)
        assert data.P.shape == (1, n, n)
        np.testing.assert_allclose(data.P.data.numpy()[0], P.data)

    def test_batched_values(self):
        B, n = 4, 5
        P = _make_spd_csr(n, seed=10)
        vals = np.random.default_rng(11).standard_normal((B, P.nnz))
        data = _sparse_data(P=_triple(P, vals), c=np.zeros(n))
        assert data.batch_size == B
        assert data.P.shape == (B, n, n)
        np.testing.assert_allclose(data.P.data.numpy(), vals)

    def test_batch_size_from_a_vector(self):
        """Shared matrix values with a batched vector: B comes from the vector
        and the values are copied into every problem."""
        B, n = 3, 5
        P = _make_spd_csr(n, seed=12)
        data = _sparse_data(P=P, c=np.zeros((B, n)))
        assert data.batch_size == B
        np.testing.assert_allclose(data.P.data.numpy(), np.broadcast_to(P.data, (B, P.nnz)))

    def test_values_are_copied(self):
        n = 6
        P = _make_spd_csr(n, seed=20)
        indptr, indices, vals = _triple(P)
        data = _sparse_data(P=(indptr, indices, vals), c=np.ones(n))
        vals.zero_()
        assert not np.allclose(data.P.data.numpy(), 0.0)

    def test_all_blocks(self):
        B, n, p, m = 2, 5, 3, 4
        P = _make_spd_csr(n, seed=40)
        A = _make_rect_csr(p, n, seed=41)
        G = _make_rect_csr(m, n, seed=42)
        data = _sparse_data(
            P=P, c=np.zeros((B, n)),
            A=A, b=np.zeros((B, p)),
            G=G, h_u=np.ones((B, m)), h_l=-np.ones((B, m)),
        )
        assert data.batch_size == B
        assert (data.n, data.p, data.m) == (n, p, m)
        assert data.A.shape == (B, p, n) and data.G.shape == (B, m, n)

    def test_empty_A_and_G(self):
        """Omitting A and G yields empty batched CSR placeholders."""
        B, n = 2, 4
        P = _make_spd_csr(n, seed=50)
        data = _sparse_data(P=P, c=np.zeros((B, n)))
        assert data.p == 0 and data.m == 0
        assert data.A.shape == (B, 0, n) and data.G.shape == (B, 0, n)
        assert data.A.nnz == 0 and data.G.nnz == 0

    def test_batch_mismatch_raises(self):
        n = 4
        P = _make_spd_csr(n, seed=51)
        vals = np.ones((3, P.nnz))
        with pytest.raises(ValueError, match="disagree on the batch size"):
            _sparse_data(P=_triple(P, vals), c=np.zeros((2, n)))


class TestSetters:
    """``set_P`` / ``set_A`` / ``set_G`` take only the stored values, as a
    ``(B, nnz)`` array (or ``(nnz,)`` broadcast), in the setup CSR order."""

    def test_set_P_batched_values(self):
        B, n = 3, 5
        P = _make_spd_csr(n, seed=60)
        data = _sparse_data(P=P, c=np.zeros((B, n)))
        vals = np.random.default_rng(61).standard_normal((B, P.nnz))
        data.set_P(to_wp(vals))
        np.testing.assert_allclose(data.P.data.numpy(), vals)

    def test_set_P_broadcast_values(self):
        B, n = 3, 5
        P = _make_spd_csr(n, seed=70)
        data = _sparse_data(P=P, c=np.zeros((B, n)))
        data.set_P(to_wp(np.ones(P.nnz)))
        np.testing.assert_allclose(data.P.data.numpy(), np.ones((B, P.nnz)))

    def test_set_P_keeps_buffer_address(self):
        n = 5
        P = _make_spd_csr(n, seed=71)
        data = _sparse_data(P=P, c=np.zeros(n))
        ptr = data.P.data.ptr
        data.set_P(to_wp(np.ones(P.nnz)))
        assert data.P.data.ptr == ptr

    def test_set_P_rejects_wrong_nnz(self):
        n = 5
        P = _make_spd_csr(n, seed=80)
        data = _sparse_data(P=P, c=np.zeros(n))
        with pytest.raises(ValueError, match="shape mismatch"):
            data.set_P(to_wp(np.ones(P.nnz + 1)))

    def test_set_A_and_G_values(self):
        n = 4
        P = _make_spd_csr(n, seed=83)
        A = sp_cpu.csr_matrix(np.array([[1.0, 0.0, 2.0, 0.0]]))
        G = sp_cpu.csr_matrix(np.eye(n))
        data = _sparse_data(P=P, c=np.zeros(n), A=A, b=np.zeros(1),
                            G=G, h_u=np.ones(n))
        data.set_A(to_wp([3.0, 4.0]))
        data.set_G(to_wp(2.0 * np.ones(n)))
        np.testing.assert_allclose(data.A.data.numpy(), [[3.0, 4.0]])
        np.testing.assert_allclose(data.G.data.numpy(), 2.0 * np.ones((1, n)))

    def test_inplace_mutation_visible_in_data(self):
        """``data.P.data`` is the stored values buffer: in-place changes are seen."""
        B, n = 2, 4
        P = _make_spd_csr(n, seed=90)
        data = _sparse_data(P=P, c=np.zeros((B, n)))
        wp.copy(data.P.data, to_wp(3.0 * data.P.data.numpy()))
        np.testing.assert_allclose(data.P.data.numpy()[0], 3.0 * P.data)


# ===========================================================================
# SparseSolver.setup
# ===========================================================================

def _solver():
    return quiet(SparseSolver())


class TestSolverSetup:

    def test_all_shared_is_one_problem(self):
        n = 4
        P = _make_spd_csr(n, seed=90)
        s = _solver()
        s.setup(P=_triple(P), c=to_wp(np.arange(n, dtype=np.float64)))
        assert s.data.batch_size == 1
        s.solve()

    def test_batched_setup_matches_individual_solves(self):
        """Per-problem values given at setup solve each problem."""
        B, n = 3, 4
        P = _make_spd_csr(n, seed=91)
        rng = np.random.default_rng(92)
        P_vals = (1.0 + rng.random((B, 1))) * P.data[None, :]
        c = rng.standard_normal((B, n))

        s = _solver()
        s.setup(P=_triple(P, P_vals), c=to_wp(c))
        assert s.data.batch_size == B
        s.solve()
        for k in range(B):
            ref = _solver()
            ref.setup(P=_triple(P, P_vals[k]), c=to_wp(c[k]))
            ref.solve()
            np.testing.assert_allclose(
                s.result.x.numpy()[k], ref.result.x.numpy()[0], atol=1e-8,
            )

    def test_broadcast_setup_then_update_matches_batched_setup(self):
        """Equal problems at setup (a zero-stride batched view) then per-problem
        values through update give the same result as a batched setup."""
        B, n = 3, 4
        P = _make_spd_csr(n, seed=93)
        rng = np.random.default_rng(94)
        P_vals = (1.0 + rng.random((B, 1))) * P.data[None, :]
        c = rng.standard_normal((B, n))

        direct = _solver()
        direct.setup(P=_triple(P, P_vals), c=to_wp(c))
        direct.solve()

        staged = _solver()
        staged.setup(P=_triple(P), c=broadcast(to_wp(c[0]), B))
        staged.update(P=to_wp(P_vals), c=to_wp(c))
        staged.solve()
        np.testing.assert_allclose(staged.result.x.numpy(), direct.result.x.numpy(), atol=1e-10)

    def test_int64_indices(self):
        n = 5
        P = _make_spd_csr(n, seed=95)
        s = _solver()
        s.setup(P=(to_wp(P.indptr, np.int64), to_wp(P.indices, np.int64), to_wp(P.data)),
                c=to_wp(np.ones(n)))
        s.solve()

    def test_torch_csr_tensor_triple(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("torch+CUDA required")
        n = 5
        P = _make_spd_csr(n, seed=96)
        t = torch.sparse_csr_tensor(
            torch.from_numpy(P.indptr.astype(np.int32)).cuda(),
            torch.from_numpy(P.indices.astype(np.int32)).cuda(),
            torch.from_numpy(P.data).cuda(),
            size=P.shape,
        )
        s = _solver()
        s.setup(P=(t.crow_indices(), t.col_indices(), t.values()), c=to_wp(np.ones(n)))
        s.solve()
        ref = _solver()
        ref.setup(P=_triple(P), c=to_wp(np.ones(n)))
        ref.solve()
        np.testing.assert_allclose(s.result.x.numpy(), ref.result.x.numpy(), atol=1e-12)


class TestSolverSetupRejects:

    def test_csr_object(self):
        n = 4
        P = _make_spd_csr(n, seed=100)
        with pytest.raises(TypeError, match="CSR triple"):
            _solver().setup(P=P, c=to_wp(np.zeros(n)))

    def test_host_arrays(self):
        n = 4
        P = _make_spd_csr(n, seed=101)
        with pytest.raises(TypeError, match="GPU array"):
            _solver().setup(P=(P.indptr, P.indices, P.data), c=to_wp(np.zeros(n)))

    def test_float_indices(self):
        n = 4
        P = _make_spd_csr(n, seed=102)
        indptr, indices, values = _triple(P)
        with pytest.raises(TypeError, match="int32 or int64"):
            _solver().setup(P=(to_wp(P.indptr, np.float64), indices, values), c=to_wp(np.zeros(n)))

    def test_unsorted_indices(self):
        indptr = to_wp([0, 2, 3], np.int32)
        indices = to_wp([1, 0, 1], np.int32)
        with pytest.raises(ValueError, match="strictly increasing"):
            _solver().setup(P=(indptr, indices, to_wp(np.ones(3))), c=to_wp(np.zeros(2)))

    def test_bad_indptr(self):
        n = 4
        P = _make_spd_csr(n, seed=103)
        _, indices, values = _triple(P)
        bad = P.indptr.copy()
        bad[-1] += 1
        with pytest.raises(ValueError, match="indptr"):
            _solver().setup(P=(to_wp(bad, np.int32), indices, values), c=to_wp(np.zeros(n)))

    def test_column_out_of_range(self):
        n = 3
        P = _make_spd_csr(n, seed=104)
        A = (to_wp([0, 1], np.int32), to_wp([n], np.int32), to_wp(np.ones(1)))
        with pytest.raises(ValueError, match="column indices"):
            _solver().setup(P=_triple(P), c=to_wp(np.zeros(n)), A=A, b=to_wp(np.zeros(1)))

    def test_values_shape(self):
        n = 4
        P = _make_spd_csr(n, seed=105)
        with pytest.raises(ValueError, match="values must have shape"):
            _solver().setup(P=_triple(P, np.ones(P.nnz + 1)), c=to_wp(np.zeros(n)))

    def test_batch_mismatch(self):
        n = 4
        P = _make_spd_csr(n, seed=106)
        with pytest.raises(ValueError, match="disagree on the batch size"):
            _solver().setup(P=_triple(P, np.ones((2, P.nnz))), c=to_wp(np.zeros((3, n))))

    def test_update_rejects_sparse_matrix(self):
        """The pattern is fixed at setup: update takes values, not matrices."""
        n = 4
        P = _make_spd_csr(n, seed=107)
        s = _solver()
        s.setup(P=_triple(P), c=to_wp(np.zeros(n)))
        with pytest.raises(TypeError, match="nonzero values"):
            s.update(P=P)
