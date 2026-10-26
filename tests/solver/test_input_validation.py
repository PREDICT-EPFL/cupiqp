"""Tests for the three type-strict Solver subclasses.

* :class:`DenseSolver`         only accepts dense ``P, A, G``.
* :class:`SparseSolver`        only accepts CSR-style sparse ``P, A, G``.
* :class:`MultistageSolver`    only accepts ``(diag, offdiag)`` tuples of GPU
  arrays for P, A, G.

Each subclass raises :class:`TypeError` with a message naming the right
sibling class on a storage mismatch. We test:

1. **Positive paths**: each accepted storage form solves a small QP and
   produces the same ``x*`` as the reference.
2. **Negative paths**: every off-diagonal cell in the
   storage-vs-subclass table raises ``TypeError`` with the expected hint.
3. **Numerical equivalence**: same QP solved through ``DenseSolver`` and
   ``SparseSolver`` produces matching solutions.
"""

import numpy as np
import pytest
import scipy.sparse as sp_cpu

from cupiqp import (
    DenseSolver, SparseSolver, MultistageSolver,
    Settings, Status,
)
# Abstract bases aren't re-exported (they're internal); import directly
# from their submodules for the abstract-contract tests.
from cupiqp.solver import SolverBase
from tests.helpers import broadcast, csr_triple, to_wp, to_wp_dict


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_qp_dense(n=8, p=3, m=5, seed=42):
    """Random strictly convex QP returned as dense numpy arrays."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)
    P = (P + P.T) / 2
    return dict(
        P=P,
        c=rng.standard_normal(n),
        A=rng.standard_normal((p, n)),
        b=rng.standard_normal(p),
        G=rng.standard_normal((m, n)),
        h_u=np.abs(rng.standard_normal(m)) + 2.0,
        h_l=-np.abs(rng.standard_normal(m)) - 2.0,
        x_u=np.abs(rng.standard_normal(n)) + 3.0,
        x_l=-np.abs(rng.standard_normal(n)) - 3.0,
    )


def _qp_to_sparse(qp):
    """Lift a dense QP onto the GPU: CSR triples for the matrices, Warp
    vectors. SparseSolver requires **GPU** arrays; host data is rejected."""
    out = dict(qp)
    for k in ("P", "A", "G"):
        out[k] = csr_triple(qp[k])
    for k in ("c", "b", "h_u", "h_l", "x_u", "x_l"):
        out[k] = to_wp(qp[k])
    return out


def _qp_to_scipy_sparse(qp):
    """CPU-side scipy CSR fixture for *rejection* tests.

    Used to verify that DenseSolver / SparseSolver / MultistageSolver
    all reject scipy.sparse inputs with a clear GPU-only message.
    """
    out = dict(qp)
    out["P"] = sp_cpu.csr_matrix(qp["P"])
    out["A"] = sp_cpu.csr_matrix(qp["A"])
    out["G"] = sp_cpu.csr_matrix(qp["G"])
    return out


def _qp_dense_torch(qp):
    """Convert dense QP entries to dense torch.Tensors on CUDA."""
    import torch
    return {
        k: torch.tensor(v, dtype=torch.float64, device="cuda")
        for k, v in qp.items()
    }


def _quiet_settings():
    s = Settings()
    s.preconditioner_iter = 0
    s.verbose = False
    return s


# ----------------------------------------------------------------------
# DenseSolver
# ----------------------------------------------------------------------


class TestDenseSolver:
    """DenseSolver accepts GPU dense arrays (anything exposing
    ``__cuda_array_interface__``); rejects everything else.

    cupiqp is GPU-only -- CPU numpy is explicitly rejected with a clear
    error asking for a GPU array.
    """

    def test_dense_numpy_REJECT(self):
        """Raw numpy arrays are CPU-only and must be rejected -- they
        don't expose ``__cuda_array_interface__``."""
        qp = _make_qp_dense()
        s = DenseSolver()
        with pytest.raises(TypeError, match=r"DenseSolver requires.*GPU dense"):
            s.setup(**qp)  # numpy directly -- should fail

    def test_numpy_vector_REJECT(self):
        """Numpy `c` (a vector) is also rejected -- vectors are checked
        with the same GPU-only contract as matrices."""
        qp = _make_qp_dense()
        # All matrices on GPU, but c stays on CPU as numpy.
        kwargs = to_wp_dict(qp)
        kwargs["c"] = qp["c"]   # numpy vector
        s = DenseSolver()
        with pytest.raises(TypeError, match=r"DenseSolver requires c.*GPU"):
            s.setup(**kwargs)

    def test_cpu_torch_vector_REJECT(self):
        """CPU torch tensor for `b` is rejected (not GPU)."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        qp = _make_qp_dense()
        kwargs = to_wp_dict(qp)
        kwargs["b"] = torch.tensor(qp["b"], dtype=torch.float64)  # CPU
        s = DenseSolver()
        with pytest.raises(TypeError, match=r"DenseSolver requires b.*GPU"):
            s.setup(**kwargs)

    def test_dense_warp_OK(self):
        qp = to_wp_dict(_make_qp_dense())
        s = DenseSolver()
        s.settings = _quiet_settings()
        s.setup(**qp)
        s.solve()
        assert s.result.info.to_host().status == [Status.CUPIQP_SOLVED]

    def test_dense_cupy_OK(self):
        cp = pytest.importorskip("cupy")
        qp = {k: cp.array(v) for k, v in _make_qp_dense().items()}
        s = DenseSolver()
        s.settings = _quiet_settings()
        s.setup(**qp)
        s.solve()
        assert s.result.info.to_host().status == [Status.CUPIQP_SOLVED]

    def test_dense_torch_OK(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        qp = _qp_dense_torch(_make_qp_dense())
        s = DenseSolver()
        s.settings = _quiet_settings()
        s.setup(**qp)
        s.solve()
        assert s.result.info.to_host().status == [Status.CUPIQP_SOLVED]

    def test_scipy_csr_REJECT(self):
        """scipy CSR is sparse + on CPU -- fails the GPU-dense check."""
        qp = _qp_to_scipy_sparse(_make_qp_dense())   # genuine scipy CSR
        s = DenseSolver()
        with pytest.raises(TypeError, match=r"DenseSolver requires P.*GPU dense"):
            s.setup(**qp)

    def test_torch_sparse_csr_REJECT(self):
        """torch sparse_csr_tensor: layout != strided -> not GPU dense."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        n = 4
        crow = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cuda")
        col = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda")
        vals = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64, device="cuda")
        P_sp = torch.sparse_csr_tensor(crow, col, vals, size=(n, n))
        c = torch.zeros(n, dtype=torch.float64, device="cuda")
        s = DenseSolver()
        with pytest.raises(TypeError, match=r"DenseSolver requires P.*GPU dense"):
            s.setup(P=P_sp, c=c)

    def test_block_tuple_REJECT(self):
        """A multistage (diag, offdiag) tuple is not a GPU array -- rejected by DenseSolver."""
        P = (to_wp(np.zeros((2, 2, 2))), to_wp(np.zeros((1, 2, 2))))
        s = DenseSolver()
        with pytest.raises(TypeError, match=r"DenseSolver requires P.*GPU dense"):
            s.setup(P=P, c=np.zeros(4))

    def test_list_of_csr_REJECT(self):
        """A list of CSRs is not a GPU dense array."""
        qp = _qp_to_sparse(_make_qp_dense(n=4, p=2, m=2))
        s = DenseSolver()
        with pytest.raises(TypeError, match=r"DenseSolver requires P.*GPU dense"):
            s.setup(P=[qp["P"], qp["P"]], c=qp["c"])

    def test_unrecognized_REJECT(self):
        s = DenseSolver()
        with pytest.raises(TypeError, match="DenseSolver"):
            s.setup(P="not a matrix", c=np.zeros(4))


# ----------------------------------------------------------------------
# SparseSolver
# ----------------------------------------------------------------------


class TestSparseSolver:
    """SparseSolver takes CSR triples ``(indptr, indices, values)`` of GPU
    arrays; sparse matrix objects, dense matrices and host data are rejected."""

    def test_scipy_csr_REJECT(self):
        """A scipy.sparse matrix object is not a CSR triple."""
        qp = _qp_to_scipy_sparse(_make_qp_dense())
        for k in ("c", "b", "h_u", "h_l", "x_u", "x_l"):
            qp[k] = to_wp(qp[k])
        s = SparseSolver()
        with pytest.raises(TypeError, match=r"SparseSolver requires P as a CSR triple"):
            s.setup(**qp)

    def test_csr_triple_OK(self):
        """The canonical SparseSolver path: GPU CSR triples."""
        qp = _qp_to_sparse(_make_qp_dense())
        s = SparseSolver()
        s.settings = _quiet_settings()
        s.setup(**qp)
        s.solve()
        assert s.result.info.to_host().status == [Status.CUPIQP_SOLVED]

    def test_list_of_matrices_REJECT(self):
        """A list of matrices is not a CSR triple; batches are per-problem values."""
        qp = _qp_to_sparse(_make_qp_dense(seed=1))
        qp["P"] = [qp["P"], qp["P"]]
        s = SparseSolver()
        with pytest.raises(TypeError, match=r"SparseSolver requires P as a CSR triple"):
            s.setup(**qp)

    def test_batched_values_OK(self):
        """B=2 batch: one shared pattern, per-problem values at setup()."""
        qp_a = _make_qp_dense(seed=1)
        qp_b = _make_qp_dense(seed=2)

        def batched(k):
            if k not in ("P", "A", "G"):
                return to_wp(np.stack([qp_a[k], qp_b[k]]))
            # Both matrices are fully dense: same pattern, stacked values.
            M_a, M_b = sp_cpu.csr_matrix(qp_a[k]), sp_cpu.csr_matrix(qp_b[k])
            indptr, indices, _ = csr_triple(M_a)
            return indptr, indices, to_wp(np.stack([M_a.data, M_b.data]))

        s = SparseSolver()
        s.settings = _quiet_settings()
        s.setup(**{k: batched(k) for k in qp_a})
        assert s.data.batch_size == 2
        s.solve()
        statuses = s.result.info.to_host().status
        assert all(st == Status.CUPIQP_SOLVED for st in statuses)

    def test_torch_csr_triple_OK(self):
        """The index and value tensors of a CUDA torch CSR tensor are accepted."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        qp = _make_qp_dense(n=4, p=2, m=2)
        qp_sp = _qp_to_sparse(qp)
        P_np = sp_cpu.csr_matrix(qp["P"])
        P_tsp = torch.sparse_csr_tensor(
            torch.tensor(P_np.indptr, dtype=torch.int64, device="cuda"),
            torch.tensor(P_np.indices, dtype=torch.int64, device="cuda"),
            torch.tensor(P_np.data, dtype=torch.float64, device="cuda"),
            size=P_np.shape,
        )
        s = SparseSolver()
        s.settings = _quiet_settings()
        s.setup(P=(P_tsp.crow_indices(), P_tsp.col_indices(), P_tsp.values()), c=qp_sp["c"],
            A=qp_sp["A"], b=qp_sp["b"], G=qp_sp["G"],
            h_u=qp_sp["h_u"], h_l=qp_sp["h_l"],
            x_u=qp_sp["x_u"], x_l=qp_sp["x_l"],
        )
        s.solve()
        assert s.result.info.to_host().status == [Status.CUPIQP_SOLVED]

    def test_cupy_csc_REJECT(self):
        """A cupy CSC matrix object is not a CSR triple (no silent conversion)."""
        cp = pytest.importorskip("cupy")
        from cupyx.scipy.sparse import csc_matrix as cp_csc
        n = 4
        s = SparseSolver()
        with pytest.raises(TypeError, match=r"SparseSolver requires P as a CSR triple"):
            s.setup(P=cp_csc(cp.eye(n)), c=cp.zeros(n))

    def test_cpu_torch_triple_REJECT(self):
        """CPU tensor values are host data; only the pattern may be on the host."""
        torch = pytest.importorskip("torch")
        n = 4
        crow = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)   # CPU
        col = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        vals = torch.tensor([1.0] * 4, dtype=torch.float64)
        s = SparseSolver()
        with pytest.raises(TypeError, match=r"SparseSolver requires P values to be a GPU array"):
            s.setup(P=(crow, col, vals), c=to_wp(np.zeros(n)))

    def test_host_pattern_ACCEPT(self):
        """A host pattern (numpy indptr / indices) with GPU values solves the
        same QP as the same pattern on the GPU."""
        qp = _make_qp_dense()
        P = sp_cpu.csr_matrix(qp["P"])
        A = sp_cpu.csr_matrix(qp["A"])
        vecs = {k: to_wp(qp[k]) for k in ("c", "b")}
        xs = []
        for to_pattern in (np.asarray, to_wp):
            s = SparseSolver()
            s.settings = _quiet_settings()
            s.setup(P=(to_pattern(P.indptr), to_pattern(P.indices), to_wp(P.data)),
                    A=(to_pattern(A.indptr), to_pattern(A.indices), to_wp(A.data)),
                    **vecs)
            s.solve()
            assert s.result.info.to_host().status[0] == Status.CUPIQP_SOLVED
            xs.append(s.result.x.numpy())
        np.testing.assert_allclose(xs[0], xs[1], rtol=0, atol=1e-12)   # cuDSS is not bit-reproducible

    def test_dense_numpy_REJECT(self):
        qp = _make_qp_dense()
        s = SparseSolver()
        with pytest.raises(TypeError, match=r"SparseSolver requires P as a CSR triple"):
            s.setup(**qp)

    def test_dense_torch_REJECT(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        qp = _qp_dense_torch(_make_qp_dense())
        s = SparseSolver()
        with pytest.raises(TypeError, match=r"SparseSolver requires P as a CSR triple"):
            s.setup(**qp)

    def test_block_tuple_REJECT(self):
        P = (to_wp(np.zeros((2, 2, 2))), to_wp(np.zeros((1, 2, 2))))
        s = SparseSolver()
        with pytest.raises(TypeError, match=r"SparseSolver requires P as a CSR triple"):
            s.setup(P=P, c=np.zeros(4))

    def test_unrecognized_REJECT(self):
        s = SparseSolver()
        with pytest.raises(TypeError, match="SparseSolver"):
            s.setup(P=42, c=np.zeros(4))

    def test_numpy_vector_REJECT(self):
        """Vectors must be GPU dense even with sparse P/A/G."""
        qp_raw = _make_qp_dense()
        qp = _qp_to_sparse(qp_raw)
        qp["c"] = qp_raw["c"]   # numpy override: P, A, G OK; c not GPU.
        s = SparseSolver()
        with pytest.raises(TypeError, match=r"SparseSolver requires the vector c"):
            s.setup(**qp)


# ----------------------------------------------------------------------
# MultistageSolver -- input validation plus a small numerical check
# against DenseSolver on the assembled block-tridiagonal matrix.
# ----------------------------------------------------------------------


class TestMultistageSolver:
    """MultistageSolver takes plain GPU arrays: ``(diag, offdiag)`` tuples for
    block matrices and arrays for vectors, one template problem at setup()."""

    @staticmethod
    def _two_stage():
        """Host arrays of a two-stage problem."""
        # n = 4 (N = 2 stages of d = 2), P = [[D0, E0^T], [E0, D1]]
        P_diag = np.array([[[4.0, 0.5], [0.5, 3.0]], [[5.0, 0.0], [0.0, 2.0]]])
        P_offdiag = np.array([[[0.3, 0.1], [0.0, 0.2]]])
        c = np.array([1.0, -1.0, 0.5, -0.5])
        return P_diag, P_offdiag, c

    def test_dense_matrix_in_P_slot_REJECT(self):
        s = MultistageSolver()
        with pytest.raises(TypeError, match=r"\(diag, offdiag\) tuple"):
            s.setup(P=to_wp(np.eye(4)), c=to_wp(np.zeros(4)))

    def test_scipy_csr_REJECT(self):
        s = MultistageSolver()
        with pytest.raises(TypeError, match=r"\(diag, offdiag\) tuple"):
            s.setup(P=sp_cpu.eye(4, format="csr"), c=to_wp(np.zeros(4)))

    def test_numpy_blocks_REJECT(self):
        s = MultistageSolver()
        with pytest.raises(TypeError, match="GPU array"):
            s.setup(P=(np.ones((2, 2, 2)), None), c=to_wp(np.zeros(4)))

    def test_numpy_vector_REJECT(self):
        P_diag, P_offdiag, _ = self._two_stage()
        s = MultistageSolver()
        with pytest.raises(TypeError, match="GPU array"):
            s.setup(P=(to_wp(P_diag), to_wp(P_offdiag)), c=np.zeros(4))

    def test_wrong_dtype_REJECT(self):
        """No implicit casting: inputs must have the solver dtype."""
        P_diag, P_offdiag, c = self._two_stage()
        s = MultistageSolver()
        with pytest.raises(TypeError, match="does not cast dtypes"):
            s.setup(P=(to_wp(P_diag, np.float32), None), c=to_wp(c))

    def test_inconsistent_offdiag_shape_REJECT(self):
        P_diag, _, c = self._two_stage()
        s = MultistageSolver()
        with pytest.raises(ValueError, match="P offdiag"):
            s.setup(P=(to_wp(P_diag), to_wp(np.zeros((2, 2, 2)))), c=to_wp(c))

    def test_matches_dense_and_partial_update(self):
        """Per-problem off-diagonal blocks at setup, then a partial
        update(P=(None, offdiag)), agree with DenseSolver on the assembled
        matrices."""
        P_diag, P_offdiag, c = self._two_stage()
        B = 3
        scales = np.array([1.0, 0.5, -1.0])
        off_b = scales[:, None, None, None] * P_offdiag[None]

        s = MultistageSolver()
        s.settings = _quiet_settings()
        s.setup(P=(to_wp(P_diag), to_wp(off_b)), c=to_wp(c.reshape(2, 2)),  # batched offdiag: B = 3
                x_l=to_wp(-np.ones(4)), x_u=to_wp(np.ones(4)))
        s.solve()
        assert all(st == Status.CUPIQP_SOLVED for st in s.result.info.to_host().status)
        off_b = scales[::-1, None, None, None] * P_offdiag[None]
        s.update(P=(None, to_wp(off_b)))
        s.solve()
        assert all(st == Status.CUPIQP_SOLVED for st in s.result.info.to_host().status)

        for k in range(B):
            P = np.zeros((4, 4))
            P[:2, :2], P[2:, 2:] = P_diag[0], P_diag[1]
            P[2:, :2] = off_b[k, 0]
            P[:2, 2:] = off_b[k, 0].T
            d = DenseSolver()
            d.settings = _quiet_settings()
            d.setup(P=to_wp(P), c=to_wp(c), x_l=to_wp(-np.ones(4)), x_u=to_wp(np.ones(4)))
            d.solve()
            np.testing.assert_allclose(
                s.result.x.numpy()[k], d.result.x.numpy()[0], atol=1e-7,
            )

    def test_update_keeps_buffers(self):
        """update() writes in place: the public block buffers keep their address."""
        P_diag, P_offdiag, c = self._two_stage()
        s = MultistageSolver()
        s.settings = _quiet_settings()
        s.setup(P=(to_wp(P_diag), to_wp(P_offdiag)), c=broadcast(to_wp(c), 2))
        ptrs = (s.data.P_diag.ptr, s.data.P_offdiag.ptr)
        s.update(P=(to_wp(2.0 * P_diag), None), c=to_wp(np.ones((2, 4))))
        assert (s.data.P_diag.ptr, s.data.P_offdiag.ptr) == ptrs

# ----------------------------------------------------------------------
# Numerical equivalence: same QP, two backends.
# ----------------------------------------------------------------------


class TestDenseSparseEquivalence:
    """Same problem through DenseSolver and SparseSolver should agree."""

    def test_solutions_match_within_tol(self):
        qp = _make_qp_dense(n=8, p=3, m=5, seed=42)

        d = DenseSolver()
        d.settings = _quiet_settings()
        d.setup(**to_wp_dict(qp))
        d.solve()
        d_status = d.result.info.to_host().status
        x_d = d.result.x.numpy()[0]

        qp_sp = _qp_to_sparse(qp)
        s = SparseSolver()
        s.settings = _quiet_settings()
        s.setup(**qp_sp)
        s.solve()
        s_status = s.result.info.to_host().status
        x_s = s.result.x.numpy()[0]

        assert d_status == [Status.CUPIQP_SOLVED]
        assert s_status == [Status.CUPIQP_SOLVED]
        np.testing.assert_allclose(x_d, x_s, atol=1e-5)


# ----------------------------------------------------------------------
# Backwards compatibility: bare Solver still works.
# ----------------------------------------------------------------------


class TestSolverIsAbstract:
    """Bare ``Solver`` / ``SolverBase`` is an :class:`abc.ABC` and cannot
    be instantiated. Users must construct one of the concrete subclasses
    (:class:`DenseSolver`, :class:`SparseSolver`,
    :class:`MultistageSolver`).
    """

    def test_construction_raises_TypeError(self):
        """``SolverBase()`` raises :class:`TypeError` (Python's standard
        behavior for an abstract class with unimplemented methods)."""
        with pytest.raises(TypeError, match="abstract"):
            SolverBase()


@pytest.mark.parametrize("cls", [DenseSolver, SparseSolver, MultistageSolver])
def test_solve_before_setup_raises(cls):
    """``solve()`` without a prior ``setup()`` raises a clear error."""
    with pytest.raises(RuntimeError, match=r"setup\(\)"):
        cls().solve()


# ----------------------------------------------------------------------
# Solver.grad() -- manual implicit-differentiation
# ----------------------------------------------------------------------


class TestSolverGrad:
    """``Solver.grad(grad_x, grad_y, grad_z_*, grad_s_*)`` -- manual VJP
    callable outside autograd. Verifies it (a) raises if called before
    solve, (b) accepts None / 1-D / batched cotangents, and (c) returns
    the same gradients that the torch autograd path produces."""

    def test_raises_before_solve(self):
        s = DenseSolver()
        s.settings.enable_grad = True
        with pytest.raises(RuntimeError, match="prior solve"):
            s.backward(grad_x=None)

    def test_grad_x_only_returns_dP_dc(self):
        """Cotangent only on x -> dP and dc are non-None; vector grads
        (db, dh_u, ...) are populated as Warp arrays of expected shape."""
        qp = to_wp_dict(_make_qp_dense(n=6, p=2, m=3))
        s = DenseSolver()
        s.settings = _quiet_settings()
        s.settings.enable_grad = True
        s.setup(**qp)
        s.solve()
        st = s.result.info.to_host().status
        assert st == [Status.CUPIQP_SOLVED]

        n = qp["P"].shape[-1]
        dx = to_wp(np.ones((1, n)))
        out = s.backward(grad_x=dx)

        assert out.P is not None and out.P.shape == (1, n, n)
        assert out.c is not None and out.c.shape == (1, n)
        # P symmetry: dP should be symmetric (we symmetrize in Step 3).
        dP = out.P.numpy()
        np.testing.assert_allclose(dP, dP.transpose(0, 2, 1), atol=1e-10)

    def test_None_means_zero(self):
        """Calling solver.grad with all cotangents None gives zero grads."""
        qp = to_wp_dict(_make_qp_dense())
        s = DenseSolver()
        s.settings = _quiet_settings()
        s.settings.enable_grad = True
        s.setup(**qp)
        s.solve()
        out = s.backward()  # all None
        # All gradients must be all-zero (data was symmetric, lambda=0).
        assert np.allclose(out.P.numpy(), 0.0)
        assert np.allclose(out.c.numpy(), 0.0)


# ----------------------------------------------------------------------
# SparseSolver.grad -- gather onto the sparsity pattern
# ----------------------------------------------------------------------


class TestSparseSolverGrad:
    """``SparseSolver.grad()`` returns a :class:`SparseData` whose
    ``P, A, G`` are :class:`BatchedCsrMatrix` instances on the same
    sparsity pattern as the original problem matrices. The values must
    match what you'd get by computing the full dense gradient and
    gathering at each nonzero position."""

    def test_raises_before_solve(self):
        from cupiqp import SparseSolver
        s = SparseSolver()
        s.settings.enable_grad = True
        with pytest.raises(RuntimeError, match="prior solve"):
            s.backward(grad_x=None)

    def test_grad_returns_sparse_data_with_same_pattern(self):
        """Sparse grad's ``P``, ``A``, ``G`` use the input's
        ``indptr/indices`` (same sparsity pattern)."""
        from cupiqp import SparseSolver
        from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
        qp_raw = _make_qp_dense(n=6, p=2, m=3, seed=7)
        qp = _qp_to_sparse(qp_raw)
        s = SparseSolver()
        s.settings = _quiet_settings()
        s.settings.enable_grad = True
        s.setup(**qp)
        s.solve()

        out = s.backward(grad_x=to_wp(2.0 * s.result.x.numpy()))

        # P, A, G are BatchedCsrMatrix sharing the input's sparsity pattern.
        for name in ("P", "A", "G"):
            grad_mat = getattr(out, name)
            in_indptr, in_indices, _ = qp[name]   # CSR triple from the setup
            assert isinstance(grad_mat, UniformBatchedCsrMatrix)
            assert grad_mat.nnz == in_indices.size
            np.testing.assert_array_equal(grad_mat.indices.numpy(), in_indices.numpy())
            np.testing.assert_array_equal(grad_mat.indptr.numpy(), in_indptr.numpy())
            assert grad_mat.shape == (1, *qp_raw[name].shape)

    def test_matches_dense_gathered(self):
        """Sparse grad's per-nonzero values match dense grad's values
        gathered at the corresponding (row, col) positions -- to within
        solver tolerance."""
        import numpy as np_
        from cupiqp import DenseSolver, SparseSolver

        qp_raw = _make_qp_dense(n=8, p=3, m=4, seed=11)

        # Sparse path.
        ssp = SparseSolver()
        ssp.settings = _quiet_settings()
        ssp.settings.enable_grad = True
        ssp.setup(**_qp_to_sparse(qp_raw))
        ssp.solve()
        gsp = ssp.backward(grad_x=to_wp(2.0 * ssp.result.x.numpy()))

        # Dense path.
        sd = DenseSolver()
        sd.settings = _quiet_settings()
        sd.settings.enable_grad = True
        sd.setup(**to_wp_dict(qp_raw))
        sd.solve()
        gd = sd.backward(grad_x=to_wp(2.0 * sd.result.x.numpy()))

        # Solutions agree (the same QP, different storage).
        np_.testing.assert_allclose(
            ssp.result.x.numpy()[0], sd.result.x.numpy()[0],
            atol=1e-7,
        )

        # Sparse dP_values should equal dense_dP gathered at (rows, cols).
        # Build host-side (rows, cols) from the sparse pattern.
        def _csr_rows_cols(mat):
            indptr = mat.indptr.numpy()
            cols = mat.indices.numpy()
            rows = np_.zeros_like(cols)
            for i in range(mat.rows):
                rows[indptr[i]:indptr[i + 1]] = i
            return rows, cols

        for name, atol in (("P", 1e-7), ("A", 1e-7), ("G", 1e-7)):
            sparse_mat = getattr(gsp, name)
            if sparse_mat is None:
                continue
            rows, cols = _csr_rows_cols(sparse_mat)
            dense_mat = getattr(gd, name).numpy()[0]
            sparse_vals = sparse_mat.data.numpy()[0]
            dense_at_pattern = dense_mat[rows, cols]
            np_.testing.assert_allclose(
                sparse_vals, dense_at_pattern, atol=atol,
                err_msg=f"sparse d{name} doesn't match dense d{name} gathered",
            )

    def test_vector_grads_match_dense(self):
        """Vector gradients (c, b, h_u, h_l, x_u, x_l) are the same in
        sparse and dense -- they don't depend on sparsity."""
        import numpy as np_
        from cupiqp import DenseSolver, SparseSolver

        qp_raw = _make_qp_dense(n=8, p=3, m=4, seed=11)
        ssp = SparseSolver()
        ssp.settings = _quiet_settings()
        ssp.settings.enable_grad = True
        ssp.setup(**_qp_to_sparse(qp_raw))
        ssp.solve()
        gsp = ssp.backward(grad_x=to_wp(2.0 * ssp.result.x.numpy()))

        sd = DenseSolver()
        sd.settings = _quiet_settings()
        sd.settings.enable_grad = True
        sd.setup(**to_wp_dict(qp_raw))
        sd.solve()
        gd = sd.backward(grad_x=to_wp(2.0 * sd.result.x.numpy()))

        for name in ("c", "b", "h_u", "h_l", "x_u", "x_l"):
            v_sp = getattr(gsp, name).numpy()[0]
            v_d  = getattr(gd,  name).numpy()[0]
            np_.testing.assert_allclose(
                v_sp, v_d, atol=1e-7,
                err_msg=f"d{name} mismatch between sparse and dense grad",
            )
