"""Tests for cupiqp.torch.SparseSolverLayer: CSR patterns at setup, values-only
calls, agreement of gradients with the dense layer, and the pending-solve
policy (max_pending_solves)."""

import warnings

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from cupiqp import Status
from cupiqp.torch import DenseSolverLayer, SparseSolverLayer, StructureMismatchError

from tests.helpers import layer_statuses


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")

DEVICE = "cuda"
DTYPE = torch.float64


def _t(values):
    return torch.tensor(values, dtype=DTYPE, device=DEVICE)


def _make(layer_cls, **settings):
    layer = layer_cls()
    layer.settings.preconditioner_iter = 0
    layer.settings.verbose = False
    layer.settings.eps_abs = 1e-11
    layer.settings.eps_rel = 1e-11
    for name, value in settings.items():
        setattr(layer.settings, name, value)
    return layer


def _diag_csr(values: torch.Tensor) -> torch.Tensor:
    n = values.numel()
    return torch.sparse_csr_tensor(
        torch.arange(n + 1, dtype=torch.int64, device=DEVICE),
        torch.arange(n, dtype=torch.int64, device=DEVICE),
        values, size=(n, n),
    )


def _tridiag_qp(n=5, seed=0):
    rng = np.random.default_rng(seed)
    main = 4.0 + rng.random(n)
    off = 0.5 * rng.standard_normal(n - 1)
    P = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    c = rng.standard_normal(n)
    G = rng.standard_normal((2, n))
    h_u = 1.0 + rng.random(2)
    h_l = -1.0 - rng.random(2)
    return P, c, G, h_u, h_l


def _to_csr(dense: np.ndarray) -> torch.Tensor:
    return _t(dense).to_sparse_csr()


def _gather(dense_grad, csr):
    """Dense gradient at the stored positions of ``csr``."""
    crow, col = csr.crow_indices(), csr.col_indices()
    rows = torch.repeat_interleave(torch.arange(csr.shape[0], device=DEVICE), torch.diff(crow))
    return dense_grad[rows, col]


# ---------------------------------------------------------------------------
# Setup and call contract
# ---------------------------------------------------------------------------
def test_setup_requires_csr_and_call_requires_values():
    values = _t([2.0, 3.0, 4.0])
    c = torch.ones(3, dtype=DTYPE, device=DEVICE)
    layer = _make(SparseSolverLayer)
    with pytest.raises(TypeError, match="sparse CSR"):
        layer.setup(torch.diag(values), c)
    layer.setup(_diag_csr(values), c)
    with pytest.raises(TypeError, match="dense"):
        layer(P=_diag_csr(values))
    with pytest.raises(StructureMismatchError, match="shape"):
        layer(P=torch.ones(4, dtype=DTYPE, device=DEVICE))
    x = layer(P=values + 1.0)
    assert x.shape == (1, 3)
    torch.testing.assert_close(x, (-c / (values + 1.0)).unsqueeze(0), atol=1e-8, rtol=1e-8)
    assert layer_statuses(layer) == (Status.CUPIQP_SOLVED,)


def test_setup_validation_of_csr_and_vectors():
    values = _t([2.0, 3.0, 4.0])
    c = torch.ones(3, dtype=DTYPE, device=DEVICE)
    layer = _make(SparseSolverLayer)
    with pytest.raises(ValueError, match="dtype"):
        layer.setup(_diag_csr(values.float()), c)
    with pytest.raises(ValueError, match="must be on cuda"):
        layer.setup(_diag_csr(values).cpu(), c)
    rect = torch.ones(2, 3, dtype=DTYPE, device=DEVICE).to_sparse_csr()
    with pytest.raises(ValueError, match="'P' must be square"):
        layer.setup(rect, c)
    with pytest.raises(ValueError, match="'c' must be 1-D or 2-D"):
        layer.setup(_diag_csr(values), c.reshape(1, 1, 3))
    with pytest.raises(TypeError, match="'c' must be a dense"):
        layer.setup(_diag_csr(values), c.to_sparse())
    with pytest.raises(TypeError, match="torch.Tensor"):
        layer.setup(_diag_csr(values), [1.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="not set up"):
        layer(c=c)
    assert not layer.is_setup
    layer.setup(_diag_csr(values), c)
    with pytest.raises(StructureMismatchError, match="not given at setup"):
        layer(h_u=c)
    torch.testing.assert_close(layer(c=c), (-c / values).unsqueeze(0), atol=1e-8, rtol=1e-8)


def test_forward_matches_dense_layer():
    P, c, G, h_u, h_l = _tridiag_qp()
    sparse = _make(SparseSolverLayer)
    sparse.setup(_to_csr(P), _t(c), G=_to_csr(G), h_u=_t(h_u), h_l=_t(h_l))
    dense = _make(DenseSolverLayer)
    dense.setup(_t(P), _t(c), G=_t(G), h_u=_t(h_u), h_l=_t(h_l))
    torch.testing.assert_close(sparse(), dense(), atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(sparse.result.z_u, dense.result.z_u, atol=1e-7, rtol=1e-7)


def test_batched_csr_setup_and_batched_values():
    B, n = 3, 4
    values = _t([[1.0, 2.0, 3.0, 4.0]] * B) * torch.arange(1, B + 1, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    P = torch.stack([_diag_csr(values[b]).to_dense() for b in range(B)]).to_sparse_csr()
    c = torch.ones(B, n, dtype=DTYPE, device=DEVICE)
    layer = _make(SparseSolverLayer)
    layer.setup(P, c)
    assert layer.batch_size == B
    torch.testing.assert_close(layer(P=values), -c / values, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(layer(P=values[0]), -c / values[0], atol=1e-8, rtol=1e-8)


def test_forward_and_backward_allocate_nothing():
    values = _t([2.0, 3.0, 4.0])
    c = torch.ones(3, dtype=DTYPE, device=DEVICE)
    v = values.clone().requires_grad_(True)
    layer = _make(SparseSolverLayer)
    layer.setup(_diag_csr(values), c)
    layer(P=v, c=c).sum().backward()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    layer(P=v, c=c).sum().backward()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------
def test_value_gradients_match_dense_layer():
    P, c, G, h_u, h_l = _tridiag_qp(seed=1)
    P_csr, G_csr = _to_csr(P), _to_csr(G)
    sparse = _make(SparseSolverLayer)
    sparse.setup(P_csr, _t(c), G=G_csr, h_u=_t(h_u), h_l=_t(h_l))
    P_vals = P_csr.values().clone().requires_grad_(True)
    G_vals = G_csr.values().clone().requires_grad_(True)
    c_s = _t(c).requires_grad_(True)
    h_s = _t(h_u).requires_grad_(True)
    x = sparse(P=P_vals, c=c_s, G=G_vals, h_u=h_s)
    (x.sum() + sparse.result.z_u.sum()).backward()

    dense = _make(DenseSolverLayer)
    dense.setup(_t(P), _t(c), G=_t(G), h_u=_t(h_u), h_l=_t(h_l))
    P_d = _t(P).requires_grad_(True)
    G_d = _t(G).requires_grad_(True)
    c_d = _t(c).requires_grad_(True)
    h_d = _t(h_u).requires_grad_(True)
    x_d = dense(P=P_d, c=c_d, G=G_d, h_u=h_d)
    (x_d.sum() + dense.result.z_u.sum()).backward()

    tol = dict(atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(P_vals.grad, _gather(P_d.grad, P_csr), **tol)
    torch.testing.assert_close(G_vals.grad, _gather(G_d.grad, G_csr), **tol)
    torch.testing.assert_close(c_s.grad, c_d.grad, **tol)
    torch.testing.assert_close(h_s.grad, h_d.grad, **tol)


def test_shared_values_gradient_is_summed_over_batch():
    B, n = 3, 3
    values = _t([2.0, 3.0, 4.0])
    c = torch.arange(1, B * n + 1, dtype=DTYPE, device=DEVICE).reshape(B, n)
    layer = _make(SparseSolverLayer)
    layer.setup(_diag_csr(values), c)
    v_shared = values.clone().requires_grad_(True)
    layer(P=v_shared, c=c).sum().backward()
    v_batched = values.expand(B, n).clone().requires_grad_(True)
    layer(P=v_batched, c=c).sum().backward()
    tol = dict(atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v_shared.grad, v_batched.grad.sum(dim=0), **tol)
    torch.testing.assert_close(v_shared.grad, (c / values**2).sum(dim=0), **tol)


def test_pending_backward_blocks_reset_until_graph_is_dropped():
    import gc
    values = _t([1.0, 2.0])
    c = torch.ones(2, dtype=DTYPE, device=DEVICE).requires_grad_(True)
    layer = _make(SparseSolverLayer)
    layer.setup(_diag_csr(values), c.detach())
    x = layer(c=c)
    with pytest.raises(RuntimeError, match="pending backward"):
        layer(c=c)
    with pytest.raises(RuntimeError, match="pending backward"):
        layer.reset()
    del x
    gc.collect()
    layer.reset()
    layer.setup(_diag_csr(values), c.detach())
    layer(c=c).sum().backward()
    assert c.grad is not None


def test_multi_pending_rollout_matches_single_calls():
    P, c, G, h_u, h_l = _tridiag_qp(seed=2)
    P_csr, G_csr = _to_csr(P), _to_csr(G)
    T = 3
    layer = _make(SparseSolverLayer)
    layer.setup(P_csr, _t(c), G=G_csr, h_u=_t(h_u), h_l=_t(h_l), max_pending_solves=T,
                per_call_fields=("P", "c"))
    vals = [(P_csr.values() * (1.0 + 0.1 * k)).requires_grad_(True) for k in range(T)]
    cs = [(_t(c) + 0.2 * k).requires_grad_(True) for k in range(T)]
    outs = [layer(P=vals[k], c=cs[k]) for k in range(T)]
    assert len({o.data_ptr() for o in outs}) == T
    sum((k + 1) * o.sum() for k, o in enumerate(outs)).backward()
    for k in range(T):
        ref = _make(SparseSolverLayer)
        ref.setup(P_csr, _t(c), G=G_csr, h_u=_t(h_u), h_l=_t(h_l))
        vk = vals[k].detach().clone().requires_grad_(True)
        ck = cs[k].detach().clone().requires_grad_(True)
        ((k + 1) * ref(P=vk, c=ck).sum()).backward()
        torch.testing.assert_close(vals[k].grad, vk.grad, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(cs[k].grad, ck.grad, atol=1e-6, rtol=1e-6)


def test_pending_solve_policy_matches_dense():
    """The policy lives in the shared base: the sparse layer follows it too."""
    from unittest import mock
    import cupiqp
    values = _t([1.0, 2.0])
    c0 = _t([1.0, -1.0])
    layer = _make(SparseSolverLayer)
    with pytest.raises(ValueError, match="per_call_fields is only used"):
        layer.setup(_diag_csr(values), c0, per_call_fields=("c",))
    with pytest.raises(ValueError, match="max_pending_solves must be a positive integer"):
        layer.setup(_diag_csr(values), c0, max_pending_solves=0, per_call_fields=("c",))

    # Capacity one: a no_grad call is refused while a solve is pending.
    layer.setup(_diag_csr(values), c0)
    c = c0.clone().requires_grad_(True)
    x = layer(c=c)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="pending backward"):
            layer(c=c0)
    x.sum().backward()
    torch.testing.assert_close(c.grad, -1.0 / values, atol=1e-6, rtol=1e-6)

    # Capacity two: no_grad calls are free, every backward refactorizes.
    layer.reset()
    layer.setup(_diag_csr(values), c0, max_pending_solves=2, per_call_fields=("c",))
    cs = [(c0 + k).requires_grad_(True) for k in range(2)]
    x0 = layer(c=cs[0])
    with torch.no_grad():
        layer(c=c0 * 5.0)
        layer(c=c0 * 7.0)
    x1 = layer(c=cs[1])
    with pytest.raises(RuntimeError, match="max_pending_solves"):
        layer(c=c)
    torch.testing.assert_close(x0.detach(), -(c0 / values).unsqueeze(0), atol=1e-6, rtol=1e-6)
    with mock.patch.object(cupiqp.SparseSolver, "backward",
                           side_effect=AssertionError("reused the factorization")):
        (x0.sum() + 3.0 * x1.sum()).backward()
    torch.testing.assert_close(cs[0].grad, -1.0 / values, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(cs[1].grad, -3.0 / values, atol=1e-6, rtol=1e-6)
