"""Tests for cupiqp.torch.MultistageSolverLayer: forward and every block and
vector gradient against the dense layer on the assembled QP, shared and
batched blocks, rollouts with several pending solves, validation, and no
allocation in a call and its backward."""

import gc

import numpy as np
import pytest
import torch

from cupiqp import Status
from cupiqp.torch import DenseSolverLayer, MultistageSolverLayer, StructureMismatchError


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

DEVICE = "cuda"
DTYPE = torch.float64
N, D, RA, RG = 3, 2, 1, 2          # stages, stage size, equality and inequality rows per block


def _t(a):
    return torch.tensor(np.asarray(a), dtype=DTYPE, device=DEVICE)


def _blocks(seed=0):
    """Blocks of a strictly convex multistage QP with coupling in P, A and G,
    feasible with a strictly interior point, as numpy arrays."""
    rng = np.random.default_rng(seed)
    P_diag = np.stack([(lambda M: M @ M.T + 3.0 * np.eye(D))(rng.standard_normal((D, D)))
                       for _ in range(N)])
    P_off = 0.2 * rng.standard_normal((N - 1, D, D))
    A_diag, A_off = rng.standard_normal((N, RA, D)), rng.standard_normal((N, RA, D))
    G_diag, G_off = rng.standard_normal((N, RG, D)), rng.standard_normal((N, RG, D))
    blocks = dict(P_diag=P_diag, P_off=P_off, A_diag=A_diag, A_off=A_off, G_diag=G_diag, G_off=G_off)
    dense = _assemble(blocks)
    x_star = rng.standard_normal(N * D)
    mid = dense["G"] @ x_star
    rad = 0.5 + rng.random(mid.shape)
    vectors = dict(c=rng.standard_normal(N * D), b=dense["A"] @ x_star,
                   h_l=mid - rad, h_u=mid + rad, x_l=x_star - 1.5, x_u=x_star + 1.5)
    return blocks, vectors, dense


def _assemble(blk):
    """Dense P, A, G of the block structure (block row k of A and G holds
    offdiag[k-1] on stage k-1 and diag[k] on stage k; row N only offdiag[N-1])."""
    n = N * D
    P = np.zeros((n, n))
    for k in range(N):
        P[k*D:(k+1)*D, k*D:(k+1)*D] = blk["P_diag"][k]
    for k in range(N - 1):
        P[(k+1)*D:(k+2)*D, k*D:(k+1)*D] = blk["P_off"][k]
        P[k*D:(k+1)*D, (k+1)*D:(k+2)*D] = blk["P_off"][k].T

    def bidiagonal(diag, off, r):
        M = np.zeros(((N + 1) * r, n))
        for k in range(N):
            M[k*r:(k+1)*r, k*D:(k+1)*D] = diag[k]
            M[(k+1)*r:(k+2)*r, k*D:(k+1)*D] = off[k]
        return M

    return dict(P=P, A=bidiagonal(blk["A_diag"], blk["A_off"], RA),
                G=bidiagonal(blk["G_diag"], blk["G_off"], RG))


def _settings(layer):
    layer.settings.verbose = False
    layer.settings.preconditioner_iter = 0
    layer.settings.eps_abs = 1e-11
    layer.settings.eps_rel = 1e-11
    return layer


def _multistage_layer(blocks, vectors, c=None, **setup_kwargs):
    """``c`` (default: the shared ``vectors["c"]``) fixes the batch size."""
    layer = _settings(MultistageSolverLayer())
    c = _t(vectors["c"]) if c is None else c
    layer.setup(P=(_t(blocks["P_diag"]), _t(blocks["P_off"])), c=c,
                A=(_t(blocks["A_diag"]), _t(blocks["A_off"])), b=_t(vectors["b"]),
                G=(_t(blocks["G_diag"]), _t(blocks["G_off"])), h_u=_t(vectors["h_u"]),
                h_l=_t(vectors["h_l"]), x_u=_t(vectors["x_u"]), x_l=_t(vectors["x_l"]),
                **setup_kwargs)
    return layer


def _dense_layer(dense, vectors, c=None):
    layer = _settings(DenseSolverLayer())
    c = _t(vectors["c"]) if c is None else c
    layer.setup(_t(dense["P"]), c, A=_t(dense["A"]), b=_t(vectors["b"]),
                G=_t(dense["G"]), h_u=_t(vectors["h_u"]), h_l=_t(vectors["h_l"]),
                x_u=_t(vectors["x_u"]), x_l=_t(vectors["x_l"]))
    return layer


def _statuses(layer):
    return tuple(Status(int(v)) for v in layer.result.status.cpu().tolist())


# ---------------------------------------------------------------------------
# Forward and gradients against the dense layer
# ---------------------------------------------------------------------------
def test_forward_and_every_gradient_match_dense_layer():
    blocks, vectors, dense = _blocks(seed=1)
    B = 2
    c_batch = _t(np.stack([vectors["c"], vectors["c"] + 0.3]))
    weights = _t(np.random.default_rng(2).standard_normal((B, N * D)))

    ms = _multistage_layer(blocks, vectors, c=c_batch)
    ms_in = dict(P_diag=_t(blocks["P_diag"]), P_offdiag=_t(blocks["P_off"]), c=c_batch.clone(),
                 A_diag=_t(blocks["A_diag"]), A_offdiag=_t(blocks["A_off"]), b=_t(vectors["b"]),
                 G_diag=_t(blocks["G_diag"]), G_offdiag=_t(blocks["G_off"]),
                 h_u=_t(vectors["h_u"]), h_l=_t(vectors["h_l"]),
                 x_u=_t(vectors["x_u"]), x_l=_t(vectors["x_l"]))
    for t in ms_in.values():
        t.requires_grad_(True)
    x_ms = ms(**ms_in)
    assert x_ms.shape == (B, N * D)
    assert _statuses(ms) == (Status.CUPIQP_SOLVED,) * B
    (weights * x_ms).sum().backward()

    de = _dense_layer(dense, vectors, c=c_batch)
    de_in = dict(P=_t(dense["P"]), c=c_batch.clone(), A=_t(dense["A"]), b=_t(vectors["b"]),
                 G=_t(dense["G"]), h_u=_t(vectors["h_u"]), h_l=_t(vectors["h_l"]),
                 x_u=_t(vectors["x_u"]), x_l=_t(vectors["x_l"]))
    for t in de_in.values():
        t.requires_grad_(True)
    x_de = de(**de_in)
    (weights * x_de).sum().backward()

    tol = dict(atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(x_ms.detach(), x_de.detach(), **tol)
    for name in ("c", "b", "h_u", "h_l", "x_u", "x_l"):
        torch.testing.assert_close(ms_in[name].grad, de_in[name].grad, **tol, msg=name)

    dP, dA, dG = (de_in[k].grad.cpu().numpy() for k in ("P", "A", "G"))
    exp = {k: np.zeros_like(v) for k, v in blocks.items()}
    for k in range(N):
        exp["P_diag"][k] = dP[k*D:(k+1)*D, k*D:(k+1)*D]
        exp["A_diag"][k] = dA[k*RA:(k+1)*RA, k*D:(k+1)*D]
        exp["A_off"][k] = dA[(k+1)*RA:(k+2)*RA, k*D:(k+1)*D]
        exp["G_diag"][k] = dG[k*RG:(k+1)*RG, k*D:(k+1)*D]
        exp["G_off"][k] = dG[(k+1)*RG:(k+2)*RG, k*D:(k+1)*D]
    for k in range(N - 1):
        # P_offdiag[k] is block (k+1, k) and, transposed, block (k, k+1).
        exp["P_off"][k] = dP[(k+1)*D:(k+2)*D, k*D:(k+1)*D] + dP[k*D:(k+1)*D, (k+1)*D:(k+2)*D].T
    pairs = dict(P_diag="P_diag", P_offdiag="P_off", A_diag="A_diag", A_offdiag="A_off",
                 G_diag="G_diag", G_offdiag="G_off")
    for field, key in pairs.items():
        torch.testing.assert_close(ms_in[field].grad, _t(exp[key]), **tol, msg=field)


def test_shared_block_gradient_is_summed_over_batch():
    blocks, vectors, _ = _blocks(seed=3)
    B = 3
    c_batch = _t(np.stack([vectors["c"] + 0.2 * k for k in range(B)]))
    layer = _multistage_layer(blocks, vectors, c=c_batch)
    shared = _t(blocks["A_off"]).requires_grad_(True)
    layer(c=c_batch, A_offdiag=shared).sum().backward()
    batched = _t(blocks["A_off"]).expand(B, N, RA, D).clone().requires_grad_(True)
    layer(c=c_batch, A_offdiag=batched).sum().backward()
    torch.testing.assert_close(shared.grad, batched.grad.sum(dim=0), atol=1e-8, rtol=1e-8)


def test_block_absent_at_setup_cannot_be_passed():
    blocks, vectors, _ = _blocks(seed=4)
    layer = _settings(MultistageSolverLayer())
    layer.setup(P=(_t(blocks["P_diag"]), None), c=_t(vectors["c"]),
                x_l=_t(vectors["x_l"]), x_u=_t(vectors["x_u"]))
    with pytest.raises(StructureMismatchError, match="'P_offdiag' was not given at setup"):
        layer(P_offdiag=_t(blocks["P_off"]))
    with pytest.raises(StructureMismatchError, match="'A_diag' was not given at setup"):
        layer(A_diag=_t(blocks["A_diag"]))
    c = _t(vectors["c"]).requires_grad_(True)
    layer(c=c).sum().backward()
    assert c.grad.shape == (N * D,)


# ---------------------------------------------------------------------------
# Several pending solves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_rollout_matches_single_calls(order):
    blocks, vectors, _ = _blocks(seed=5)
    T = 3
    layer = _multistage_layer(blocks, vectors, max_pending_solves=T,
                              per_call_fields=("c", "A_offdiag"))
    cs = [(_t(vectors["c"]) + 0.2 * k).requires_grad_(True) for k in range(T)]
    dyn = [(_t(blocks["A_off"]) * (1.0 + 0.05 * k)).requires_grad_(True) for k in range(T)]
    outs = [layer(c=cs[k], A_offdiag=dyn[k]) for k in range(T)]
    assert len({o.data_ptr() for o in outs}) == T
    terms = [(k + 1) * o.sum() for k, o in enumerate(outs)]
    sum(terms[::-1] if order == "reverse" else terms).backward()
    for k in range(T):
        ref = _multistage_layer(blocks, vectors)
        ck = cs[k].detach().clone().requires_grad_(True)
        dk = dyn[k].detach().clone().requires_grad_(True)
        ((k + 1) * ref(c=ck, A_offdiag=dk).sum()).backward()
        torch.testing.assert_close(cs[k].grad, ck.grad, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(dyn[k].grad, dk.grad, atol=1e-6, rtol=1e-6)
        del ref
        gc.collect()        # socu keeps dropped multistage solvers in cycles


def test_forward_and_backward_allocate_nothing():
    blocks, vectors, _ = _blocks(seed=6)
    layer = _multistage_layer(blocks, vectors)
    c = _t(vectors["c"]).requires_grad_(True)
    dyn = _t(blocks["A_off"]).requires_grad_(True)
    layer(c=c, A_offdiag=dyn).sum().backward()          # warm up
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    layer(c=c, A_offdiag=dyn).sum().backward()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_setup_and_call_validation():
    blocks, vectors, _ = _blocks(seed=7)
    layer = _settings(MultistageSolverLayer())
    with pytest.raises(TypeError, match=r"'P' must be a \(diag, offdiag\) pair"):
        layer.setup(P=_t(blocks["P_diag"]), c=_t(vectors["c"]))
    with pytest.raises(ValueError, match="'P' needs its diagonal blocks"):
        layer.setup(P=(None, _t(blocks["P_off"])), c=_t(vectors["c"]))
    with pytest.raises(ValueError, match="'P_diag' must be 3-D or 4-D"):
        layer.setup(P=(_t(blocks["P_diag"][0]), None), c=_t(vectors["c"]))
    with pytest.raises(ValueError, match=r"not given \['P_offdiag'\]"):
        layer.setup(P=(_t(blocks["P_diag"]), None), c=_t(vectors["c"]),
                    max_pending_solves=2, per_call_fields=("c", "P_offdiag"))
    with pytest.raises(ValueError, match=r"unknown \['P'\]"):
        layer.setup(P=(_t(blocks["P_diag"]), None), c=_t(vectors["c"]),
                    max_pending_solves=2, per_call_fields=("P",))
    assert not layer.is_setup
    layer = _multistage_layer(blocks, vectors)
    with pytest.raises(StructureMismatchError, match=r"'A_offdiag' must have shape"):
        layer(A_offdiag=_t(blocks["A_off"][:2]))
    with pytest.raises(StructureMismatchError, match=r"'c' must have shape \(6,\)"):
        layer(c=_t(vectors["c"]).reshape(N, D))       # vectors are flat in calls
    layer(c=_t(vectors["c"]).reshape(N, D).reshape(-1))
