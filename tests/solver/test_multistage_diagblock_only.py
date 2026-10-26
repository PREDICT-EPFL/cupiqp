"""Multistage problems without off-diagonal blocks.

An ``offdiag`` of ``None`` at setup removes those blocks: P becomes block
diagonal, and A / G become block diagonal with ``N`` block rows (``b`` /
``h_l`` / ``h_u`` have ``N`` blocks). Every combination must solve the same
QP as the dense solver given the assembled matrices.
"""
import itertools

import numpy as np
import pytest

from cupiqp import DenseSolver, MultistageSolver, Status
from tests.helpers import to_wp


def _assemble(N, d, P_D, P_E, A_D, A_E, G_D, G_E):
    """Dense P, A, G of one problem; A / G have N + 1 block rows with off-diagonals, N without."""
    n = N * d
    P = np.zeros((n, n))
    for k in range(N):
        P[k * d:(k + 1) * d, k * d:(k + 1) * d] = P_D[k]
    if P_E is not None:
        for k in range(N - 1):
            P[(k + 1) * d:(k + 2) * d, k * d:(k + 1) * d] = P_E[k]
            P[k * d:(k + 1) * d, (k + 1) * d:(k + 2) * d] = P_E[k].T

    def bidiag(D, E):
        r = D.shape[1]
        M = np.zeros(((N + (E is not None)) * r, n))
        for k in range(N):
            M[k * r:(k + 1) * r, k * d:(k + 1) * d] = D[k]
        if E is not None:
            for k in range(N):
                M[(k + 1) * r:(k + 2) * r, k * d:(k + 1) * d] = E[k]
        return M

    return P, bidiag(A_D, A_E), bidiag(G_D, G_E)


def _problem(N, d, rA, rG, B, has_P_E, has_A_E, has_G_E, seed):
    rng = np.random.default_rng(seed)
    P_D = rng.standard_normal((B, N, d, d))
    P_D = P_D @ P_D.transpose(0, 1, 3, 2) + 3 * d * np.eye(d)
    blocks = dict(
        P_D=P_D, P_E=0.3 * rng.standard_normal((B, N - 1, d, d)) if has_P_E else None,
        A_D=rng.standard_normal((B, N, rA, d)), A_E=rng.standard_normal((B, N, rA, d)) if has_A_E else None,
        G_D=rng.standard_normal((B, N, rG, d)), G_E=rng.standard_normal((B, N, rG, d)) if has_G_E else None,
    )
    dense = [_assemble(N, d, *(None if v is None else v[i] for v in blocks.values())) for i in range(B)]
    x0 = rng.uniform(-0.3, 0.3, (B, N * d))
    b = np.stack([dense[i][1] @ x0[i] for i in range(B)])
    Gx = np.stack([dense[i][2] @ x0[i] for i in range(B)])
    vecs = dict(c=rng.standard_normal((B, N * d)), b=b, h_l=Gx - 1.0, h_u=Gx + 0.5,
                x_l=-np.ones((B, N * d)), x_u=np.ones((B, N * d)))
    return blocks, dense, vecs


def _solvers(N, d, blocks, dense, vecs):
    B = vecs["c"].shape[0]
    ms = MultistageSolver()
    ms.setup(P=(to_wp(blocks["P_D"]), to_wp(blocks["P_E"])), A=(to_wp(blocks["A_D"]), to_wp(blocks["A_E"])),
             G=(to_wp(blocks["G_D"]), to_wp(blocks["G_E"])), **{k: to_wp(v) for k, v in vecs.items()})
    ds = DenseSolver()
    ds.setup(P=to_wp(np.stack([m[0] for m in dense])), A=to_wp(np.stack([m[1] for m in dense])),
             G=to_wp(np.stack([m[2] for m in dense])), **{k: to_wp(v) for k, v in vecs.items()})
    return ms, ds


def _assert_same_solution(ms, ds):
    ms.solve()
    ds.solve()
    assert all(s == Status.CUPIQP_SOLVED for s in ms.result.info.to_host().status)
    assert all(s == Status.CUPIQP_SOLVED for s in ds.result.info.to_host().status)
    np.testing.assert_allclose(ms.result.x.numpy(), ds.result.x.numpy(), atol=1e-8)


@pytest.mark.parametrize("has_P_E,has_A_E,has_G_E", list(itertools.product([True, False], repeat=3)))
@pytest.mark.parametrize("N", [1, 5])
def test_absent_offdiag_matches_dense(N, has_P_E, has_A_E, has_G_E):
    d, B = 3, 3
    blocks, dense, vecs = _problem(N, d, 2, 4, B, has_P_E, has_A_E, has_G_E,
                                   seed=N * 10 + 4 * has_P_E + 2 * has_A_E + has_G_E)
    ms, ds = _solvers(N, d, blocks, dense, vecs)
    _assert_same_solution(ms, ds)

    # update() keeps the structure: new costs, same comparison.
    c = np.random.default_rng(99).standard_normal((B, N * d))
    ms.update(c=to_wp(c))
    ds.update(c=to_wp(c))
    _assert_same_solution(ms, ds)


def test_absent_offdiag_cannot_be_added_in_update():
    N, d = 4, 2
    blocks, dense, vecs = _problem(N, d, 1, 2, 2, False, False, False, seed=7)
    ms, _ = _solvers(N, d, blocks, dense, vecs)
    with pytest.raises(ValueError, match="without off-diagonal"):
        ms.update(P=(None, to_wp(np.zeros((N - 1, d, d)))))
    with pytest.raises(ValueError, match="without off-diagonal"):
        ms.update(G=(None, to_wp(np.zeros((N, 2, d)))))


def test_absent_offdiag_vector_shapes():
    """Without A / G off-diagonals, b and the bounds have N blocks, not N + 1."""
    N, d = 4, 2
    blocks, dense, vecs = _problem(N, d, 1, 2, 1, True, False, False, seed=3)
    ms = MultistageSolver()
    with pytest.raises(ValueError):
        ms.setup(P=(to_wp(blocks["P_D"][0]), to_wp(blocks["P_E"][0])), c=to_wp(vecs["c"][0]),
                 A=(to_wp(blocks["A_D"][0]), None), b=to_wp(np.zeros((N + 1, 1))))
