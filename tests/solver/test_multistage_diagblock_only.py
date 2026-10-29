"""Multistage problems without off-diagonal blocks.

An ``offdiag`` of ``None`` at setup removes those blocks: P becomes block
diagonal, and A / G become block diagonal with ``N`` block rows (``b`` /
``h_l`` / ``h_u`` have ``N`` blocks). Every combination must solve the same
QP as the dense solver given the assembled matrices.
"""
import itertools

import numpy as np
import pytest

from cupiqp import DenseSolver, MultistageSolver
from tests.helpers import assemble_multistage, assert_all_solved, map_arrays, random_multistage_qp, to_wp


def _problem(N, d, rA, rG, B, has_P_E, has_A_E, has_G_E, seed):
    """A batched multistage QP in the solver's layout and, per problem, the
    dense ``(P, A, G)`` it assembles to."""
    qp = random_multistage_qp(seed, N, d, rA, rG, B, has_P_E, has_A_E, has_G_E)
    blocks = [M for pair in (qp["P"], qp["A"], qp["G"]) for M in pair]
    dense = [assemble_multistage(N, d, *(None if M is None else M[i] for M in blocks)) for i in range(B)]
    return qp, dense


def _solvers(qp, dense):
    B = qp["c"].shape[0]
    ms = MultistageSolver()
    ms.setup(**map_arrays(qp, to_wp))
    ds = DenseSolver()
    vecs = {k: to_wp(v.reshape(B, -1)) for k, v in qp.items() if not isinstance(v, tuple)}
    ds.setup(P=to_wp(np.stack([M[0] for M in dense])), A=to_wp(np.stack([M[1] for M in dense])),
             G=to_wp(np.stack([M[2] for M in dense])), **vecs)
    return ms, ds


def _assert_same_solution(ms, ds):
    ms.solve()
    ds.solve()
    assert_all_solved(ms)
    assert_all_solved(ds)
    np.testing.assert_allclose(ms.result.x.numpy(), ds.result.x.numpy(), atol=1e-8)


@pytest.mark.parametrize("has_P_E,has_A_E,has_G_E", list(itertools.product([True, False], repeat=3)))
@pytest.mark.parametrize("N", [1, 5])
def test_absent_offdiag_matches_dense(N, has_P_E, has_A_E, has_G_E):
    d, B = 3, 3
    qp, dense = _problem(N, d, 2, 4, B, has_P_E, has_A_E, has_G_E,
                         seed=N * 10 + 4 * has_P_E + 2 * has_A_E + has_G_E)
    ms, ds = _solvers(qp, dense)
    _assert_same_solution(ms, ds)

    # update() keeps the structure: new costs, same comparison.
    c = np.random.default_rng(99).standard_normal((B, N * d))
    ms.update(c=to_wp(c.reshape(B, N, d)))
    ds.update(c=to_wp(c))
    _assert_same_solution(ms, ds)


def test_absent_offdiag_cannot_be_added_in_update():
    N, d = 4, 2
    qp, dense = _problem(N, d, 1, 2, 2, False, False, False, seed=7)
    ms, _ = _solvers(qp, dense)
    with pytest.raises(ValueError, match="without off-diagonal"):
        ms.update(P=(None, to_wp(np.zeros((N - 1, d, d)))))
    with pytest.raises(ValueError, match="without off-diagonal"):
        ms.update(G=(None, to_wp(np.zeros((N, 2, d)))))


def test_absent_offdiag_vector_shapes():
    """Without A / G off-diagonals, b and the bounds have N blocks, not N + 1."""
    N, d = 4, 2
    qp, _ = _problem(N, d, 1, 2, 1, True, False, False, seed=3)
    ms = MultistageSolver()
    with pytest.raises(ValueError):
        ms.setup(P=(to_wp(qp["P"][0][0]), to_wp(qp["P"][1][0])), c=to_wp(qp["c"][0]),
                 A=(to_wp(qp["A"][0][0]), None), b=to_wp(np.zeros((N + 1, 1))))
