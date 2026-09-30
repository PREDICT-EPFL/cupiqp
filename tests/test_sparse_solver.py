"""End-to-end correctness of the sparse backend under the full-length dual layout.

The sparse KKT backend solves the augmented system with the explicit inequality
diagonal magnitude ``z_reg = 1 / weight`` (where the condensed row weight is
``z_reg_inv = w_l + w_u``); inactive inequality rows (both bounds infinite) are
decoupled. These tests check the *whole* sparse solve, not just the KKT
round-trip:

  * the primal solution matches PIQP (the CPU reference cuPIQP is modeled on),
    across all-finite / one-sided / mixed-infinite bound patterns;
  * duals are returned full length (``z_l``/``z_u`` length m, ``z_bl``/``z_bu``
    length n) with exactly 0 at infinite-bound positions;
  * a batched solve whose elements have *different* finite/infinite patterns
    matches independent single-problem sparse solves.

The reference is PIQP on CPU (not the GPU dense backend): mixing the cuDSS
(sparse) and cuSOLVER (dense) backends in one process trips a pre-existing
nvmath/cuSOLVER handle-ordering issue unrelated to this layout, so the
cross-check is kept backend-pure on the GPU side.
"""
import numpy as np
import cupy as cp
import pytest
from cupyx.scipy.sparse import csr_matrix

piqp = pytest.importorskip("piqp")

from cupiqp import SparseSolver, Status, PIQP_INF


def _make_qp(n, p, m, pattern, seed):
    """Strictly-feasible convex QP with a chosen finite/infinite bound pattern.

    ``pattern`` maps any of 'hl','hu','xl','xu' to a boolean finite-mask;
    missing keys default to all-finite.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)
    P = 0.5 * (P + P.T)
    c = rng.standard_normal(n)
    x0 = rng.standard_normal(n)

    A = rng.standard_normal((p, n)) if p else np.zeros((0, n))
    b = A @ x0 if p else np.zeros((0,))
    G = rng.standard_normal((m, n)) if m else np.zeros((0, n))
    Gx0 = G @ x0 if m else np.zeros((0,))

    def fin(key, length):
        msk = pattern.get(key)
        return np.ones(length, bool) if msk is None else np.asarray(msk, bool)

    h_l = Gx0 - (0.5 + rng.random(m)); h_u = Gx0 + (0.5 + rng.random(m))
    h_l[~fin('hl', m)] = -np.inf; h_u[~fin('hu', m)] = np.inf
    x_l = x0 - (0.5 + rng.random(n)); x_u = x0 + (0.5 + rng.random(n))
    x_l[~fin('xl', n)] = -np.inf; x_u[~fin('xu', n)] = np.inf
    return dict(P=P, c=c, A=A, b=b, G=G, h_l=h_l, h_u=h_u, x_l=x_l, x_u=x_u)



def _csr(dense) -> tuple:
    """CSR triple ``(indptr, indices, values)`` of a dense matrix, on the GPU."""
    M = csr_matrix(cp.asarray(dense))
    return M.indptr, M.indices, M.data

def _solve_piqp(qp):
    n, p, m = qp['P'].shape[0], qp['A'].shape[0], qp['G'].shape[0]
    s = piqp.DenseSolver()
    s.settings.eps_abs = 1e-10
    s.settings.eps_rel = 1e-10
    s.settings.verbose = False
    s.setup(
        np.asfortranarray(qp['P']), qp['c'],
        np.asfortranarray(qp['A']) if p else None, qp['b'] if p else None,
        np.asfortranarray(qp['G']) if m else None,
        qp['h_l'] if m else None, qp['h_u'] if m else None,
        qp['x_l'], qp['x_u'],
    )
    status = s.solve()
    return np.asarray(s.result.x)


def _solve_sparse(qp):
    n, p, m = qp['P'].shape[0], qp['A'].shape[0], qp['G'].shape[0]
    s = SparseSolver()
    s.settings.eps_abs = 1e-10
    s.settings.eps_rel = 1e-10
    s.settings.verbose = False
    kw = dict(P=_csr(qp['P']), c=cp.asarray(qp['c']))
    if p:
        kw.update(A=_csr(qp['A']), b=cp.asarray(qp['b']))
    if m:
        kw.update(G=_csr(qp['G']),
                  h_l=cp.asarray(qp['h_l']), h_u=cp.asarray(qp['h_u']))
    kw.update(x_l=cp.asarray(qp['x_l']), x_u=cp.asarray(qp['x_u']))
    s.setup(**kw)
    s.solve()
    return s


def _setup_batched(s, qps, blocks=('P', 'c', 'A', 'b', 'G', 'h_l', 'h_u', 'x_l', 'x_u')):
    """Set up ``s`` with every problem of ``qps``: the CSR pattern of
    ``qps[0]`` with stacked ``(B, nnz)`` values for the matrices, stacked
    ``(B, k)`` vectors."""
    inputs = {}
    for k in blocks:
        if k in ('P', 'A', 'G'):
            csrs = [csr_matrix(cp.asarray(q[k])) for q in qps]
            assert all(m.nnz == csrs[0].nnz for m in csrs), "patterns must match"
            inputs[k] = (csrs[0].indptr, csrs[0].indices, cp.stack([m.data for m in csrs]))
        else:
            inputs[k] = cp.asarray(np.stack([q[k] for q in qps]))
    s.setup(**inputs)


_CASES = {
    'all_finite': dict(n=8, p=2, m=5, pattern={}, seed=1),
    'oneside_hu': dict(n=8, p=2, m=5, pattern={'hl': [False] * 5}, seed=2),
    'oneside_xl': dict(n=7, p=1, m=4, pattern={'xu': [False] * 7}, seed=3),
    'mixed': dict(n=9, p=2, m=6,
                  pattern={'hl': [True, False, True, False, True, False],
                           'hu': [False, True, True, False, True, True],
                           'xl': [True, False, True, True, False, True, False, True, True],
                           'xu': [False, True, True, False, True, False, True, True, False]},
                  seed=4),
    'only_box': dict(n=8, p=0, m=0, pattern={}, seed=5),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_sparse_matches_piqp(name):
    cfg = _CASES[name]
    qp = _make_qp(cfg['n'], cfg['p'], cfg['m'], cfg['pattern'], cfg['seed'])
    n, p, m = cfg['n'], cfg['p'], cfg['m']

    x_ref = _solve_piqp(qp)
    s = _solve_sparse(qp)
    assert int(np.asarray(s.result.info.status_value)[0]) == Status.CUPIQP_SOLVED.value

    x = cp.asnumpy(s.result.x)[0]
    # Cross-solver primal agreement (both stop on residuals, not on x directly).
    np.testing.assert_allclose(x, x_ref, atol=2e-5, rtol=1e-4)

    # Duals are full length, with exact zeros at infinite-bound positions.
    z_l, z_u = cp.asnumpy(s.result.z_l)[0], cp.asnumpy(s.result.z_u)[0]
    z_bl, z_bu = cp.asnumpy(s.result.z_bl)[0], cp.asnumpy(s.result.z_bu)[0]
    assert z_l.shape == (m,) and z_u.shape == (m,)
    assert z_bl.shape == (n,) and z_bu.shape == (n,)
    np.testing.assert_array_equal(z_l[qp['h_l'] <= -PIQP_INF], 0.0)
    np.testing.assert_array_equal(z_u[qp['h_u'] >= PIQP_INF], 0.0)
    np.testing.assert_array_equal(z_bl[qp['x_l'] <= -PIQP_INF], 0.0)
    np.testing.assert_array_equal(z_bu[qp['x_u'] >= PIQP_INF], 0.0)

    # Stationarity in cuPIQP's convention: Px + c + A^T y + G^T(z_u - z_l) + (z_bu - z_bl) = 0.
    y = cp.asnumpy(s.result.y)[0]
    stat = qp['P'] @ x + qp['c'] + (qp['A'].T @ y if p else 0.0) \
        + (qp['G'].T @ (z_u - z_l) if m else 0.0) + (z_bu - z_bl)
    assert np.max(np.abs(stat)) < 1e-5


def test_sparse_batched_per_element_masks():
    """Batched sparse solve with different per-element infinite patterns must
    match independent single-problem sparse solves."""
    n, p, m = 6, 1, 4
    qps = [
        _make_qp(n, p, m, {}, 10),
        _make_qp(n, p, m, {'hl': [False] * m}, 11),
        _make_qp(n, p, m, {'xu': [False] * n}, 12),
    ]
    B = len(qps)

    refs = [cp.asnumpy(_solve_sparse(qp).result.x[0]) for qp in qps]

    # Shared sparsity pattern across the batch (same P/A/G structure here);
    # bounds differ per element, including which entries are infinite.
    s = SparseSolver()
    s.settings.eps_abs = 1e-10
    s.settings.verbose = False
    _setup_batched(s, qps)
    s.solve()
    for i in range(B):
        np.testing.assert_allclose(cp.asnumpy(s.result.x[i]), refs[i], atol=1e-5, rtol=1e-4)


def _setup_sparse(qp):
    n, p, m = qp['P'].shape[0], qp['A'].shape[0], qp['G'].shape[0]
    s = SparseSolver()
    s.settings.eps_abs = 1e-10
    s.settings.eps_rel = 1e-10
    s.settings.verbose = False
    kw = dict(P=_csr(qp['P']), c=cp.asarray(qp['c']),
              x_l=cp.asarray(qp['x_l']), x_u=cp.asarray(qp['x_u']))
    if p:
        kw.update(A=_csr(qp['A']), b=cp.asarray(qp['b']))
    if m:
        kw.update(G=_csr(qp['G']),
                  h_l=cp.asarray(qp['h_l']), h_u=cp.asarray(qp['h_u']))
    s.setup(**kw)
    return s


def _box_qp(h_u_pattern):
    """min 0.5||x||^2 - 5*sum(x) s.t. h_l <= I x <= h_u, with G = I.

    The unconstrained optimum is x = 5 in every coordinate; each finite upper
    bound (set to 1.0) binds, so ``h_u_pattern`` (True = finite 1.0, False =
    +inf) directly selects which coordinates are pinned to 1.0 vs free at 5.0.
    This makes toggling a row active/inactive *change the solution*, which is
    what exercises the sparse G-mask refresh.
    """
    n = len(h_u_pattern)
    P = np.eye(n)
    c = -5.0 * np.ones(n)
    G = np.eye(n)
    h_l = np.full(n, -np.inf)
    h_u = np.where(np.asarray(h_u_pattern, bool), 1.0, np.inf)
    x_l = np.full(n, -100.0)
    x_u = np.full(n, 100.0)
    return dict(P=P, c=c, A=np.zeros((0, n)), b=np.zeros((0,)),
                G=G, h_l=h_l, h_u=h_u, x_l=x_l, x_u=x_u)


def test_sparse_update_toggles_active_inequality_rows():
    """update(h_l/h_u) that flips inequality rows finite<->infinite must give
    the same answer as a fresh setup with the new bounds, with no re-setup().

    This guards the sparse KKT G-block mask: an inactive row's G coupling is
    zeroed in the factorized matrix, so flipping a row active<->inactive via a
    bound update must re-scatter the masked G - even though G itself is
    unchanged. Both toggle directions are exercised in one update.
    """
    # patt1: row 0 inactive (+inf), rows 1,2 pinned -> x = [5, 1, 1].
    # patt2: row 0 pinned, row 1 inactive -> x = [1, 5, 1]. Both rows toggle.
    qp1 = _box_qp([False, True, True])
    qp2 = _box_qp([True, False, True])

    s_ref = _setup_sparse(qp2)
    s_ref.solve()
    x_ref = cp.asnumpy(s_ref.result.x[0])

    s1 = _setup_sparse(qp1)
    s1.solve()
    x_patt1 = cp.asnumpy(s1.result.x[0])
    # Sanity: toggling these rows actually moves the solution, so the test
    # genuinely exercises a stale mask rather than a no-op.
    assert not np.allclose(x_patt1, x_ref, atol=1e-3)

    # Reuse the same solver: only the bound vectors change.
    s1.update(h_l=cp.asarray(qp2['h_l']), h_u=cp.asarray(qp2['h_u']))
    s1.solve()
    assert int(np.asarray(s1.result.info.status_value)[0]) == Status.CUPIQP_SOLVED.value
    np.testing.assert_allclose(cp.asnumpy(s1.result.x[0]), x_ref, atol=2e-5, rtol=1e-4)

    # The now-inactive row's upper dual must be exactly 0; the now-active row's
    # must be non-zero (its bound binds).
    z_u = cp.asnumpy(s1.result.z_u[0])
    assert z_u[1] == 0.0
    assert abs(z_u[0]) > 1e-3


def test_sparse_batched_update_toggles_active_inequality_rows():
    """Batched analogue: per-element bound updates that flip active rows must
    match independent fresh setups (different toggle per batch element)."""
    # Each element starts with a different inactive row and ends with a
    # different one, so the per-batch active masks genuinely differ.
    patt1 = [[False, True, True], [True, False, True], [True, True, False]]
    patt2 = [[True, False, True], [True, True, False], [False, True, True]]
    qps1 = [_box_qp(p) for p in patt1]
    qps2 = [_box_qp(p) for p in patt2]
    B = len(patt1)
    n = 3

    refs = []
    for q in qps2:
        sr = _setup_sparse(q)
        sr.solve()
        refs.append(cp.asnumpy(sr.result.x[0]))

    s = SparseSolver()
    s.settings.eps_abs = 1e-10
    s.settings.eps_rel = 1e-10
    s.settings.verbose = False
    _setup_batched(s, qps1, blocks=('P', 'c', 'G', 'h_l', 'h_u', 'x_l', 'x_u'))
    s.solve()
    s.update(
        h_l=cp.asarray(np.stack([q['h_l'] for q in qps2])),
        h_u=cp.asarray(np.stack([q['h_u'] for q in qps2])),
    )
    s.solve()
    for i in range(B):
        np.testing.assert_allclose(cp.asnumpy(s.result.x[i]), refs[i], atol=2e-5, rtol=1e-4)
