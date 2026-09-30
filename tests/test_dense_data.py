"""Tests for ``DenseData``."""
import cupy as cp
import warp as wp
import numpy as np
import pytest

from cupiqp.dense.dense_data import DenseData
from cupiqp.typedef import PIQP_INF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _w(a):
    """Zero-copy Warp view: Data objects take Warp arrays only (the solver converts user arrays)."""
    return wp.array(a, copy=False)


def _dense_data(**kw) -> DenseData:
    """Build a ``DenseData`` from kwargs (``init`` doesn't return self)."""
    d = DenseData()
    d.init(**{k: _w(v) for k, v in kw.items()})
    return d


def _random_batched_qp(
    B: int = 4, n: int = 6, p: int = 3, m: int = 4,
    seed: int = 42, mixed_inf: bool = True,
) -> dict[str, cp.ndarray]:
    """Random batched dense QP data with a consistent bound structure.

    Bound structure (which positions are ±∞) is shared across the batch — that
    is what DenseData requires. When ``mixed_inf`` is False, all bounds are
    finite.
    """
    rng = np.random.default_rng(seed)

    if mixed_inf:
        h_u_finite = rng.random(m) > 0.3 if m > 0 else np.zeros(m, dtype=bool)
        h_l_finite = rng.random(m) > 0.3 if m > 0 else np.zeros(m, dtype=bool)
        x_u_finite = rng.random(n) > 0.3
        x_l_finite = rng.random(n) > 0.3
    else:
        h_u_finite = np.ones(m, dtype=bool)
        h_l_finite = np.ones(m, dtype=bool)
        x_u_finite = np.ones(n, dtype=bool)
        x_l_finite = np.ones(n, dtype=bool)

    # DenseData only stores buffers (no factorization / PD check), so plain
    # random matrices are enough to exercise every code path here.
    Ps, cs, As, bs, Gs = [], [], [], [], []
    h_us, h_ls, x_us, x_ls = [], [], [], []
    for _ in range(B):
        Ps.append(rng.standard_normal((n, n)))
        cs.append(rng.standard_normal(n))
        As.append(rng.standard_normal((p, n)))
        bs.append(rng.standard_normal(p))
        Gs.append(rng.standard_normal((m, n)))
        h_us.append(np.where(h_u_finite, np.abs(rng.standard_normal(m)) + 1.0, np.inf))
        h_ls.append(np.where(h_l_finite, -np.abs(rng.standard_normal(m)) - 1.0, -np.inf))
        x_us.append(np.where(x_u_finite, np.abs(rng.standard_normal(n)) + 1.0, np.inf))
        x_ls.append(np.where(x_l_finite, -np.abs(rng.standard_normal(n)) - 1.0, -np.inf))

    return dict(
        P=cp.array(np.stack(Ps)),
        c=cp.array(np.stack(cs)),
        A=cp.array(np.stack(As)),
        b=cp.array(np.stack(bs)),
        G=cp.array(np.stack(Gs)),
        h_u=cp.array(np.stack(h_us)),
        h_l=cp.array(np.stack(h_ls)),
        x_u=cp.array(np.stack(x_us)),
        x_l=cp.array(np.stack(x_ls)),
    )


# (B, n, p, m) tuples: one single-problem and one batched case, each with all
# constraint types. Zero-constraint edge cases have dedicated tests below.
BNPM_SHAPES: list[tuple[int, int, int, int]] = [
    (1, 4, 1, 2),     # single problem
    (4, 6, 3, 4),     # batched
]


# ===========================================================================
# Construction & shapes
# ===========================================================================

@pytest.mark.parametrize("B,n,p,m", BNPM_SHAPES)
def test_construction_shapes(B: int, n: int, p: int, m: int) -> None:
    qp = _random_batched_qp(B=B, n=n, p=p, m=m)
    data = _dense_data(**qp)
    assert data.P.shape == (B, n, n)
    assert data.c.shape == (B, n)
    assert data.A.shape == (B, p, n)
    assert data.b.shape == (B, p)
    assert data.G.shape == (B, m, n)
    assert data.h_u.shape == (B, m)
    assert data.h_l.shape == (B, m)
    assert data.x_u.shape == (B, n)
    assert data.x_l.shape == (B, n)


def test_dtype_is_float64() -> None:
    data = _dense_data(**_random_batched_qp())
    for name in ("P", "c", "A", "b", "G", "h_u", "h_l", "x_u", "x_l"):
        assert cp.asarray(getattr(data, name)).dtype == cp.float64
    assert data.dtype is wp.float64


def test_2d_inputs_promoted_to_3d() -> None:
    """``init`` accepts un-batched 2-D / 1-D inputs and treats them as B=1."""
    n, p, m = 5, 2, 3
    rng = np.random.default_rng(7)
    data = _dense_data(
        P=cp.array(rng.standard_normal((n, n))),
        c=cp.array(rng.standard_normal(n)),
        A=cp.array(rng.standard_normal((p, n))),
        b=cp.array(rng.standard_normal(p)),
        G=cp.array(rng.standard_normal((m, n))),
        h_u=cp.ones(m),
        h_l=-cp.ones(m),
    )
    assert data.batch_size == 1
    assert data.P.shape == (1, n, n)
    assert data.c.shape == (1, n)
    assert data.A.shape == (1, p, n)


def test_no_equality_constraints() -> None:
    qp = _random_batched_qp()
    del qp['A'], qp['b']
    data = _dense_data(**qp)
    assert data.p == 0
    assert data.A.shape == (4, 0, 6)
    assert data.b.shape == (4, 0)


def test_no_inequality_constraints() -> None:
    qp = _random_batched_qp()
    del qp['G'], qp['h_u'], qp['h_l']
    data = _dense_data(**qp)
    assert data.m == 0
    assert data.G.shape == (4, 0, 6)


def test_no_variable_bounds() -> None:
    """Omitting x_l / x_u creates no box block: zero-width storage and views."""
    qp = _random_batched_qp()
    del qp['x_u'], qp['x_l']
    data = _dense_data(**qp)
    assert data.has_x_l is False
    assert data.has_x_u is False
    assert data.num_xl == 0
    assert data.num_xu == 0
    assert data.x_l.shape == (data.batch_size, 0)
    assert data.x_u.shape == (data.batch_size, 0)
    assert data.finite_mask_xl.shape == (data.batch_size, 0)
    assert data.finite_mask_xu.shape == (data.batch_size, 0)
    # No box bounds => no variable has an active box bound.
    cp.testing.assert_array_equal(data.active_x_bound, cp.zeros((data.batch_size, data.n), dtype=wp.dtype_to_numpy(data.dtype)))


# ===========================================================================
# Bound structure: indices + counts
# ===========================================================================

def test_bound_masks_for_h_u_x_l_x_u() -> None:
    """Finite-bound masks record per-batch activity; counts stay full-length."""
    qp = _random_batched_qp(B=2, n=6, p=1, m=4, seed=55)
    data = _dense_data(**qp)

    expected_hu = cp.asarray(cp.asnumpy(data.h_u) < PIQP_INF, dtype=wp.dtype_to_numpy(data.dtype))
    expected_xl = cp.asarray(cp.asnumpy(data.x_l) > -PIQP_INF, dtype=wp.dtype_to_numpy(data.dtype))
    expected_xu = cp.asarray(cp.asnumpy(data.x_u) < PIQP_INF, dtype=wp.dtype_to_numpy(data.dtype))

    cp.testing.assert_array_equal(data.finite_mask_hu, expected_hu)
    cp.testing.assert_array_equal(data.finite_mask_xl, expected_xl)
    cp.testing.assert_array_equal(data.finite_mask_xu, expected_xu)

    assert data.num_hu == data.m
    assert data.num_xl == data.n
    assert data.num_xu == data.n


def test_num_ineq_sums_all_active_bounds() -> None:
    qp = _random_batched_qp(B=2, n=4, p=1, m=3, seed=99)
    data = _dense_data(**qp)
    assert data.num_ineq == data.num_hl + data.num_hu + data.num_xl + data.num_xu


def test_per_batch_bound_structure_is_allowed() -> None:
    """Batch elements may now have different finite/infinite bound masks."""
    B, n, m = 2, 4, 3
    rng = np.random.default_rng(42)
    P = rng.standard_normal((B, n, n))

    G = cp.array(rng.standard_normal((B, m, n)))
    h_l = np.array([[-1.0, -1.0, -1.0],
                    [-np.inf, -1.0, -1.0]])
    h_u = np.ones((B, m)) * 2.0

    data = _dense_data(
        P=cp.array(P), c=cp.zeros((B, n)),
        G=G, h_u=cp.array(h_u), h_l=cp.array(h_l),
    )
    cp.testing.assert_array_equal(data.finite_mask_hl[:, 0], cp.asarray([1.0, 0.0], dtype=wp.dtype_to_numpy(data.dtype)))


def test_inf_constraints_keep_G_rows_and_use_masks() -> None:
    """Rows with both inequality bounds infinite keep their original G values.

    The finite masks, not destructive G-row zeroing, make those rows inert.
    """
    B, n, m = 2, 4, 3
    rng = np.random.default_rng(42)
    P = rng.standard_normal((B, n, n))

    G = rng.standard_normal((B, m, n))
    h_l = -np.inf * np.ones((B, m))
    h_u = np.inf * np.ones((B, m))
    h_l[:, 1] = -2.0
    h_u[:, 1] = 2.0

    data = _dense_data(
        P=cp.array(P), c=cp.zeros((B, n)),
        G=cp.array(G), h_u=cp.array(h_u), h_l=cp.array(h_l),
    )
    np.testing.assert_allclose(cp.asnumpy(data.G), G)
    cp.testing.assert_array_equal(data.active_G_row, cp.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.dtype_to_numpy(data.dtype)))


# ===========================================================================
# Setters (in-place via ``set_*``)
# ===========================================================================

def _assert_inplace_set(
    data: DenseData,
    name: str,
    new_value: cp.ndarray,
) -> None:
    """Apply ``data.set_<name>(new_value)`` and assert: same buffer ptr,
    content equals new_value."""
    original = getattr(data, name)
    old_ptr = original.ptr
    getattr(data, f"set_{name}")(_w(new_value))
    after = getattr(data, name)
    assert after.ptr == old_ptr, f"{name} pointer changed"
    cp.testing.assert_array_equal(after, new_value)


@pytest.mark.parametrize("field", ["P", "c", "A", "b", "G"])
def test_set_field_inplace(field: str) -> None:
    qp = _random_batched_qp(B=2, n=4, p=2, m=3)
    data = _dense_data(**qp)
    # New value with the same shape as the current field, distinct content.
    current = getattr(data, field)
    new_value = cp.ones_like(cp.asarray(current))
    _assert_inplace_set(data, field, new_value)


@pytest.mark.parametrize("field", ["h_u", "h_l", "x_u", "x_l"])
def test_set_bound_inplace(field: str) -> None:
    """Bound setters mutate in place — only finite slots are exercised here
    so ``check=True`` doesn't complain about a structural mismatch."""
    qp = _random_batched_qp(B=2, n=4, p=1, m=2, mixed_inf=False)
    data = _dense_data(**qp)
    current = getattr(data, field)
    sign = -1.0 if field.endswith("_l") else 1.0
    new_value = sign * (cp.abs(cp.ones_like(cp.asarray(current))) + 5.0)
    _assert_inplace_set(data, field, new_value)


@pytest.mark.parametrize(
    "field, finite_value, infinite_value",
    [
        ("x_l", -1.0, -cp.inf),
        ("x_u", 1.0, cp.inf),
    ],
)
@pytest.mark.parametrize("finite_to_infinite", [True, False])
def test_set_variable_bound_updates_changed_structure(
    field: str,
    finite_value: float,
    infinite_value: float,
    finite_to_infinite: bool,
) -> None:
    qp = _random_batched_qp(B=2, n=4, p=1, m=2, mixed_inf=False)
    if not finite_to_infinite:
        qp[field][:, 0] = infinite_value
    data = _dense_data(**qp)

    new_value = cp.asarray(getattr(data, field)).copy()
    new_value[:, 0] = infinite_value if finite_to_infinite else finite_value
    getattr(data, f"set_{field}")(_w(new_value), check=True)

    cp.testing.assert_array_equal(getattr(data, field), new_value)
    mask = data.finite_mask_xl if field == "x_l" else data.finite_mask_xu
    expected = 0.0 if finite_to_infinite else 1.0
    cp.testing.assert_array_equal(mask[:, 0], cp.full((2,), expected, dtype=wp.dtype_to_numpy(data.dtype)))


@pytest.mark.parametrize(
    "field, finite_value, infinite_value",
    [
        ("h_l", -1.0, -cp.inf),
        ("h_u", 1.0, cp.inf),
    ],
)
@pytest.mark.parametrize("finite_to_infinite", [True, False])
def test_set_inequality_bound_updates_changed_structure(
    field: str,
    finite_value: float,
    infinite_value: float,
    finite_to_infinite: bool,
) -> None:
    """``set_h_l`` / ``set_h_u`` may change finite/infinite masks in place."""
    qp = _random_batched_qp(B=2, n=4, p=1, m=2, mixed_inf=False)
    if not finite_to_infinite:
        qp[field][:, 0] = infinite_value
    data = _dense_data(**qp)

    new_value = cp.asarray(getattr(data, field)).copy()
    new_value[:, 0] = infinite_value if finite_to_infinite else finite_value
    getattr(data, f"set_{field}")(_w(new_value), check=True)

    cp.testing.assert_array_equal(getattr(data, field), new_value)
    mask = data.finite_mask_hl if field == "h_l" else data.finite_mask_hu
    expected = 0.0 if finite_to_infinite else 1.0
    cp.testing.assert_array_equal(mask[:, 0], cp.full((2,), expected, dtype=wp.dtype_to_numpy(data.dtype)))


def test_set_P_wrong_shape_raises() -> None:
    qp = _random_batched_qp(B=2, n=4, p=1, m=2)
    data = _dense_data(**qp)
    with pytest.raises(ValueError, match="P must have shape"):
        data.set_P(_w(cp.zeros((3, 4, 4))))


def test_set_c_wrong_shape_raises() -> None:
    qp = _random_batched_qp(B=2, n=4, p=1, m=2)
    data = _dense_data(**qp)
    with pytest.raises(ValueError, match="c must have shape"):
        data.set_c(_w(cp.zeros((2, 5))))


# ===========================================================================
# Input validation on ``init``
# ===========================================================================

def test_P_not_3d_raises() -> None:
    """1-D ``P`` can't be promoted to 3-D and must fail validation.

    A 2-D ``(n, n)`` ``P`` reshapes to ``(1, n, n)`` and is valid as a
    single-problem QP — see ``test_2d_inputs_promoted_to_3d``.
    """
    with pytest.raises(ValueError, match="P must have shape"):
        _dense_data(P=cp.zeros((4,)), c=cp.zeros((1, 4)))


def test_P_not_square_raises() -> None:
    with pytest.raises(ValueError, match="P must be square"):
        _dense_data(P=cp.zeros((2, 3, 4)), c=cp.zeros((2, 3)))


def test_batch_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="disagree on the batch size"):
        _dense_data(P=cp.zeros((2, 3, 3)), c=cp.zeros((3, 3)))


def test_dim_mismatch_P_c_raises() -> None:
    with pytest.raises(ValueError, match="c must have shape"):
        _dense_data(P=cp.zeros((2, 3, 3)), c=cp.zeros((2, 4)))


def test_G_without_h_raises() -> None:
    """``G`` requires at least one of ``h_l`` / ``h_u``."""
    B, n, m = 2, 3, 2
    P = cp.tile(cp.eye(n), (B, 1, 1))
    with pytest.raises(ValueError, match="h_l or h_u"):
        _dense_data(P=P, c=cp.zeros((B, n)), G=cp.zeros((B, m, n)))


def test_h_without_G_raises() -> None:
    """``h_l`` / ``h_u`` are not meaningful without ``G``."""
    B, n, m = 2, 3, 2
    P = cp.tile(cp.eye(n), (B, 1, 1))
    with pytest.raises(ValueError, match="h_l and h_u must be None"):
        _dense_data(P=P, c=cp.zeros((B, n)), h_u=cp.ones((B, m)))
