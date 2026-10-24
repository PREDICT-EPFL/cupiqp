"""The dense QP solver of the JAX interface."""

from __future__ import annotations

import threading
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from jax.custom_batching import custom_vmap

from ...settings import Settings
from .primitives import (
    DATA_FIELDS,
    DATA_NDIM,
    DenseSignature,
    INFO_FIELDS,
    RESULT_FIELDS,
    get_dense_primitive,
    settings_key,
)
from .types import QPSolution


def _to_warp_dtype(dtype: Any):
    """``wp.float32`` or ``wp.float64`` for a JAX / NumPy floating dtype."""
    if dtype is wp.float32 or dtype is wp.float64:
        return dtype
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        np_dtype = None
    if np_dtype == np.float32:
        return wp.float32
    if np_dtype == np.float64:
        return wp.float64
    raise TypeError(f"dtype must be float32 or float64; got {dtype!r}.")


class DenseSolver:
    r"""A dense convex QP as a pure JAX function, solved on the GPU by the
    core :class:`cupiqp.DenseSolver`. This is ``cupiqp.jax.DenseSolver``: it
    takes and returns JAX arrays, and :meth:`solve` is a pure function.

    :meth:`solve` solves

    $$
    \begin{aligned}
    \min_{x}\quad & \tfrac{1}{2}\, x^\top P x + c^\top x \\
    \text{s.t.}\quad & A x = b, \\
    & h_l \le G x \le h_u, \\
    & x_l \le x \le x_u,
    \end{aligned}
    $$

    and returns a :class:`~cupiqp.jax.QPSolution` of JAX arrays. The call is
    a pure function of its array arguments: it works eagerly, under
    ``jax.jit`` and under ``jax.vmap``, and keeps no state the caller can
    observe. The solve itself runs the core solver through an XLA custom
    call on the GPU; nothing is re-implemented in ``jax.numpy``.

    **What is fixed where.** The solver object holds the floating-point
    dtype and the :attr:`settings`. Everything else is read from the arrays
    of each call: the dimensions, which constraint blocks are given, and the
    batch size. Under ``jax.jit`` these are static shapes, so a function
    that calls :meth:`solve` is compiled once per distinct problem
    structure, as any JAX function is. Which entries of a bound are infinite
    is *not* structure: it may change from call to call without
    recompilation. Settings are read at each call; like any non-array value,
    a function compiled by ``jax.jit`` keeps the settings it was traced
    with, so set them before the first call.

    **Batching.** Every argument is either *shared*, in the shape of one
    problem (``P`` of shape ``(n, n)``, ``c`` of shape ``(n,)``, ...), or
    *batched*, with a leading axis of size ``B`` (``(B, n, n)``,
    ``(B, n)``, ...). The batched arguments must agree on ``B``; shared
    arguments are used by every problem of the batch without being copied
    ``B`` times by you. If every argument is shared, one problem is solved
    and the solution has no batch axis. ``jax.vmap`` over the operator
    produces the same single batched solve: mapped arguments become batched,
    unmapped ones stay shared. Mapping over an argument that already has a
    batch axis (nested batching) is not supported.

    **Gradients** are not supported yet; ``jax.grad`` through :meth:`solve`
    raises.

    **Failures do not raise.** A problem that does not converge, or is
    detected infeasible, is reported in ``solution.status`` with a code of
    :class:`cupiqp.Status`; the other fields hold the last iterate. Read
    ``status`` where you need it, for instance ``jnp.where(status == 0, ...)``.

    Parameters
    ----------
    dtype : jax.numpy.float64 or jax.numpy.float32, default: float64
        Floating-point precision of the solve. All array arguments must
        have this dtype; the operator never casts. ``float64`` requires
        ``jax.config.update("jax_enable_x64", True)`` before any array is
        created.
    settings : cupiqp.Settings, optional
        Solver configuration; by default the defaults of the core solver
        for ``dtype``. Also available, and mutable, as :attr:`settings`.
        ``enable_grad`` is ignored.

    Examples
    --------
    One problem, then the same structure under ``jit`` with a batch of
    right-hand sides:

    ```python
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import cupiqp
    from cupiqp.jax import DenseSolver

    solver = DenseSolver()
    solver.settings.eps_abs = 1e-6
    solver.settings.eps_rel = 1e-6

    P = jnp.eye(2)
    c = jnp.array([-1.0, -4.0])
    G = jnp.array([[1.0, 1.0]])
    h_u = jnp.array([1.0])
    h_l = jnp.array([-jnp.inf])            # one-sided row

    sol = solver.solve(P, c, G=G, h_l=h_l, h_u=h_u)   # sol.x has shape (2,)
    print(cupiqp.Status(int(sol.status)).name)

    @jax.jit
    def solve_many(cs):                    # cs: (B, 2), everything else shared
        return solver.solve(P, cs, G=G, h_l=h_l, h_u=h_u).x

    xs = solve_many(jnp.stack([c, 2 * c]))  # (2, 2), one custom call
    xs2 = jax.vmap(lambda c: solver.solve(P, c, G=G, h_l=h_l, h_u=h_u).x)(jnp.stack([c, 2 * c]))
    ```

    Notes
    -----
    The solver object owns no GPU memory. For every distinct problem signature
    (dtype, dimensions, given blocks, batch layout, settings) the interface
    keeps one core solver per device alive in a process-wide cache, so that
    repeated calls reuse its allocations and factorization workspace;
    :func:`cupiqp.jax.clear_cache` releases them. Calls of the same
    signature are serialized.

    JAX reserves most of the GPU memory when its backend starts, which can
    leave the core solver unable to create its cuSOLVER or cuDSS handles.
    Set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (or a small
    ``XLA_PYTHON_CLIENT_MEM_FRACTION``) in the environment before JAX is
    initialized.

    Every call copies its inputs into the solver and its results into
    JAX-owned outputs. Both copies run on the GPU in stream order; the
    interface never waits on the host.
    """

    __slots__ = ("_dtype", "_settings")

    def __init__(self, dtype: Any = jnp.float64, settings: Settings | None = None) -> None:
        wp_dtype = _to_warp_dtype(dtype)
        if wp_dtype is wp.float64 and not jax.config.jax_enable_x64:
            raise RuntimeError(
                "DenseSolver with dtype float64 requires 64-bit JAX arrays: call "
                "jax.config.update('jax_enable_x64', True) before creating any "
                "array, or use dtype=jnp.float32."
            )
        self._dtype = wp_dtype
        self.settings = Settings.for_dtype(wp_dtype) if settings is None else settings

    @property
    def dtype(self) -> np.dtype:
        """The floating dtype of every argument and result."""
        return np.dtype(wp.dtype_to_numpy(self._dtype))

    @property
    def settings(self) -> Settings:
        """Solver configuration (a ``Settings`` dataclass), read at every
        call of :meth:`solve`. Change its fields before a solve, e.g.
        ``solver.settings.max_iter = 50``; a function compiled by ``jax.jit``
        keeps the settings it was traced with."""
        return self._settings

    @settings.setter
    def settings(self, value: Settings) -> None:
        if not isinstance(value, Settings):
            raise TypeError(f"settings must be a cupiqp.Settings; got {type(value).__name__}.")
        if value.dtype is not self._dtype:
            raise ValueError(
                f"settings.dtype is {value.dtype.__name__} but the solver dtype is "
                f"{self._dtype.__name__}; build the settings with Settings.for_dtype()."
            )
        self._settings = value

    def __repr__(self) -> str:
        return f"DenseSolver(dtype={self.dtype})"

    def solve(
        self,
        P: jax.Array,
        c: jax.Array,
        A: jax.Array | None = None,
        b: jax.Array | None = None,
        G: jax.Array | None = None,
        h_u: jax.Array | None = None,
        h_l: jax.Array | None = None,
        x_u: jax.Array | None = None,
        x_l: jax.Array | None = None,
    ) -> QPSolution:
        """Solve the QP (or the batch of QPs) given by the arrays.

        Parameters
        ----------
        P : jax.Array
            Quadratic cost, ``(n, n)`` or ``(B, n, n)``, symmetric positive
            semidefinite. Required.
        c : jax.Array
            Linear cost, ``(n,)`` or ``(B, n)``. Required.
        A, b : jax.Array, optional
            Equality constraints ``A x = b``: ``A`` of shape ``(p, n)`` or
            ``(B, p, n)``, ``b`` of shape ``(p,)`` or ``(B, p)``. Give both
            or neither.
        G, h_l, h_u : jax.Array, optional
            Inequalities ``h_l <= G x <= h_u``: ``G`` of shape ``(m, n)`` or
            ``(B, m, n)``, bounds ``(m,)`` or ``(B, m)``. At least one bound
            is required with ``G``; a bound that is not given is absent
            (its multipliers have zero width). Use ``-inf`` / ``+inf``
            entries for rows that are one-sided in a given bound.
        x_l, x_u : jax.Array, optional
            Box bounds ``x_l <= x <= x_u``, ``(n,)`` or ``(B, n)``.

        Returns
        -------
        QPSolution
            The solution, with a leading batch axis if any argument was
            batched.

        Raises
        ------
        TypeError
            If an argument does not have the operator dtype.
        ValueError
            If an argument has the wrong rank or shape, the batched
            arguments disagree on ``B``, or the blocks are inconsistent
            (``A`` without ``b``, bounds without ``G``, ``G`` without any
            bound).
        """
        values = (P, c, A, b, G, h_u, h_l, x_u, x_l)
        given = {name: value for name, value in zip(DATA_FIELDS, values) if value is not None}
        return _solve_dense(self._dtype, settings_key(self._settings), given)


# ---------------------------------------------------------------------------
# Signature detection and the jitted, vmap-aware function of each signature
# ---------------------------------------------------------------------------

def _dense_signature(wp_dtype, settings: tuple, given: dict[str, Any]) -> DenseSignature:
    """Validate the arrays of one call and describe its static structure."""
    if ("A" in given) != ("b" in given):
        raise ValueError("A and b must be given together.")
    if "G" in given and not ("h_l" in given or "h_u" in given):
        raise ValueError("G requires at least one of h_l and h_u.")
    if "G" not in given and ("h_l" in given or "h_u" in given):
        raise ValueError("h_l and h_u require G.")

    expected = np.dtype(wp.dtype_to_numpy(wp_dtype))
    batched = {}
    for name, value in given.items():
        dtype = getattr(value, "dtype", None)
        shape = getattr(value, "shape", None)
        if dtype is None or shape is None:
            raise TypeError(f"{name} must be a JAX array; got {type(value).__name__}.")
        if np.dtype(dtype) != expected:
            raise TypeError(
                f"{name} has dtype {np.dtype(dtype)}; this operator uses {expected}. "
                f"The operator never casts: convert {name} with .astype() first."
            )
        base = DATA_NDIM[name]
        if len(shape) == base:
            batched[name] = False
        elif len(shape) == base + 1:
            batched[name] = True
        elif len(shape) == base + 2:
            raise ValueError(
                f"{name} has {len(shape)} dimensions: it carries two batch axes. "
                "Nested batching (jax.vmap over an argument that already has a "
                "batch axis) is not supported; flatten the batch axes first."
            )
        else:
            kind = "matrix" if base == 2 else "vector"
            raise ValueError(
                f"{name} must be a {kind}: {base}-D for one problem, or {base + 1}-D "
                f"with a leading batch axis; got {len(shape)}-D."
            )

    batch_sizes = {int(given[name].shape[0]) for name, is_batched in batched.items() if is_batched}
    if len(batch_sizes) > 1:
        detail = ", ".join(
            f"{name}: {int(given[name].shape[0])}" for name, is_batched in batched.items() if is_batched
        )
        raise ValueError(f"the batched arguments disagree on the batch size ({detail}).")
    batch_size = batch_sizes.pop() if batch_sizes else 1
    if batch_size < 1:
        raise ValueError("the batch size must be at least 1.")

    def problem_shape(name: str) -> tuple[int, ...]:
        shape = tuple(int(s) for s in given[name].shape)
        return shape[1:] if batched[name] else shape

    n = problem_shape("c")[0]
    if n < 1:
        raise ValueError("c must have at least one entry.")
    p = problem_shape("b")[0] if "b" in given else 0
    m = problem_shape("G")[0] if "G" in given else 0
    expected_shapes = {
        "P": (n, n), "c": (n,), "A": (p, n), "b": (p,), "G": (m, n),
        "h_u": (m,), "h_l": (m,), "x_u": (n,), "x_l": (n,),
    }
    for name in given:
        if problem_shape(name) != expected_shapes[name]:
            batch = "(B, " if batched[name] else "("
            want = batch + ", ".join(str(s) for s in expected_shapes[name]) + ")"
            raise ValueError(
                f"{name} has shape {tuple(given[name].shape)}; expected {want} "
                f"with n={n}, p={p}, m={m}."
            )

    fields = tuple(name for name in DATA_FIELDS if name in given)
    return DenseSignature(
        dtype=wp_dtype, n=n, m=m, p=p, fields=fields,
        batched=tuple(batched[name] for name in fields),
        batch_size=batch_size, settings=settings,
    )


def _assemble(signature: DenseSignature, raw: dict[str, jax.Array]) -> QPSolution:
    """Build the public solution from the raw FFI outputs: absent blocks
    become zero-width arrays and an unbatched call loses the batch axis."""
    B = signature.batch_size
    float_dtype = wp.dtype_to_numpy(signature.dtype)
    values = {}
    for name in RESULT_FIELDS:
        if name in raw:
            values[name] = raw[name]
        else:
            values[name] = jnp.zeros((B, 0), dtype=float_dtype)
    for name in INFO_FIELDS:
        values[name] = raw[name]
    if signature.unbatched:
        values = {name: value[0] for name, value in values.items()}
    return QPSolution(**values)


_FUNCTIONS_LOCK = threading.Lock()
_FUNCTIONS: dict[DenseSignature, Callable[..., QPSolution]] = {}


def _solver_function(signature: DenseSignature) -> Callable[..., QPSolution]:
    """The jitted function of ``signature``: takes the present data
    arguments positionally, returns a ``QPSolution``. Under ``jax.vmap`` it
    becomes one batched call of the same operator instead of a loop."""
    with _FUNCTIONS_LOCK:
        function = _FUNCTIONS.get(signature)
    if function is not None:
        return function

    primitive = get_dense_primitive(signature)
    output_names = primitive.output_names

    @custom_vmap
    def solve(*arrays):
        raw = primitive(*arrays)
        return _assemble(signature, dict(zip(output_names, raw)))

    @solve.def_vmap
    def solve_vmap_rule(axis_size, in_batched, *arrays):
        if not signature.unbatched:
            raise NotImplementedError(
                "jax.vmap over a DenseSolver.solve call whose arguments already carry a "
                "batch axis (nested batching) is not supported; pass the batch "
                "to a single call instead."
            )
        given = dict(zip(signature.fields, arrays))
        solution = _solve_dense(signature.dtype, signature.settings, given)
        return solution, QPSolution(*([True] * len(QPSolution._fields)))

    function = jax.jit(solve)
    with _FUNCTIONS_LOCK:
        return _FUNCTIONS.setdefault(signature, function)


def _solve_dense(wp_dtype, settings: tuple, given: dict[str, Any]) -> QPSolution:
    signature = _dense_signature(wp_dtype, settings, given)
    function = _solver_function(signature)
    return function(*(given[name] for name in signature.fields))
