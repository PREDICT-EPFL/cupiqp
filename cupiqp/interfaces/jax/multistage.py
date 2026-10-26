"""The multistage QP solver of the JAX interface."""

from __future__ import annotations

from typing import Any, Mapping

import jax
import warp as wp

from ...multistage.multistage_solver import MultistageSolver as _CoreMultistageSolver
from .base import JaxSolverBase, check_constraint_blocks
from .types import QPSolution


def _split_pair(name: str, value: Any) -> tuple[Any, Any]:
    """``(diag, offdiag)`` of the block matrix ``name``; ``None`` passes through."""
    if value is None:
        return None, None
    if not (isinstance(value, (tuple, list)) and len(value) == 2):
        raise TypeError(
            f"{name} must be a (diag, offdiag) tuple of JAX arrays; offdiag may be "
            f"None (no off-diagonal blocks). Got {type(value).__name__}."
        )
    if value[0] is None:
        raise ValueError(f"{name} diag is required when {name} is given.")
    return value[0], value[1]


def _rows_per_block(name: str, diag: Any, N: int, d: int) -> int:
    """Rows per block of the constraint matrix ``name`` from its diagonal
    blocks, ``(N, r, d)`` or ``(B, N, r, d)``."""
    shape = tuple(int(s) for s in getattr(diag, "shape", ()))
    if len(shape) not in (3, 4) or shape[-3] != N or shape[-1] != d:
        raise ValueError(
            f"{name} diag must have shape ({N}, r, {d}) or (B, {N}, r, {d}); got {shape}."
        )
    return shape[-2]


def _flatten(value: Any, num_blocks: int, rows: int) -> Any:
    """``value`` in the flat layout, ``(k,)`` or ``(B, k)`` with
    ``k = num_blocks * rows``: the block layout ``(num_blocks, rows)`` or
    ``(B, num_blocks, rows)`` is reshaped (free in JAX); anything else is
    returned as is for the shape check to report."""
    shape = tuple(int(s) for s in getattr(value, "shape", ()))
    if len(shape) == 3 and shape[1:] == (num_blocks, rows):
        return value.reshape(shape[0], num_blocks * rows)
    if len(shape) == 2 and shape == (num_blocks, rows):
        return value.reshape(num_blocks * rows)
    return value


class MultistageSolver(JaxSolverBase):
    r"""A multistage (block-structured) convex QP as a pure JAX function,
    solved on the GPU by the core :class:`cupiqp.MultistageSolver`. This is
    ``cupiqp.jax.MultistageSolver``: it takes and returns JAX arrays, and
    :meth:`solve` is a pure function.

    It solves the same QP as :class:`DenseSolver`,

    $$
    \begin{aligned}
    \min_{x}\quad & \tfrac{1}{2}\, x^\top P x + c^\top x \\
    \text{s.t.}\quad & A x = b, \\
    & h_l \le G x \le h_u, \\
    & x_l \le x \le x_u,
    \end{aligned}
    $$

    where the matrices are block-structured as in optimal control: ``N``
    stages of ``d`` variables (``n = N * d``), ``P`` symmetric
    block-tridiagonal, ``A`` and ``G`` block lower-bidiagonal. The arguments
    of :meth:`solve` are exactly those of the core ``setup()``: each matrix
    is a ``(diag, offdiag)`` pair of block arrays, and each vector is in the
    block layout ``(num_blocks, rows)`` or the flat layout ``(k,)``. The
    block sizes, and which matrices have off-diagonal blocks, are read from
    the shapes of each call; under ``jax.jit`` they are static, like the
    dimensions of :class:`DenseSolver`.

    Everything else is as for :class:`DenseSolver`: arrays are shared or
    batched (with a leading batch axis), the dtype and :attr:`settings` are
    held by the solver object, :meth:`solve` works eagerly and under
    ``jax.jit`` and ``jax.vmap``, failures are reported in
    ``solution.status``, and gradients flow to every array argument,
    including each block array. The solution
    is in the flat layout, as in the core: ``x`` has shape ``(B, n)``; use
    ``x.reshape(B, N, d)`` for the per-stage view.

    Parameters
    ----------
    dtype : jax.numpy.float64 or jax.numpy.float32, default: float64
        Floating-point precision of the solve; see :class:`DenseSolver`.
    settings : cupiqp.Settings, optional
        Solver configuration; see :class:`DenseSolver`.

    Examples
    --------
    ```python
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from cupiqp.jax import MultistageSolver

    N, d = 10, 3
    P_diag = jnp.broadcast_to(2.0 * jnp.eye(d), (N, d, d))
    P_offdiag = jnp.broadcast_to(-0.5 * jnp.eye(d), (N - 1, d, d))

    solver = MultistageSolver()

    @jax.jit
    def solve_many(cs):                                  # (B, N, d), the rest shared
        return solver.solve(P=(P_diag, P_offdiag), c=cs,
                            x_l=-jnp.ones((N, d)), x_u=jnp.ones((N, d))).x

    xs = solve_many(jnp.ones((8, N, d)))                 # (8, N * d)
    ```

    See Also
    --------
    DenseSolver : the same interface for dense problems, with the full
        description of batching, settings, caching and memory.
    """

    _core_class = _CoreMultistageSolver

    def _input_rank(self, name: str) -> int:
        return 3 if name.endswith(("_diag", "_offdiag")) else 1

    def _setup_arguments(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """The core takes each matrix as a ``(diag, offdiag)`` pair, with
        ``None`` for absent off-diagonal blocks; the vectors are flat."""
        arguments = {}
        for name, value in inputs.items():
            if name.endswith("_diag"):
                block = name[: -len("_diag")]
                arguments[block] = (value, inputs.get(f"{block}_offdiag"))
            elif not name.endswith("_offdiag"):
                arguments[name] = value
        return arguments

    _update_arguments = _setup_arguments

    def _gradient_arrays(self, gradients: Any, inputs: tuple[str, ...]) -> dict[str, wp.array]:
        """Block gradients by their FFI names; vector gradients flat."""
        arrays = {}
        for name in inputs:
            if name.endswith(("_diag", "_offdiag")):
                arrays[name] = getattr(gradients, name)
            else:
                value = getattr(gradients, name)
                arrays[name] = value.reshape((value.shape[0], -1)) if value.ndim == 3 else value
        return arrays

    def solve(
        self,
        P: tuple[jax.Array, jax.Array | None],
        c: jax.Array,
        A: tuple[jax.Array, jax.Array | None] | None = None,
        b: jax.Array | None = None,
        G: tuple[jax.Array, jax.Array | None] | None = None,
        h_u: jax.Array | None = None,
        h_l: jax.Array | None = None,
        x_u: jax.Array | None = None,
        x_l: jax.Array | None = None,
    ) -> QPSolution:
        """Solve the QP (or the batch of QPs) given by the arrays.

        Every array is given for one problem, shared by the whole batch, or
        with a leading batch axis ``B``. The shapes below are for one problem.

        Parameters
        ----------
        P : tuple of jax.Array
            ``(P_diag, P_offdiag)``: diagonal blocks ``(N, d, d)`` and lower
            off-diagonal blocks ``(N - 1, d, d)`` (block ``(k + 1, k)``) of
            the symmetric block-tridiagonal cost. ``P_offdiag`` may be
            ``None``: P is block diagonal. Required.
        c : jax.Array
            Linear cost, ``(N, d)`` or flat ``(N * d,)``. Required.
        A, b : optional
            Equality constraints ``A x = b``: ``A = (A_diag, A_offdiag)``,
            both ``(N, r, d)``, so ``A`` has ``N + 1`` block rows, and ``b``
            of shape ``(N + 1, r)`` or flat. ``A_offdiag`` may be ``None``
            (per-stage equalities, ``N`` block rows): ``b`` is then
            ``(N, r)`` or flat. Give both or neither.
        G, h_l, h_u : optional
            Inequalities ``h_l <= G x <= h_u``: ``G = (G_diag, G_offdiag)``
            laid out like ``A``, bounds like ``b``. At least one bound is
            required with ``G``. Use ``-inf`` / ``+inf`` entries for
            one-sided rows.
        x_l, x_u : jax.Array, optional
            Box bounds, ``(N, d)`` or flat ``(N * d,)``.

        Returns
        -------
        QPSolution
            The solution in the flat layout, with a leading batch axis if any
            argument was batched.

        Raises
        ------
        TypeError
            If a matrix is not a ``(diag, offdiag)`` tuple or an argument
            does not have the solver dtype.
        ValueError
            If the block shapes are inconsistent, an argument has the wrong
            rank or shape, the batched arguments disagree on ``B``, or the
            blocks are inconsistent (``A`` without ``b``, bounds without
            ``G``, ``G`` without any bound).
        """
        P_diag, P_offdiag = _split_pair("P", P)
        if P_diag is None:
            raise TypeError("MultistageSolver.solve requires P.")
        A_diag, A_offdiag = _split_pair("A", A)
        G_diag, G_offdiag = _split_pair("G", G)

        check_constraint_blocks(A is not None, b is not None, G is not None, h_l is not None, h_u is not None)
        P_shape = tuple(int(s) for s in getattr(P_diag, "shape", ()))
        if len(P_shape) not in (3, 4) or P_shape[-1] != P_shape[-2]:
            raise ValueError(f"P diag must have shape (N, d, d) or (B, N, d, d); got {P_shape}.")
        N, d = P_shape[-3], P_shape[-1]
        if N < 1:
            raise ValueError("P diag must contain at least one block.")
        r_A = _rows_per_block("A", A_diag, N, d) if A_diag is not None else 0
        r_G = _rows_per_block("G", G_diag, N, d) if G_diag is not None else 0
        # A and G have N + 1 block rows with off-diagonal blocks, N without.
        rows_A = N + int(A_offdiag is not None)
        rows_G = N + int(G_offdiag is not None)

        matrix_shapes = {"P_diag": (N, d, d), "P_offdiag": (N - 1, d, d),
                         "A_diag": (N, r_A, d), "A_offdiag": (N, r_A, d),
                         "G_diag": (N, r_G, d), "G_offdiag": (N, r_G, d)}
        vector_blocks = {"c": (N, d), "x_u": (N, d), "x_l": (N, d),
                         "b": (rows_A, r_A), "h_u": (rows_G, r_G), "h_l": (rows_G, r_G)}
        arguments = {
            "P_diag": P_diag, "P_offdiag": P_offdiag, "c": c,
            "A_diag": A_diag, "A_offdiag": A_offdiag, "b": b,
            "G_diag": G_diag, "G_offdiag": G_offdiag, "h_u": h_u, "h_l": h_l,
            "x_u": x_u, "x_l": x_l,
        }
        given = {}
        for name, value in arguments.items():
            if value is not None:
                given[name] = _flatten(value, *vector_blocks[name]) if name in vector_blocks else value
        return self._compiled(given, N * d, rows_A * r_A, rows_G * r_G, matrix_shapes).function(*given.values())
