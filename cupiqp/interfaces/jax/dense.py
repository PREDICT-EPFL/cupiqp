"""The dense QP solver of the JAX interface."""

from __future__ import annotations

import jax

from ...dense.dense_solver import DenseSolver as _CoreDenseSolver
from .base import JaxSolverBase, check_constraint_blocks
from .types import QPSolution


class DenseSolver(JaxSolverBase):
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
    and the solution has no batch axis. ``jax.vmap`` over :meth:`solve`
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
        have this dtype; the solver never casts. ``float64`` requires
        ``jax.config.update("jax_enable_x64", True)`` before any array is
        created.
    settings : cupiqp.Settings, optional
        Solver configuration; by default the defaults of the core solver
        for ``dtype``. Also available, and mutable, as :attr:`settings`.
        ``preconditioner_reuse_on_update`` and ``verbose`` must stay
        ``False``; gradient settings are ignored.

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
    (dtype, dimensions, given blocks, batch layout, forward settings) the
    interface keeps one core solver alive in a process-wide
    cache, so that repeated calls reuse its allocations and factorization
    workspace. The first call of a signature sets that solver up and is
    slow; keep shapes, batch sizes and settings stable, and warm up with one
    call. :func:`cupiqp.jax.release_solver_memory` frees the solvers.
    Calls of the same signature are serialized.

    Two settings are rejected because they would break the pure-function
    contract: ``preconditioner_reuse_on_update`` (results would depend on
    earlier calls) and ``verbose`` (printing is a side effect XLA may skip).
    Settings used only by the core's ``backward()`` are ignored.

    JAX reserves most of the GPU memory when its backend starts, which can
    leave the core solver unable to create its cuSOLVER or cuDSS handles.
    Set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (or a small
    ``XLA_PYTHON_CLIENT_MEM_FRACTION``) in the environment before JAX is
    initialized.

    Every call copies its inputs into the solver and its results into
    JAX-owned outputs. Both copies run on the GPU in stream order; the
    interface never waits on the host.
    """

    _core_class = _CoreDenseSolver

    def _input_rank(self, name: str) -> int:
        return 2 if name in ("P", "A", "G") else 1

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
            If an argument does not have the solver dtype.
        ValueError
            If a setting is not supported by the JAX interface, an argument
            has the wrong rank or shape, the batched arguments disagree on
            ``B``, or the blocks are inconsistent (``A`` without ``b``,
            bounds without ``G``, ``G`` without any bound).
        """
        arguments = dict(P=P, c=c, A=A, b=b, G=G, h_u=h_u, h_l=h_l, x_u=x_u, x_l=x_l)
        given = {name: value for name, value in arguments.items() if value is not None}
        check_constraint_blocks(A is not None, b is not None, G is not None, h_l is not None, h_u is not None)
        n = int(c.shape[-1])
        p = int(b.shape[-1]) if b is not None else 0
        m = int(G.shape[-2]) if G is not None else 0
        return self._solve(given, n, p, m, {"P": (n, n), "A": (p, n), "G": (m, n)})
