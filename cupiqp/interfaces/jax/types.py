"""Public JAX-facing result type of the cuPIQP JAX interface."""

from __future__ import annotations

from typing import NamedTuple

import jax


class QPSolution(NamedTuple):
    """Solution of one QP, or of a batch of QPs, returned by the ``solve``
    method of a JAX solver such as :class:`~cupiqp.jax.DenseSolver`.

    Every field is a ``jax.Array`` owned by JAX: it can be used in further
    JAX work, inside or outside ``jax.jit``, and is never overwritten by a
    later solve. For a batched call the leading axis of every field is the
    batch axis of size ``B``; for an unbatched call there is no leading
    axis, and the shapes below lose their ``B``.

    Dual and slack variables of a constraint block that was not passed to
    the call have zero width (``(B, 0)``) rather than being ``None``, so the
    structure of the tuple is the same for every problem. The width of a
    block does not depend on which of its bound entries are infinite.

    Attributes
    ----------
    x : jax.Array
        Primal solution, ``(B, n)``.
    y : jax.Array
        Multipliers of the equality constraints ``A x = b``, ``(B, p)``.
    z_l, z_u : jax.Array
        Multipliers of the lower and upper inequality bounds
        ``h_l <= G x <= h_u``, ``(B, m)`` each.
    z_bl, z_bu : jax.Array
        Multipliers of the lower and upper box bounds ``x_l <= x <= x_u``,
        ``(B, n)`` each.
    s_l, s_u, s_bl, s_bu : jax.Array
        Slacks of the corresponding inequality and box bounds.
    status : jax.Array
        One int32 code of :class:`cupiqp.Status` per problem, ``(B,)``.
        ``cupiqp.Status.CUPIQP_SOLVED`` (0) means converged to tolerance. A
        failed solve never raises; read this field instead::

            solved = sol.status == cupiqp.Status.CUPIQP_SOLVED   # (B,) bool
    iterations : jax.Array
        Number of interior-point iterations per problem, ``(B,)`` int32.
    primal_res, dual_res : jax.Array
        Final primal and dual residuals per problem, ``(B,)``, in the
        solver dtype.
    """

    x: jax.Array
    y: jax.Array
    z_l: jax.Array
    z_u: jax.Array
    z_bl: jax.Array
    z_bu: jax.Array
    s_l: jax.Array
    s_u: jax.Array
    s_bl: jax.Array
    s_bu: jax.Array
    status: jax.Array
    iterations: jax.Array
    primal_res: jax.Array
    dual_res: jax.Array
