"""JAX interface of cuPIQP: QP solves as pure functions for ``jit`` and ``vmap``.

A solver object such as :class:`DenseSolver` holds the dtype and the solver
settings. Its ``solve`` method takes JAX arrays, solves the QP on the GPU through an XLA custom
call and returns a :class:`QPSolution` of JAX arrays. The call behaves like
any pure JAX function: it can be traced by ``jax.jit``, batched by
``jax.vmap`` (one native batched solve, not a loop), and used inside larger
JAX programs without leaving the device::

    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from cupiqp.jax import DenseSolver

    solver = DenseSolver()
    solver.settings.max_iter = 50

    @jax.jit
    def control(c):                           # c: (B, n), P and bounds shared
        sol = solver.solve(P, c, x_l=x_l, x_u=x_u)
        return jnp.where(sol.status[:, None] == 0, sol.x, 0.0)

Arguments are shared (single-problem shape) or batched (leading batch axis),
exactly as for the core solvers. Gradients are not available yet.

Importing this module imports JAX. The interface keeps one core solver per
distinct problem signature and device; :func:`clear_cache` releases them.

JAX reserves most of the GPU memory for itself when its backend starts,
which can leave the core solver unable to create its cuSOLVER or cuDSS
handles. Set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (or a small
``XLA_PYTHON_CLIENT_MEM_FRACTION``) in the environment before JAX is
initialized.
"""

from .dense import DenseSolver
from .primitives import clear_solvers as _clear_solvers, solver_count
from .types import QPSolution


def clear_cache() -> None:
    """Release every core solver the JAX interface keeps, and their GPU
    memory. Functions that were already compiled stay valid; their next
    call creates the solvers again."""
    _clear_solvers()


def cache_size() -> int:
    """Number of distinct problem signatures (dtype, dimensions, given
    blocks, batch layout, settings) that currently have a live core solver."""
    return solver_count()


__all__ = ["DenseSolver", "QPSolution", "clear_cache", "cache_size"]
