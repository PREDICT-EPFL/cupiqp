"""JAX interface of cuPIQP: QP solves as pure functions for ``jit`` and ``vmap``.

A solver object such as :class:`DenseSolver`, :class:`SparseSolver` or
:class:`MultistageSolver` holds the dtype and the solver settings (and, for
the sparse one, the CSR patterns). Its ``solve`` method takes JAX arrays,
solves the QP on the GPU through an XLA custom call and returns a :class:`QPSolution` of JAX arrays. The call behaves like
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
exactly as for the core solvers. ``jax.grad`` and ``jax.vjp`` work through
``solve()`` with respect to every array argument.

Importing this module imports JAX. The interface is written for one CUDA
device: every compiled signature keeps one core solver, on the device XLA
runs on.

**Caching.** Every distinct problem signature (dtype, dimensions, given
blocks, sparsity pattern, which arguments are batched, batch size, forward
settings) gets its own compiled function, XLA custom call and core solver.
The core solver holds the GPU memory and is created by the first call of
its signature, which is therefore slow; later calls reuse it. Keep shapes,
batch sizes and settings stable in a loop, and call once to warm up before
timing. :func:`release_solver_memory` frees the core solvers and their memory;
the compiled functions and custom calls stay alive for the whole process,
as JAX's own compilation cache does.

JAX reserves most of the GPU memory for itself when its backend starts,
which can leave the core solver unable to create its cuSOLVER or cuDSS
handles. Set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (or a small
``XLA_PYTHON_CLIENT_MEM_FRACTION``) in the environment before JAX is
initialized.
"""

from .base import release_solver_memory
from .dense import DenseSolver
from .multistage import MultistageSolver
from .sparse import SparseSolver
from .types import QPSolution

__all__ = [
    "DenseSolver", "SparseSolver", "MultistageSolver", "QPSolution",
    "release_solver_memory",
]
