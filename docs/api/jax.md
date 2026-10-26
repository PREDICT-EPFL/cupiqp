# JAX Interface

Pure-function wrappers of the solvers for `jax.jit` and `jax.vmap`, available
as `cupiqp.jax`. See the [JAX Interface guide](../guide/jax.md) for the
workflow and the argument layouts.

## DenseSolver

The JAX operator `cupiqp.jax.DenseSolver`, distinct from the core
[`cupiqp.DenseSolver`](solvers.md#densesolver) it runs.

::: cupiqp.interfaces.jax.DenseSolver
    options:
      members: [solve, dtype, settings]

## SparseSolver

::: cupiqp.interfaces.jax.SparseSolver
    options:
      members: [solve, dtype, settings]

## MultistageSolver

::: cupiqp.interfaces.jax.MultistageSolver
    options:
      members: [solve, dtype, settings]

## QPSolution

::: cupiqp.interfaces.jax.QPSolution

## Cache

::: cupiqp.interfaces.jax.release_solver_memory


