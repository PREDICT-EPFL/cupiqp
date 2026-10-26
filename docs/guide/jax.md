# JAX Interface

`cupiqp.jax` exposes each solver as a pure JAX function. The `solve` method of
`cupiqp.jax.DenseSolver`, `cupiqp.jax.SparseSolver` or
`cupiqp.jax.MultistageSolver` (not to be confused with the core solvers of the
same names) takes JAX arrays, solves on the GPU through an XLA custom call, and returns a
`QPSolution` of JAX arrays. The call works eagerly, under `jax.jit` and under
`jax.vmap`; nothing is re-implemented in `jax.numpy`, the solve is the core
[`DenseSolver`](../api/solvers.md#densesolver),
[`SparseSolver`](../api/solvers.md#sparsesolver) or
[`MultistageSolver`](../api/solvers.md#multistagesolver).

`jax.grad` and `jax.vjp` work through every solve; see
[Gradients](#gradients) below.

## Installation

The interface ships with every install of cuPIQP; the `jax` extra installs
JAX itself. Install a CUDA build of JAX that matches your driver (see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html)),
for example together with the CUDA extra you need:

```bash
python -m pip install ".[cuda12,jax]"
python -m pip install "jax[cuda12]"
```

`import cupiqp` does not import JAX. It is imported when you first access
`cupiqp.jax`.

cuPIQP solves in float64 by default. JAX creates float32 arrays unless 64-bit
mode is enabled, so turn it on before creating any array:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

`cupiqp.jax.DenseSolver(dtype=jnp.float32)` solves in float32 and does not need this.

JAX reserves most of the GPU memory for itself when its backend starts, which
can leave the solver unable to create its cuSOLVER or cuDSS handles. Let JAX
allocate on demand instead, before it is initialized:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

(or give it a fraction with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`).

## One problem

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import cupiqp
from cupiqp.jax import DenseSolver

P = jnp.eye(2)
c = jnp.array([-1.0, -4.0])
G = jnp.array([[1.0, 1.0]])          # constrain x1 + x2
h_l = jnp.array([-jnp.inf])          # no lower bound on the row
h_u = jnp.array([1.0])               # x1 + x2 <= 1

solver = DenseSolver()                   # float64, default settings
sol = solver.solve(P, c, G=G, h_l=h_l, h_u=h_u)

print(cupiqp.Status(int(sol.status)).name)   # CUPIQP_SOLVED
print(sol.x)                                  # (2,)
```

The call takes the same arguments as the core `cupiqp.DenseSolver.setup()`: `P`, `c`, and any
of the blocks `A, b`, `G, h_l, h_u`, `x_l, x_u`. The solver object holds only the
dtype and the settings; the dimensions and which blocks are given are read from
the arrays. Under `jax.jit` those are static shapes, so a function is compiled
once per problem structure, as any JAX function is. Which entries of a bound
are `inf` is a value, not structure: it may change from call to call.

## Results

`QPSolution` is a `NamedTuple` of JAX arrays: `x`, the multipliers `y`,
`z_l`, `z_u`, `z_bl`, `z_bu`, the slacks `s_l`, `s_u`, `s_bl`, `s_bu`, and the
per-problem diagnostics `status`, `iterations`, `primal_res`, `dual_res`.
Blocks that were not given have zero width (`(B, 0)`), never `None`. Every
field is owned by JAX: a later solve does not change it.

A solve that does not converge, or finds the problem infeasible, does not
raise: `status` carries the [`Status`](../api/results.md) code of each problem
and the other fields hold the last iterate. Inside `jit`, read it with array
operations:

```python
@jax.jit
def safe_solve(c):
    sol = solver.solve(P, c, G=G, h_l=h_l, h_u=h_u)
    ok = sol.status == cupiqp.Status.CUPIQP_SOLVED
    return jnp.where(ok, sol.x, jnp.nan)
```

## Batches: shared and batched arguments

Every argument is either **shared**, in the shape of one problem, or
**batched**, with a leading axis of size `B`. Batched arguments must agree on
`B`; shared ones are used by every problem without being copied `B` times by
you. The solution then has a leading batch axis. This is the layout of the
core solvers (see [Batched Solving](batched.md)), and one call solves the whole
batch in a single custom call:

```python
cs = jnp.stack([c, 2 * c, 3 * c])          # (3, 2): batched
sol = solver.solve(P, cs, G=G, h_l=h_l, h_u=h_u)   # P, G, h_l, h_u shared
sol.x.shape, sol.status.shape              # (3, 2), (3,)
```

`jax.vmap` gives the same batched solve, not a loop: mapped arguments become
batched, unmapped ones (`in_axes=None`, or closed over) stay shared.
`jax.jit(jax.vmap(...))` lowers to exactly one custom call.

```python
solve_one = lambda c: solver.solve(P, c, G=G, h_l=h_l, h_u=h_u).x
xs = jax.vmap(solve_one)(cs)               # (3, 2), same as sol.x above
```

Mapping over an argument that already carries a batch axis (nested batching)
is not supported; flatten the batch axes into one.

## Sparse problems

`SparseSolver` takes the CSR patterns of `P`, `A` and `G` once, in the
constructor, and every `solve` call takes only the **stored values** of each
matrix, `(nnz,)` shared or `(B, nnz)` batched, in the order of the pattern. A
pattern is the `(indptr, indices)` pair of the matrix, the core's CSR triple
without its values, as host arrays (NumPy, or concrete JAX arrays). Other GPU
arrays are rejected; copy them to the host first. Store the full symmetric `P`, keep column
indices sorted within each row, and store every entry that may be nonzero in
any problem (explicit zeros are fine). A different pattern is a different
solver, and a new symbolic analysis in the core, so build the solver outside
your loop.

```python
import scipy.sparse as sp
from cupiqp.jax import SparseSolver

P = sp.csr_matrix(...); A = sp.csr_matrix(...)          # patterns and values
solver = SparseSolver(P=(P.indptr, P.indices),          # patterns fixed here
                      A=(A.indptr, A.indices))

sol = solver.solve(jnp.asarray(P.data), c, A=jnp.asarray(A.data), b=b)

@jax.jit
def solve_many(P_values, cs):                           # (B, nnz_P), (B, n)
    return solver.solve(P_values, cs, A=jnp.asarray(A.data), b=b).x
```

The vectors, batching, `vmap`, settings and results are exactly as for the
dense solver. One difference comes from the core: the sparse factorization is
not bitwise reproducible from run to run unless
`settings.use_deterministic_mode_for_cudss` is on, so two sparse solves of the
same data agree to rounding, not bit for bit.

## Multistage problems

`MultistageSolver.solve` takes exactly the arguments of the core
`MultistageSolver.setup()`: each matrix is a `(diag, offdiag)` pair of block
arrays, and each vector is in the block layout `(num_blocks, rows)` or the flat
layout `(k,)`, shared or with a leading batch axis. An `offdiag` of `None`
removes those blocks. The block sizes and which matrices have off-diagonal
blocks are read from the shapes, so nothing is passed to the constructor.

```python
from cupiqp.jax import MultistageSolver

solver = MultistageSolver()

@jax.jit
def mpc(x_ref):                                     # (B, N, d): one reference per problem
    sol = solver.solve(P=(P_diag, None), c=-x_ref,  # block-diagonal cost
                       A=(A_diag, A_offdiag), b=b,  # dynamics couple the stages
                       x_l=x_l, x_u=x_u)
    return sol.x.reshape(-1, N, d)
```

The solution is in the flat layout, as in the core: `x` is `(B, N * d)`.

## Settings

`solver.settings` is a [`Settings`](../api/settings.md) object, configured like
the core solver's: set its fields before solving, or pass a `Settings` to the
constructor. They are read at every call. A function compiled by `jax.jit`
keeps the settings it was traced with, like any Python value it closes over,
so set them before its first call.

```python
solver = DenseSolver()
solver.settings.max_iter = 50
solver.settings.eps_abs = 1e-6
```

Two settings would break the pure-function contract and raise a `ValueError`
before anything runs. `preconditioner_reuse_on_update = True` reuses the
scaling of an earlier call, so a result would depend on what was solved before.
`verbose = True` prints, which is a side effect XLA may skip or repeat.

## Gradients

`jax.grad`, `jax.vjp` and `jax.value_and_grad` work through `solve()` with
respect to every array argument, by implicit differentiation of the KKT
conditions at the solution (the core's `backward_from_state()`).

```python
solver = DenseSolver()

def loss(c):
    return jnp.sum(solver.solve(P, c, G=G, h_l=h_l, h_u=h_u).x ** 2)

g = jax.grad(loss)(c)                       # (n,)
gs = jax.jit(jax.vmap(jax.grad(loss)))(cs)  # (B, n): one batched forward, one batched backward
```

- **What is differentiable.** The ten solution fields `x`, `y`, `z_*` and
  `s_*`. The diagnostics `status`, `iterations`, `primal_res` and `dual_res`
  are not: a loss that uses them gets no gradient through them.
- **Shapes.** Each gradient has the shape its argument was given in. A shared
  argument's gradient is the sum over the batch. For the sparse solver a
  matrix gradient is the gradient of its stored values; for the multistage
  solver each block array gets its own gradient.
- **Failed problems** follow `settings.gradient_on_failure`, as in the core:
  `"differentiate"` (default) differentiates the last iterate, `"zero"` gives
  zero gradients. Read `status` from the forward to know which problems failed.
- **Backward settings.** `gradient_on_failure` and the `gradient_smoothing*`
  settings are part of the compiled call, like the forward settings.
- **Cost and memory.** The backward refactorizes the KKT system at the saved
  solution, about the cost of one forward iteration. JAX keeps the inputs and
  the solution of every differentiated call until the backward runs; a
  `lax.scan` over `T` solves keeps `T` of them.
- **Purity.** The backward is a pure function of the forward's inputs and
  solution, which JAX carries as residuals. It works under `jit`, `vmap`,
  `lax.scan` and any order in which XLA runs forwards and backwards.
- **Not supported:** second derivatives, forward mode (`jax.jvp`), and
  gradients with respect to settings or sparsity patterns.

## What happens underneath

- The solver object owns no GPU memory. For every distinct problem signature (dtype,
  dimensions, given blocks, sparsity pattern, which arguments are batched, batch
  size, forward settings) the interface keeps one core solver in a
  process-wide cache, so repeated calls reuse its buffers. The first call of a
  signature runs the core `setup()` (allocation, symbolic analysis, kernel
  compilation) and is slow: warm up with one call before timing, and keep
  shapes, batch sizes and settings stable in a loop.
- `cupiqp.jax.release_solver_memory()` waits for pending solves and frees every core
  solver; compiled functions stay valid and recreate their solver on the next
  call. The compiled functions and XLA custom calls themselves live for the
  whole process, like JAX's compilation cache.
- Each call copies its inputs into the solver and its results into JAX-owned
  outputs, all on the GPU in stream order. The solver runs on its own CUDA
  stream, ordered after the XLA stream on entry and before it on exit; the
  interface never waits on the host.
- The custom call is registered as side-effect free: XLA may drop a solve
  whose results are unused, and calls of one signature are serialized.
- A call is a pure function of its arrays: the same inputs give the same
  outputs, eagerly or compiled (bitwise for the dense backend; to rounding
  for the sparse one unless cuDSS deterministic mode is on).

| | `cupiqp.jax` solvers | core solvers |
|---|---|---|
| State | none visible; cache of solvers behind the call | `setup()` once, `update()` + `solve()` |
| Inputs | JAX arrays, shared or batched | any GPU array, shared or batched |
| Outputs | new JAX arrays per call | Warp views, overwritten by the next solve |
| `jit` / `vmap` | yes | no (eager only; JAX arrays accepted as inputs) |
| Gradients | `jax.grad`, `jax.vjp` | `backward()` |
