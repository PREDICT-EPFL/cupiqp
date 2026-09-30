# Solvers

All solver classes share the `setup` / `solve` / `update` workflow and use the same
[`Settings`](settings.md#settings). See
[Re-solving with new data](../getting-started.md#re-solving-with-new-data) for the
fixed-structure update pattern and [Differentiation](../guide/differentiation.md) for
the `backward()` workflow.
They differ only in the accepted storage format for `P`, `A`, `G` and the KKT
factorization used. See [Backends](../guide/backends.md) for guidance on choosing one.

## DenseSolver

::: cupiqp.DenseSolver
    options:
      inherited_members: true
      members: [setup, solve, update, backward]

## SparseSolver

::: cupiqp.SparseSolver
    options:
      inherited_members: true
      members: [setup, solve, update, backward]

`SparseSolver.setup` takes `P`, `A`, `G` as CSR triples `(indptr, indices, values)` of
GPU arrays; each value array and vector is shared or batched, and the batch size is read
from the batched ones. `update` takes each matrix's values alone, `(nnz,)` or `(B, nnz)`
(see `setup` and `update` above).

## MultistageSolver

::: cupiqp.MultistageSolver
    options:
      inherited_members: true
      members: [setup, solve, update, backward]

`MultistageSolver` takes plain GPU arrays: each block-structured matrix is a
`(diag, offdiag)` tuple and each vector an array, shared by every problem or with a
leading batch axis, at `setup` and at `update` (see `setup` and `update` above for the
block layout).

