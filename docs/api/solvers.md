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

`SparseSolver` takes a batch of sparse matrices as a single
`UniformBatchedCsrMatrix` — cuPIQP's own batched CSR container (the preferred,
fastest batched input; see `setup` above):

::: cupiqp.UniformBatchedCsrMatrix
    options:
      show_if_no_docstring: true
      members: false

## MultistageSolver

::: cupiqp.MultistageSolver
    options:
      inherited_members: true
      members: [setup, solve, update, backward]

`MultistageSolver` takes its problem data as the block-structured objects below —
build them, fill in their data, and pass them to `setup` (see above for which
argument expects which type):

::: cupiqp.BlockTridiagMat
    options:
      show_if_no_docstring: true
      members: false

::: cupiqp.BlockBidiagMat
    options:
      show_if_no_docstring: true
      members: false

::: cupiqp.BlockVec
    options:
      show_if_no_docstring: true
      members: false

