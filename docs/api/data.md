# Problem Data

`setup()` builds the right `Data` subclass from your inputs and exposes it as
`solver.data`; you rarely construct these directly. The user-built input
containers are documented alongside their solver: `UniformBatchedCsrMatrix` under
[SparseSolver](solvers.md#sparsesolver) and the `Block*` types under
[MultistageSolver](solvers.md#multistagesolver).

## Data

The abstract base class shared by all backends.

::: cupiqp.Data
    options:
      show_if_no_docstring: true
      members: [n, p, m, batch_size, dtype, device, P, c, A, b, G, h_l, h_u, x_l, x_u,
                num_hl, num_hu, num_xl, num_xu, num_ineq,
                finite_mask_hl, finite_mask_hu, finite_mask_xl, finite_mask_xu,
                active_G_row, active_x_bound, num_finite_bounds]

## DenseData

::: cupiqp.DenseData
    options:
      show_if_no_docstring: true
      members: false

## SparseData

::: cupiqp.SparseData
    options:
      show_if_no_docstring: true
      members: false

## MultistageData

::: cupiqp.MultistageData
    options:
      show_if_no_docstring: true
      members: false
