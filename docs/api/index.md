# API Reference

This reference is generated **directly from the docstrings** in the cuPIQP source.
Everything here is part of the public API — the names exported from the top-level
`cupiqp` package.

```python
import cupiqp
```

## Public surface

| Group | Names | Page |
|---|---|---|
| **Solvers** | `DenseSolver`, `SparseSolver`, `MultistageSolver`, `DenseLargeProblemSolver`, `SparseLargeProblemSolver`, `MultistageLargeProblemSolver` | [Solvers](solvers.md) |
| **Problem data** | `Data`, `DenseData`, `SparseData`, `MultistageData` | [Problem Data](data.md) |
| **Solver input containers** | `UniformBatchedCsrMatrix` (with `SparseSolver`); `BlockTridiagMat`, `BlockBidiagMat`, `BlockVec` (with `MultistageSolver`) | [Solvers](solvers.md) |
| **Configuration & results** | `Settings`, `Result`, `Status`, `PIQP_INF` | [Settings & Results](settings-results.md) |

The package also exposes the PIQP-style status aliases
(`cupiqp.CUPIQP_SOLVED`, `cupiqp.CUPIQP_MAX_ITER_REACHED`, …) — see [`Status`](settings-results.md#status) —
and `cupiqp.__version__`.

!!! tip "Looking for narrative docs?"
    For task-oriented guides (rather than the symbol-by-symbol reference), see
    [Getting Started](../getting-started.md), [Backends](../guide/backends.md), and
    [Differentiation](../guide/differentiation.md).
