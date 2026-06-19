# Large-Scale Problems

For a **single large QP** (or a small batch of large QPs), cuPIQP ships a companion
to each backend — `DenseLargeProblemSolver`, `SparseLargeProblemSolver`, and
`MultistageLargeProblemSolver`. They solve exactly the same problem as the standard
backends, with the same API and the same numerical result, but use a different
inner-loop kernel strategy that pays off once the problem gets big.

!!! tip "Worked example"
    The mean-variance **portfolio optimization** notebook scales a sparse QP up to
    tens of thousands of variables and is the running example used below:
    [`examples/portfolio_optimization.ipynb`](https://github.com/PREDICT-EPFL/cupiqp/blob/main/examples/portfolio_optimization.ipynb).

## Why a separate class?

The standard backends fuse the interior-point inner loop (step length, barrier
parameter `mu`, centering `sigma`, residual and merit evaluation) into
**shape-specialized Warp tile kernels**. Those kernels are JIT-compiled and
specialized to the problem dimensions, which makes them very fast per launch and lets
them amortize beautifully across a **batch** and across IPM iterations.

The catch is the compile step. The tile-kernel families are specialized to
`max(n, p, m)`, so as the problem grows the **compilation time on the first solve
grows with it** and eventually dominates the first-solve latency — you can spend far
more time compiling kernels than actually solving.

The `*LargeProblemSolver` variants replace those tile kernels with **CuPy
axis-reduction kernels** (`cp.min`, `cp.sum`, `cp.max` over the data axis). CuPy's
reductions are generic and need no shape-specialized compilation, so the compile cliff
disappears. They carry more per-launch overhead than a fused tile kernel, but when each
reduction runs over a long axis — i.e. a large problem — that overhead is amortized and
the trade is worth it. The variants also build their Ruiz preconditioner with the tile
kernels switched off, for the same reason.

Everything else is identical. The algorithm, the stopping criteria, the regularization
and scaling are unchanged, and the result agrees with the standard backend **to solver
tolerance** — only the kernel implementation differs.

## The three variants

There is one large-problem companion per backend, so the choice of variant follows the
[same rule as choosing a backend](backends.md): match the class to how your `P` / `A` /
`G` are stored.

| Standard backend | Large-problem variant | Matrices `P, A, G` |
|---|---|---|
| `DenseSolver` | `DenseLargeProblemSolver` | dense `cupy` arrays |
| `SparseSolver` | `SparseLargeProblemSolver` | [`UniformBatchedCsrMatrix`](../api/solvers.md#cupiqp.UniformBatchedCsrMatrix) / CSR |
| `MultistageSolver` | `MultistageLargeProblemSolver` | block-structured objects |

```python
from cupiqp import (
    DenseLargeProblemSolver,
    SparseLargeProblemSolver,
    MultistageLargeProblemSolver,
)
```

## Using it: the portfolio QP

The portfolio problem is a **sparse** QP — a factor risk model
$\Sigma = F F^\top + D$ with the substitution $y = F^\top x$ keeps the quadratic term
sparse — so the relevant variant is `SparseLargeProblemSolver`. Switching from the
standard sparse backend is a **one-line change**; `setup` / `solve` / `update` /
`result` are exactly the same:

```python
from cupyx.scipy.sparse import csr_matrix
from cupiqp import SparseLargeProblemSolver   # was: SparseSolver

solver = SparseLargeProblemSolver()
solver.settings.verbose = False
solver.setup(
    P=csr_matrix(prob["P"].tocsr()), c=cp.asarray(prob["c"]),
    A=csr_matrix(prob["A"].tocsr()), b=cp.asarray(prob["b"]),
    x_l=cp.asarray(prob["z_l"]),     x_u=cp.asarray(prob["z_u"]),
)
solver.solve()

z = solver.result.x.get()[0]
x = z[:prob["n"]]                    # portfolio weights (y-part is auxiliary)
```

At the small end of the notebook (`n = 1000`, `k = 100`) the standard `SparseSolver`
is the better pick — the problem is small enough that tile-kernel compilation is cheap
and the fused kernels win. As the notebook scales the QP up to `n = 10000`,
`k = 1000` (`N = 11000` variables), the tile-kernel compile time starts to dominate the
first solve, and this is exactly where reaching for `SparseLargeProblemSolver` removes
that cost.

## When to use it

!!! note "Rule of thumb"
    Reach for a `*LargeProblemSolver` when first-solve latency on a **large**, single
    (or small-batch) problem is dominated by kernel compilation — i.e. when
    `max(n, p, m)` is large. For **batched small-to-medium** problems, prefer the
    standard backends: the tile kernels amortize across the batch, and CuPy's
    per-launch reduction overhead is not worth paying on many tiny problems.

Because the API is identical, the cheapest way to decide is to try both on a
representative instance and compare the first-solve (and steady-state) timing — swap
the class name and nothing else. See [Backends](backends.md#large-problem-variants) for
the same variants in the backend-selection context.
