# Problem Formulation

cuPIQP is a GPU-accelerated **proximal interior-point** solver for convex **quadratic
programs (QPs)** of the form

$$
\begin{aligned}
\min_{x}\quad & \tfrac{1}{2}\, x^\top P x + c^\top x \\
\mathrm{s.t.}\quad & A x = b, \\
                  & h_l \le G x \le h_u, \\
                  & x_l \le x \le x_u,
\end{aligned}
$$

## Data and shapes

| symbol | meaning | shape |
|---|---|---|
| $P $ | quadratic cost (symmetric positive semidefinite) | $n \times n$ |
| $c$ | linear cost | $n$ |
| $A,\, b$ | equality constraints | $p \times n$, $\;p$ |
| $G,\, h_l,\, h_u$ | two-sided inequality constraints | $m \times n$, $\;m$, $\;m$ |
| $x_l,\, x_u$ | element-wise box bounds on $x$ | $n$, $\;n$ |

Only `P` and `c` are required. Every other block is optional — pass `None` (or simply
omit it) when the corresponding constraint is absent. A QP with only `P` and `c` is an
unconstrained quadratic minimization.

!!! info "Convexity"
    cuPIQP solves **convex** QPs: `P` must be symmetric positive semidefinite
    ($P \succeq 0$). Indefinite `P` is outside the problem class and is not supported.

## One-sided constraints and free variables

Unbounded entries — one-sided inequalities, one-sided box bounds, and free variables —
are set to $\pm\infty$ (use `cupy.inf`). cuPIQP **detects these and drops the
corresponding rows/bounds automatically**, so an infinite bound costs nothing
numerically.

For example, to express $2x_1 \le -1$ as a row of $h_l \le G x \le h_u$ with no lower
bound, set that entry of `h_l` to `-cupy.inf`:

```python
import cupy as cp

G   = cp.array([[2.0, 0.0]])
h_l = cp.array([-cp.inf])   # no lower bound on this row
h_u = cp.array([-1.0])      # 2*x1 <= -1
```

The package constant `PIQP_INF` marks the threshold above which a bound is treated as
infinite; you can import it from the top-level package:

```python
from cupiqp import PIQP_INF
```

## Storage formats

cuPIQP accepts the matrices `P`, `A`, `G` in three storage formats, each handled by a
dedicated backend:

| Format | Matrices as | Solver | Notes |
|---|---|---|---|
| **Dense** | `cupy` dense arrays | [`DenseSolver`](guide/backends.md#densesolver) | small-to-medium, fully dense problems |
| **Sparse** | `cupyx.scipy.sparse` CSR | [`SparseSolver`](guide/backends.md#sparsesolver) | large, structurally sparse problems |
| **Multistage** | block-structured objects | [`MultistageSolver`](guide/backends.md#multistagesolver) | block-tridiagonal/-arrow KKT, e.g. OCPs |

The accepted input types depend on the backend:

- The dense backend accepts matrices and vectors exposing the
  [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html),
  including CuPy arrays, dense CUDA `torch.Tensor` objects, CUDA JAX arrays, and Numba
  CUDA device arrays.
- The sparse backend requires CSR containers for `P`, `A`, and `G`; its vector inputs
  (`c`, `b`, `h_l`, `h_u`, `x_l`, `x_u`) are dense GPU arrays.
- The multistage backend requires the corresponding `BlockTridiagMat`,
  `BlockBidiagMat`, and `BlockVec` objects.

See [Backends](guide/backends.md) for the exact per-backend contracts.

## Batching

cuPIQP is **natively batched**: every array carries a **leading batch dimension**
`(B, …)`, and `B` independent QPs are solved in one batched solver call, with no Python
loop over the batch. A single problem is just `B = 1`. See
[Batched Solving](guide/batched.md) for the full shape table and the rules that apply
across a batch.

## The solution

After [`solve()`](api/solvers.md), the primal solution and per-problem
diagnostics are exposed through `solver.result`:

```python
solver.result.x                  # (B, n) optimal primal solution
solver.result.info.status        # list of B Status enums (one per problem)
solver.result.info.primal_obj    # (B,) objective values
```

The dual variables (equality multipliers `y`, inequality multipliers `z_*`) and slacks
(`s_*`) are also available. See [Settings & Results](api/settings-results.md) for the complete
list.
