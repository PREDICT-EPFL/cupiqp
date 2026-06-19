# Getting Started

This walkthrough solves a small QP with both the dense and the sparse backend, then
shows how to solve a whole **batch** of QPs in one call. Everything runs on the GPU.

!!! tip "Notebook version"
    The same material is available as a runnable notebook:
    [`examples/getting_started.ipynb`](https://github.com/PREDICT-EPFL/cupiqp/blob/main/examples/getting_started.ipynb).

## Part 1 — A single QP

We solve the two-variable QP

$$
\begin{aligned}
\min_{x_1,\,x_2}\quad & \tfrac12\bigl(6 x_1^2 + 4 x_2^2\bigr) - x_1 - 4 x_2 \\
\text{s.t.}\quad
  & x_1 - 2 x_2 = 1, \\
  & -10 \le x_1 - x_2 \le 0.2, \\
  & 2 x_1 \le -1, \\
  & x_1 \le 1, \\
  & x_2 \ge -1.
\end{aligned}
$$

Note the **one-sided** pieces: the inequality $2x_1 \le -1$ has no lower bound
($h_l = -\infty$), and each variable is bounded on only one side. We build everything
**directly as cupy arrays** so the data already lives on the GPU.

```python
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupiqp import DenseSolver, SparseSolver, Status

# quadratic + linear cost
P = cp.array([[6.0, 0.0],
              [0.0, 4.0]])
c = cp.array([-1.0, -4.0])

# equality constraint:  A x = b
A = cp.array([[1.0, -2.0]])
b = cp.array([1.0])

# two-sided inequalities:  h_l <= G x <= h_u
# For a one-sided block, either set the unused side to -inf / +inf, or omit it
# entirely by passing only the side you need (e.g. h_u alone for G x <= h_u).
# An omitted side is fixed at setup() and stores no duals/slacks for it.
G   = cp.array([[1.0, -1.0],
                [2.0,  0.0]])
h_l = cp.array([-10.0, -cp.inf])
h_u = cp.array([  0.2,  -1.0])

# box bounds:  x_l <= x <= x_u
x_l = cp.array([-cp.inf, -1.0])
x_u = cp.array([   1.0,  cp.inf])
```

### Dense backend

`DenseSolver` works with **dense** cupy arrays for `P`, `A`, `G`. Create the solver,
optionally tweak `solver.settings`, then `setup(...)` the problem and `solve()`. The
solution and per-problem info are exposed through `solver.result`.

```python
solver = DenseSolver()
solver.settings.verbose = True        # print the banner + interior-point iteration log

solver.setup(P=P, c=c, A=A, b=b, G=G, h_l=h_l, h_u=h_u, x_l=x_l, x_u=x_u)
solver.solve()

# result.x carries a leading batch dimension (B, n); here B = 1
x_dense = solver.result.x.get()[0]
print("status  :", solver.result.info.status[0].name)
print("solution:", x_dense)
```

### Sparse backend

`SparseSolver` expects `P`, `A`, `G` as **GPU CSR** matrices
(`cupyx.scipy.sparse.csr_matrix`); the vectors stay as cupy arrays. We reuse the exact
same data, just wrapping the matrices as CSR. For larger, structurally sparse problems
this is far more efficient than the dense backend.

```python
solver = SparseSolver()
solver.settings.verbose = True

solver.setup(
    P=csr_matrix(P), c=c,
    A=csr_matrix(A), b=b,
    G=csr_matrix(G), h_l=h_l, h_u=h_u,
    x_l=x_l, x_u=x_u,
)
solver.solve()

x_sparse = solver.result.x.get()[0]
print("status  :", solver.result.info.status[0].name)
print("solution:", x_sparse)

assert solver.result.info.status[0] == Status.CUPIQP_SOLVED
assert cp.allclose(cp.asarray(x_dense), cp.asarray(x_sparse), atol=1e-6)
print("Both backends converged to the same optimum.")
```

## Part 2 — A batch of QPs

cuPIQP is **natively batched**: it solves `B` independent QPs in a *single* GPU call.
The only rule is that **the batch size is the leading dimension** of every array:

| array | single | batched |
|---|---|---|
| `P` | `(n, n)` | `(B, n, n)` |
| `c`, `x_l`, `x_u` | `(n,)` | `(B, n)` |
| `A` / `G` | `(p, n)` / `(m, n)` | `(B, p, n)` / `(B, m, n)` |
| `b`, `h_l`, `h_u` | `(p,)` / `(m,)` | `(B, p)` / `(B, m)` |

`solver.result.x` then has shape `(B, n)` and `solver.result.info.status` is a list of
`B` statuses (one per problem).

Here we reuse the **same QP structure** from Part 1 but give each problem a different
**equality target** `b` — like solving the same controller for several set-points at
once.

```python
B = 4

# only the equality target b differs across the batch; everything else is shared
b_batch = cp.array([[0.9], [1.0], [1.1], [1.2]])      # shape (B, p) with p = 1

# replicate the shared data along the leading batch dimension -> (B, ...)
stack = lambda M: cp.stack([M] * B)
P_b, c_b, A_b, G_b = stack(P), stack(c), stack(A), stack(G)
h_l_b, h_u_b, x_l_b, x_u_b = stack(h_l), stack(h_u), stack(x_l), stack(x_u)
```

### Dense backend (batched)

Exactly the same call as Part 1 — just hand `setup` the batched `(B, …)` arrays.

```python
dense_solver = DenseSolver()
dense_solver.setup(P=P_b, c=c_b, A=A_b, b=b_batch, G=G_b,
                   h_l=h_l_b, h_u=h_u_b, x_l=x_l_b, x_u=x_u_b)
dense_solver.solve()

X_dense = dense_solver.result.x.get()                 # (B, n)
for i, st in enumerate(dense_solver.result.info.status):
    print(f"problem {i}:  b = {float(b_batch[i, 0]):.1f}   status = {st.name}   x = {X_dense[i]}")
```

### Sparse backend (batched)

For a batch, the **preferred** input is a
[`UniformBatchedCsrMatrix`](api/solvers.md#cupiqp.UniformBatchedCsrMatrix) — cuPIQP's
own container holding `B` matrices that share **one** sparsity pattern, with the values
stacked as a `(B, nnz)` array. Build one from the shared pattern (`indices` / `indptr`)
and the per-problem values; the vectors are stacked `(B, …)` just like the dense case.

```python
from cupiqp import UniformBatchedCsrMatrix

def batched_csr(M):
    """Pack B copies of a cupy matrix into a UniformBatchedCsrMatrix."""
    m = csr_matrix(M)
    values = cp.broadcast_to(m.data, (B, m.nnz)).copy()   # (B, nnz); here all equal
    return UniformBatchedCsrMatrix(B, m.indices, m.indptr, values, shape=m.shape)

sparse_solver = SparseSolver()
sparse_solver.setup(
    P=batched_csr(P), c=c_b,
    A=batched_csr(A), b=b_batch,
    G=batched_csr(G), h_l=h_l_b, h_u=h_u_b,
    x_l=x_l_b, x_u=x_u_b,
)
sparse_solver.solve()

X_sparse = sparse_solver.result.x.get()
assert all(st == Status.CUPIQP_SOLVED for st in sparse_solver.result.info.status)
assert cp.allclose(cp.asarray(X_dense), cp.asarray(X_sparse), atol=1e-6)
print("All problems solved; dense and sparse batches agree.")
```

!!! warning "Avoid passing a raw `list` of `csr_matrix`"
    `setup` *does* also accept a plain `list` / `tuple` of
    `cupyx.scipy.sparse.csr_matrix` (one per batch element, all sharing the same
    pattern), but it is **discouraged**: separate matrix objects are not laid out with
    the uniform stride that batched linear-algebra routines need, so cuPIQP has to copy
    them into a single
    [`UniformBatchedCsrMatrix`](api/solvers.md#cupiqp.UniformBatchedCsrMatrix) at
    `setup`. Build and pass one yourself to skip that copy.

## Re-solving with new data

`setup()` fixes the problem structure: shapes, sparsity patterns, and which constraint
blocks are present. Which individual bounds are finite is **not** part of that structure —
see below. Call `setup()` once per solver instance.

For new numerical values with the same structure, call `update()` and solve again.
Arguments left as `None` keep their current values:

```python
solver.setup(P=P, c=c, A=A, b=b0)
solver.solve()

for b_k in trajectory:
    solver.update(b=b_k)
    solver.solve()
```

**It is allowed to change which bounds are finite** in `update()`: pass new `h_l`, `h_u`, `x_l`, `x_u` arrays that mark different entries as `±inf` (cuPIQP keeps a full-length dual/slack vector
for each *present* side and masks the infinite entries), so toggling a bound between finite and `±inf` does not need a new `setup()`. Which bound sides are **present** is structural, however: each of `h_l`, `h_u`, `x_l`, `x_u` is either provided at `setup()` (full-length block) or omitted (no storage, `(B, 0)` duals/slacks), and that choice is fixed. Calling `update()`/`set_*` on a side that was omitted at `setup()` raises — adding a side, like any change to dimensions, sparsity, or which constraint blocks are present, requires a new solver instance.

## Next steps

- [Backends](guide/backends.md) — choose dense, sparse, or multistage storage.
- [Batched Solving](guide/batched.md) — the rules that apply across a batch.
- [Differentiation](guide/differentiation.md) — compute VJPs through a solved QP.
- [Settings](api/settings.md) — tolerances, regularization, and more.
- [Results & Status](api/results.md) — solution fields, per-problem diagnostics, and status codes.
