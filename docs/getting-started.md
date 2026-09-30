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
**directly as GPU arrays** (cupy here; Warp, CUDA torch or JAX arrays work the same way) so the data already lives on the GPU. Results come back as `warp.array` objects on the GPU; call `.numpy()` for a host copy or view them from your framework of choice (`cupy.asarray(x)`, `torch.from_dlpack(x)`).

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

`DenseSolver` works with **dense** GPU arrays for `P`, `A`, `G`. Create the solver,
optionally tweak `solver.settings`, then `setup(...)` the problem and `solve()`. The solution and per-problem info are exposed through `solver.result`.

```python
solver = DenseSolver()
solver.settings.verbose = True        # print the banner + interior-point iteration log

solver.setup(P=P, c=c, A=A, b=b, G=G, h_l=h_l, h_u=h_u, x_l=x_l, x_u=x_u)   # one problem
solver.solve()

# result.x carries a leading batch dimension (B, n); here B = 1
x_dense = solver.result.x.numpy()[0]        # result.x is a warp.array on the GPU
print("status  :", solver.result.info.status[0].name)
print("solution:", x_dense)
```

### Sparse backend

`SparseSolver` takes `P`, `A`, `G` as **CSR triples** `(indptr, indices, values)` of
GPU arrays; the vectors are dense GPU arrays. We reuse the exact same data, converting
the matrices to CSR with `cupyx.scipy.sparse.csr_matrix` and passing its three arrays.
For larger, structurally sparse problems this is far more efficient than the dense
backend.

```python
solver = SparseSolver()
solver.settings.verbose = True

def csr(M):
    """CSR triple (indptr, indices, values) of a dense GPU matrix."""
    M = csr_matrix(M)
    return M.indptr, M.indices, M.data

solver.setup(
    P=csr(P), c=c,
    A=csr(A), b=b,
    G=csr(G), h_l=h_l, h_u=h_u,
    x_l=x_l, x_u=x_u,
)
solver.solve()

x_sparse = solver.result.x.numpy()[0]
print("status  :", solver.result.info.status[0].name)
print("solution:", x_sparse)

assert solver.result.info.status[0] == Status.CUPIQP_SOLVED
assert cp.allclose(cp.asarray(x_dense), cp.asarray(x_sparse), atol=1e-6)
print("Both backends converged to the same optimum.")
```

## Part 2 — A batch of QPs

cuPIQP is **natively batched**: it solves `B` independent QPs in a *single* GPU call.
All `B` problems share one structure. Every argument of `setup` is either **batched**,
with a leading batch axis (one value per problem), or **shared**, in the shape of one
problem (one value for every problem); `setup` reads `B` from the batched arguments.
`update()` takes the same two forms:

| array | shared | batched |
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
```

### Dense backend (batched)

The same call as Part 1, with `b` batched: its leading axis makes this a batch of `B`
problems, and every other argument is shared.

```python
dense_solver = DenseSolver()
dense_solver.setup(P=P, c=c, A=A, b=b_batch,         # b_batch: (B, p), one target per problem
                   G=G, h_l=h_l, h_u=h_u, x_l=x_l, x_u=x_u)
dense_solver.solve()

X_dense = dense_solver.result.x.numpy()               # (B, n)
for i, st in enumerate(dense_solver.result.info.status):
    print(f"problem {i}:  b = {float(b_batch[i, 0]):.1f}   status = {st.name}   x = {X_dense[i]}")
```

### Sparse backend (batched)

All `B` sparse problems share **one** sparsity pattern per matrix. The values of a
CSR triple follow the same rule as the vectors: `(nnz,)` shared, or `(B, nnz)` with one
row per problem, in the order of the pattern's `indices`. Here only `b` differs.

```python
sparse_solver = SparseSolver()
sparse_solver.setup(
    P=csr(P), c=c,
    A=csr(A), b=b_batch,                 # (B, p): one target per problem
    G=csr(G), h_l=h_l, h_u=h_u,
    x_l=x_l, x_u=x_u,
)
sparse_solver.solve()

X_sparse = sparse_solver.result.x.numpy()
assert all(st == Status.CUPIQP_SOLVED for st in sparse_solver.result.info.status)
assert cp.allclose(cp.asarray(X_dense), cp.asarray(X_sparse), atol=1e-6)
print("All problems solved; dense and sparse batches agree.")
```

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
