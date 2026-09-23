# Batched Solving

CuPIQP is **natively batched**. A single solver instance solves `B` independent QPs in
one solver call. 

## Setup from one problem, batch as the leading dimension

All `B` problems in a batch share one structure: the dimensions, and which constraint
blocks and bound sides exist. So `setup(batch_size, ...)` takes the batch size and
**one** template problem (2-D matrices, 1-D vectors); its values are copied into every
problem. Per-problem numbers are then set with `update`, where every array carries
**the batch size as its leading dimension** (or the single-problem shape, to share one
value across the batch). Every array in the results has the batch size as its leading
dimension too.

| field | `setup` (one problem) | `update` (per problem) |
|---|---|---|
| `P`, `c` | `(n, n)`, `(n,)` | `(B, n, n)`, `(B, n)` |
| `A`, `b` | `(p, n)`, `(p,)` | `(B, p, n)`, `(B, p)` |
| `G`, `h_l`, `h_u` | `(m, n)`, `(m,)`, `(m,)` | `(B, m, n)`, `(B, m)`, `(B, m)` |
| `x_l`, `x_u` | `(n,)`, `(n,)` | `(B, n)`, `(B, n)` |

Each problem in the batch carries its own information and converges independently. Read per-problem diagnostics from `solver.result.info` — every field
is a `(B,)` array. See [Results & Status](../api/results.md).


```python
solver = DenseSolver()
solver.setup(
  B,               # batch size
  P=P_batch[0],    # one template problem: 2-dim matrices ...
  c=c_batch[0],    # ... and 1-dim vectors
  A=A_batch[0],
  b=b_batch[0],
  G=G_batch[0],
  h_l=h_l_batch[0],
  h_u=h_u_batch[0],
  x_l=x_l_batch[0],
  x_u=x_u_batch[0],
  )
solver.update(     # per-problem data, batch size as the leading dim
  P=P_batch, c=c_batch, A=A_batch, b=b_batch, G=G_batch,
  h_l=h_l_batch, h_u=h_u_batch, x_l=x_l_batch, x_u=x_u_batch,
  )
solver.solve()

x_sol = solver.result.x                  # cupy array of shape (B, n)
status = solver.result.info.status       # list of length B
for i, st in enumerate(status):
    print(f"problem {i}: status = {st.name}, x = {x_sol[i]}")
```

### Single problem case:

A single problem is simply the `B=1` case: `setup(1, ...)`, with no `update` needed.
cuPIQP internally treats it as a batch of one, so **the returned result still carries
the batch size 1 as the leading dimension**.

```python
P_single = cp.eye(2)                 # 2-dim array, no batch dim
c_single = cp.ones(2)                # 1-dim array, no batch dim
A_single = cp.array([[1.0, -2.0]])   # 2-dim array, no batch dim
b_single = cp.array([1.0])           # 1-dim array, no batch dim

solver = DenseSolver()
solver.setup(
  1,
  P=P_single, c=c_single,
  A=A_single, b=b_single
  )
solver.solve()

# result.x still carries a leading batch dimension (B, n); here B = 1
x_sol = solver.result.x[0]
```

### Sparse problems
For **sparse** problems, all `B` problems share one sparsity pattern. `setup` takes the
batch size and **one** template problem, with `P`, `A`, `G` as single 2-D GPU CSR
matrices ([`cupyx.scipy.sparse.csr_matrix`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csr_matrix.html#cupyx.scipy.sparse.csr_matrix)
or a 2-D CUDA [`torch.sparse_csr_tensor`](https://docs.pytorch.org/docs/2.12/generated/torch.sparse_csr_tensor.html))
and 1-D vectors; its values are copied into every problem. Per-problem numbers are then
set with `update`, where each matrix is given by its nonzero values as a dense
`(B, nnz)` array in the CSR order of the template:

```python
solver = SparseSolver()
solver.setup(B, P=P_csr, c=c_single, G=G_csr, h_l=h_l_single, h_u=h_u_single)
solver.update(P=P_values, c=c_batch, h_l=h_l_batch, h_u=h_u_batch)  # (B, P_csr.nnz), (B, n), (B, m)
solver.solve()
```

In `update`, the vectors $c, b, h_l, h_u, x_l, x_u$ are stacked `(B, ...)` dense arrays just like the dense case (or 1-D to share one value across the batch).

See [this example](https://github.com/PREDICT-EPFL/cupiqp/blob/main/examples/getting_started.ipynb) for more details.


## Uniform structure across the batch

All problems in a batch share the same **structure**, even though their numerical data
differ:

- Same shapes `n`, `p`, `m` and (for the sparse backend) the same sparsity pattern.
- For sparse solver, all $P$ matrices in the batch must have the same sparsity pattern. This also applies to $A$ and $G$.

!!! note "Bound patterns can differ per problem"
    There are two separate questions about bounds, and they behave differently:

    **1. Which entries are finite — free to differ per problem, and changeable between solves.**
    Within a bound array you pass, any entry set to `±inf` means "no bound there". This
    pattern can be different for each problem in the batch, and you can change it later with
    `update()` without calling `setup()` again. For example, in a batch of 2 problems with
    `m = 2` inequality rows:

    ```python
    # problem 0: row 0 is one-sided (only upper), row 1 is two-sided
    # problem 1: both rows two-sided
    h_l = cp.array([[-cp.inf, -3.0],    # problem 0
                    [  -2.0,  -3.0]])   # problem 1
    h_u = cp.array([[   1.0,   4.0],
                    [   1.0,   4.0]])
    ```

    cuPIQP keeps a full-length slot for every row and just ignores the `±inf` ones, so no
    common pattern is needed.

    **2. Which bound *sides* exist — fixed at `setup()`, shared by the whole batch.**
    Each of the four sides `h_l`, `h_u`, `x_l`, `x_u` is either passed at `setup()` or left
    out (`None`). A side you leave out is gone for *every* problem in the batch and uses no
    memory (its result duals/slacks are `(B, 0)`); you cannot add it back later without a
    fresh `setup()`. So if you only ever need an upper inequality, pass `h_u` and omit `h_l`
    entirely. If *some* problems need the lower side, instead keep `h_l` present for the
    whole batch and set it to `-inf` on the problems that don't (case 1 above).


