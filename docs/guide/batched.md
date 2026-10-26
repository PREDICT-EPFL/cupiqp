# Batched Solving

CuPIQP is **natively batched**. A single solver instance solves `B` independent QPs in
one solver call. 

## Batch as the leading dimension

All `B` problems in a batch share one structure: the dimensions, and which constraint
blocks and bound sides exist. Every argument of `setup` and `update` is either
**batched**, with the batch size as its leading dimension (one value per problem), or
**shared**, in the shape of a single problem (one value for every problem). `setup`
reads `B` from the batched arguments, which must agree; if every argument is shared, it
sets up a single problem. Every array in the results has the batch size as its leading
dimension.

| field | shared | batched |
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
  P=P_batch, c=c_batch,       # batched: (B, n, n), (B, n)
  A=A_single, b=b_batch,      # A shared by every problem, b batched
  x_l=x_l_single, x_u=x_u_single,
  )
solver.solve()

x_sol = solver.result.x.numpy()                # (B, n); result.x is a warp.array on the GPU
status = solver.result.info.status.numpy()     # (B,) Status codes
for i, st in enumerate(status):
    print(f"problem {i}: status = {Status(st).name}, x = {x_sol[i]}")
```

To change values later, pass new arrays in the same shared or batched form to
`update`, then `solve` again; the structure stays fixed. To set up `B` problems that
are all equal for now, give at least one argument its batch axis, for example
`cupy.broadcast_to(c, (B, n))`, and fill in per-problem values with `update`.

### Single problem case:

A single problem is simply the case where every argument is shared. cuPIQP internally
treats it as a batch of one, so **the returned result still carries the batch size 1 as
the leading dimension**.

```python
P_single = cp.eye(2)                 # 2-dim array, no batch dim
c_single = cp.ones(2)                # 1-dim array, no batch dim
A_single = cp.array([[1.0, -2.0]])   # 2-dim array, no batch dim
b_single = cp.array([1.0])           # 1-dim array, no batch dim

solver = DenseSolver()
solver.setup(P=P_single, c=c_single, A=A_single, b=b_single)
solver.solve()

# result.x still carries a leading batch dimension (B, n); here B = 1
x_sol = solver.result.x[0]
```

### Sparse problems
For **sparse** problems, `P`, `A`, `G` are CSR triples `(indptr, indices, values)` of GPU
arrays. The pattern, `indptr` and `indices`, is shared by all `B` problems; the values
follow the same rule as the vectors: `(nnz,)` shared, or `(B, nnz)` per problem, in the
order of `indices`. `update` takes the values alone. For a
[`cupyx.scipy.sparse.csr_matrix`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csr_matrix.html#cupyx.scipy.sparse.csr_matrix)
`M` pass `(M.indptr, M.indices, M.data)`; for a CUDA
[`torch.sparse_csr_tensor`](https://docs.pytorch.org/docs/2.12/generated/torch.sparse_csr_tensor.html)
`T`, `(T.crow_indices(), T.col_indices(), T.values())`.

```python
solver = SparseSolver()
solver.setup(
  P=(P_csr.indptr, P_csr.indices, P_values),   # P_values: (B, P_csr.nnz)
  c=c_batch,                                   # (B, n)
  G=(G_csr.indptr, G_csr.indices, G_csr.data), # values shared by every problem
  h_l=h_l_single, h_u=h_u_single,
  )
solver.solve()
solver.update(P=P_values_new, h_u=h_u_batch)   # (B, P_csr.nnz), (B, m)
solver.solve()
```

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


