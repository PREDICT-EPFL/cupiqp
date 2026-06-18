# Batched Solving

CuPIQP is **natively batched**. A single solver instance solves `B` independent QPs in
one solver call. 

## Batch as the leading dimension

CuPIQP is natively designed to solve batched problems. Therefore, the problem data should be passed as batchl, and every array in the results has **the batch size as its leading dimension**.

| field |  shape |
|---|---|
| `P`, `c` |  `(B, n, n)`, `(B, n)` |
| `A`, `b` | `(B, p, n)` , `(B, p)` |
| `G`, `h_l`, `h_u` | `(B, m, n)` , `(B, m)`, `(B, m)` |
| `x_l`, `x_u` | `(B, n)`, `(B, n)` |


The solver detects the batch size at `setup()` time from the input shape.

Each problem in the batch carries its own information and converges independently. Read per-problem diagnostics from `solver.result.info` — every field
is a `(B,)` array. See [Results & Status](../api/results.md).


```python
solver = DenseSolver()
solver.setup(
  P=P_batch,       # 3-dim array, with batch size as leading dim
  c=c_batch,       # 2-dim array, with batch size as leading dim
  A=A_batch,       # 3-dim array, with batch size as leading dim
  b=b_batch,       # 2-dim array, with batch size as leading dim
  G=G_batch,       # 3-dim array, with batch size as leading dim
  h_l=h_l_batch,   # 2-dim array, with batch size as leading dim
  h_u=h_u_batch,   # 2-dim array, with batch size as leading dim
  x_l=x_l_batch,   # 2-dim array, with batch size as leading dim
  x_u=x_u_batch,   # 2-dim array, with batch size as leading dim
  )
solver.solve()

x_sol = solver.result.x                  # cupy array of shape (B, n)
status = solver.result.info.status       # list of length B
for i, st in enumerate(status):
    print(f"problem {i}: status = {status.name}, x = {x_sol[i]}")
```

### Single problem case:

A single problem is simply the `B=1` case. CuPIQP accepts problem data as single problem, i.e., without the batch size dimension.

However, cuPIQP internally treat this case as `B=1` and **the returned result still carries the batch size 1 as the leading dimension**.

```python
P_single = cp.eye(2)                 # 2-dim array, no batch dim
c_single = cp.ones(2)                # 1-dim array, no batch dim
A_single = cp.array([[1.0, -2.0]])   # 2-dim array, no batch dim
b_single = cp.array([1.0])           # 1-dim array, no batch dim

solver = DenseSolver()
solver.setup(
  P=P_single, c=c_single,
  A=A_single, b=b_single
  )
solver.solve()

# result.x still carries a leading batch dimension (B, n); here B = 1
x_sol = solver.result.x[0]
```

### Sparse problems
For the **sparse** problems, pass `P`, `A`, `G` as one of the following types:
- [`cupyx.scipy.sparse.csr_matrix`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.csr_matrix.html#cupyx.scipy.sparse.csr_matrix). It only works for batch size 1.
- List or tuple of `cupyx.scipy.sparse.csr_matrix` objects. However, it is **discouraged** when the batch size is big because this can cause long setup time and low performance.
- `UniformBatchedCsrMatrix`. This is a class introduced and used internally by cuPIQP itself to represent a batch of sparse CSR matrices with uniform sparsity. It has `indices` and `indptr` attributes which are two `cupy.ndarray` of shape (nnz,) to represent the sparsity pattern. Its `data` attribute is a `cupy.ndarray` of shape (batch_size, nnz) that stores the non-zeros values of all matrices in the batch. We **encourage** users to import it from cuPIQP to store their batched CSR matrices and pass to the `SparseSolver`.
- [`torch.sparse_csr_tensor`](https://docs.pytorch.org/docs/2.12/generated/torch.sparse_csr_tensor.html), which is the CSR matrix representation in Torch and can express batched CSR matrices. CuPIQP requires that  `crow_indices` and `col_indices` must be identical for all matrices to enforce uniform sparsity.

The vectors $c, b, h_l, h_u, x_l, x_u$ stay stacked `(B, ...)` dense arrays just like the dense case.

See [this example](https://github.com/PREDICT-EPFL/cupiqp/blob/main/examples/getting_started.ipynb) for more details.


## Uniform structure across the batch

All problems in a batch share the same **structure**, even though their numerical data
differ:

- Same shapes `n`, `p`, `m` and (for the sparse backend) the same sparsity pattern.
- For sparse solver, all $P$ matrices in the batch must have the same sparsity pattern. This also applies to $A$ and $G$.

!!! note "Bound patterns may differ per problem"
    The finite/infinite **bound pattern** does **not** have to be shared across the batch.
    Each problem may mark different entries of `h_l`, `h_u`, `x_l`, `x_u` as `±inf`, and you
    may even toggle a bound between finite and `±inf` across solves with `update()` without
    calling `setup()` again. cuPIQP stores a full-length dual/slack vector — one slot per
    row of `G` and per variable — and masks the infinite entries per problem, so no common
    bound pattern is required.


