# Backends

cuPIQP runs the **same** proximal interior-point algorithm regardless of how the KKT
linear systems are factorized. The backend is chosen by picking the matching
**type-strict solver class**, each of which enforces a one-to-one mapping between its
KKT factorization and the storage category of your `P` / `A` / `G` inputs.

| Solver | Matrices `P, A, G` | KKT backend | Use when |
|---|---|---|---|
| [`DenseSolver`](#densesolver) | dense `cupy` arrays | dense Cholesky | small-to-medium, dense problems |
| [`SparseSolver`](#sparsesolver) | [`UniformBatchedCsrMatrix`](../api/solvers.md#cupiqp.UniformBatchedCsrMatrix) (or CSR) | sparse LDLᵀ (cuDSS) | large, structurally sparse problems |
| [`MultistageSolver`](#multistagesolver) | block-structured objects | block Cholesky | block-tridiagonal/-arrow KKT (e.g. OCPs) |

All three accept GPU-resident inputs only and share the same `setup` / `solve` /
`update` workflow and [`Settings`](../api/settings.md). See
[Re-solving with new data](../getting-started.md#re-solving-with-new-data) for the
fixed-structure update pattern.

!!! info "GPU arrays only — no silent host copies"
    Every non-`None` input must already be a GPU array. CPU arrays
    (`numpy.ndarray`, CPU torch tensors, CPU JAX arrays) are **rejected** with an
    actionable `TypeError` rather than copied to the device. Convert first:

    ```python
    P_cuda = cupy.asarray(P_numpy)                  # cupy
    P_cuda = torch.tensor(P_numpy, device="cuda")   # torch
    ```

    Dense inputs are accepted via the
    [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)
    protocol, which unifies CuPy, CUDA `torch.Tensor`, CUDA JAX arrays, and Numba CUDA
    device arrays behind one check.

---

## `DenseSolver`

The dense backend factorizes a condensed KKT matrix with a batched dense Cholesky. It
is the right choice for small-to-medium problems whose matrices are essentially dense.

```python
import cupy as cp
from cupiqp import DenseSolver

P = cp.eye(4)
c = cp.zeros(4)

s = DenseSolver()
s.setup(P=P, c=c)
s.solve()
```

- `P`, `A`, `G` must be **dense** GPU arrays (2D for a single problem, 3D `(B, …)` for a
  batch).
- The vector inputs (`c`, `b`, `h_l`, `h_u`, `x_l`, `x_u`) are dense GPU vectors.

---

## `SparseSolver`

The sparse backend uses a sparse LDLᵀ direct factorization (cuDSS) and is far more
efficient than the dense backend for large, structurally sparse problems.

```python
from cupyx.scipy.sparse import csr_matrix
from cupiqp import SparseSolver

s = SparseSolver()
s.setup(
    P=csr_matrix(P), c=c,
    A=csr_matrix(A), b=b,
    G=csr_matrix(G), h_l=h_l, h_u=h_u,
)
s.solve()
```

- `P`, `A`, `G` are **GPU CSR** matrices (`cupyx.scipy.sparse.csr_matrix`); a single CSR
  is treated as the `B = 1` case.
- For a **batch**, the preferred input is a
  [`UniformBatchedCsrMatrix`](../api/solvers.md#cupiqp.UniformBatchedCsrMatrix) —
  cuPIQP's own container holding `B` matrices that share **one** sparsity pattern, with
  the values stacked as a `(B, nnz)` array. The vectors are stacked `(B, …)`.

```python
from cupiqp import UniformBatchedCsrMatrix

# Pack each shared-pattern matrix into a (B, nnz) batched container.
# from_cupy_csr_matrix replicates one CSR across the batch; for differing
# per-problem values, build with the UniformBatchedCsrMatrix(B, indices, indptr,
# values, shape=...) constructor instead (see Getting Started).
P_b = UniformBatchedCsrMatrix.from_cupy_csr_matrix(csr_matrix(P), batch_size=B)
A_b = UniformBatchedCsrMatrix.from_cupy_csr_matrix(csr_matrix(A), batch_size=B)
G_b = UniformBatchedCsrMatrix.from_cupy_csr_matrix(csr_matrix(G), batch_size=B)
s.setup(P=P_b, c=c_b, A=A_b, b=b_b, G=G_b, h_l=h_l_b, h_u=h_u_b)
```

!!! warning "Avoid passing a raw `list` of `csr_matrix`"
    `setup` also accepts a plain `list` of `B` CSR matrices that share one pattern, but
    separate matrix objects lack the uniform stride batched routines need, so cuPIQP must
    copy them into a `UniformBatchedCsrMatrix` at `setup`. Build and pass one yourself to
    skip that copy. See [Getting Started](../getting-started.md) for the full example.

!!! tip "Bit-reproducible cuDSS"
    Set `settings.use_deterministic_mode_for_cudss = True` for bit-wise reproducible
    sparse factorizations (somewhat slower). See [Settings](../api/settings.md).

---

## `MultistageSolver`

The multistage backend exploits **block-tridiagonal / block-tridiagonal-arrow** KKT
structure — the structure that arises in optimal control problems (OCPs) and other
multistage programs — with a block Cholesky factorization. It requires the
[`socu`](https://github.com/PREDICT-EPFL/socu) extra (install with
`pip install ".[cuda13,multistage]"`).

It accepts **block-structured storage end-to-end**: generic CSR is *not* auto-promoted
to block form, because if you have not built the block matrices the multistage solver
cannot exploit the structure anyway.

```python
from cupiqp import MultistageSolver
from cupiqp.multistage.multistage_utils import (
    BlockTridiagMat, BlockBidiagMat, BlockVec,
)

P = BlockTridiagMat(num_diag_blocks=N, block_size=d)
A = BlockBidiagMat(rows_of_blocks=d, cols_of_blocks=d, N=N)
c = BlockVec(num_blocks=N, rows=d)
b = BlockVec(num_blocks=N, rows=d)
# ... fill block data ...

s = MultistageSolver()
s.setup(P=P, c=c, A=A, b=b)
s.solve()
```

| Input | Type |
|---|---|
| `P` | `BlockTridiagMat` |
| `A`, `G` | `BlockBidiagMat` (or `None`) |
| `c`, `b`, `h_u`, `h_l`, `x_u`, `x_l` | `BlockVec` (or `None`) |

---

## Large-problem variants

For each backend there is a `*LargeProblemSolver` companion:

```python
from cupiqp import (
    DenseLargeProblemSolver,
    SparseLargeProblemSolver,
    MultistageLargeProblemSolver,
)
```

These replace the shape-specialized Warp **tile kernels** in the inner IPM loop with
CuPy axis-reduction kernels. Use them when `max(n, p, m)` is large enough that the Warp
tile-kernel **compile time dominates first-solve latency**, and the per-launch overhead
of CuPy reductions amortizes well at that scale. They are numerically equivalent to the
regular backends (agree to solver tolerance) and share the same API — only the kernel
strategy differs.

!!! note "Rule of thumb"
    Reach for a `*LargeProblemSolver` only when first-solve latency on a large,
    single (or small-batch) problem is dominated by kernel compilation. For batched
    small-to-medium problems, the standard solvers are faster.
