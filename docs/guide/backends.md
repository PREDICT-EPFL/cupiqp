# Backends

cuPIQP runs the **same** proximal interior-point algorithm regardless of how the KKT
linear systems are factorized. The backend is chosen by picking the matching
**type-strict solver class**, each of which enforces a one-to-one mapping between its
KKT factorization and the storage category of your `P` / `A` / `G` inputs.

| Solver | Matrices `P, A, G` | KKT backend | Use when |
|---|---|---|---|
| [`DenseSolver`](#densesolver) | dense `cupy` arrays | dense Cholesky | small-to-medium, dense problems |
| [`SparseSolver`](#sparsesolver) | one GPU CSR template + `(B, nnz)` values | sparse LDLᵀ (cuDSS) | large, structurally sparse problems |
| [`MultistageSolver`](#multistagesolver) | `(diag, offdiag)` block arrays | block Cholesky | block-tridiagonal/-arrow KKT (e.g. OCPs) |

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
s.setup(1, P=P, c=c)      # batch size, then one problem
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
    B,                                   # batch size
    P=csr_matrix(P), c=c,                # ONE template problem: 2-D CSR + 1-D vectors
    A=csr_matrix(A), b=b,
    G=csr_matrix(G), h_l=h_l, h_u=h_u,
)
s.update(P=P_values, c=c_b)              # per-problem numbers: (B, nnz) and (B, n)
s.solve()
```

- `setup(batch_size, ...)` takes **one** problem: `P`, `A`, `G` as single 2-D **GPU
  CSR** matrices (`cupyx.scipy.sparse.csr_matrix` or a 2-D CUDA `torch.sparse_csr_tensor`)
  and 1-D vectors. It fixes the structure shared by all `B` problems (sparsity patterns,
  which blocks and bound sides exist) and copies its values into every problem.
- `update` sets per-problem numbers. A matrix is given by its **nonzero values** only, a
  dense `(B, nnz)` array in the CSR order of the template (or `(nnz,)` to share one set of
  values); vectors are `(B, k)` or `(k,)`. The sparsity pattern never changes after
  `setup`, so store every entry that may be nonzero in *any* problem (explicit zeros are
  fine).

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

It takes **plain GPU arrays, block by block**: each block-structured matrix is a
`(diag, offdiag)` tuple of arrays and each vector an array. Generic dense or CSR
matrices are *not* converted to block form, because the structure can only be
exploited if you provide it. With `N` stages of size `d` and `r` constraint rows per
block row:

```python
from cupiqp import MultistageSolver

s = MultistageSolver()
s.setup(
    B,                                  # batch size, then ONE template problem
    P=(P_diag, P_offdiag),              # (N, d, d), (N-1, d, d)
    c=c,                                # (N, d) or flat (N*d,)
    A=(A_diag, A_offdiag), b=b,         # (N, r, d) each; b: (N+1, r) or flat
)
s.update(P=(None, P_offdiag_batch))     # per-problem data (B, ...); None = unchanged
s.solve()
```

| Input | Layout (one problem) |
|---|---|
| `P` | `(P_diag, P_offdiag)`: symmetric block-tridiagonal, lower off-diagonal blocks |
| `A`, `G` | `(diag, offdiag)`, both `(N, r, d)`: block lower-bidiagonal with `N + 1` block rows |
| `c`, `x_l`, `x_u` | `(N, d)` or flat `(N*d,)` |
| `b`, `h_l`, `h_u` | `(N + 1, r)` or flat `((N + 1) * r,)` |

Block row `k` of `A` is `[A_offdiag[k-1], A_diag[k]]` acting on stages `k-1, k`; the last
block row holds only `A_offdiag[N-1]`. Every array passed to `update` is copied once,
in place, into the solver's buffers.

---

## Kernel strategy

cuPIQP runs the same interior-point algorithm at every problem size; only the
inner-loop kernel implementation changes, and the solver picks it automatically at
`setup()` — there is no setting to tune and nothing to change in your code. The inner
loop (step length, barrier parameter `mu`, centering `sigma`, residual and merit
evaluation) is evaluated one of two ways:

- **Fused Warp tile kernels** — JIT-compiled and specialized to the problem dimensions,
  very fast per launch, and they amortize across a batch and across IPM iterations.
  The catch is the compile step: because the kernels are specialized to the problem
  width, the **first-solve compile time grows with the problem** and eventually
  dominates — you can spend more time compiling kernels than actually solving.
- **CuPy axis-reduction kernels** (`cp.min`, `cp.sum`, `cp.max` over the data axis) —
  generic, so they need no shape-specialized compilation and the compile cliff
  disappears. They carry more per-launch overhead, but over a long reduction axis — i.e.
  a wide problem — that overhead is amortized and the trade is worth it. This path also
  builds the Ruiz preconditioner with the tile kernels switched off, for the same reason.

The choice is made from the problem width at `setup()`: with
`tile_width = max(n + p + m, p + num_ineq)`, the solver uses the CuPy axis-reduction
kernels when `tile_width >= 1024` and the fused Warp tile kernels otherwise, where
`num_ineq` is the total number of finite inequality + box-bound rows. It depends
on width only — independent of batch size and dtype, because the thing it avoids
(tile-kernel compile time) depends on shape, not on how many problems you batch. The
threshold is an internal heuristic, not a precisely calibrated constant, and both paths
run the same algorithm and agree to solver tolerance.

!!! note "Batched workloads use the tile kernels"
    The selection is by problem width, not batch size: a large batch of moderately-wide
    problems still uses the fused tile kernels (they amortize across the batch), which is
    the intended behavior. The CuPy path targets a single (or small-batch) wide
    problem where compile time of warp tile-based kernels would dominate the run time.
