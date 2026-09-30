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
s.setup(P=P, c=c)      # one problem
s.solve()
```

- `P`, `A`, `G` must be **dense** GPU arrays: 2D shared by every problem, or 3D `(B, …)`
  with one matrix per problem.
- The vector inputs (`c`, `b`, `h_l`, `h_u`, `x_l`, `x_u`) are dense GPU arrays: 1D shared,
  or 2D `(B, k)` per problem. The batch size `B` is read from the batched inputs.

---

## `SparseSolver`

The sparse backend uses a sparse LDLᵀ direct factorization (cuDSS) and is far more
efficient than the dense backend for large, structurally sparse problems.

```python
from cupiqp import SparseSolver

s = SparseSolver()
s.setup(
    P=(P_indptr, P_indices, P_values),   # values (nnz,) shared, or (B, nnz) per problem
    c=c,                                 # (n,) or (B, n)
    A=(A_indptr, A_indices, A_values), b=b,
    G=(G_indptr, G_indices, G_values), h_l=h_l, h_u=h_u,
)
s.solve()
s.update(P=P_values_new, c=c_new)        # values only: (B, nnz) and (B, n)
s.solve()
```

- `P`, `A`, `G` are **CSR triples** `(indptr, indices, values)` of GPU arrays. `indptr`
  and `indices` are `int32` or `int64`, with column indices sorted within each row; for a
  cupyx `csr_matrix` `M` pass `(M.indptr, M.indices, M.data)`, for a CUDA torch CSR
  tensor `T` pass `(T.crow_indices(), T.col_indices(), T.values())`. `setup` checks the
  pattern once.
- The pattern is shared by every problem and fixed by `setup`: store every entry that may
  be nonzero in *any* problem (explicit zeros are fine). Values and vectors are shared
  (`(nnz,)`, `(k,)`) or batched (`(B, nnz)`, `(B, k)`), at `setup` and at `update`;
  `update` takes a matrix's values only.

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
    P=(P_diag, P_offdiag),              # (N, d, d), (N-1, d, d): shared by every problem
    c=c_batch,                          # (B, N, d): batched, so this is a batch of B
    A=(A_diag, A_offdiag), b=b,         # (N, r, d) each; b: (N+1, r) or flat
)
s.solve()
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

## Streams

Each solver instance runs on **one CUDA stream** of its own: `setup`, `update`,
`solve`, `backward` (and `OcpSolver.set`) make it Warp's current stream for their
duration, so every kernel, copy, library call and CUDA graph of the solver is
issued there. The stream is exposed as `solver.stream` (a `warp.Stream`).

By default the solver creates the stream. It is created blocking with respect to
the legacy default stream, so plain cupy / torch / numba code on the default stream
is ordered with the solver automatically. To run the solver on a stream of your own,
pass a `warp.Stream` to the constructor:

```python
stream = wp.Stream("cuda:0")
solver = DenseSolver(stream=stream)      # borrowed: the solver never destroys it
```

Only `warp.Stream` objects are accepted by the core solvers. Producers or consumers
on other non-default streams must order themselves with `solver.stream` (for
example with `wp.ScopedStream(solver.stream)` around the work, or Warp events).
The framework adapters do this for you: `cupiqp.torch` orders every call with
the `torch.cuda.Stream` given to its constructor (or torch's current stream).
The sparse backend enters a non-owning cupy view of the same stream during
`setup()`, where it builds the KKT sparsity pattern with `cupyx`; it is an
implementation detail of that backend.

## Inner-loop kernels

cuPIQP runs the same interior-point algorithm at every problem size, with one set
of Warp kernels. The per-element steps of the iteration (variable updates, the
Newton right-hand sides, the regularization terms) are Warp kernels specialized to
the block widths fixed at `setup()`, and every reduction of the iteration (step
lengths, the barrier parameter `mu`, the centering parameter `sigma`, residual norms
and objectives) is a fixed-size block reduction: one CUDA block per problem of the
batch, each thread striding over the row. These reduction kernels read the problem
width from the array shapes, so they are compiled once per dtype and serve a
3-variable QP and a QP whose KKT width is in the thousands alike - there is no
compile time that grows with the problem width, no width-dependent code path, and
nothing to select.

Every backend stores its data as Warp arrays: problem data, iterates and
workspaces, and for the sparse backend also the CSR pattern (`int32` row pointers
and column indices) and the `(B, nnz)` value buffers. The dense linear algebra
calls cuBLAS / cuSOLVER and the sparse backend calls cuSPARSE / cuDSS on those
buffers directly, and CUDA graphs are captured with Warp. CuPy appears only in
the sparse backend at `setup()`, where `cupyx.scipy.sparse` assembles the KKT
pattern and the index maps that the Warp kernels use afterwards.
