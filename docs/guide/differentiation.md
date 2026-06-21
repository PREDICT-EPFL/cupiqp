# Differentiation

cuPIQP computes reverse-mode **vector-Jacobian products (VJPs)** through a solved QP
using implicit differentiation of the KKT conditions. The backward pass reuses the KKT
factorization from the forward solve instead of constructing the full solution Jacobian.

If an outer scalar loss $\ell$ depends on the solution $x^\star$, call
`backward()` with the upstream gradient $\partial \ell / \partial x^\star$. The returned
data object contains the corresponding gradients with respect to `P`, `c`, `A`, `b`,
`G`, `h_l`, `h_u`, `x_l`, and `x_u`. A bound side that was omitted at `setup()` has no
gradient block (its gradient is absent / `(B, 0)`), mirroring the inputs you provided.

## Basic workflow

Enable gradient support **before** `setup()` by setting `solver.settings.enable_grad=True`, because setup allocates the backward
buffers and prepares backend-specific kernels:

```python
import cupy as cp
from cupiqp import DenseSolver, Status

P = cp.eye(2)
c = cp.array([-1.0, 2.0])

solver = DenseSolver()
solver.settings.enable_grad = True  # enable gradient computation
solver.setup(P=P, c=c)

status = solver.solve()              # always a list, one Status per problem
assert status[0] == Status.CUPIQP_SOLVED

x_star = solver.result.x

# l = 0.5 * ||x*||^2, so dl/dx* = x*
grad_data = solver.backward(grad_x=x_star)

grad_c = grad_data.c
grad_P = grad_data.P
```

For a single problem, solution and gradient arrays still carry a leading batch
dimension, so `x_star` and `grad_c` above have shape `(1, 2)`.

## Upstream gradients

`backward()` accepts upstream gradients for every solution variable:

```python
grad_data = solver.backward(
    grad_x=grad_x,
    grad_y=grad_y,
    grad_z_l=grad_z_l,
    grad_z_u=grad_z_u,
    grad_z_bl=grad_z_bl,
    grad_z_bu=grad_z_bu,
    grad_s_l=grad_s_l,
    grad_s_u=grad_s_u,
    grad_s_bl=grad_s_bl,
    grad_s_bu=grad_s_bu,
)
```

Each upstream gradient must have the same shape as its corresponding field in `solver.result`.
Omitted upstream gradients are treated as zero, so losses that depend only on `x` usually need
only `grad_x`.

The returned object matches the selected backend:

| Solver | Return type | Matrix-gradient storage |
|---|---|---|
| `DenseSolver` | `DenseData` | dense arrays with a leading batch dimension |
| `SparseSolver` | `SparseData` | values at the original CSR structural nonzeros |
| `MultistageSolver` | `MultistageData` | block-structured matrices and vectors |

All returned gradients are expressed in the original, unscaled problem coordinates;
the backward pass handles any solver preconditioning internally.

!!! warning "The high-level `OcpSolver` is not differentiable yet"
    Gradients are supported on `DenseSolver`, `SparseSolver`, and `MultistageSolver`
    (over their raw QP data). The high-level [`OcpSolver`](backends.md#multistagesolver)
    is **not** differentiable: it has no mapping from solution gradients back to the OCP
    fields you set (`Q`, `R`, `A`, `B`, `x0`, ...), and `enable_grad` is unsupported on
    it. To differentiate an optimal-control problem today, assemble the QP yourself and
    use `DenseSolver` or `SparseSolver`.

## Repeated solves

The backward pass can follow each solve in a fixed-structure update loop:

```python
for c_k in costs:
    solver.update(c=c_k)
    solver.solve()

    grad_x = outer_loss_gradient(solver.result.x)
    grad_data = solver.backward(grad_x=grad_x)
```

As with the forward solution, every problem in a batch is differentiated independently
and gradients carry the leading batch dimension `(B, ...)`.

!!! warning "Gradient buffers are reused"
    Each solver returns the same internal gradient-data object on every `backward()`
    call. Its buffers are overwritten by the next backward pass. Copy any gradient that
    must be retained:

    ```python
    grad_c = grad_data.c.copy()
    ```

!!! note "Differentiability at active-set changes"
    The VJP describes the local solution map at the converged QP. At points where the
    active set changes, that map can be nonsmooth and the gradient may be discontinuous
    or undefined.

See the [solver API](../api/solvers.md) for the full `backward()` signature and
[Settings](../api/settings.md) for `enable_grad` and solver
tolerances.
