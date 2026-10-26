# PyTorch Layers

`cupiqp.torch` wraps each solver backend in a differentiable `torch.nn.Module`.
A layer solves a batch of QPs on the GPU and returns the primal solution as a
CUDA tensor. `loss.backward()` then propagates gradients through the solve, by
implicit differentiation of the KKT conditions, to every tensor you passed in.

| Layer | Core solver | Matrices at `setup()` | Matrices per call |
|---|---|---|---|
| `DenseSolverLayer` | [`DenseSolver`](../api/solvers.md#densesolver) | dense `(rows, cols)` or `(B, rows, cols)` | same as at setup |
| `SparseSolverLayer` | [`SparseSolver`](../api/solvers.md#sparsesolver) | CUDA sparse CSR tensors | stored values, `(nnz,)` or `(B, nnz)` |
| `MultistageSolverLayer` | [`MultistageSolver`](../api/solvers.md#multistagesolver) | `(diag, offdiag)` block pairs | one tensor per block, e.g. `A_offdiag` |

## Installation

The layers ship with every install of cuPIQP; the `torch` extra installs PyTorch
itself. Combine it with the CUDA extra you need (see
[Installation](../installation.md)):

```bash
python -m pip install "cupiqp[cuda12,torch]"
```

`import cupiqp` does not import PyTorch. It is imported when you first access
`cupiqp.torch`.

## `setup()` and calls

A layer has two phases:

- **`setup()`** fixes the problem structure (dimensions, which constraint blocks
  exist, sparsity patterns, batch size) and allocates every GPU buffer the layer
  will ever need. It takes the same arguments as the core solver's `setup()`.
- **Calling the layer** (`layer(...)`, i.e. `forward()`) loads the tensors you
  pass, solves, and returns `x` of shape `(B, n)`. Every argument you leave out
  keeps its current value.

```python
import torch
import cupiqp

dev, dt = "cuda", torch.float64
P = torch.eye(2, device=dev, dtype=dt)
c = torch.tensor([-1.0, -2.0], device=dev, dtype=dt)
x_l = torch.zeros(2, device=dev, dtype=dt)
x_u = torch.ones(2, device=dev, dtype=dt)

layer = cupiqp.torch.DenseSolverLayer()        # dtype=torch.float64 by default
layer.setup(P, c, x_l=x_l, x_u=x_u)            # structure + allocation, once

c_k = torch.tensor([-0.5, -0.3], device=dev, dtype=dt, requires_grad=True)
x = layer(c=c_k)                               # (1, 2): solve with a new cost
x.sum().backward()
print(c_k.grad)                                # dx/dc = -P^{-1} 1 = (-1, -1)
```

Every argument, at `setup()` and at each call, is either **shared** (single-problem
shape, one value for all `B` problems) or **batched** (leading batch axis `B`),
independently of the others. The batch size is read from the batched arguments of
`setup()` (1 if all are shared). Results always carry the batch axis, also for
`B = 1`.

A call cannot change the structure: passing a block that was not given at
`setup()`, or a tensor of a different shape, raises `StructureMismatchError`.
Call `layer.reset()` and `setup()` again, or use another layer, to change it.

Settings work as on the core solvers, through `layer.settings`. Settings that the
core solver fixes at `setup()` (`enable_grad`, `gradient_smoothing` and the
preconditioner settings) raise at the next call if changed; the others may change
between calls.

## Which tensors receive gradients

Gradients flow to **every tensor passed to the call** that requires grad:
matrices, the linear cost and all constraint vectors. Tensors given only to
`setup()` receive no gradient; pass a tensor to the call to differentiate with
respect to it.

- A **batched** input gets a gradient of the same shape, one per problem.
- A **shared** input gets the gradient **summed over the batch**.
- `SparseSolverLayer` matrices are passed and differentiated as their stored CSR
  values: `layer(P=P_csr.values())` gives a `(nnz,)` gradient in the order of the
  pattern.
- `MultistageSolverLayer` takes each block as its own argument (`P_diag`,
  `P_offdiag`, `A_diag`, `A_offdiag`, `G_diag`, `G_offdiag`), so each block is
  shared, batched and differentiated on its own.

Only `x` is returned by the call and differentiated. The duals, slacks, status and
iteration counts are available in `layer.result` (see below).

Problems whose solve did not converge are still differentiated by default.
Set `layer.settings.gradient_on_failure = "zero"` to give them zero gradients
instead, and check `layer.result.status` to see which problems failed.

## Results

`layer.result` is a `TorchSolution` with the outcome of the latest call, all as
CUDA tensors:

- `x` `(B, n)`; the duals `y`, `z_l`, `z_u`, `z_bl`, `z_bu` and slacks `s_l`, `s_u`,
  `s_bl`, `s_bu` as `(B, k)` tensors, or `None` when that constraint block was not
  given at `setup()`;
- `status` and `iterations`, int32 `(B,)` tensors;
- `info`, the core solver's per-problem diagnostics as Warp arrays.

```python
solved = layer.result.status == cupiqp.Status.CUPIQP_SOLVED   # (B,) bool, on the GPU
if not bool(solved.all()):                                     # explicit host read
    print(layer.result.status.cpu())
```

!!! warning "Results are views of layer-owned memory"
    A call and its backward allocate no GPU memory. The tensors they return, `x`
    included, are views of buffers that the **next call overwrites**. To keep a
    result past the next call, copy it:

    ```python
    x_keep = layer(c=c_k).clone()          # one tensor
    sol = layer.result.clone()             # the whole solution
    ```

    The autograd graph is not affected: `clone()` of `x` is differentiable as usual.
    With `max_pending_solves > 1` (below) each pending call returns views of its own
    saved state, so their outputs stay valid until their backward has run.

## Several solves before one backward: `max_pending_solves`

A differentiable call is *pending* from its forward until its backward has run, or
until its output graph is dropped. `setup(..., max_pending_solves=T)` sets how many
calls may be pending at once.

**`max_pending_solves=1` (default).** The backward reuses the KKT factorization left
by the forward solve, which is the fastest backward and needs no extra memory. While
a call is pending, the layer refuses every further call, even under
`torch.no_grad()`, since any solve would overwrite that factorization. Run
`backward()` first, drop the output, or use separate layer instances.

**`max_pending_solves=T > 1`.** For example the `T` steps of a rollout with one
`backward()` at the end. `setup()` preallocates `T` saved states, each about the size
of one batched solution (all primal, dual and slack variables). Every differentiable
call saves its solution into a free state and returns views of it. Every backward
restores the inputs of its call, refactorizes the KKT system at the saved point and
differentiates there, so the backward costs one extra factorization per call.

`max_pending_solves > 1` requires `per_call_fields`, the names of the arguments every
call passes. Every call, also one under `torch.no_grad()`, must pass exactly those;
the other data keep their setup values. Inputs must be contiguous. Calls that need no
gradient save no state and do not count against the limit.

```python
T = 3
rollout = cupiqp.torch.DenseSolverLayer()
rollout.setup(P, c, x_l=x_l, x_u=x_u, max_pending_solves=T, per_call_fields=("c",))

theta = torch.tensor([-0.4, -0.2], device=dev, dtype=dt, requires_grad=True)
loss, state = 0.0, theta
for k in range(T):
    x_k = rollout(c=state)                 # each call keeps its own output
    loss = loss + (k + 1) * x_k.sum()
    state = -x_k[0]                        # the next cost depends on this solution
loss.backward()
print(theta.grad)
```

A `(T + 1)`-th pending call raises a `RuntimeError`. A solve cannot be
differentiated twice: use a new call instead of `retain_graph=True`.

## Devices and streams

- All tensors must be CUDA tensors of the layer's `dtype` (`torch.float32` or
  `torch.float64`) on the layer's device. CPU tensors and other dtypes are rejected.
- The device is, in this order: the `device` argument of the constructor, the device
  of the `stream` argument, or PyTorch's current CUDA device.
- Pass contiguous tensors. A non-contiguous input is made contiguous by PyTorch,
  which allocates.

The core solver runs on a CUDA stream of its own. At every `setup()`, call and
backward, the layer orders it with a PyTorch stream: the solver waits for the work
already queued on that stream, and the stream waits for the solver before the call
returns. No host synchronization is involved. By default the layer uses
`torch.cuda.current_stream()` at each call; pass `stream=` to the constructor to
pin it to a specific `torch.cuda.Stream`:

```python
s = torch.cuda.Stream()
layer = cupiqp.torch.DenseSolverLayer(stream=s)
layer.setup(P, c, x_l=x_l, x_u=x_u)
with torch.cuda.stream(s):
    x = layer(c=c_k)
```

Results are therefore safe to use in further PyTorch work on that stream right away.
To read them on the host, use `.cpu()` or `.item()`, which wait for the solve.

## Example

A runnable script with both a single differentiable call and a rollout with
`max_pending_solves`:
[`examples/torch_interface_example.py`](https://github.com/PREDICT-EPFL/cupiqp/blob/main/examples/torch_interface_example.py).
See the [API reference](../api/torch.md) for every argument.
