# cuPIQP

[![Institution](https://img.shields.io/badge/Institution-Automatic%20Control%20Laboratory,%20EPFL-%23E1251B?style=flat)](https://www.epfl.ch)
[![Funding](https://img.shields.io/badge/Grant-NCCR%20Automation%20(51NF40__180545)-90e3dc.svg)](https://nccr-automation.ch/)
![License](https://img.shields.io/badge/License-BSD--2--Clause-brightgreen.svg)

**cuPIQP** is a GPU-accelerated convex **Quadratic Programming (QP)** solver that
implements the [PIQP](https://github.com/PREDICT-EPFL/piqp) (Proximal Interior Point
Quadratic Programming) algorithm **entirely on NVIDIA GPUs**.

Its core strength is solving **large batches** of small-to-medium QPs in one batched
solver call — the inner kernels operate on `(B, …)` tensors with no Python loop over the
batch — while also exposing the solve as a **differentiable** operation via implicit
differentiation. It **also scales** to large sparse and dense QPs, in the same class as GPU solvers such as
[cuClarabel](https://github.com/cvxgrp/CuClarabel),
[cuOpt](https://github.com/NVIDIA/cuopt), and
[QOCO-GPU](https://github.com/qoco-org/qoco).


## What cuPIQP solves

cuPIQP solves convex QPs of the form

$$
\begin{aligned}
\min_{x} \quad & \tfrac{1}{2} x^\top P x + c^\top x \\
\text{s.t.} \quad & A x = b, \\
& h_l \leq G x \leq h_u, \\
& x_l \leq x \leq x_u,
\end{aligned}
$$

with primal decision variables $x \in \mathbb{R}^n$, matrices $P\in \mathbb{S}_+^n$, $A \in \mathbb{R}^{p \times n}$,  $G \in \mathbb{R}^{m \times n}$, and vectors $c \in \mathbb{R}^n$, $b \in \mathbb{R}^p$, $h_l \in \mathbb{R}^m$, $h_u \in \mathbb{R}^m$, $x_l \in \mathbb{R}^n$, and $x_u \in \mathbb{R}^n$.

See [Problem Formulation](problem-formulation.md) for the full data layout.

<!-- ## A 10-line example

```python
import cupy as cp
from cupiqp import DenseSolver

P = cp.array([[6.0, 0.0],
              [0.0, 4.0]])
c = cp.array([-1.0, -4.0])

solver = DenseSolver()
solver.setup(P=P, c=c)
solver.solve()

print(solver.result.info.status[0].name)   # CUPIQP_SOLVED
print(solver.result.x.get()[0])             # optimal x
```

All problem data lives **on the GPU** — dense arrays as `cupy` arrays, sparse matrices
as `cupyx.scipy.sparse` CSR, or block-structured objects for multistage problems. -->

## Features

- **Native batched solving** — solve $B$ independent QPs in parallel from a single
  solver instance by stacking inputs along a leading batch axis; the inner kernels
  operate on `(B, …)` tensors with no Python-side loop. Built for sampling-based
  control, RL rollouts, and parameter sweeps.
- **Differentiable** — efficiently compute VJPs via implicit differentiation
  by reusing the condensed factor from the forward solve.
- **Scales to large QPs** — the same solver handles large sparse and dense QPs.
- **GPU-resident solver** — the IPM iterations, KKT factorizations, and linear algebra
  run on the GPU; each iteration reads back only a small `(B, …)` diagnostics buffer for
  the host-side convergence check.
- **Versatile problem types** — general dense and sparse QPs, as well as multistage
  optimization problems such as optimal control problems (OCPs).

## Comparison with PIQP

cuPIQP implements the same
[Proximal Interior Point](https://doi.org/10.1007/s12532-024-00263-9) algorithm as
[PIQP](https://github.com/PREDICT-EPFL/piqp), targeting large-scale QPs on NVIDIA GPUs:

|                       | **PIQP** (CPU)                                          | **cuPIQP** (GPU)                                |
| --------------------- | ------------------------------------------------------- | ----------------------------------------------- |
| **Language**          | C++ (with C / Python / Matlab / Julia / Rust bindings)  | Python (CuPy + Warp)                            |
| **Execution**         | CPU (multi-threaded via OpenMP)                         | GPU-resident (CUDA)                             |
| **Batched solving**   | Designed for single solves                              | Designed for batched solves, massive parallelism |
| **Differentiable**    | No                                                      | Yes, via implicit differentiation               |

## Where to next

- New here? Start with [Installation](installation.md) and
  [Getting Started](getting-started.md).
- Building a controller or learning loop? See [Batched Solving](guide/batched.md).
- Differentiating through a solution? See [Differentiation](guide/differentiation.md).
- Looking for a specific class or setting? See the [API Reference](api/index.md).

## Citing

If you use cuPIQP in academic work, please cite the underlying PIQP algorithm paper and
this implementation. A BibTeX entry will be provided once a cuPIQP-specific publication
is available.

## License

BSD-2-Clause. cuPIQP is developed at the
[Automatic Control Laboratory, EPFL](https://www.epfl.ch), with support from
[NCCR Automation](https://nccr-automation.ch/).
