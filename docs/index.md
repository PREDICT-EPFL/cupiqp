# cuPIQP

[![Institution](https://img.shields.io/badge/Institution-Automatic%20Control%20Laboratory,%20EPFL-%23E1251B?style=flat)](https://www.epfl.ch)
[![Funding](https://img.shields.io/badge/Grant-NCCR%20Automation%20(51NF40__180545)-90e3dc.svg)](https://nccr-automation.ch/)
![License](https://img.shields.io/badge/License-BSD--2--Clause-brightgreen.svg)

**cuPIQP** is a GPU-native convex **Quadratic Programming (QP)** solver that
implements the [PIQP](https://github.com/PREDICT-EPFL/piqp) (Proximal Interior Point
Quadratic Programming) algorithm **entirely on NVIDIA GPUs**.

The core strength of cuPIQP is to solve **large batches** of small-to-medium QPs in parallel while also exposing the solve as a **differentiable** operation via implicit differentiation. 

In addition to batched workloads, cuPIQP can also solve large-scale sparse and dense QPs, alongside GPU solvers such as [cuClarabel](https://github.com/cvxgrp/CuClarabel), [cuOpt](https://github.com/NVIDIA/cuopt), and [QOCO-GPU](https://github.com/qoco-org/qoco).


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


## Features

- **Native batched solving** — solve multiple independent QPs with the same dimension and sparsity pattern in parallel from a single solver instance by stacking inputs along a leading batch axis.
- **Differentiable** — efficiently compute VJPs via implicit differentiation
  by reusing the condensed factor from the forward solve.
- **Scales to large QPs** — the same solver handles large sparse and dense QPs.
- **Robust** - implements proximal iterior point method, with modern techniques including preconditioner, iterative refinement, etc.
- **GPU-resident** — the IPM iterations, KKT factorizations, and linear algebra run on the GPU.
- **Versatile problem types** — supports general dense and sparse QPs, as well as multistage optimization problems such as optimal control problems (OCPs).

## Comparison with PIQP

cuPIQP implements the same
[Proximal Interior Point](https://doi.org/10.1007/s12532-024-00263-9) algorithm as
[PIQP](https://github.com/PREDICT-EPFL/piqp), targeting large-scale QPs on NVIDIA GPUs:

|                       | **PIQP** (CPU)                                          | **cuPIQP** (GPU)                                |
| --------------------- | ------------------------------------------------------- | ----------------------------------------------- |
| **Language**          | C++ (with C / Python / Matlab / Julia / Rust bindings)  | Python (CuPy + Warp)                            |
| **Execution**         | CPU (multi-threaded via OpenMP)                         | GPU-resident (CUDA)                             |
| **Batched solving**   | Limited                             | Massive |
| **Differentiable**    | No                                                      | Yes, via implicit differentiation               |

## Dependencies

CuPIQP is built on multiple existing libraries, including:

- [CuPy](https://cupy.dev/) — GPU array library (`cupy-cuda12x` or `cupy-cuda13x`).
- [Warp](https://github.com/NVIDIA/warp) — JIT-compiled CUDA kernels.
- [nvmath-python](https://developer.nvidia.com/nvmath-python) — cuBLAS / cuSOLVER /
  cuSPARSE / cuDSS bindings and CUDA runtime packages via the selected CUDA extra.
- [NVTX](https://github.com/NVIDIA/NVTX) — profiling annotations.
- [socu](https://github.com/PREDICT-EPFL/socu) — required by `MultistageSolver` as the
  block-structured linear-system solver (installed by default as a core dependency).

Framework-independent dependencies (`numpy`, `scipy`, `warp-lang`, `nvmath-python`,
`nvtx`) resolve cleanly from PyPI; the CUDA-bound packages come from the CUDA extras.

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
