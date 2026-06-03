# cuPIQP

[![Institution](https://img.shields.io/badge/Institution-Automatic%20Control%20Laboratory,%20EPFL-%23E1251B?style=flat)](https://www.epfl.ch)
[![Funding](https://img.shields.io/badge/Grant-NCCR%20Automation%20(51NF40__180545)-90e3dc.svg)](https://nccr-automation.ch/)
![License](https://img.shields.io/badge/License-BSD--2--Clause-brightgreen.svg)

CuPIQP is a GPU-accelerated convex Quadratic Programming (QP) solver implementing the [PIQP](https://github.com/PREDICT-EPFL/piqp) (Proximal Interior Point Quadratic Programming) algorithm entirely on NVIDIA GPUs. Its core strength is solving **large batches** of small-to-medium QPs in a single GPU launch, while exposing the solve as a **differentiable** layer for PyTorch and JAX. It **also scales to large-scale** sparse and dense QPs, in the same class as GPU solvers such as [cuClarabel](https://github.com/cvxgrp/CuClarabel), [cuOpt](https://github.com/NVIDIA/cuopt), and [QOCO-GPU](https://github.com/qoco-org/qoco).

## Problem Formulation

cuPIQP solves convex QPs of the form:

$$
\begin{aligned}
\min_{x} \quad & \tfrac{1}{2} x^\top P x + c^\top x \\
\text{s.t.} \quad & A x = b \\
& h_l \leq G x \leq h_u \\
& x_l \leq x \leq x_u
\end{aligned}
$$

where $P \succeq 0$ is positive semidefinite, $x \in \mathbb{R}^n$ is the decision variable, $A \in \mathbb{R}^{p \times n}$ defines equality constraints, and $G \in \mathbb{R}^{m \times n}$ defines two-sided inequality constraints. Any bound may be $\pm\infty$ and is handled without numerical penalty.

## Features

- **Native batched solving** — solve $B$ independent QPs in parallel from a single solver instance by stacking inputs along a leading batch axis; the inner kernels operate on `(B, …)` tensors with no Python-side loop. Built for sampling-based control, RL rollouts, and parameter sweeps.
- **Differentiable** — efficient computation of the VJPs via implicit differentiation by reusing the condensed factor from the forward solve. Integration into PyTorch and JAX are on the way!
- **Scales to large QPs** — the same solver handles large sparse and dense QPs, competing with GPU solvers such as cuClarabel, cuOpt, and QOQO-GPU.
- **Fully GPU-resident solver** — all iterations, KKT factorizations, and linear algebra run on the GPU with very few host–device synchronization during solve.
- **CUDA Graph capture** — solver iterations are recorded as CUDA graphs and replayed with near-zero kernel-launch overhead.
- **Versatile problem types** — supports general dense and sparse QPs, as well as multistage optimization problems like optimal control problems (OCPs).

## Installation

### Requirements

- Python 3.10 or later.
- Linux with an NVIDIA GPU and a working CUDA driver/runtime stack.
- CUDA Python packages compatible with the installed CUDA stack. This repository
  defines extras for CUDA 12.x and CUDA 13.x, including CuPy and nvmath runtime libraries.

cuPIQP is not currently published on PyPI. From a local clone, install it
with one CUDA extra:

```bash
git clone https://github.com/PREDICT-EPFL/cupiqp.git
cd cupiqp
python -m pip install ".[cuda12]"  # choose for a CUDA 12.x CuPy environment
# or:
python -m pip install ".[cuda13]"  # choose for a CUDA 13.x CuPy environment
```

If an appropriate CuPy installation is already present in the environment,
the base local install is:

```bash
python -m pip install .
```


### Verifying the install

```python
import cupy as cp
from cupiqp import DenseSolver

solver = DenseSolver()
solver.settings.verbose = True
solver.setup(P=cp.eye(3), c=cp.zeros(3))
solver.solve()
```

### Runtime dependencies (for reference)

Pulled automatically by the relevant extras above:

- [CuPy](https://cupy.dev/) — GPU array library (`cupy-cuda12x` or `cupy-cuda13x`).
- [Warp](https://github.com/NVIDIA/warp) — JIT-compiled CUDA kernels.
- [nvmath-python](https://developer.nvidia.com/nvmath-python) — cuBLAS / cuSOLVER / cuSPARSE / cuDSS bindings and CUDA runtime packages via the selected CUDA extra.
- [NVTX](https://github.com/NVIDIA/NVTX) — profiling annotations.
- [socu](https://github.com/PREDICT-EPFL/socu) — required by the `MultistageSolver` as the linear system solver.

## Quick Start


## Comparison with PIQP

CuPIQP implements the same [Proximal Interior Point](https://doi.org/10.1007/s12532-024-00263-9) algorithm as [PIQP](https://github.com/PREDICT-EPFL/piqp), targeting large-scale QPs on NVIDIA GPUs:

| | **PIQP** (CPU) | **CuPIQP** (GPU) |
|---|---|---|
| **Language** | C++ (with C / Python / Matlab / Julia / Rust bindings) | Python (CuPy + Warp) |
| **Execution** | CPU (multi-threaded via OpenMP) | Fully GPU-resident (CUDA) |
| **Batched solving** | Designed for single solves | Designed for batched solves with massive parallelism |
| **Differentiable** | No | Yes, via implicit differentiation |


## Citing

If you use cuPIQP in academic work, please cite the underlying PIQP algorithm
paper and this implementation. A BibTeX entry will be provided once a
cuPIQP-specific publication is available.

## License

BSD-2-Clause. See `LICENSE`.
