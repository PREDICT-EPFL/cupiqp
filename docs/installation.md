# Installation

## Requirements

- **Python 3.10 or later.**
- **Linux** with an **NVIDIA GPU** and a working CUDA driver/runtime stack.
- CUDA Python packages compatible with the installed CUDA stack. cuPIQP defines extras
  for **CUDA 12.x** and **CUDA 13.x**, which pull in the matching CuPy and nvmath
  runtime libraries.

!!! warning "GPU-only"
    cuPIQP runs **only on the GPU**. All problem data must be GPU-resident
    (`cupy` arrays, `cupyx.scipy.sparse` CSR matrices, or CUDA `torch.Tensor` /
    JAX arrays). CPU arrays such as `numpy.ndarray` are **rejected**, not silently
    copied to the device — convert them explicitly first.

## Install from source

cuPIQP is not currently published on PyPI. Clone the repository and install it with the
CUDA extra that matches the CUDA version reported by `nvidia-smi`:

```bash
git clone https://github.com/PREDICT-EPFL/cupiqp.git
cd cupiqp

python -m pip install ".[cuda12]"   # for a CUDA 12.x CuPy environment
# or:
python -m pip install ".[cuda13]"   # for a CUDA 13.x CuPy environment
```

If a suitable CuPy installation is already present in your environment, the bare local
install is enough:

```bash
python -m pip install .
```

## Optional extras

The CUDA extra can be combined with feature extras (comma-separated, no spaces):

| Extra         | Enables                                                              | Example                              |
| ------------- | ------------------------------------------------------------------- | ------------------------------------ |
| `cuda12`      | CuPy + nvmath runtime for CUDA 12.x                                 | `pip install ".[cuda12]"`            |
| `cuda13`      | CuPy + nvmath runtime for CUDA 13.x                                 | `pip install ".[cuda13]"`            |
| `multistage`  | `MultistageSolver` (pulls [`socu`](https://github.com/PREDICT-EPFL/socu)) | `pip install ".[cuda13,multistage]"` |

!!! note "Choosing the CUDA extra"
    `cuda12` and `cuda13` are mutually exclusive in practice. Installing both resolves,
    but only the wheel matching your driver actually loads at runtime. Pick the one that
    matches `nvidia-smi`.

## Verifying the install

```python
import cupy as cp
from cupiqp import DenseSolver

solver = DenseSolver()
solver.settings.verbose = True
solver.setup(P=cp.eye(3), c=cp.zeros(3))
solver.solve()
```

With `verbose = True` you should see the cuPIQP banner followed by the interior-point
iteration log, ending in a `CUPIQP_SOLVED` status.

## Runtime dependencies

Pulled automatically by the extras above:

- [CuPy](https://cupy.dev/) — GPU array library (`cupy-cuda12x` or `cupy-cuda13x`).
- [Warp](https://github.com/NVIDIA/warp) — JIT-compiled CUDA kernels.
- [nvmath-python](https://developer.nvidia.com/nvmath-python) — cuBLAS / cuSOLVER /
  cuSPARSE / cuDSS bindings and CUDA runtime packages via the selected CUDA extra.
- [NVTX](https://github.com/NVIDIA/NVTX) — profiling annotations.
- [socu](https://github.com/PREDICT-EPFL/socu) — required by `MultistageSolver` as the
  block-structured linear-system solver (installed via the `multistage` extra).

Framework-independent dependencies (`numpy`, `scipy`, `warp-lang`, `nvmath-python`,
`nvtx`) resolve cleanly from PyPI; the CUDA-bound packages come from the CUDA extras.
