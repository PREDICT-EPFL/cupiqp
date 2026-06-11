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
CUDA extra that matches the CUDA version reported by `nvidia-smi`. **Pick your CUDA
version once** in the tabs below — every install command on this page then follows the
same choice.

=== "CUDA 12.x"

    ```bash
    git clone https://github.com/PREDICT-EPFL/cupiqp.git
    cd cupiqp
    python -m pip install ".[cuda12]"
    ```

=== "CUDA 13.x"

    ```bash
    git clone https://github.com/PREDICT-EPFL/cupiqp.git
    cd cupiqp
    python -m pip install ".[cuda13]"
    ```

If a suitable CuPy installation is already present in your environment, the bare local
install is enough:

```bash
python -m pip install .
```

## Optional extras

cuPIQP ships all three solver backends — **dense, sparse, and multistage** — by
default, so the plain install above (with the matching CUDA extra) is all you need.
There is deliberately no separate `multistage` extra. The available install extras are:

| Extra    | Enables                             |
| -------- | ----------------------------------- |
| `cuda12` | CuPy + nvmath runtime for CUDA 12.x |
| `cuda13` | CuPy + nvmath runtime for CUDA 13.x |

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
  block-structured linear-system solver (installed by default as a core dependency).

Framework-independent dependencies (`numpy`, `scipy`, `warp-lang`, `nvmath-python`,
`nvtx`) resolve cleanly from PyPI; the CUDA-bound packages come from the CUDA extras.
