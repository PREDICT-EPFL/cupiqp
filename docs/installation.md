# Installation

## Requirements

- **Python 3.10 or later.**
- **Linux** with an **NVIDIA GPU** and a working CUDA driver.
- The NVIDIA runtime libraries (cuBLAS, cuSOLVER, cuSPARSE, cuDSS) matching your
  CUDA version. The `cuda12` and `cuda13` extras install them for you through
  nvmath-python.

!!! warning "GPU-only"
    cuPIQP runs **only on the GPU**. All problem data must be GPU-resident: a
    `warp.array`, or any CUDA array that exposes `__cuda_array_interface__` or DLPack
    (CuPy arrays, CUDA `torch.Tensor`, JAX arrays on a GPU, ...). CPU arrays such as
    `numpy.ndarray` are **rejected**, not silently copied to the device. Convert them
    explicitly first, e.g. with `wp.array(x_numpy, device="cuda")`.

## Install from source

cuPIQP is not currently published on PyPI. Clone the repository and install it with
the CUDA extra that matches the CUDA version reported by `nvidia-smi`. **Pick your
CUDA version once** in the tabs below; every install command on this page then
follows the same choice.

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

If the NVIDIA runtime libraries matching your CUDA version are already installed in
your environment, the bare local install is enough:

```bash
python -m pip install .
```

## Optional extras

cuPIQP ships all three solver backends (**dense, sparse, and multistage**) and the
PyTorch layers in every install. The extras only add third-party dependencies:

| Extra    | Installs                                                        |
| -------- | --------------------------------------------------------------- |
| `cuda12` | NVIDIA runtime libraries for CUDA 12.x, through nvmath-python   |
| `cuda13` | NVIDIA runtime libraries for CUDA 13.x, through nvmath-python   |
| `torch`  | PyTorch, for the differentiable layers in [`cupiqp.torch`](guide/torch.md) |

Extras combine, for example (from the repository root):

=== "CUDA 12.x"

    ```bash
    python -m pip install ".[cuda12,torch]"
    ```

=== "CUDA 13.x"

    ```bash
    python -m pip install ".[cuda13,torch]"
    ```

!!! note "Choosing the CUDA extra"
    `cuda12` and `cuda13` are mutually exclusive in practice. Installing both resolves,
    but only the libraries matching your driver actually load at runtime. Pick the one
    that matches `nvidia-smi`.

!!! note "PyTorch with CUDA"
    The `torch` extra installs PyTorch from PyPI. If you need a PyTorch build for a
    specific CUDA version, install it first by following the
    [PyTorch instructions](https://pytorch.org/get-started/locally/); cuPIQP then
    uses the installed one.

## Verifying the install

```python
import numpy as np
import warp as wp
from cupiqp import DenseSolver, Status

P = wp.array(np.eye(3), dtype=wp.float64, device="cuda")
c = wp.array(np.ones(3), dtype=wp.float64, device="cuda")

solver = DenseSolver()
solver.settings.verbose = True
solver.setup(P=P, c=c)
solver.solve()                                   # asynchronous: queues the solve

status = solver.result.info.status.numpy()       # .numpy() waits for the solve
assert status[0] == Status.CUPIQP_SOLVED
print(solver.result.x.numpy())                   # [[-1. -1. -1.]]
```

With `verbose = True` you should see the cuPIQP banner followed by the per-iteration
info, ending in a `CUPIQP_SOLVED` status report.
