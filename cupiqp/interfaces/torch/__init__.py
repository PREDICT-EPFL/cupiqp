"""PyTorch interface of cuPIQP: differentiable QP layers.

Each layer is a ``torch.nn.Module`` owning one core solver. ``setup()``
fixes the problem structure and allocates GPU memory once; every call
loads the values it is given, solves on the GPU, and returns the primal
solution. Gradients with respect to every tensor passed to the call flow
through the solve by implicit differentiation of the KKT conditions::

    layer = cupiqp.torch.DenseSolverLayer()
    layer.setup(P, c, A=A, b=b)
    x = layer(c=c_k, b=b_k)
    loss(x).backward()           # c_k.grad, b_k.grad
    layer.result.status          # (B,) int32 status codes, on the device

``layer.result`` holds the duals, slacks and per-problem status of the
latest call, all as CUDA tensors.
"""

from .dense import DenseSolverLayer
from .sparse import SparseSolverLayer
from .types import StructureMismatchError, TorchSolution

__all__ = [
    "DenseSolverLayer",
    "SparseSolverLayer",
    "TorchSolution",
    "StructureMismatchError",
]
