# PyTorch Layers

Differentiable `torch.nn.Module` wrappers of the solvers, available as
`cupiqp.torch`. See the [PyTorch Layers guide](../guide/torch.md) for the workflow,
the gradient rules and `max_pending_solves`.

## DenseSolverLayer

::: cupiqp.interfaces.torch.DenseSolverLayer
    options:
      inherited_members: true
      members: [setup, forward, reset, settings, result]

## SparseSolverLayer

::: cupiqp.interfaces.torch.SparseSolverLayer
    options:
      inherited_members: true
      members: [setup, forward, reset, settings, result]

## MultistageSolverLayer

::: cupiqp.interfaces.torch.MultistageSolverLayer
    options:
      inherited_members: true
      members: [setup, forward, reset, settings, result]

## TorchSolution

::: cupiqp.interfaces.torch.TorchSolution
    options:
      members: [clone]

## StructureMismatchError

::: cupiqp.interfaces.torch.StructureMismatchError
