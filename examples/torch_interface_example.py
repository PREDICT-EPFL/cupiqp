"""Minimal example of the PyTorch interface: a differentiable QP layer.

Solves the box-constrained QP

    minimize    0.5 x^T P x + c^T x
    subject to  x_l <= x <= x_u

with P = I, whose unconstrained minimizer is x = -c, differentiates the
solution with respect to c, and shows max_pending_solves for a rollout of
several solves with one backward pass.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import cupiqp

device = torch.device("cuda")
P = torch.eye(2, device=device, dtype=torch.float64)
c = torch.tensor([-1.0, -2.0], device=device, dtype=torch.float64)
x_l = torch.zeros(2, device=device, dtype=torch.float64)
x_u = torch.ones(2, device=device, dtype=torch.float64)

# setup() fixes the structure and allocates once; a call loads new values,
# solves, and returns x of shape (B, n).
layer = cupiqp.torch.DenseSolverLayer()
layer.settings.verbose = False
layer.setup(P, c, x_l=x_l, x_u=x_u)

x = layer()
print(f"status: {cupiqp.Status(layer.result.status[0].item()).name}")
print(f"x:      {x}")                      # (1, 1): the bound is active

# Gradients flow to every tensor passed to the call, here c, by implicit
# differentiation of the KKT system.
c_grad = torch.tensor([-0.5, -0.3], device=device, dtype=torch.float64, requires_grad=True)
x = layer(c=c_grad)
print(f"x:      {x}")                      # (0.5, 0.3): interior solution
x.sum().backward()
print(f"dx/dc:  {c_grad.grad}")            # -P^{-1} 1 = (-1, -1) at an interior solution

# Rollout: several differentiable calls, one backward. With
# max_pending_solves=T, up to T calls may await their backward: every call
# saves its solution (the size of one solution) into storage allocated at
# setup, and its backward refactorizes the KKT system at that point.
T = 3
rollout = cupiqp.torch.DenseSolverLayer()
rollout.settings.verbose = False
rollout.setup(P, c, x_l=x_l, x_u=x_u, max_pending_solves=T, per_call_fields=("c",))
theta = torch.tensor([-0.4, -0.2], device=device, dtype=torch.float64, requires_grad=True)
loss = 0.0
state = theta
for k in range(T):
    x_k = rollout(c=state)                 # each call keeps its own output
    loss = loss + (k + 1) * x_k.sum()
    state = -x_k[0]                        # the next cost depends on this solution
loss.backward()
print(f"rollout dloss/dtheta: {theta.grad}")
