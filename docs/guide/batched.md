# Batched Solving

cuPIQP is **natively batched**. A single solver instance solves `B` independent QPs in
one batched solver call — the inner kernels operate on `(B, …)` tensors with **no
Python-side loop** over the batch. This is the feature cuPIQP is built around: sampling-based
control, RL rollouts, scenario MPC, and parameter sweeps all map directly onto it.

A single problem is simply the `B = 1` case.

## The one rule: batch is the leading axis

Every array's **leading dimension is the batch size** `B`:

| array | single | batched |
|---|---|---|
| `P` | `(n, n)` | `(B, n, n)` |
| `c`, `x_l`, `x_u` | `(n,)` | `(B, n)` |
| `A` / `G` | `(p, n)` / `(m, n)` | `(B, p, n)` / `(B, m, n)` |
| `b`, `h_l`, `h_u` | `(p,)` / `(m,)` | `(B, p)` / `(B, m)` |

The solver detects a batch from the matrix rank (`P.ndim == 3`, or a list of matrices)
at `setup()` time. `solver.result.x` then has shape `(B, n)` and
`solver.result.info.status` is a list of `B` `Status` enums — one per problem.

```python
dense_solver.setup(P=P_b, c=c_b, A=A_b, b=b_b, G=G_b,
                   h_l=h_l_b, h_u=h_u_b, x_l=x_l_b, x_u=x_u_b)
dense_solver.solve()

X = dense_solver.result.x.get()                  # (B, n)
for i, st in enumerate(dense_solver.result.info.status):
    print(f"problem {i}: status = {st.name}, x = {X[i]}")
```

For the **sparse** backend, pass `P`, `A`, `G` as **lists of `B` CSR matrices that
share the same sparsity pattern**; the vectors are stacked `(B, …)` like the dense case.
See [Backends](backends.md#sparsesolver).

## Per-problem solving

Each problem in the batch carries its own regularization (`rho`, `delta`), residuals,
and status, and converges independently. Problems that have already terminated are held
fixed while the others keep iterating, so a fast-converging problem does not waste work
once it is solved. Read per-problem diagnostics from `solver.result.info` — every field
is a `(B,)` array. See [Results & Status](../api/settings-results.md).

## Shared structure across the batch

All problems in a batch share the same **structure**, even though their numerical data
differ:

- Same shapes `n`, `p`, `m` and (for the sparse backend) the same sparsity pattern.
- **Same finite/infinite bound pattern.** Every problem in the batch must mark the same
  entries of `h_l`, `h_u`, `x_l`, `x_u` as `±inf`. cuPIQP validates this at `setup()`
  and raises a `ValueError` on a mismatch:

    ```text
    Bound structure mismatch in 'h_l': all problems in the batch
    must have the same set of finite bounds.
    ```

  Vary the bound *values* freely across the batch — just keep the *pattern* of which
  bounds are finite identical.

## Typical use cases

- **Sampling-based / scenario MPC** — solve one QP per sampled disturbance or scenario,
  all at once.
- **Reinforcement learning** — batch the QP layer over a rollout or a minibatch.
- **Parameter sweeps & tuning** — solve the same controller for many set-points,
  weights, or initial states in parallel.

## Warm re-solving a batch

Pair batching with [`update()`](../getting-started.md#re-solving-with-new-data) to
re-solve a batch with new numerical data while reusing every GPU allocation — the standard inner loop for batched
MPC:

```python
solver.setup(P=P_b, c=c_b, A=A_b, b=b0_b, G=G_b, h_l=h_l_b, h_u=h_u_b)
solver.solve()

for b_k in trajectory:           # b_k: (B, p), e.g. B sampled initial states
    solver.update(b=b_k)
    solver.solve()
    U = solver.result.x          # (B, n)
```
