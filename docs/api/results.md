After `solver.solve()`, all post-solve information is stored in `solver.result`. Every
field is a `warp.array` on the **device**, at an address that stays fixed for the lifetime
of the solver; call `.numpy()` to read one on the host.

```python
solver.solve()

x        = solver.result.x.numpy()                 # (B, n) optimal primal solution
status   = solver.result.info.status.numpy()       # (B,) Status codes
obj      = solver.result.info.primal_obj.numpy()   # (B,) objective values
n_iter   = solver.result.info.iter.numpy()         # (B,) per-problem iteration counts
n_total  = solver.result.info.iter_total.numpy()[0]  # IPM iterations the batch ran
```

`solve()` is asynchronous: it queues the solve on `solver.stream` and returns. `.numpy()`
waits for the solve to finish before copying, so the code above is always correct. Your own
GPU work queued on `solver.stream` after `solve()` (Warp kernels, or a CUDA graph you
capture) can read the fields directly, without any synchronization.

## Status

`solver.result.info.status` holds one int32 status code per problem. `Status` is an
`IntEnum`, so the codes compare directly with its members:

```python
from cupiqp import Status

status = solver.result.info.status.numpy()
assert (status == Status.CUPIQP_SOLVED).all()
print(Status(status[0]).name)        # CUPIQP_SOLVED
```

| `Status` member | Value |  Meaning |
|----|--|---|
| `CUPIQP_UNSOLVED` | -1 | not yet solved |
| `CUPIQP_SOLVED` | 0 | converged to tolerance |
| `CUPIQP_MAX_ITER_REACHED` | 1 | hit max number of iterations |
| `CUPIQP_PRIMAL_INFEASIBLE` | 2 |  detected primal infeasible |
| `CUPIQP_DUAL_INFEASIBLE` | 3 | detected dual infeasible |
| `CUPIQP_NUMERICAL_ISSUES` | 4 | numerical failure |


## Info

`solver.result.info` holds the per-problem diagnostics. Each field below is a `(B,)`
`warp.array` on the device; the floating-point fields use the solver's dtype (`float64` by
default).

| Field | dtype | Meaning |
|---|---|---|
| `status` | `int32` | the `Status` code of each problem |
| `iter` | `int32` | per-problem iteration count |
| `factor_retries` | `int32` | KKT factorization retries |
| `primal_obj`, `dual_obj` | float | primal / dual objective values |
| `duality_gap`, `duality_gap_rel` | float | absolute / relative duality gap |
| `primal_res`, `primal_res_rel` | float | primal residual (abs / rel) |
| `dual_res`, `dual_res_rel` | float | dual residual (abs / rel) |
| `rho`, `delta` | float | final regularization terms |
| `mu` | float | final complementarity measure |
| `primal_step`, `dual_step` | float | final step sizes |

`iter_total` is a `(1,)` `int32` array: the iterations the whole batch ran (the slowest
problem's count).

### All fields at once: `to_host()`

`solver.result.info.to_host()` waits for the solve and copies every field to the host in
one transfer. It returns an `InfoSnapshot` with the same field names as NumPy arrays,
`status` as a list of `Status` and `iter_total` as an `int`. The snapshot owns its data, so
later solves do not change it.

```python
info = solver.result.info.to_host()
print(info.status, info.iter, info.primal_res)
```

!!! note "Avoid host reads in hot loops"
    Each `.numpy()` or `to_host()` waits for the GPU. Inside a tight control or training
    loop, keep the diagnostics on the GPU and only fetch them when you need to inspect
    them.


### Solution variables

`solver.result` exposes the full primal–dual–slack variables as zero-copy `(B, …)` `warp.array` views of the internal states on **device** (view them from cupy / torch / JAX with `cupy.asarray(x)`, `torch.from_dlpack(x)`, `jax.dlpack.from_dlpack(x)`). A *present* block is **full-length**: one entry per row of `G` for the inequality variables and one entry per decision variable for the box-bound variables. A block is empty `(B, 0)` when that bound side was **omitted (passed as `None`) at `setup()`** — each of the four bound sides (`h_l`, `h_u`, `x_l`, `x_u`) is independent, so e.g. a one-sided problem `G x <= h_u` (with `h_l=None`) gives `z_l`, `s_l` of shape `(B, 0)` while `z_u`, `s_u` stay `(B, m)`. If no `G` is given at all (`m = 0`), then `z_l`, `z_u`, `s_l`, `s_u` are all `(B, 0)`.

| Attribute | Shape | Meaning |
|---|---|---|
| `x` | `(B, n)` | primal solution |
| `y` | `(B, p)` | equality-constraint multipliers |
| `z_l` | `(B, m)`, or `(B, 0)` if `h_l` is `None` at `setup()` | dual variables for $h_l \leq Gx$ |
| `z_u` | `(B, m)`, or `(B, 0)` if `h_u` is `None` at `setup()` | dual variables for $Gx \leq h_u$ |
| `z_bl` | `(B, n)`, or `(B, 0)` if `x_l` is `None` at `setup()` | dual variables for $x_l \leq x$ |
| `z_bu` | `(B, n)`, or `(B, 0)` if `x_u` is `None` at `setup()` | dual variables for $x \leq x_u$ |
| `s_l` | `(B, m)`, or `(B, 0)` if `h_l` is `None` at `setup()` | slack variables for $h_l \leq Gx$ |
| `s_u` | `(B, m)`, or `(B, 0)` if `h_u` is `None` at `setup()` | slack variables for $Gx \leq h_u$ |
| `s_bl` | `(B, n)`, or `(B, 0)` if `x_l` is `None` at `setup()` | slack variables for $x_l \leq x$ |
| `s_bu` | `(B, n)`, or `(B, 0)` if `x_u` is `None` at `setup()` | slack variables for $x \leq x_u$ |

Note the difference between an *inactive* bound and an *absent* side. For example, with
`m` inequality rows:

- If you pass `h_l` as an array with some (or all) entries `-inf`, the lower side is
  **present**: `z_l` and `s_l` keep their full `(B, m)` shape, and the entries for the
  `-inf` rows are simply held at zero. The column for each row keeps a stable position, so
  you can flip a bound between finite and `±inf` across solves via `update()` without the
  shape changing.
- If you instead pass `h_l=None` (or omit it) at `setup()`, the lower side is **absent**:
  `z_l` and `s_l` are `(B, 0)` and use no memory. This is fixed for the lifetime of the
  solver — you cannot add the side back with `update()`.

The convenience views `primals_all` and `duals_all` expose the packed primal and dual
buffers, concatenated in this order along the last axis:

- `primals_all`: `[x | s_l | s_u | s_bl | s_bu]` — shape `(B, n + num_ineq)`
- `duals_all`: `[y | z_l | z_u | z_bl | z_bu]` — shape `(B, p + num_ineq)`

where `num_ineq = num_hl + num_hu + num_xl + num_xu`. Each bound block contributes its
full width (`m` or `n`) when present and `0` when absent, so an omitted side simply drops
out of the concatenation and the following block shifts up.

!!! warning "Views, not copies"
    The solution attributes are views into the solver's internal GPU buffers and are
    **overwritten by the next `solve()`**. Copy what you need to keep:

    ```python
    x = wp.clone(solver.result.x)      # device copy that survives the next solve
    x_host = solver.result.x.numpy()   # numpy copy on the host
    ```




