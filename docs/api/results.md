
After `solver.solve()`, all post-solve information is stored in `solver.result`. 

```python
solver.solve()

x        = solver.result.x                  # (B, n) optimal primal solution
status   = solver.result.info.status        # list of B Status enums
obj      = solver.result.info.primal_obj    # (B,) objective values
n_iter   = solver.result.info.iter          # (B,) per-problem iteration counts
n_total  = solver.result.info.iter_total    # int: total IPM iterations the solver ran
```

## Status

After calling `solver.solve()`, the per-problem status can be obtained from `solver.result.info.status` as a **list of `Status`** enums (one per problem in the batch). The meaning of status is listed in the following table:

| `Status` member | Value |  Meaning |
|----|--|---|
| `CUPIQP_UNSOLVED` | -1 | not yet solved |
| `CUPIQP_SOLVED` | 0 | converged to tolerance |
| `CUPIQP_MAX_ITER_REACHED` | 1 | hit max number of iterations |
| `CUPIQP_PRIMAL_INFEASIBLE` | 2 |  detected primal infeasible |
| `CUPIQP_DUAL_INFEASIBLE` | 3 | detected dual infeasible |
| `CUPIQP_NUMERICAL_ISSUES` | 4 | numerical failure |


## Info

`solver.result.info` holds the numeric fields in a cupy array buffer of the solver's float dtype (`float64` by default) on **device**. `status` and `iter` are numpy arrays on **host**.

| Field | Type | Shape/length | Meaning |
|---|---|---|---|
| `status` | `list[Status]` | `B` | the status of each problem |
| `iter` | `numpy.ndarray(dtype=np.int32)` | `(B,)` | per-problem iteration count |
| `primal_obj`, `dual_obj` | `cupy.ndarray` | `(B,)` | primal / dual objective values |
| `duality_gap`, `duality_gap_rel` | `cupy.ndarray` | `(B,)` | absolute / relative duality gap |
| `primal_res`, `primal_res_rel` | `cupy.ndarray` | `(B,)` | primal residual (abs / rel) |
| `dual_res`, `dual_res_rel` | `cupy.ndarray` | `(B,)` | dual residual (abs / rel) |
| `rho`, `delta` | `cupy.ndarray` | `(B,)` | final regularization terms |
| `mu` | `cupy.ndarray` | `(B,)` | final complementarity measure |
| `primal_step`, `dual_step` | `cupy.ndarray` | `(B,)` | final step sizes |

<!-- !!! note "Avoid host syncs in hot loops"
    Reading `.get()` / `.item()` on these `(B,)` device arrays forces a host
    synchronization. Inside a tight control or training loop, keep diagnostics on the
    GPU and only fetch them when you actually need to inspect them. -->


### Solution variables

`solver.result` exposes the full primal–dual–slack variables as zero-copy `(B, …)` views of the internal states as cupy arrays on **device**. A *present* block is **full-length**: one entry per row of `G` for the inequality variables and one entry per decision variable for the box-bound variables. A block is empty `(B, 0)` when that bound side was **omitted (passed as `None`) at `setup()`** — each of the four bound sides (`h_l`, `h_u`, `x_l`, `x_u`) is independent, so e.g. a one-sided problem `G x <= h_u` (with `h_l=None`) gives `z_l`, `s_l` of shape `(B, 0)` while `z_u`, `s_u` stay `(B, m)`. If no `G` is given at all (`m = 0`), then `z_l`, `z_u`, `s_l`, `s_u` are all `(B, 0)`.

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
    x = solver.result.x.copy()       # cupy copy that survives the next solve
    x_host = solver.result.x.get()   # numpy copy on the host
    ```




