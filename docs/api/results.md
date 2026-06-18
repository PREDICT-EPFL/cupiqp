
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

`solver.result` exposes the full primal–dual–slack variables as zero-copy `(B, …)` views of the internal states as cupy arrays on **device**. The blocks are **full-length**: one entry per row of `G` for the inequality variables and one entry per decision variable for the box-bound variables. A block is empty `(B, 0)` only when the corresponding constraint type is absent entirely — e.g. if no `G` is given (`m = 0`), then `z_l`, `z_u`, `s_l`, `s_u` are all `(B, 0)`.

| Attribute | Shape | Meaning |
|---|---|---|
| `x` | `(B, n)` | primal solution |
| `y` | `(B, p)` | equality-constraint multipliers |
| `z_l`, `z_u` | `(B, m)` | inequality multipliers (lower / upper rows of `G x`) |
| `z_bl`, `z_bu` | `(B, n)` | box-bound multipliers (lower / upper) |
| `s_l`, `s_u` | `(B, m)` | inequality slacks |
| `s_bl`, `s_bu` | `(B, n)` | box-bound slacks |

Every inequality block has one column per row of `G`, and every box block one column per
variable, **whether or not** that individual bound is finite. Entries for infinite
(inactive) bounds are held at zero rather than dropped, so each column keeps a stable
position across solves even if you toggle a bound between finite and `±inf` via
`update()`. The convenience views `primals_all` and `duals_all` expose the packed primal
and dual buffers.

!!! warning "Views, not copies"
    The solution attributes are views into the solver's internal GPU buffers and are
    **overwritten by the next `solve()`**. Copy what you need to keep:

    ```python
    x = solver.result.x.copy()       # cupy copy that survives the next solve
    x_host = solver.result.x.get()   # numpy copy on the host
    ```




