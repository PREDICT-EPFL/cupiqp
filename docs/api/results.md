
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

`solver.result` exposes the full primal–dual–slack variables as zero-copy `(B, …)` views of the internal states as cupy arrays on **device**. Blocks tied to absent constraints are empty (e.g., if $h_l$ is `None`, then the shapes `z_l` and `s_l` are both `(B, 0)`).

| Attribute | Shape | Meaning |
|---|---|---|
| `x` | `(B, n)` | primal solution |
| `y` | `(B, p)` | equality-constraint multipliers |
| `z_l`, `z_u` | `(B, num_hl)`, `(B, num_hu)` | inequality multipliers (lower / upper rows of `G x`) |
| `z_bl`, `z_bu` | `(B, num_xl)`, `(B, num_xu)` | box-bound multipliers (lower / upper) |
| `s_l`, `s_u` | `(B, num_hl)`, `(B, num_hu)` | inequality slacks |
| `s_bl`, `s_bu` | `(B, num_xl)`, `(B, num_xu)` | box-bound slacks |

The counts `num_hl`, `num_hu`, `num_xl`, `num_xu` are the numbers of **finite** lower/
upper inequality and box bounds — infinite bounds are dropped, so these blocks only
cover active bound rows. The convenience views `primals_all` and `duals_all` expose the
packed primal and dual buffers.

!!! warning "Views, not copies"
    The solution attributes are views into the solver's internal GPU buffers and are
    **overwritten by the next `solve()`**. Copy what you need to keep:

    ```python
    x = solver.result.x.copy()       # cupy copy that survives the next solve
    x_host = solver.result.x.get()   # numpy copy on the host
    ```




