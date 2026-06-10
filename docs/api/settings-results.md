# Settings & Results

## Settings

Every solver owns a `Settings` dataclass at `solver.settings`. Mutate its fields before
`setup()` or between solves:

```python
solver = DenseSolver(dtype="float64")
solver.settings.verbose = True
solver.settings.max_iter = 100
solver.settings.eps_abs = 1e-6
solver.setup(P=P, c=c)
solver.solve()
```

You can also build a `Settings` object directly. The factory
`Settings.for_dtype(dtype)` returns settings with **dtype-appropriate tolerance
defaults** (see [Precision](#precision-and-dtype)):

```python
from cupiqp import Settings

settings = Settings.for_dtype("float32")
settings.max_iter = 200
```

!!! warning "`dtype` is fixed at construction"
    `Settings.dtype` cannot be reassigned after the object is created. Pass `dtype=` to
    the solver constructor, or build a fresh `Settings` via `Settings.for_dtype(dtype)`.
    Assigning `settings.dtype = ...` raises `AttributeError`.

### Precision and dtype

| Field | Type | Default | Description |
|---|---|---|---|
| `dtype` | `"float32"` \| `"float64"` | `"float64"` | Solver arithmetic precision (fixed at construction). |
| `device` | `str` | `"cuda"` | Compute device. |

`float32` and `float64` carry **different default tolerances**, since you cannot ask for
`float64`-level accuracy in `float32` arithmetic. `Settings.for_dtype("float32")` loosens
the tolerances accordingly. If you tighten a `float32` tolerance below its recommended
floor, the solver emits a warning that convergence may fail.

Representative defaults:

| Field | `float64` default | `float32` default |
|---|---|---|
| `eps_abs` | `1e-8` | `1e-4` |
| `eps_rel` | `1e-9` | `1e-4` |
| `eps_duality_gap_abs` | `1e-8` | `1e-4` |
| `eps_duality_gap_rel` | `1e-9` | `1e-4` |
| `reg_lower_limit` | `1e-10` | `1e-5` |

### Convergence tolerances

| Field | Default (f64) | Description |
|---|---|---|
| `eps_abs` | `1e-8` | Absolute tolerance on the primal/dual residuals. |
| `eps_rel` | `1e-9` | Relative tolerance on the primal/dual residuals. |
| `check_duality_gap` | `True` | Also require the duality gap to satisfy its tolerances before declaring convergence. |
| `eps_duality_gap_abs` | `1e-8` | Absolute duality-gap tolerance. |
| `eps_duality_gap_rel` | `1e-9` | Relative duality-gap tolerance. |
| `max_iter` | `250` | Maximum interior-point iterations before returning `CUPIQP_MAX_ITER_REACHED`. |
| `infeasibility_threshold` | `0.9` | Threshold used in the primal/dual infeasibility detection. |

### Proximal regularization

cuPIQP is a **proximal** interior-point method: it regularizes the KKT system with a
primal regularization `rho` and a dual regularization `delta`, driving both down as the
iterates converge.

| Field | Default (f64) | Description |
|---|---|---|
| `rho_init` | `1e-6` | Initial primal proximal regularization. |
| `delta_init` | `1e-4` | Initial dual proximal regularization. |
| `reg_lower_limit` | `1e-10` | Lower limit on the proximal regularization. |
| `reg_finetune_lower_limit` | `1e-13` | Tighter lower limit used during fine-tuning. |
| `reg_finetune_primal_update_threshold` | `7` | Stagnated-primal-update count that triggers regularization fine-tuning. |
| `reg_finetune_dual_update_threshold` | `7` | Stagnated-dual-update count that triggers regularization fine-tuning. |
| `tau` | `0.99` | Fraction-to-the-boundary parameter for the interior-point step. |
| `max_factor_retires` | `10` | Max KKT-factorization retries (with increased regularization) before a numerical failure. |

### Preconditioner (Ruiz equilibration)

cuPIQP equilibrates the problem with a Ruiz preconditioner before solving.

| Field | Default | Description |
|---|---|---|
| `preconditioner_iter` | `10` | Number of Ruiz equilibration sweeps. Set to `0` to disable scaling entirely. |
| `preconditioner_scale_cost` | `False` | Also scale the cost (`P`, `c`) during equilibration. |
| `preconditioner_reuse_on_update` | `False` | On `update()`, reuse the existing scaling instead of recomputing it. |

!!! tip "Exact gradients"
    When differentiating through the solve, setting `preconditioner_iter = 0` yields
    exact gradients (no scaling to differentiate through).

### Iterative refinement

| Field | Default (f64) | Description |
|---|---|---|
| `iterative_refinement_always_enabled` | `False` | Always run iterative refinement after each KKT solve. |
| `iterative_refinement_eps_abs` | `1e-12` | Absolute target residual for refinement. |
| `iterative_refinement_eps_rel` | `1e-12` | Relative target residual for refinement. |
| `iterative_refinement_max_iter` | `10` | Max refinement iterations per KKT solve. |
| `iterative_refinement_min_improvement_rate` | `5.0` | Required residual-improvement rate to keep refining. |
| `iterative_refinement_static_regularization_eps` | `1e-8` | Static regularization added to the factorized system for refinement. |
| `iterative_refinement_static_regularization_rel` | `≈ε²` | Relative static regularization (`float`-eps squared). |

### Execution and CUDA graphs

| Field | Default | Description |
|---|---|---|
| `enable_cuda_graph` | `True` | Capture the repeated IPM iteration as a CUDA graph and replay it with near-zero launch overhead. |
| `use_deterministic_mode_for_cudss` | `False` | Bit-wise reproducible cuDSS factorizations (slower); sparse backend only. |
| `kkt_solver` | backend-specific | KKT factorization: `"dense_cholesky"`, `"sparse_ldlt"`, or `"multistage_block_cholesky"`. Set automatically by the chosen solver class. |

!!! note "`kkt_solver` is set by the solver class"
    Each solver subclass fixes `kkt_solver` to match its backend
    (`DenseSolver → "dense_cholesky"`, etc.). You normally do not set it by hand.

### Differentiation, diagnostics, and logging

| Field | Default | Description |
|---|---|---|
| `enable_grad` | `False` | Allocate backward buffers; see [Differentiation](../guide/differentiation.md). |
| `verbose` | `False` | Print the banner and the per-iteration log during `solve()`. |
| `debug` | `False` | Extra debug checks. |

### Validation

`settings.verify_settings()` returns `True` when every field is within its valid range
(positive tolerances, `0 < tau ≤ 1`, a recognized `kkt_solver` and `dtype`, etc.). Use
it as a quick sanity check after programmatically constructing settings.

## Result

After `solve()`, read everything from `solver.result`. It bundles the primal/dual/slack
variables together with per-problem solver info. Every field carries a **leading batch
dimension** `(B, …)`; for a single problem `B = 1`.

```python
solver.solve()

x        = solver.result.x                  # (B, n) optimal primal solution
status   = solver.result.info.status        # list of B Status enums
obj      = solver.result.info.primal_obj    # (B,) objective values
n_iter   = solver.result.info.iter          # (B,) iteration counts
```

::: cupiqp.Result
    options:
      inherited_members: true
      show_if_no_docstring: true
      members: [x, y, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu,
                primals_all, duals_all, batch_size]

### Solution variables

`solver.result` exposes the full primal–dual–slack iterate as zero-copy `(B, …)` views.
Blocks tied to absent constraints are empty.

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

## Status

Per-problem solver outcome. `solver.result.info.status` is a **list of `Status`**
enums (one per problem in the batch).

| `Status` member | Value |  Meaning |
|----|--|---|
| `CUPIQP_UNSOLVED` | `-1` | not yet solved |
| `CUPIQP_SOLVED` | `0` | converged to tolerance |
| `CUPIQP_MAX_ITER_REACHED` | `1` | hit `max_iter` |
| `CUPIQP_PRIMAL_INFEASIBLE` | `2` |  detected primal infeasible |
| `CUPIQP_DUAL_INFEASIBLE` | `3` | detected dual infeasible |
| `CUPIQP_NUMERICAL_ISSUES` | `4` | numerical failure |


## Info — per-problem diagnostics

`solver.result.info` holds a `(B, num_fields)` buffer; each field is a `(B,)` array. The
most useful fields:

| Field | Meaning |
|---|---|
| `status` | list of `B` `Status` enums (see above) |
| `iter` | iterations taken |
| `primal_obj`, `dual_obj` | primal / dual objective values |
| `duality_gap`, `duality_gap_rel` | absolute / relative duality gap |
| `primal_res`, `primal_res_rel` | primal residual (abs / rel) |
| `dual_res`, `dual_res_rel` | dual residual (abs / rel) |
| `rho`, `delta` | final primal / dual proximal regularization |
| `mu` | final complementarity measure |
| `primal_step`, `dual_step` | last fraction-to-the-boundary step sizes |

!!! note "Avoid host syncs in hot loops"
    Reading `.get()` / `.item()` on these `(B,)` device arrays forces a host
    synchronization. Inside a tight control or training loop, keep diagnostics on the
    GPU and only fetch them when you actually need to inspect them.

## PIQP_INF

Threshold above which a bound is treated as `±∞` and dropped. See
[Problem Formulation](../problem-formulation.md#one-sided-constraints-and-free-variables)
for details.

::: cupiqp.PIQP_INF
