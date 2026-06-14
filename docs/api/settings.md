# Settings

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

### Validation

`settings.verify_settings()` returns `True` when every field is within its valid range
(positive tolerances, `0 < tau ≤ 1`, a recognized `kkt_solver` and `dtype`, etc.). Use
it as a quick sanity check after programmatically constructing settings.
