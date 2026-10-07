# Settings

## Settings

Every solver owns a `Settings` dataclass at `solver.settings`. Mutate its fields before
`setup()` or between solves:

```python
import warp as wp
from cupiqp import DenseSolver

solver = DenseSolver(dtype=wp.float64)
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
import warp as wp
from cupiqp import Settings

settings = Settings.for_dtype(wp.float32)
settings.max_iter = 200
```

!!! warning "`dtype` is fixed at construction"
    `Settings.dtype` cannot be reassigned after the object is created. Pass `dtype=` to
    the solver constructor, or build a fresh `Settings` via `Settings.for_dtype(dtype)`.
    Assigning `settings.dtype = ...` raises `AttributeError`.

### Precision and dtype

| Field | Type | Default | Description |
|---|---|---|---|
| `dtype` | `wp.float32` \| `wp.float64` | `wp.float64` | Solver arithmetic precision (fixed at construction). |

`float32` and `float64` carry **different default tolerances**, since you cannot ask for
`float64`-level accuracy in `float32` arithmetic. `Settings.for_dtype(wp.float32)` loosens
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
| `enable_cuda_graph` | `True` | Record the solve as a CUDA graph on the first `solve()` and replay it afterwards: the whole solve, with the IPM loop running on the GPU, for the dense and multistage backends; one IPM iteration for the sparse backend. Recording does not slow the first solve, and later solves are typically several times faster for small problems. Set it to `False` only for debugging: every kernel is then launched from Python, so an error is reported at the kernel that caused it. |
| `use_deterministic_mode_for_cudss` | `False` | Bit-wise reproducible cuDSS factorizations (slower); sparse backend only. |

### Differentiation, diagnostics, and logging

| Field | Default | Description |
|---|---|---|
| `enable_grad` | `False` | Allocate backward buffers; see [Differentiation](../guide/differentiation.md). |
| `gradient_smoothing` | `False` | Enable qpax-style relaxed-KKT gradient smoothing. Fixed at `setup()`. Requires `enable_grad`. See [Differentiation](../guide/differentiation.md#gradient-smoothing). |
| `gradient_smoothing_mu` | `1e-3` | Target complementarity `s * z = mu` at the relaxed point. Larger smooths more (more bias); smaller approaches the unsmoothed gradient. |
| `gradient_smoothing_tol` | `1e-5` | Convergence tolerance of the inner relaxed-Newton solve (separate from the forward `eps_abs`). |
| `gradient_smoothing_max_iter` | `5` | Maximum relaxed-Newton iterations per backward (separate from the forward `max_iter`). |
| `verbose` | `False` | Print the banner and the per-iteration log during `solve()`. |

### Validation

`settings.verify_settings()` returns `True` when every field is within its valid range
(positive tolerances, `0 < tau ≤ 1`, a recognized `dtype`, etc.). Use
it as a quick sanity check after programmatically constructing settings.
