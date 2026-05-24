import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np


_F32_DEFAULTS = {
    "eps_abs":                                        1e-4,
    "eps_rel":                                        1e-4,
    "eps_duality_gap_abs":                            1e-4,
    "eps_duality_gap_rel":                            1e-4,
    "reg_lower_limit":                                1e-5,
    "reg_finetune_lower_limit":                       1e-7,
    "iterative_refinement_eps_abs":                   1e-6,
    "iterative_refinement_eps_rel":                   1e-6,
    "iterative_refinement_static_regularization_eps": 1e-4,
    "iterative_refinement_static_regularization_rel": float(np.finfo("float32").eps) ** 2,  # ≈ 1.42e-14
}

_F64_DEFAULTS = {
    "eps_abs":                                        1e-8,
    "eps_rel":                                        1e-9,
    "eps_duality_gap_abs":                            1e-8,
    "eps_duality_gap_rel":                            1e-9,
    "reg_lower_limit":                                1e-10,
    "reg_finetune_lower_limit":                       1e-13,
    "iterative_refinement_eps_abs":                   1e-12,
    "iterative_refinement_eps_rel":                   1e-12,
    "iterative_refinement_static_regularization_eps": 1e-8,
    "iterative_refinement_static_regularization_rel": float(np.finfo("float64").eps) ** 2,  # ≈ 4.93e-32
}

assert _F32_DEFAULTS.keys() == _F64_DEFAULTS.keys(), (
    "_F32_DEFAULTS and _F64_DEFAULTS must list exactly the same fields."
)


@dataclass
class Settings:
    dtype: Literal["float32", "float64"] = "float64"
    device: str = "cuda"

    rho_init: float = 1e-6
    delta_init: float = 1e-4

    eps_abs: float = 1e-8
    eps_rel: float = 1e-9

    check_duality_gap: bool = True
    eps_duality_gap_abs: float = 1e-8
    eps_duality_gap_rel: float = 1e-9

    infeasibility_threshold: float = 0.9

    reg_lower_limit: float = 1e-10
    reg_finetune_lower_limit: float = 1e-13
    reg_finetune_primal_update_threshold: int = 7
    reg_finetune_dual_update_threshold: int = 7

    max_iter: int = 250
    max_factor_retires: int = 10

    preconditioner_scale_cost: bool = False
    preconditioner_reuse_on_update: bool = False
    preconditioner_iter: int = 10

    tau: float = 0.99

    kkt_solver: Literal["sparse_ldlt", "dense_cholesky", "multistage_block_cholesky"] = "sparse_ldlt"

    iterative_refinement_always_enabled: bool = False
    iterative_refinement_eps_abs: float = 1e-12
    iterative_refinement_eps_rel: float = 1e-12
    iterative_refinement_max_iter: int = 10
    iterative_refinement_min_improvement_rate: float = 5.0
    iterative_refinement_static_regularization_eps: float = 1e-8
    iterative_refinement_static_regularization_rel: float = 4.930380657631324e-32

    use_deterministic_mode_for_cudss: bool = False  # bit-wise reproducible cuDSS (slower)
    enable_cuda_graph: bool = True

    verbose: bool = False
    debug: bool = False
    compute_timings: bool = False
    enable_grad: bool = False


    @classmethod
    def for_dtype(cls, dtype) -> "Settings":
        if dtype == "float32":
            defaults = _F32_DEFAULTS
        elif dtype == "float64":
            defaults = _F64_DEFAULTS
        else:
            raise ValueError(
                f"Unsupported dtype {dtype!r}; expected 'float32' or 'float64'."
            )
        return cls(dtype=dtype, **defaults)

    def __setattr__(self, name, value):
        if name == "dtype" and "dtype" in self.__dict__:
            raise AttributeError(
                "Settings.dtype is fixed at construction; build a new "
                "Settings via Settings.for_dtype(dtype) or pass dtype= "
                "to the solver constructor."
            )
        if (name in _F32_DEFAULTS
                and self.__dict__.get("dtype") == "float32"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0 < value < _F32_DEFAULTS[name]):
            warnings.warn(
                f"Settings.{name} = {value:g} is tighter than the float32 "
                f"recommended {_F32_DEFAULTS[name]:g}, convergence may fail.",
                stacklevel=2,
            )
        super().__setattr__(name, value)

    def verify_settings(self) -> bool:
        return (self.rho_init > 0 and
               self.delta_init > 0 and
               self.eps_abs > 0 and
               self.eps_rel >= 0 and
               self.eps_duality_gap_abs > 0 and
               self.eps_duality_gap_rel >= 0 and
               self.infeasibility_threshold >= 0 and
               self.reg_lower_limit > 0 and
               self.reg_finetune_primal_update_threshold >= 0 and
               self.reg_finetune_dual_update_threshold >= 0 and
               self.max_iter > 0 and
               self.max_factor_retires > 0 and
               self.preconditioner_iter >= 0 and
               self.tau > 0 and self.tau <= 1 and
               self.iterative_refinement_eps_abs > 0 and
               self.iterative_refinement_eps_rel >= 0 and
               self.iterative_refinement_max_iter >= 0 and
               self.iterative_refinement_min_improvement_rate >= 1.0 and
               self.iterative_refinement_static_regularization_eps > 0 and
               self.iterative_refinement_static_regularization_rel >= 0 and
               self.kkt_solver in ["dense_cholesky", "sparse_ldlt", "multistage_block_cholesky"]
               and self.dtype in ("float32", "float64")
               )
