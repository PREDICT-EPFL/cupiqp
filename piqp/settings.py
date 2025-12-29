
from dataclasses import dataclass


@dataclass
class Settings:
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

    kkt_solver: str = "dense_cholesky"  # Assuming KKTSolver is an enum converted to string

    iterative_refinement_always_enabled: bool = False
    iterative_refinement_eps_abs: float = 1e-12
    iterative_refinement_eps_rel: float = 1e-12
    iterative_refinement_max_iter: int = 10
    iterative_refinement_min_improvement_rate: float = 5.0
    iterative_refinement_static_regularization_eps: float = 1e-8
    iterative_refinement_static_regularization_rel: float = 2.220446049250313e-32  # Approximation of epsilon squared

    verbose: bool = False
    debug: bool = False
    compute_timings: bool = False

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
               self.kkt_solver in ["dense_cholesky", "sparse_ldlt"]
               )