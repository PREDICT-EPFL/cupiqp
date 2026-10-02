from abc import ABC, abstractmethod

import warp as wp

from .data import Data
from .solver_kernels import create_solution_finite_kernel, REDUCTION_BLOCK_DIM


class KKTSolverBase(ABC):
    """Base interface for backend KKT solvers.

    Represent the system with the following form:
    [P+x_reg     A^T      G^T    ] [Delta_x] = [rhs_x]
    [A         -delta*I     0    ] [Delta_y] = [rhs_y]
    [G           0     -(z_reg)  ] [Delta_z] = [rhs_z]
    """
    def __init__(self, batch_size: int, dtype, device):
        # Per-problem outcome flags, written on the device and never read on
        # the host by the solver: 0 = success, nonzero = failed. Their
        # addresses are fixed for the lifetime of the solver.
        self._factor_status = wp.zeros(batch_size, dtype=wp.int32, device=device)
        self._solve_status = wp.zeros(batch_size, dtype=wp.int32, device=device)
        self._check_solution_finite_kernel = create_solution_finite_kernel(dtype)
        self._device = device

    @abstractmethod
    def update_data(self, data: Data, update_P: bool, update_A: bool, update_G: bool) -> None:
        """Notify the KKT solver that problem data has changed.
        """
        pass

    @abstractmethod
    def update_kkt(self, data: Data, delta: float, x_reg: wp.array, z_reg: wp.array, z_reg_inv: wp.array) -> bool:
        """Refresh the KKT matrix with new regularization scalings."""
        pass

    @abstractmethod
    def factor(self) -> None:
        """Enqueue the factorization of the KKT matrix.

        Nothing is read on the host; once the enqueued work has run,
        ``factor_status`` holds one entry per problem.
        """
        pass

    @property
    def factor_status(self) -> wp.array:
        """``(B,)`` int32 device array written by ``factor()``: zero where the
        factorization of that problem succeeded, nonzero where it failed."""
        return self._factor_status

    @property
    def solve_status(self) -> wp.array:
        """``(B,)`` int32 device array written by ``solve()``: zero where the
        last KKT solution of that problem is finite, nonzero where it is not."""
        return self._solve_status

    def _write_solve_status(self, lhs_x: wp.array, lhs_y: wp.array, lhs_z: wp.array) -> None:
        """Called by every backend at the end of ``solve()``: flag, per
        problem, a solution with a NaN or infinite entry in ``solve_status``.

        No library reports this on the device: cuSOLVER's ``potrs`` info stays
        zero for a NaN factor or right-hand side, cuDSS exposes its solve
        info only through a host query, and the block Cholesky reports
        nothing. A factorization can also report success on a matrix that
        contains NaN, so this finite check is what catches poisoned data. The
        kernel uses ``wp.isfinite``, which is false for NaN as well as for
        +inf and -inf, so no separate NaN test is needed."""
        wp.launch_tiled(
            kernel=self._check_solution_finite_kernel,
            dim=[lhs_x.shape[0]],
            inputs=[lhs_x, lhs_y, lhs_z, self._solve_status],
            block_dim=REDUCTION_BLOCK_DIM,
            device=self._device
        )

    @abstractmethod
    def solve(self, data: Data, rhs_x: wp.array, rhs_y: wp.array, rhs_z: wp.array, lhs_x: wp.array, lhs_y: wp.array, lhs_z: wp.array) -> None:
        """Solve the KKT system with the current factor and write the
        per-problem outcome to ``solve_status`` (via ``_write_solve_status``)."""
        pass

    @abstractmethod
    def eval_P_x(self, data: Data, alpha: float, x: wp.array, z: wp.array) -> None:
        """
        Evaluate z = alpha * P * x
        """
        pass

    @abstractmethod
    def eval_A_xn(self, data: Data, alpha_n: float, xn: wp.array, zn: wp.array) -> None:
        """
        Evaluate Ax with scaling factor alpha_n
        zn = alpha_n * A * xn
        """
        pass

    @abstractmethod
    def eval_AT_xt(self, data: Data, alpha_t: float, xt: wp.array, zt: wp.array) -> None:
        """
        Evaluate A^T xt with scaling factor alpha_t
        zt = alpha_t * A^T * xt
        """
        pass
    
    @abstractmethod
    def eval_G_xn(self, data: Data, alpha_n: float, xn: wp.array, zn: wp.array) -> None:
        """
        Evaluate Gx with scaling factor alpha_n
        zn = alpha_n * G * xn
        """
        pass

    @abstractmethod
    def eval_GT_xt(self, data: Data, alpha_t: float, xt: wp.array, zt: wp.array) -> None:
        """
        Evaluate G^T xt with scaling factor alpha_t
        zt = alpha_t * G^T * xt
        """
        pass

