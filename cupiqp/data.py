from abc import ABC, abstractmethod
import cupy as cp

from .typedef import PIQP_INF


class Data(ABC):
    """Abstract base class for QP problem data.

    All arrays carry a leading batch dimension ``(B, ...)``.
    For single-problem inputs, ``B = 1``.
    """

    @abstractmethod
    def _disable_inf_constraints(self):
        """Zero out G rows where both h_l and h_u are infinite."""
        ...

    @abstractmethod
    def extract_P_diag(self, diag_P: cp.ndarray) -> None:
        """Extract the diagonal of P into *diag_P* — shape ``(B, n)``."""
        ...

    @abstractmethod
    def set_P(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_c(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_A(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_b(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_G(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_h_l(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_h_u(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_x_l(self, value, check: bool = True):
        ...

    @abstractmethod
    def set_x_u(self, value, check: bool = True):
        ...

    def _finalize(self):
        """Shared post-init: preprocessing + constraints RHS norm.

        Must be called by every subclass ``__init__`` after ``_batch_size``,
        ``_n``, ``_P``, ``_c``, ``_A``, ``_b``, ``_G``, ``_h_l``, ``_h_u``,
        ``_x_l``, ``_x_u`` have been populated.
        """
        self._preprocess()
        self._constraints_rhs_inf_norm = cp.empty(self._batch_size, dtype=cp.float64)
        self._compute_constraints_rhs_inf_norm()

    def _preprocess(self):
        self._init_h_l()
        self._init_h_u()
        self._disable_inf_constraints()
        self._init_h_l()
        self._init_h_u()
        self._init_x_l()
        self._init_x_u()

    # ------------------------------------------------------------------
    # Bound index initialization — works on (B, k) dense vectors
    # ------------------------------------------------------------------

    def _init_h_l(self):
        B, m = self._batch_size, self.m
        if m > 0 and self._h_l.shape[-1] == m:
            mask0 = self._h_l[0] > -PIQP_INF
            self._validate_bound_consistency(self._h_l, -PIQP_INF, '>', 'h_l')
            self._idx_hl = cp.where(mask0)[0].astype(cp.int32)
        else:
            self._idx_hl = cp.empty((0,), dtype=cp.int32)
            if self._h_l.shape[-1] == 0:
                self._h_l = -2 * PIQP_INF * cp.ones((B, m), dtype=cp.float64)

    def _init_h_u(self):
        B, m = self._batch_size, self.m
        if m > 0 and self._h_u.shape[-1] == m:
            mask0 = self._h_u[0] < PIQP_INF
            self._validate_bound_consistency(self._h_u, PIQP_INF, '<', 'h_u')
            self._idx_hu = cp.where(mask0)[0].astype(cp.int32)
        else:
            self._idx_hu = cp.empty((0,), dtype=cp.int32)
            if self._h_u.shape[-1] == 0:
                self._h_u = 2 * PIQP_INF * cp.ones((B, m), dtype=cp.float64)

    def _init_x_l(self):
        n = self._n
        if self._x_l.shape[-1] == n:
            mask0 = self._x_l[0] > -PIQP_INF
            self._validate_bound_consistency(self._x_l, -PIQP_INF, '>', 'x_l')
            self._idx_xl = cp.where(mask0)[0].astype(cp.int32)
        else:
            self._idx_xl = cp.empty((0,), dtype=cp.int32)
            self._x_l = -2 * PIQP_INF * cp.ones((self._batch_size, n), dtype=cp.float64)

    def _init_x_u(self):
        n = self._n
        if self._x_u.shape[-1] == n:
            mask0 = self._x_u[0] < PIQP_INF
            self._validate_bound_consistency(self._x_u, PIQP_INF, '<', 'x_u')
            self._idx_xu = cp.where(mask0)[0].astype(cp.int32)
        else:
            self._idx_xu = cp.empty((0,), dtype=cp.int32)
            self._x_u = 2 * PIQP_INF * cp.ones((self._batch_size, n), dtype=cp.float64)

    @staticmethod
    def _validate_bound_consistency(bounds: cp.ndarray, threshold: float,
                                    direction: str, name: str):
        """Ensure all problems in the batch have the same finite/infinite pattern."""
        if bounds.shape[0] <= 1:
            return  # nothing to validate for B=1
        if direction == '>':
            mask = bounds > -abs(threshold)
        else:
            mask = bounds < abs(threshold)
        if not bool(cp.all(mask == mask[0:1])):
            raise ValueError(
                f"Bound structure mismatch in '{name}': all problems in the batch "
                f"must have the same set of finite bounds."
            )

    # ------------------------------------------------------------------
    # Constraints RHS inf-norm — shape (B,)
    # ------------------------------------------------------------------

    def _compute_constraints_rhs_inf_norm(self):
        """Per-problem inf-norm of constraint RHS vectors — shape ``(B,)``."""
        self._constraints_rhs_inf_norm[:] = 0.0
        if self.p > 0:
            cp.maximum(self._constraints_rhs_inf_norm,
                       cp.max(cp.abs(self._b), axis=1),
                       out=self._constraints_rhs_inf_norm)
        if self.num_hu > 0:
            cp.maximum(self._constraints_rhs_inf_norm,
                       cp.max(cp.abs(self._h_u[:, self._idx_hu]), axis=1),
                       out=self._constraints_rhs_inf_norm)
        if self.num_hl > 0:
            cp.maximum(self._constraints_rhs_inf_norm,
                       cp.max(cp.abs(self._h_l[:, self._idx_hl]), axis=1),
                       out=self._constraints_rhs_inf_norm)
        if self.num_xu > 0:
            cp.maximum(self._constraints_rhs_inf_norm,
                       cp.max(cp.abs(self._x_u[:, self._idx_xu]), axis=1),
                       out=self._constraints_rhs_inf_norm)
        if self.num_xl > 0:
            cp.maximum(self._constraints_rhs_inf_norm,
                       cp.max(cp.abs(self._x_l[:, self._idx_xl]), axis=1),
                       out=self._constraints_rhs_inf_norm)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def n(self) -> int:
        """Number of variables."""
        return self._n

    @property
    def p(self) -> int:
        """Number of equality constraints."""
        return self._A.shape[-2] if hasattr(self._A, 'shape') else 0

    @property
    def m(self) -> int:
        """Number of inequality constraints."""
        return self._G.shape[-2] if hasattr(self._G, 'shape') else 0

    @property
    def P(self):
        return self._P

    @property
    def c(self):
        return self._c

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def G(self):
        return self._G

    @property
    def h_u(self):
        return self._h_u

    @property
    def h_l(self):
        return self._h_l

    @property
    def x_u(self):
        return self._x_u

    @property
    def x_l(self):
        return self._x_l

    @property
    def num_hl(self) -> int:
        return cp.size(self._idx_hl)

    @property
    def idx_hl(self):
        """Indices of lower inequality constraints."""
        return self._idx_hl

    @property
    def num_hu(self) -> int:
        return cp.size(self._idx_hu)

    @property
    def idx_hu(self):
        """Indices of upper inequality constraints."""
        return self._idx_hu

    @property
    def num_xl(self) -> int:
        """Number of lower bound constraints."""
        return cp.size(self._idx_xl)
    
    @property
    def idx_xl(self):
        """Indices of lower bound constraints."""
        return self._idx_xl

    @property
    def num_xu(self) -> int:
        """Number of upper bound constraints."""
        return cp.size(self._idx_xu)

    @property
    def idx_xu(self):
        """Indices of upper bound constraints."""
        return self._idx_xu

    @property
    def num_ineq(self) -> int:
        return self.num_hl + self.num_hu + self.num_xl + self.num_xu
