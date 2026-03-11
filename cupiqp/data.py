from typing import Any
from abc import ABC, abstractmethod
import cupy as cp
from typing import Optional, Union
import cupyx.scipy.sparse as sp

from .typedef import PIQP_INF

ArrayLike = Union[cp.ndarray, sp.spmatrix]


class Data(ABC):
    def __init__(self, 
                 P: ArrayLike, 
                 c: ArrayLike, 
                 A: Optional[ArrayLike] = None, 
                 b: Optional[ArrayLike] = None, 
                 G: Optional[ArrayLike] = None, 
                 h_u: Optional[ArrayLike] = None, 
                 h_l: Optional[ArrayLike] = None, 
                 x_u: Optional[ArrayLike] = None, 
                 x_l: Optional[ArrayLike] = None):
                
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be a square matrix.")
        if c.ndim != 1:
            raise ValueError("c must be a one-dimensional array.")
        if P.shape[0] != c.shape[0]:
            raise ValueError("Dimension mismatch between P and c.")
        self._P = self._as_float64_mat(P)
        self._c = self._as_float64_vec(c)
        
        if A is not None and b is not None:
            if A.ndim != 2:
                raise ValueError("A must be a two-dimensional array.")
            if b.ndim != 1:
                raise ValueError("b must be a one-dimensional array.")
            if A.shape[0] != b.shape[0]:
                raise ValueError("Dimension mismatch between A and b.")
        self._A = self._as_float64_mat(A)
        self._b = self._as_float64_vec(b)
        
        if G is not None:
            if G.ndim != 2:
                raise ValueError("G must be a two-dimensional array.")
            if h_l is None and h_u is None:
                raise ValueError("Either h_l or h_u should be provided.")
            if h_l is not None and cp.shape(h_l) != (G.shape[0],):
                raise ValueError(f"h_l must have shape {(G.shape[0],)}, got {h_l.shape}")
            if h_u is not None and cp.shape(h_u) != (G.shape[0],):
                raise ValueError(f"h_u must have shape {(G.shape[0],)}, got {h_u.shape}")
        else:
            if h_u is not None or h_l is not None:
                raise ValueError("h_l and h_u should be None when G is None.")
        self._G = self._as_float64_mat(G)
        self._h_u = self._as_float64_vec(h_u)
        self._h_l = self._as_float64_vec(h_l)

        if x_l is not None and x_u is not None:
            if x_l.ndim != 1 or x_u.ndim != 1:
                raise ValueError("x_l and x_u must be one-dimensional arrays.")
            if x_l.shape[0] != P.shape[0] or x_u.shape[0] != P.shape[0]:
                raise ValueError("Dimension mismatch between x_l, x_u, and P.")
        self._x_u = self._as_float64_vec(x_u)
        self._x_l = self._as_float64_vec(x_l)
        self._finalize()

    @staticmethod
    def _as_float64_mat(M: Union[Any, None]) -> Any:
        pass

    @staticmethod
    def _as_float64_vec(v: Union[cp.ndarray, None]) -> cp.ndarray:
        return v.astype(cp.float64) if v is not None else cp.zeros((0,), dtype=cp.float64)

    def _finalize(self):
        """Shared post-init: preprocessing + constraints RHS norm.

        Called at the end of every subclass __init__ after _P, _c, _A, _b,
        _G, _h_l, _h_u, _x_l, _x_u have been populated.
        """
        self._preprocess()
        self._constraints_rhs_inf_norm = cp.empty(1, dtype=cp.float64)
        self._compute_constraints_rhs_inf_norm()

    def _preprocess(self):
        self._init_h_l()
        self._init_h_u()
        self.disable_inf_constraints()
        self._init_h_l()
        self._init_h_u()
        self._init_x_l()
        self._init_x_u()

    def _compute_constraints_rhs_inf_norm(self):
        """
        Compute the infinity norm of the right-hand side of the constraints,
        used in computing the relative residuals in the solver.
        """
        inf_norm = self._constraints_rhs_inf_norm
        inf_norm[:] = 0.
        if self.p > 0:
            cp.maximum(inf_norm, cp.linalg.norm(self.b, ord=cp.inf), out=inf_norm)
        if self.num_hu > 0:
            cp.maximum(inf_norm, cp.linalg.norm(self.h_u[self.idx_hu], ord=cp.inf), out=inf_norm)
        if self.num_hl > 0:
            cp.maximum(inf_norm, cp.linalg.norm(self.h_l[self.idx_hl], ord=cp.inf), out=inf_norm)
        if self.num_xu > 0:
            cp.maximum(inf_norm, cp.linalg.norm(self.x_u[self.idx_xu], ord=cp.inf), out=inf_norm)
        if self.num_xl > 0:
            cp.maximum(inf_norm, cp.linalg.norm(self.x_l[self.idx_xl], ord=cp.inf), out=inf_norm)

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

    @abstractmethod
    def set_P(self, value: Any, check: bool = True):
        """Update P values in-place. If check=True, validate dimensions/sparsity."""
        pass

    @abstractmethod
    def set_c(self, value: Any, check: bool = True):
        pass

    @abstractmethod
    def set_A(self, value: Any, check: bool = True):
        pass

    @abstractmethod
    def set_b(self, value: Any, check: bool = True):
        pass

    @abstractmethod
    def set_G(self, value: Any, check: bool = True):
        pass

    @abstractmethod
    def set_h_l(self, value: Any, check: bool = True):
        pass

    @abstractmethod
    def set_h_u(self, value: Any, check: bool = True):
        pass

    @abstractmethod
    def set_x_l(self, value: Any, check: bool = True):
        pass

    @abstractmethod
    def set_x_u(self, value: Any, check: bool = True):
        pass
    
    @property
    def n(self):
        """Number of variables."""
        return self._P.shape[0]
    
    @property
    def p(self):
        """Number of equality constraints."""
        return self._A.shape[0]
    
    @property
    def m(self):
        """Number of inequality constraints."""
        return self._G.shape[0]
    
    @property
    def num_hl(self):
        """Number of lower inequality constraints."""
        return cp.size(self._idx_hl)
    
    @property
    def idx_hl(self):
        """Indices of lower inequality constraints."""
        return self._idx_hl
    
    @property
    def num_hu(self):
        """Number of upper inequality constraints."""
        return cp.size(self._idx_hu)
    
    @property
    def idx_hu(self):
        """Indices of upper inequality constraints."""
        return self._idx_hu
    
    @property
    def num_xl(self):
        """Number of lower bound constraints."""
        return cp.size(self._idx_xl)
    
    @property
    def idx_xl(self):
        """Indices of lower bound constraints."""
        return self._idx_xl
    
    @property
    def num_xu(self):
        """Number of upper bound constraints."""
        return cp.size(self._idx_xu)
    
    @property
    def idx_xu(self):
        """Indices of upper bound constraints."""
        return self._idx_xu
    
    def _init_h_l(self):
        if self._h_l is not None:
            self._idx_hl = cp.where(self._h_l > -PIQP_INF)[0].astype(cp.int32)
        else:
            self._idx_hl = cp.empty((0,), dtype=cp.int32)
            self._h_l = -2 * PIQP_INF * cp.ones((self.m,), dtype=cp.float64)

    def _init_h_u(self):
        if self._h_u is not None:
            self._idx_hu = cp.where(self._h_u < PIQP_INF)[0].astype(cp.int32)
        else:
            self._idx_hu = cp.empty((0,), dtype=cp.int32)
            self._h_u = 2 * PIQP_INF * cp.ones((self.m,), dtype=cp.float64)
        
    def disable_inf_constraints(self):
        """
        For inequalities like -inf < g'x < +inf, set g to 0 and upper/lower bound to +1/-1
        """
        for i in range(self.m):
            if self._h_l[i] <= -PIQP_INF and self._h_u[i] >= PIQP_INF:
                self._G[i, :] = cp.zeros((self.n))
                self._h_l[i] = -1.
                self._h_u[i] = 1.
        
    def _init_x_l(self):
        if self._x_l is not None:
            self._idx_xl = cp.where(self._x_l > -PIQP_INF)[0].astype(cp.int32)
        else:
            self._idx_xl = cp.empty((0,), dtype=cp.int32)
            self._x_l = -2 * PIQP_INF * cp.ones((self.n,), dtype=cp.float64)

    def _init_x_u(self):
        if self._x_u is not None:
            self._idx_xu = cp.where(self._x_u < PIQP_INF)[0].astype(cp.int32)
        else:
            self._idx_xu = cp.empty((0,), dtype=cp.int32)
            self._x_u = 2 * PIQP_INF * cp.ones((self.n,), dtype=cp.float64)
