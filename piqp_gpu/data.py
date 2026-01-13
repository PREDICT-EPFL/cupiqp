import cupy as cp
from typing import Optional, Union
import cupyx.scipy.sparse as sp

from .typedef import PIQP_INF

ArrayLike = Union[cp.ndarray, sp.spmatrix]


class Data:
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
        
        self._P = self._as_float64(P)
        self._c = self._as_float64(c)

        if self._P.ndim != 2 or self._P.shape[0] != self._P.shape[1]:
            raise ValueError("P must be a square matrix.")
        if self._c.ndim != 1:
            raise ValueError("c must be a one-dimensional array.")
        if P.shape[0] != c.shape[0]:
            raise ValueError("Dimension mismatch between P and c.")
        
        if A is not None and b is not None:
            if A.ndim != 2:
                raise ValueError("A must be a two-dimensional array.")
            if b.ndim != 1:
                raise ValueError("b must be a one-dimensional array.")
            if A.shape[0] != b.shape[0]:
                raise ValueError("Dimension mismatch between A and b.")
        else:
            A = cp.zeros((0, P.shape[0]))
            b = cp.zeros(0)
        
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
                raise ValueError("h_l and h_l should be None when G is None.")
            G = cp.zeros((0, P.shape[0]))
            h_u = cp.zeros(0)
            h_l = cp.zeros(0)

        if x_l is not None and x_u is not None:
            if x_l.ndim != 1 or x_u.ndim != 1:
                raise ValueError("x_l and x_u must be one-dimensional arrays.")
            if x_l.shape[0] != P.shape[0] or x_u.shape[0] != P.shape[0]:
                raise ValueError("Dimension mismatch between x_l, x_u, and P.")
        
        self._A = self._as_float64(A)
        self._b = self._as_float64(b)
        self._G = self._as_float64(G)
        self._h_u = self._as_float64(h_u) if h_u is not None else None
        self._h_l = self._as_float64(h_l) if h_l is not None else None
        self._x_u = self._as_float64(x_u) if x_u is not None else None
        self._x_l = self._as_float64(x_l) if x_l is not None else None
        self._preprocess()

    @staticmethod
    def _as_float64(M: ArrayLike) -> ArrayLike:
        if sp.issparse(M):
            return M.astype(cp.float64)
        else:
            return cp.array(M, dtype=cp.float64)

    def _preprocess(self):
        self.set_h_l()
        self.set_h_u()
        self.disable_inf_constraints()
        self.set_h_l()
        self.set_h_u()
        self.set_x_l()
        self.set_x_u()
        

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
        return len(self._idx_hl)
    
    @property
    def idx_hl(self):
        """Indices of lower inequality constraints."""
        return self._idx_hl
    
    @property
    def num_hu(self):
        """Number of upper inequality constraints."""
        return len(self._idx_hu)
    
    @property
    def idx_hu(self):
        """Indices of upper inequality constraints."""
        return self._idx_hu
    
    @property
    def num_xl(self):
        """Number of lower bound constraints."""
        return len(self._idx_xl)
    
    @property
    def idx_xl(self):
        """Indices of lower bound constraints."""
        return self._idx_xl
    
    @property
    def num_xu(self):
        """Number of upper bound constraints."""
        return len(self._idx_xu)
    
    @property
    def idx_xu(self):
        """Indices of upper bound constraints."""
        return self._idx_xu
    
    def set_h_l(self):
        if self._h_l is not None:
            self._idx_hl = cp.where(self._h_l > -PIQP_INF)[0].tolist()
            self._h_l = cp.array(self._h_l, dtype=cp.float64)
        else:
            self._idx_hl = []
            self._h_l = -2 * PIQP_INF * cp.ones((self.m,))
        
    def set_h_u(self):
        if self._h_u is not None:
            self._idx_hu = cp.where(self._h_u < PIQP_INF)[0].tolist()
            self._h_u = cp.array(self._h_u, dtype=cp.float64)
        else:
            self._idx_hu = []
            self._h_u = 2 * PIQP_INF * cp.ones((self.m,))
        
    def disable_inf_constraints(self):
        """
        For inequalities like -inf < g'x < +inf, set g to 0 and upper/lower bound to +1/-1
        """
        for i in range(self.m):
            if self._h_l[i] <= -PIQP_INF and self._h_u[i] >= PIQP_INF:
                self._G[i, :] = cp.zeros((self.n))
                self._h_l[i] = -1.
                self._h_u[i] = 1.
        
    def set_x_l(self):
        if self._x_l is not None:
            self._idx_xl = cp.where(self._x_l > -PIQP_INF)[0].tolist()
            self._x_l = cp.array(self._x_l, dtype=cp.float64)
        else:
            self._idx_xl = []
            self._x_l = -2 * PIQP_INF * cp.ones((self.n,))

    def set_x_u(self):
        if self._x_u is not None:
            self._idx_xu = cp.where(self._x_u < PIQP_INF)[0].tolist()
            self._x_u = cp.array(self._x_u, dtype=cp.float64)
        else:
            self._idx_xu = []
            self._x_u = 2 * PIQP_INF * cp.ones((self.n,))
