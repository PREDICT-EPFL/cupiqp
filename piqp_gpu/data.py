import cupy as cp
from typing import Optional

class Data:
    def __init__(self, 
                 P, 
                 c, 
                 A: Optional[cp.ndarray] = None, 
                 b: Optional[cp.ndarray] = None, 
                 G: Optional[cp.ndarray] = None, 
                 h_u: Optional[cp.ndarray] = None, 
                 h_l: Optional[cp.ndarray] = None, 
                 x_u: Optional[cp.ndarray] = None, 
                 x_l: Optional[cp.ndarray] = None):
        
        self._P = P
        self._c = c

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
        
        if G is not None and h_u is not None and h_l is not None:
            if G.ndim != 2:
                raise ValueError("G must be a two-dimensional array.")
            if h_u.ndim != 1 or h_l.ndim != 1:
                raise ValueError("h_u and h_l must be one-dimensional arrays.")
            if G.shape[0] != h_u.shape[0] or G.shape[0] != h_l.shape[0]:
                raise ValueError("Dimension mismatch between G, h_u, and h_l.")
        else:
            G = cp.zeros((0, P.shape[0]))
            h_u = cp.zeros(0)
            h_l = cp.zeros(0)

        if x_l is not None and x_u is not None:
            if x_l.ndim != 1 or x_u.ndim != 1:
                raise ValueError("x_l and x_u must be one-dimensional arrays.")
            if x_l.shape[0] != P.shape[0] or x_u.shape[0] != P.shape[0]:
                raise ValueError("Dimension mismatch between x_l, x_u, and P.")
            
        x_l = -cp.inf * cp.ones(P.shape[0]) if x_l is None else x_l
        x_u = cp.inf * cp.ones(P.shape[0]) if x_u is None else x_u
        
        self._idx_xl = cp.where(cp.isfinite(x_l))[0]
        self._x_l = x_l[self._idx_xl]
        self._idx_xu = cp.where(cp.isfinite(x_u))[0]
        self._x_u = x_u[self._idx_xu]

        self._A = A
        self._b = b
        self._G = G

        self._idx_hl = cp.where(cp.isfinite(h_l))[0]
        self._idx_hu = cp.where(cp.isfinite(h_u))[0]
        self._h_l = h_l[self._idx_hl]
        self._h_u = h_u[self._idx_hu]
        

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