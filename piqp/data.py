import numpy as np
from typing import Optional

class Data:
    def __init__(self, 
                 P, 
                 c, 
                 A: Optional[np.ndarray] = None, 
                 b: Optional[np.ndarray] = None, 
                 G: Optional[np.ndarray] = None, 
                 h_u: Optional[np.ndarray] = None, 
                 h_l: Optional[np.ndarray] = None, 
                 x_u: Optional[np.ndarray] = None, 
                 x_l: Optional[np.ndarray] = None):
        
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
            A = np.zeros((0, P.shape[0]))
            b = np.zeros(0)
        
        if G is not None and h_u is not None and h_l is not None:
            if G.ndim != 2:
                raise ValueError("G must be a two-dimensional array.")
            if h_u.ndim != 1 or h_l.ndim != 1:
                raise ValueError("h_u and h_l must be one-dimensional arrays.")
            if G.shape[0] != h_u.shape[0] or G.shape[0] != h_l.shape[0]:
                raise ValueError("Dimension mismatch between G, h_u, and h_l.")
        else:
            G = np.zeros((0, P.shape[0]))
            h_u = np.zeros(0)
            h_l = np.zeros(0)

        if x_l is not None and x_u is not None:
            if x_l.ndim != 1 or x_u.ndim != 1:
                raise ValueError("x_l and x_u must be one-dimensional arrays.")
            if x_l.shape[0] != P.shape[0] or x_u.shape[0] != P.shape[0]:
                raise ValueError("Dimension mismatch between x_l, x_u, and P.")
        else:
            x_l = np.full(P.shape[0], -1.e20)
            x_u = np.full(P.shape[0], 1.e20)

        self._A = A
        self._b = b
        self._G = G
        self._h_u = h_u
        self._h_l = h_l
        # self.x_u = x_u if x_u is not None else np.full(P.shape[0], np.inf)
        # self.x_l = x_l if x_l is not None else np.full(P.shape[0], -np.inf)
        self._x_u = x_u
        self._x_l = x_l
        

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
