from typing import Optional

import cupy as cp

from ..data import Data

class DenseData(Data):
    """
    Dense data structure for optimization problems.
    """
    def __init__(self, 
                 P: cp.ndarray, 
                 q: cp.ndarray, 
                 A: Optional[cp.ndarray] = None, 
                 b: Optional[cp.ndarray] = None, 
                 G: Optional[cp.ndarray] = None, 
                 h_u: Optional[cp.ndarray] = None,
                 h_l: Optional[cp.ndarray] = None,
                 x_u: Optional[cp.ndarray] = None,
                 x_l: Optional[cp.ndarray] = None):
        super().__init__(P, q, A, b, G, h_u, h_l, x_u, x_l)
        # Additional initialization for dense data can be added here