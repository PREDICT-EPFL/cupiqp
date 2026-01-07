from typing import Optional
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, csc_matrix


from ..data import Data

class SparseData(Data):
    """
    Sparse data structure for optimization problems.
    """
    def __init__(self, 
                 P: csr_matrix, 
                 q: cp.ndarray, 
                 A: Optional[csr_matrix] = None, 
                 b: Optional[cp.ndarray] = None, 
                 G: Optional[csr_matrix] = None, 
                 h_u: Optional[cp.ndarray] = None,
                 h_l: Optional[cp.ndarray] = None,
                 x_u: Optional[cp.ndarray] = None,
                 x_l: Optional[cp.ndarray] = None):
        super().__init__(P, q, A, b, G, h_u, h_l, x_u, x_l)
        # Additional initialization for sparse data can be added here
    