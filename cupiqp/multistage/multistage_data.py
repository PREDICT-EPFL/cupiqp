from typing import Optional
import cupy as cp
from cupyx.scipy.sparse import csr_matrix

from ..sparse.sparse_data import SparseData


class MultistageData(SparseData):
    """
    Multi-stage data structure for optimization problems.

    P is a block-tri-diagonal matrix
    """
    def __init__(self, 
                 P: csr_matrix,
                 c: cp.ndarray, 
                 A: Optional[csr_matrix] = None, 
                 b: Optional[cp.ndarray] = None, 
                 G: Optional[csr_matrix] = None, 
                 h_u: Optional[cp.ndarray] = None,
                 h_l: Optional[cp.ndarray] = None,
                 x_u: Optional[cp.ndarray] = None,
                 x_l: Optional[cp.ndarray] = None):
        
        super().__init__(P, c, A, b, G, h_u, h_l, x_u, x_l)
