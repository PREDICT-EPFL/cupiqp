from typing import Optional, Union

import cupy as cp

from ..data import Data

class DenseData(Data):
    """
    Dense data structure for optimization problems.
    """
    def __init__(self,
                 P: cp.ndarray,
                 c: cp.ndarray,
                 A: Optional[cp.ndarray] = None,
                 b: Optional[cp.ndarray] = None,
                 G: Optional[cp.ndarray] = None,
                 h_u: Optional[cp.ndarray] = None,
                 h_l: Optional[cp.ndarray] = None,
                 x_u: Optional[cp.ndarray] = None,
                 x_l: Optional[cp.ndarray] = None):
        super().__init__(P, c, A, b, G, h_u, h_l, x_u, x_l)

    @staticmethod
    def _as_float64_mat(M: Union[cp.ndarray, None]) -> cp.ndarray:
        return M.astype(cp.float64) if M is not None else cp.zeros((0,), dtype=cp.float64)

    def set_P(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._P.shape:
            raise ValueError(f"P has wrong dimension. Got {value.shape} but expected {self._P.shape}")
        self._P[:] = value

    def set_c(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._c.shape:
            raise ValueError(f"c has wrong dimension. Got {value.shape} but expected {self._c.shape}")
        self._c[:] = value

    def set_A(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._A.shape:
            raise ValueError(f"A has wrong dimension. Got {value.shape} but expected {self._A.shape}")
        self._A[:] = value

    def set_b(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._b.shape:
            raise ValueError(f"b has wrong dimension. Got {value.shape} but expected {self._b.shape}")
        self._b[:] = value

    def set_G(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._G.shape:
            raise ValueError(f"G has wrong dimension. Got {value.shape} but expected {self._G.shape}")
        self._G[:] = value

    def set_h_l(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._h_l.shape:
            raise ValueError(f"h_l has wrong dimension. Got {value.shape} but expected {self._h_l.shape}")
        self._h_l[:] = value

    def set_h_u(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._h_u.shape:
            raise ValueError(f"h_u has wrong dimension. Got {value.shape} but expected {self._h_u.shape}")
        self._h_u[:] = value

    def set_x_l(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._x_l.shape:
            raise ValueError(f"x_l has wrong dimension. Got {value.shape} but expected {self._x_l.shape}")
        self._x_l[:] = value

    def set_x_u(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._x_u.shape:
            raise ValueError(f"x_u has wrong dimension. Got {value.shape} but expected {self._x_u.shape}")
        self._x_u[:] = value
