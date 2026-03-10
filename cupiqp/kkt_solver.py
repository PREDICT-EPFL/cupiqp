from abc import ABC, abstractmethod

from .data import Data
from .typedef import Vector, Matrix


class KKTSolverBase(ABC):
    """Represent the system with the following form:
    [P+x_reg     A^T      G^T    ] [Delta_x] = [rhs_x]  
    [A         -delta*I     0    ] [Delta_y] = [rhs_y]
    [G           0     -(z_reg)  ] [Delta_z] = [rhs_z]

    where W = diag(s_i/z_i) for i=1,...,m
    Actually we should note the delta*I as regularization term since it's not necessarily delta*I if there are contributions from the box constriants, which are denoted as reg_x and reg_z.
    """
    def __init__(self):
        pass

    @abstractmethod
    def update_data(self, data: Data, update_P: bool, update_A: bool, update_G: bool) -> None:
        """Notify the KKT solver that problem data has changed.
        """
        pass

    @abstractmethod
    def update_kkt(self, data: Data, delta: float, x_reg: Vector, z_reg: Vector) -> bool:
        pass

    @abstractmethod
    def factor(self) -> bool:
        pass

    @abstractmethod
    def solve(self, data: Data, rhs_x: Vector, rhs_y: Vector, rhs_z: Vector, lhs_x: Vector, lhs_y: Vector, lhs_z: Vector) -> None:
        pass

    @abstractmethod
    def eval_P_x(self, data: Data, alpha: float, x: Vector, z: Vector) -> None:
        """
        Evaluate z = alpha * P * x
        """
        pass

    @abstractmethod
    def eval_A_xn(self, data: Data, alpha_n: float, xn: Vector, zn: Vector) -> None:
        """
        Evaluate Ax with scaling factor alpha_n
        zn = alpha_n * A * xn
        """
        pass

    @abstractmethod
    def eval_AT_xt(self, data: Data, alpha_t: float, xt: Vector, zt: Vector) -> None:
        """
        Evaluate A^T xt with scaling factor alpha_t
        zt = alpha_t * A^T * xt
        """
        pass
    
    @abstractmethod
    def eval_G_xn(self, data: Data, alpha_n: float, xn: Vector, zn: Vector) -> None:
        """
        Evaluate Gx with scaling factor alpha_n
        zn = alpha_n * G * xn
        """
        pass

    @abstractmethod
    def eval_GT_xt(self, data: Data, alpha_t: float, xt: Vector, zt: Vector) -> None:
        """
        Evaluate G^T xt with scaling factor alpha_t
        zt = alpha_t * G^T * xt
        """
        pass

