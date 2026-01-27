from abc import ABC, abstractmethod
import cupy as cp


from .data import Data
from .kkt_fwd import KKTUpdateOptions
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
    def update_scalings_and_factor(self, data: Data, delta: float, x_reg: Vector, z_reg: Vector) -> bool:
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
    def eval_A_xn_and_AT_xt(self, data: Data, alpha_n: float, alpha_t: float, xn: Vector, xt: Vector, zn: Vector, zt: Vector) -> None:
        """
        Evaluate Ax and A^T xt with scaling factors alpha_n and alpha_t
        zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
        """
        pass
    
    @abstractmethod
    def eval_G_xn_and_GT_xt(self, data: Data, alpha_n: float, alpha_t: float, xn: Vector, xt: Vector, zn: Vector, zt: Vector) -> None:
        """
        Evaluate Gx and G^T xt with scaling factors alpha_n and alpha_t
        zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
        """
        pass

