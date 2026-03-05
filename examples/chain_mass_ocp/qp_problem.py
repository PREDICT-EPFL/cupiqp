from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp

class QPProblem(ABC):
    P: sp.csc_matrix
    c: np.ndarray
    Aeq: sp.csc_matrix
    beq: np.ndarray
    Aineq: sp.csc_matrix
    bineq_lb: np.ndarray
    bineq_ub: np.ndarray
    xlb: np.ndarray
    xub: np.ndarray

    @abstractmethod
    def get_solution_from_qp_solution(self, x: np.ndarray):
        pass