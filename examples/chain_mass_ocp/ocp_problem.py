from abc import ABC, abstractmethod
import numpy as np

class OCPProblem(ABC):
    N: int
    nx: int
    nu: int

    A: np.ndarray
    B: np.ndarray

    Q: np.ndarray
    QN: np.ndarray
    R: np.ndarray
    S: np.ndarray

    x0: np.ndarray
    xl: np.ndarray
    xu: np.ndarray

    ul: np.ndarray
    uu: np.ndarray

    C: np.ndarray
    D: np.ndarray
    gl: np.ndarray
    gu: np.ndarray

    @abstractmethod
    def get_solution_from_ocp_solution(self, X: np.ndarray, U: np.ndarray):
        pass