from .solver import SolverBase
from .solver_large_problem import LargeProblemSolver
from .data import Data
from .settings import Settings
from .results import Result, Status
from .kkt_systems import KKTSystem

# Type-strict, backend-specific Solver subclasses.
# Each enforces a 1-to-1 mapping between its KKT backend and the storage
# category of the user's P / A / G inputs.
from .dense.dense_solver import DenseSolver
from .sparse.sparse_solver import SparseSolver
from .multistage.multistage_solver import MultistageSolver


__all__ = [
    "SolverBase",
    "LargeProblemSolver",
    "DenseSolver",
    "SparseSolver",
    "MultistageSolver",
    "Data",
    "Settings",
    "Result",
    "Status",
    "KKTSystem",
]
