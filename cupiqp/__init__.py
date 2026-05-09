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

# Cupy-axis-reduction kernel strategy + the three (strategy x backend)
# concrete classes. Use these when max(n, p, m) is large enough that
# warp tile-kernel compile time dominates first-solve latency.
from .solver_large_problem import (
    DenseLargeProblemSolver,
    SparseLargeProblemSolver,
    MultistageLargeProblemSolver,
)


__all__ = [
    "DenseSolver",
    "SparseSolver",
    "MultistageSolver",
    "DenseLargeProblemSolver",
    "SparseLargeProblemSolver",
    "MultistageLargeProblemSolver",
    "Data",
    "Settings",
    "Result",
    "Status",
    "KKTSystem",
]
