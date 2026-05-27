from importlib.metadata import version as _pkg_version, PackageNotFoundError

from .data import Data
from .settings import Settings
from .results import Result, Status
from .kkt_systems import KKTSystem
from .typedef import PIQP_INF

from .dense.dense_data import DenseData

from .sparse.sparse_data import SparseData
from .sparse.batched_csr import UniformBatchedCsrMatrix

from .multistage.multistage_data import MultistageData
from .multistage.multistage_utils import BlockTridiagMat, BlockBidiagMat, BlockVec


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


PIQP_UNSOLVED = Status.PIQP_UNSOLVED
PIQP_SOLVED = Status.PIQP_SOLVED
PIQP_MAX_ITER_REACHED = Status.PIQP_MAX_ITER_REACHED
PIQP_PRIMAL_INFEASIBLE = Status.PIQP_PRIMAL_INFEASIBLE
PIQP_DUAL_INFEASIBLE = Status.PIQP_DUAL_INFEASIBLE
PIQP_NUMERICAL_ISSUES = Status.PIQP_NUMERICAL_ISSUES

try:
    __version__ = _pkg_version("cupiqp")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"


__all__ = [
    # Solvers
    "DenseSolver",
    "SparseSolver",
    "MultistageSolver",
    "DenseLargeProblemSolver",
    "SparseLargeProblemSolver",
    "MultistageLargeProblemSolver",
    # Problem data
    "Data",
    "DenseData",
    "SparseData",
    "MultistageData",
    "BlockTridiagMat",
    "BlockBidiagMat",
    "BlockVec",
    "UniformBatchedCsrMatrix",
    # Configuration / results
    "Settings",
    "Result",
    "Status",
    "KKTSystem",
    # Status aliases (PIQP-style)
    "PIQP_UNSOLVED",
    "PIQP_SOLVED",
    "PIQP_MAX_ITER_REACHED",
    "PIQP_PRIMAL_INFEASIBLE",
    "PIQP_DUAL_INFEASIBLE",
    "PIQP_NUMERICAL_ISSUES",
    # Constants
    "PIQP_INF",
    "__version__",
]
