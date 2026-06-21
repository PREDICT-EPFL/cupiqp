from importlib.metadata import version as _pkg_version, PackageNotFoundError

from .data import Data
from .settings import Settings
from .results import Result, Status
from .typedef import PIQP_INF

from .dense.dense_data import DenseData

from .sparse.sparse_data import SparseData
from .sparse.batched_csr import UniformBatchedCsrMatrix

from .multistage.multistage_data import MultistageData
from .multistage.ocp_data import OcpData
from .multistage.multistage_utils import BlockTridiagMat, BlockBidiagMat, BlockVec


# Type-strict, backend-specific Solver subclasses.
# Each enforces a 1-to-1 mapping between its KKT backend and the storage
# category of the user's P / A / G inputs.
from .dense.dense_solver import DenseSolver
from .sparse.sparse_solver import SparseSolver
from .multistage.multistage_solver import MultistageSolver
from .multistage.ocp_solver import OcpSolver


CUPIQP_UNSOLVED          = Status.CUPIQP_UNSOLVED
CUPIQP_SOLVED            = Status.CUPIQP_SOLVED
CUPIQP_MAX_ITER_REACHED  = Status.CUPIQP_MAX_ITER_REACHED
CUPIQP_PRIMAL_INFEASIBLE = Status.CUPIQP_PRIMAL_INFEASIBLE
CUPIQP_DUAL_INFEASIBLE   = Status.CUPIQP_DUAL_INFEASIBLE
CUPIQP_NUMERICAL_ISSUES  = Status.CUPIQP_NUMERICAL_ISSUES

try:
    __version__ = _pkg_version("cupiqp")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"


__all__ = [
    # Solvers
    "DenseSolver",
    "SparseSolver",
    "MultistageSolver",
    "OcpSolver",
    # Problem data
    "Data",
    "DenseData",
    "SparseData",
    "MultistageData",
    "OcpData",
    "BlockTridiagMat",
    "BlockBidiagMat",
    "BlockVec",
    "UniformBatchedCsrMatrix",
    # Configuration / results
    "Settings",
    "Result",
    "Status",
    # Status aliases (PIQP-style)
    "CUPIQP_UNSOLVED",
    "CUPIQP_SOLVED",
    "CUPIQP_MAX_ITER_REACHED",
    "CUPIQP_PRIMAL_INFEASIBLE",
    "CUPIQP_DUAL_INFEASIBLE",
    "CUPIQP_NUMERICAL_ISSUES",
    # Constants
    "PIQP_INF",
    "__version__",
]
