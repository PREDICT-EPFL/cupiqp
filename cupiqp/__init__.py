from importlib.metadata import version as _pkg_version, PackageNotFoundError

from .data import Data
from .settings import Settings
from .results import Result, Status, InfoSnapshot
from .typedef import PIQP_INF

from .dense.dense_data import DenseData

from .sparse.sparse_data import SparseData

from .multistage.multistage_data import MultistageData


# Type-strict, backend-specific Solver subclasses.
# Each enforces a 1-to-1 mapping between its KKT backend and the storage
# category of the user's P / A / G inputs.
from .dense.dense_solver import DenseSolver
from .sparse.sparse_solver import SparseSolver
from .multistage.multistage_solver import MultistageSolver


CUPIQP_UNSOLVED          = Status.CUPIQP_UNSOLVED
CUPIQP_SOLVED            = Status.CUPIQP_SOLVED
CUPIQP_MAX_ITER_REACHED  = Status.CUPIQP_MAX_ITER_REACHED
CUPIQP_PRIMAL_INFEASIBLE = Status.CUPIQP_PRIMAL_INFEASIBLE
CUPIQP_DUAL_INFEASIBLE   = Status.CUPIQP_DUAL_INFEASIBLE
CUPIQP_NUMERICAL_ISSUES  = Status.CUPIQP_NUMERICAL_ISSUES

# Framework interfaces are loaded on first access so that ``import cupiqp``
# imports neither torch nor jax; ``import cupiqp.torch`` and
# ``import cupiqp.jax`` work as well.
_LAZY_SUBMODULES = ("torch", "jax")


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    __version__ = _pkg_version("cupiqp")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"


__all__ = [
    # Solvers
    "DenseSolver",
    "SparseSolver",
    "MultistageSolver",
    # Problem data
    "Data",
    "DenseData",
    "SparseData",
    "MultistageData",
    # Configuration / results
    "Settings",
    "Result",
    "Status",
    "InfoSnapshot",
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
