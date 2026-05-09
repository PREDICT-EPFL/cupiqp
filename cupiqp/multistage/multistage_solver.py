from ..solver import SolverBase
from .multistage_data import MultistageData
from .multistage_preconditioner import MultistageRuizEquilibration
from .multistage_utils import BlockTridiagMat, BlockBidiagMat, BlockVec


def _check_block_tridiag(name: str, m) -> None:
    """Validate that ``m`` is a ``BlockTridiagMat`` (skip if ``None``)."""
    if m is None:
        return
    if not isinstance(m, BlockTridiagMat):
        raise TypeError(
            f"MultistageSolver requires {name} to be a BlockTridiagMat "
            f"(from cupiqp.multistage.multistage_utils); "
            f"got {type(m).__name__}. "
            f"For dense input use DenseSolver. "
            f"For generic CSR input use SparseSolver."
        )


def _check_block_bidiag(name: str, m) -> None:
    """Validate that ``m`` is a ``BlockBidiagMat`` (skip if ``None``)."""
    if m is None:
        return
    if not isinstance(m, BlockBidiagMat):
        raise TypeError(
            f"MultistageSolver requires {name} to be a BlockBidiagMat "
            f"(from cupiqp.multistage.multistage_utils); "
            f"got {type(m).__name__}."
        )


def _check_block_vec(name: str, m) -> None:
    """Validate that ``m`` is a :class:`BlockVec` (skip if ``None``).

    The multistage backend uses block-structured storage end-to-end —
    matrices *and* vectors. ``c``, ``b``, ``h_u``, ``h_l``, ``x_u``,
    ``x_l`` are all :class:`BlockVec` instances; passing a flat cupy
    array here would silently mismatch the block layout the multistage
    KKT solver expects.
    """
    if m is None:
        return
    if not isinstance(m, BlockVec):
        raise TypeError(
            f"MultistageSolver requires {name} to be a BlockVec "
            f"(from cupiqp.multistage.multistage_utils); "
            f"got {type(m).__name__}."
        )


class MultistageSolver(SolverBase):
    """Concrete :class:`SolverBase` subclass for the **multistage
    block-Cholesky** KKT backend.

    ``MultistageSolver`` is the type-strict, user-facing entry point
    for QPs whose problem data are pre-built **block-structured**
    matrices that expose the multistage (e.g. MPC) structure to the
    KKT solver. It rejects non-block ``P`` / ``A`` / ``G`` inputs with
    a clear, actionable :class:`TypeError` — generic CSR is not
    auto-promoted into block form, since if the user has not built the
    block matrices themselves the multistage solver cannot exploit the
    structure regardless.

    Block-structured storage is used **end-to-end** — matrices and
    vectors. Accepts:

    * ``P``: :class:`cupiqp.multistage.multistage_utils.BlockTridiagMat`
    * ``A``, ``G``: :class:`cupiqp.multistage.multistage_utils.BlockBidiagMat`
      (or ``None``)
    * ``c``, ``b``, ``h_u``, ``h_l``, ``x_u``, ``x_l``:
      :class:`cupiqp.multistage.multistage_utils.BlockVec` (or ``None``
      where the constraint is absent)

    Examples
    --------
    >>> from cupiqp import MultistageSolver
    >>> from cupiqp.multistage.multistage_utils import (
    ...     BlockTridiagMat, BlockBidiagMat, BlockVec,
    ... )
    >>> P = BlockTridiagMat(num_diag_blocks=N, block_size=d)
    >>> A = BlockBidiagMat(rows_of_blocks=d, cols_of_blocks=d, N=N)
    >>> c = BlockVec(num_blocks=N, rows=d)
    >>> b = BlockVec(num_blocks=N, rows=d)
    >>> # ... fill block data ...
    >>> s = MultistageSolver()
    >>> s.setup(P=P, c=c, A=A, b=b)
    >>> s.solve()
    """

    def __init__(self):
        super().__init__()
        self.settings.kkt_solver = "multistage_block_cholesky"

    def _init_data(self, P, c, A, b, G, h_u, h_l, x_u, x_l):
        # Multistage uses block-structured storage end-to-end:
        # matrices are BlockTridiag/BlockBidiag, vectors are BlockVec.
        _check_block_tridiag("P", P)
        _check_block_bidiag("A", A)
        _check_block_bidiag("G", G)
        _check_block_vec("c", c)
        _check_block_vec("b", b)
        _check_block_vec("h_u", h_u)
        _check_block_vec("h_l", h_l)
        _check_block_vec("x_u", x_u)
        _check_block_vec("x_l", x_l)
        return MultistageData(P, c, A, b, G, h_u, h_l, x_u, x_l)

    def _init_preconditioner(self):
        return MultistageRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            self._data.idx_xl, self._data.idx_xu,
            self._data.idx_hl, self._data.idx_hu,
            data=self._data,
            use_warp_tile_kernels=True,
        )
