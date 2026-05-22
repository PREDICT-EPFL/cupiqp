import cupy as cp
import warp as wp

from ..results import Variables
from ..settings import Settings
from ..solver import SolverBase
from .multistage_data import MultistageData
from .multistage_preconditioner import MultistageRuizEquilibration
from .multistage_solver_kernels import create_multistage_data_gradients_kernel
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
        self._settings.kkt_solver = "multistage_block_cholesky"

    @SolverBase.settings.setter
    def settings(self, value: Settings) -> None:
        # TODO: here we have to set the kkt solver back. That's pretty ugly. Should be improved in the future
        value.kkt_solver = "multistage_block_cholesky"
        self._settings = value


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
        data = MultistageData(dtype=self.settings.dtype, device=self.settings.device)
        data.init(P, c, A, b, G, h_u, h_l, x_u, x_l)
        return data

    def _init_preconditioner(self):
        return MultistageRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            self._data.idx_xl, self._data.idx_xu,
            self._data.idx_hl, self._data.idx_hu,
            data=self._data,
            use_warp_tile_kernels=True,
            dtype=self._data.dtype,
        )

    def setup(self, P, c, A=None, b=None, G=None,
              h_u=None, h_l=None, x_u=None, x_l=None):
        super().setup(P, c, A, b, G, h_u, h_l, x_u, x_l)
        if self.settings.enable_grad:
            d = self._data
            B = d.batch_size
            N = d.num_blocks
            d_sz = d.block_size
            dtype = d.dtype

            r_a = d._A.rows_of_blocks if d.p > 0 else 0
            N_a = d._A.N              if d.p > 0 else 0
            r_g = d._G.rows_of_blocks if d.m > 0 else 0
            N_g = d._G.N              if d.m > 0 else 0

            placeholder_P = BlockTridiagMat(
                num_diag_blocks=N, block_size=d_sz, batch_size=B, dtype=dtype,
            )
            placeholder_A = BlockBidiagMat(
                rows_of_blocks=r_a, cols_of_blocks=d_sz, N=N_a, batch_size=B, dtype=dtype,
            ) if d.p > 0 else None
            placeholder_G = BlockBidiagMat(
                rows_of_blocks=r_g, cols_of_blocks=d_sz, N=N_g, batch_size=B, dtype=dtype,
            ) if d.m > 0 else None
            # c, x_u, x_l: always (B, n)-sized buffers, matching the original
            # backward-buffer shape regardless of which bound directions are
            # active. The kernel uses index masks (idx_xu / idx_xl) internally.
            placeholder_c = BlockVec(num_blocks=N, rows=d_sz, batch_size=B, dtype=dtype)
            placeholder_xu = BlockVec(num_blocks=N, rows=d_sz, batch_size=B, dtype=dtype)
            placeholder_xl = BlockVec(num_blocks=N, rows=d_sz, batch_size=B, dtype=dtype)
            # b, h_u, h_l: presence tied to A / G existing.
            placeholder_b  = BlockVec(num_blocks=N_a + 1, rows=r_a, batch_size=B, dtype=dtype) if d.p > 0 else None
            placeholder_hu = BlockVec(num_blocks=N_g + 1, rows=r_g, batch_size=B, dtype=dtype) if d.m > 0 else None
            placeholder_hl = BlockVec(num_blocks=N_g + 1, rows=r_g, batch_size=B, dtype=dtype) if d.m > 0 else None

            self._grad_data = MultistageData(dtype=dtype, device=self.settings.device)
            self._grad_data.init(
                P=placeholder_P, c=placeholder_c,
                A=placeholder_A, b=placeholder_b,
                G=placeholder_G, h_u=placeholder_hu, h_l=placeholder_hl,
                x_u=placeholder_xu, x_l=placeholder_xl,
            )

            # Empty placeholder warp buffers for when A or G are absent —
            # the kernel still needs valid array arguments even though its
            # corresponding dispatch sub-range collapses to size 0.
            empty_blocks = wp.zeros((B, 0, 0, 0), dtype=dtype, device="cuda")
            g = self._grad_data
            self._grad_dA_D = g._A.D if g._A is not None else empty_blocks
            self._grad_dA_E = g._A.E if g._A is not None else empty_blocks
            self._grad_dG_D = g._G.D if g._G is not None else empty_blocks
            self._grad_dG_E = g._G.E if g._G is not None else empty_blocks

            # Eager-compile the fused multistage data-gradients kernel.
            self._multistage_data_gradients_kernel = create_multistage_data_gradients_kernel(
                N, d_sz, N_a, r_a, N_g, r_g, d.p, d.m, d.n, dtype=dtype)

    def _compute_data_gradients(self, adjoint_vector: Variables) -> MultistageData:
        r"""Populate ``self._grad_data`` in place and return it.

        Matrix gradients are written into ``self._grad_data._P/_A/_G``
        (block-structured). Vector grads ``c``, ``h_l``, ``x_l`` are
        copies of ``adjoint_vector.x``, ``self._lam_zl_full``,
        ``self._lam_zbl_full``.

        Returns the same instance on every call; its buffers are
        overwritten by the next backward.
        """
        data = self._data
        grad_data = self._grad_data
        B = data.batch_size
        N = data.num_blocks
        d_sz = data.block_size

        N_off = max(N - 1, 0)
        r_a   = data._A.rows_of_blocks if data.p > 0 else 0
        N_a   = data._A.N              if data.p > 0 else 0
        r_g   = data._G.rows_of_blocks if data.m > 0 else 0
        N_g   = data._G.N              if data.m > 0 else 0
        total = (
            N * d_sz * d_sz
            + N_off * d_sz * d_sz
            + 2 * N_a * r_a * d_sz
            + 2 * N_g * r_g * d_sz
            + data.n + data.p + 2 * data.m + 2 * data.n
        )
        if total > 0:
            wp.launch(
                kernel=self._multistage_data_gradients_kernel,
                dim=(B, total),
                inputs=[
                    adjoint_vector.x, adjoint_vector.y,
                    self._lam_zu_full, self._lam_zl_full,
                    self._lam_zbu_full, self._lam_zbl_full,
                    self._zu_full, self._zl_full,
                    self._result.x, self._result.y,
                    grad_data._P.diag_blocks.data,
                    grad_data._P.off_diag_blocks_lower.data,
                    self._grad_dA_D, self._grad_dA_E,
                    self._grad_dG_D, self._grad_dG_E,
                    grad_data._c, grad_data._b,
                    grad_data._h_u, grad_data._h_l,
                    grad_data._x_u, grad_data._x_l,
                ],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )

        return grad_data
