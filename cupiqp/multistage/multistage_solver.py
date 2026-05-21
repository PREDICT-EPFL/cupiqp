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
        return MultistageData(P, c, A, b, G, h_u, h_l, x_u, x_l)

    def _init_preconditioner(self):
        return MultistageRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            self._data.idx_xl, self._data.idx_xu,
            self._data.idx_hl, self._data.idx_hu,
            data=self._data,
            use_warp_tile_kernels=True,
        )

    def setup(self, P, c, A=None, b=None, G=None,
              h_u=None, h_l=None, x_u=None, x_l=None):
        super().setup(P, c, A, b, G, h_u, h_l, x_u, x_l)
        if self.settings.enable_grad:
            d = self._data
            B = d.batch_size
            N = d.num_blocks
            d_sz = d.block_size

            # Block-structured matrix-gradient containers.
            self._dP_blk = BlockTridiagMat(
                num_diag_blocks=N, block_size=d_sz, batch_size=B,
            )
            if d.p > 0:
                r_a = d._A.rows_of_blocks
                N_a = d._A.N
                self._dA_blk = BlockBidiagMat(
                    rows_of_blocks=r_a, cols_of_blocks=d_sz, N=N_a, batch_size=B,
                )
            else:
                r_a, N_a = 0, 0
                self._dA_blk = None
            if d.m > 0:
                r_g = d._G.rows_of_blocks
                N_g = d._G.N
                self._dG_blk = BlockBidiagMat(
                    rows_of_blocks=r_g, cols_of_blocks=d_sz, N=N_g, batch_size=B,
                )
            else:
                r_g, N_g = 0, 0
                self._dG_blk = None

            # BlockVec containers for the six vector grads. The kernel
            # writes to flat (B, k) DLPack views of their underlying
            # warp buffers — modifying the view mutates the BlockVec.
            self._dc_blk   = BlockVec(num_blocks=N,     rows=d_sz, batch_size=B)
            self._dx_u_blk = BlockVec(num_blocks=N,     rows=d_sz, batch_size=B)
            self._dx_l_blk = BlockVec(num_blocks=N,     rows=d_sz, batch_size=B)
            if d.p > 0:
                self._db_blk = BlockVec(num_blocks=N_a + 1, rows=r_a, batch_size=B)
            else:
                self._db_blk = None
            if d.m > 0:
                self._dh_u_blk = BlockVec(num_blocks=N_g + 1, rows=r_g, batch_size=B)
                self._dh_l_blk = BlockVec(num_blocks=N_g + 1, rows=r_g, batch_size=B)
            else:
                self._dh_u_blk = None
                self._dh_l_blk = None

            # Flat (B, k) DLPack views into BlockVec.data — what we
            # actually pass to the kernel as wp.array2d outputs.
            def _flat(blk):
                if blk is None:
                    return cp.empty((B, 0), dtype=cp.float64)
                return cp.from_dlpack(wp.to_dlpack(blk.data)).reshape(B, -1)

            self._dc_flat   = _flat(self._dc_blk)
            self._db_flat   = _flat(self._db_blk)
            self._dh_u_flat = _flat(self._dh_u_blk)
            self._dh_l_flat = _flat(self._dh_l_blk)
            self._dx_u_flat = _flat(self._dx_u_blk)
            self._dx_l_flat = _flat(self._dx_l_blk)

            # Empty placeholder warp buffers for when A or G are absent
            # — the kernel still needs valid array arguments though its
            # corresponding dispatch sub-range collapses to size 0.
            empty_A = wp.zeros((B, 0, 0, 0), dtype=wp.float64, device="cuda")
            self._dA_D_buf = self._dA_blk.D if self._dA_blk is not None else empty_A
            self._dA_E_buf = self._dA_blk.E if self._dA_blk is not None else empty_A
            self._dG_D_buf = self._dG_blk.D if self._dG_blk is not None else empty_A
            self._dG_E_buf = self._dG_blk.E if self._dG_blk is not None else empty_A

            # Eager-compile the fused multistage data-gradients kernel.
            self._multistage_data_gradients_kernel = create_multistage_data_gradients_kernel(
                N, d_sz, N_a, r_a, N_g, r_g, d.p, d.m, d.n,
            )

    def _compute_data_gradients(self, adjoint_vector: Variables) -> MultistageData:
        r"""Build a :class:`MultistageData` populated with user-space
        gradients on the same block-structured pattern as the original
        problem matrices, via a single fused warp launch.

        ``dP`` is a :class:`BlockTridiagMat` (diag + lower off-diag
        blocks; upper is implicit by symmetry). ``dA, dG`` are
        :class:`BlockBidiagMat` (D + E blocks). Vector grads
        ``dc, db, dh_*, dx_*`` are :class:`BlockVec` whose underlying
        warp buffers are written directly by the kernel via flat
        DLPack views.
        """
        data = self._data
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
                    self._dP_blk.diag_blocks.data,
                    self._dP_blk.off_diag_blocks_lower.data,
                    self._dA_D_buf, self._dA_E_buf,
                    self._dG_D_buf, self._dG_E_buf,
                    self._dc_flat, self._db_flat,
                    self._dh_u_flat, self._dh_l_flat,
                    self._dx_u_flat, self._dx_l_flat,
                ],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )

        return MultistageData(
            P=self._dP_blk, c=self._dc_blk,
            A=self._dA_blk, b=self._db_blk,
            G=self._dG_blk, h_u=self._dh_u_blk, h_l=self._dh_l_blk,
            x_u=self._dx_u_blk, x_l=self._dx_l_blk,
        )
