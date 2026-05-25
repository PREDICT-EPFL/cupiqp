from typing import Optional
import cupy as cp
import warp as wp

from ..data import Data
from ..typedef import PIQP_INF
from ..utils import to_warp_dtype
from .multistage_utils import BlockTridiagMat, BlockBidiagMat, BlockVec


class MultistageData(Data):
    """
    Multistage QP data with block-structured matrices, natively batched.

    Formulation (applied per batch index ``b``)::

        min  0.5 * x[b]^T P[b] x[b] + c[b]^T x[b]
        s.t. A[b] x[b] = b[b]
             h_l[b] <= G[b] x[b] <= h_u[b]
             x_l[b] <= x[b] <= x_u[b]

    Where P is block-tridiagonal, A and G are block-bidiagonal, and all
    matrices/vectors carry a leading batch axis. ``batch_size`` defaults to
    1 so the single-QP API is unchanged at the user level.

    All flat CuPy views (``_c``, ``_b``, ``_h_l``, …) are zero-copy DLPack
    views of the underlying Warp buffers and have shape ``(B, k)``. Updating
    the Warp buffer (e.g. via ``set_c``) automatically makes the flat view
    reflect the new values.
    """

    def __init__(self, dtype=cp.float64, device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)

    def init(
        self,
        P: BlockTridiagMat,
        c: BlockVec,
        A: Optional[BlockBidiagMat] = None,
        b: Optional[BlockVec] = None,
        G: Optional[BlockBidiagMat] = None,
        h_u: Optional[BlockVec] = None,
        h_l: Optional[BlockVec] = None,
        x_u: Optional[BlockVec] = None,
        x_l: Optional[BlockVec] = None,
    ):
        """Allocate and populate buffers from user inputs. Returns self for chaining."""

        self._require_dtype(P.diag_blocks.data)
        self._require_dtype(P.off_diag_blocks_lower.data)
        self._require_dtype(c.data)
        if A is not None:
            self._require_dtype(A.D)
            self._require_dtype(A.E)
        if b is not None:
            self._require_dtype(b.data)
        if G is not None:
            self._require_dtype(G.D)
            self._require_dtype(G.E)
        for bound in (h_u, h_l, x_u, x_l):
            if bound is not None:
                self._require_dtype(bound.data)

        B = P.batch_size
        block_size = P.block_size
        num_blocks = P.num_diag_blocks
        n = num_blocks * block_size

        self._batch_size = B
        self._n = n

        # We clone every caller-provided block container so the solver owns
        # its buffers — the preconditioner (Ruiz scaling) mutates them in
        # place, and we don't want that to bleed back into the caller's
        # arrays. Flat dlpack views are then built into the owned clones.

        # ---- P (block-tridiagonal) ----
        self._P = P.clone()

        # ---- c (linear cost) ----
        self._c_blk = c.clone()
        self._c = self._block_vec_to_flat(self._c_blk, B)
        if self._c.shape != (B, n):
            raise ValueError(
                f"c flat shape {self._c.shape} != expected ({B}, {n})"
            )

        # ---- A, b (equality constraints) ----
        if A is not None and b is not None:
            self._validate_bidiag(A, block_size, num_blocks, B)
            self._A = A.clone()
            p = (self._A.N + 1) * self._A.rows_of_blocks
            self._b_blk, self._b = self._init_block_vec(b.clone(), B, p)
        elif A is None and b is None:
            self._A = None
            self._b_blk = None
            self._b = cp.zeros((B, 0), dtype=self._dtype)
        else:
            raise ValueError("A and b must both be provided or both be None")

        # ---- G, h_u, h_l (inequality constraints) ----
        if G is not None:
            self._validate_bidiag(G, block_size, num_blocks, B)
            if h_u is None and h_l is None:
                raise ValueError(
                    "Either h_l or h_u must be provided when G is given"
                )
            self._G = G.clone()
            m = (self._G.N + 1) * self._G.rows_of_blocks
            self._h_u_blk, self._h_u = self._init_block_vec(
                h_u.clone() if h_u is not None else None, B, m
            )
            self._h_l_blk, self._h_l = self._init_block_vec(
                h_l.clone() if h_l is not None else None, B, m
            )
        else:
            if h_u is not None or h_l is not None:
                raise ValueError("h_u and h_l must be None when G is None")
            self._G = None
            self._h_u_blk = self._h_l_blk = None
            self._h_u = self._h_l = cp.zeros((B, 0), dtype=self._dtype)

        # ---- x_u, x_l (box constraints) ----
        if x_u is not None:
            self._x_u_block = x_u.clone()
            self._x_u = self._block_vec_to_flat(self._x_u_block, B)
            if self._x_u.shape != (B, n):
                raise ValueError(
                    f"x_u flat shape {self._x_u.shape} != expected ({B}, {n})"
                )
        else:
            self._x_u_block = None
            self._x_u = cp.zeros((B, 0), dtype=self._dtype)

        if x_l is not None:
            self._x_l_block = x_l.clone()
            self._x_l = self._block_vec_to_flat(self._x_l_block, B)
            if self._x_l.shape != (B, n):
                raise ValueError(
                    f"x_l flat shape {self._x_l.shape} != expected ({B}, {n})"
                )
        else:
            self._x_l_block = None
            self._x_l = cp.zeros((B, 0), dtype=self._dtype)

        # Hand off to the shared post-init: builds bound index sets,
        # constraints-RHS inf-norm, etc., all assuming (B, k) shapes — which
        # we now match.
        self._finalize()
        self._x_b_scaling = cp.ones((B, n), dtype=self._dtype)

        # Cache flat dlpack views for allocation-free in-place update paths
        # — these point directly into the Warp block-data buffers.
        self._c_flat_view = self._block_vec_to_flat(self._c_blk, B)
        if self._b_blk is not None:
            self._b_flat_view = self._block_vec_to_flat(self._b_blk, B)
        if self._h_l_blk is not None:
            self._h_l_flat_view = self._block_vec_to_flat(self._h_l_blk, B)
        if self._h_u_blk is not None:
            self._h_u_flat_view = self._block_vec_to_flat(self._h_u_blk, B)
        if self._x_l_block is not None:
            self._x_l_flat_view = self._block_vec_to_flat(self._x_l_block, B)
        if self._x_u_block is not None:
            self._x_u_flat_view = self._block_vec_to_flat(self._x_u_block, B)

    @property
    def n(self):
        return self._n

    @property
    def p(self):
        if self._A is None:
            return 0
        return (self._A.N + 1) * self._A.rows_of_blocks

    @property
    def m(self):
        if self._G is None:
            return 0
        return (self._G.N + 1) * self._G.rows_of_blocks

    @property
    def block_size(self):
        return self._P.block_size

    @property
    def num_blocks(self):
        return self._P.num_diag_blocks

    def _disable_inf_constraints(self):
        """Zero out rows of G where both h_l and h_u are infinite (per batch).

        Bound-structure consistency across batches is enforced separately by
        ``_validate_bound_consistency`` in the base class, so the finite/
        infinite mask is identical across batches and we can derive it from
        batch 0.
        """
        if self._G is None:
            return

        # All batches share the same finite/infinite pattern.
        inf_mask = (self._h_l[0] <= -PIQP_INF) & (self._h_u[0] >= PIQP_INF)
        if not bool(inf_mask.any()):
            return

        N = self._G.N
        r = self._G.rows_of_blocks

        # CuPy zero-copy views of the (B, N, r, c) Warp arrays.
        D_cp = cp.from_dlpack(wp.to_dlpack(self._G.D))
        E_cp = cp.from_dlpack(wp.to_dlpack(self._G.E))

        # mask shape (N+1, r) — same across all batches.
        mask_2d = inf_mask.reshape(N + 1, r)
        D_mask = mask_2d[:N]
        E_mask = mask_2d[1:]

        if bool(D_mask.any()):
            # Broadcast over batch axis: D_cp[:, D_mask] = 0
            D_cp[:, D_mask] = 0.0
        if bool(E_mask.any()):
            E_cp[:, E_mask] = 0.0

        # Reset the infinite-bound rows to a benign finite value across batches.
        # Use the same boolean mask shape as h_l / h_u: (B, m). Broadcast inf_mask
        # across the batch axis.
        self._h_l[:, inf_mask] = -1.0
        self._h_u[:, inf_mask] = 1.0

    def extract_P_diag(self, diag_P: cp.ndarray):
        """Extract the diagonal of every batch's P into ``diag_P`` of shape ``(B, n)``."""
        B = self._batch_size
        d = self.block_size
        N = self.num_blocks
        # (B, N, d, d) — Warp buffer.
        P_D = cp.from_dlpack(wp.to_dlpack(self._P.diag_blocks.data))
        # cp.diagonal over axes 2, 3 gives (B, N, d) → reshape to (B, N*d).
        diag_P[:] = cp.diagonal(P_D, axis1=2, axis2=3).reshape(B, N * d)

    def set_P(self, value: BlockTridiagMat, check: bool = True):
        if check:
            if value.batch_size != self._batch_size:
                raise ValueError(
                    f"P batch_size mismatch: got {value.batch_size}, expected {self._batch_size}"
                )
            if value.num_diag_blocks != self._P.num_diag_blocks:
                raise ValueError(
                    f"P num_diag_blocks mismatch: got {value.num_diag_blocks}, expected {self._P.num_diag_blocks}"
                )
            if value.block_size != self._P.block_size:
                raise ValueError(
                    f"P block_size mismatch: got {value.block_size}, expected {self._P.block_size}"
                )
        wp.copy(self._P.diag_blocks.data, value.diag_blocks.data)
        wp.copy(self._P.off_diag_blocks_lower.data, value.off_diag_blocks_lower.data)

    def set_c(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._c_blk, value)
        wp.copy(self._c_blk.data, value.data)
        self._c[:] = self._c_flat_view

    def set_A(self, value: BlockBidiagMat, check: bool = True):
        if check:
            self._check_same_bidiag(self._A, value)
        wp.copy(self._A.D, value.D)
        wp.copy(self._A.E, value.E)

    def set_b(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._b_blk, value)
        wp.copy(self._b_blk.data, value.data)
        self._b[:] = self._b_flat_view

    def set_G(self, value: BlockBidiagMat, check: bool = True):
        if check:
            self._check_same_bidiag(self._G, value)
        wp.copy(self._G.D, value.D)
        wp.copy(self._G.E, value.E)

    def set_h_l(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._h_l_blk, value)
        wp.copy(self._h_l_blk.data, value.data)
        self._h_l[:] = self._h_l_flat_view

    def set_h_u(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._h_u_blk, value)
        wp.copy(self._h_u_blk.data, value.data)
        self._h_u[:] = self._h_u_flat_view

    def set_x_l(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._x_l_block, value)
        wp.copy(self._x_l_block.data, value.data)
        self._x_l[:] = self._x_l_flat_view

    def set_x_u(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._x_u_block, value)
        wp.copy(self._x_u_block.data, value.data)
        self._x_u[:] = self._x_u_flat_view

    @staticmethod
    def _block_vec_to_flat(bv: BlockVec, B: int) -> cp.ndarray:
        """``(B, num_blocks, rows)`` Warp → ``(B, num_blocks*rows)`` cupy (zero-copy)."""
        return cp.from_dlpack(wp.to_dlpack(bv.data)).reshape(B, -1)

    def _require_dtype(self, array) -> None:
        expected = to_warp_dtype(self._dtype)
        if array.dtype != expected:
            raise TypeError(
                f"Multistage input must have dtype {expected} for this solver; "
                f"got {array.dtype}. Construct inputs with dtype={expected}."
            )

    @staticmethod
    def _validate_bidiag(mat, block_size, num_blocks, batch_size):
        if mat.batch_size != batch_size:
            raise ValueError(
                f"BlockBidiagMat batch_size ({mat.batch_size}) != P batch_size ({batch_size})"
            )
        if mat.cols_of_blocks != block_size:
            raise ValueError(
                f"BlockBidiagMat column block size ({mat.cols_of_blocks}) != P block size ({block_size})"
            )
        if mat.N != num_blocks:
            raise ValueError(
                f"BlockBidiagMat column block count ({mat.N}) != P block count ({num_blocks})"
            )

    @staticmethod
    def _check_same_bidiag(old, new):
        if new.batch_size != old.batch_size:
            raise ValueError(
                f"BlockBidiagMat batch_size mismatch: got {new.batch_size}, expected {old.batch_size}"
            )
        if new.N != old.N:
            raise ValueError(f"BlockBidiagMat N mismatch: got {new.N}, expected {old.N}")
        if new.rows_of_blocks != old.rows_of_blocks:
            raise ValueError(
                f"BlockBidiagMat rows_of_blocks mismatch: got {new.rows_of_blocks}, expected {old.rows_of_blocks}"
            )
        if new.cols_of_blocks != old.cols_of_blocks:
            raise ValueError(
                f"BlockBidiagMat cols_of_blocks mismatch: got {new.cols_of_blocks}, expected {old.cols_of_blocks}"
            )

    @staticmethod
    def _check_same_block_vec(old, new):
        if new.data.shape != old.data.shape:
            raise ValueError(
                f"BlockVec shape mismatch: got {tuple(new.data.shape)}, expected {tuple(old.data.shape)}"
            )

    def _init_block_vec(self, bv, batch_size, expected_size):
        if bv is None:
            return None, None
        if bv.batch_size != batch_size:
            raise ValueError(
                f"BlockVec batch_size ({bv.batch_size}) != P batch_size ({batch_size})"
            )
        flat = self._block_vec_to_flat(bv, batch_size)
        if flat.shape != (batch_size, expected_size):
            raise ValueError(
                f"BlockVec flat shape {flat.shape} != expected ({batch_size}, {expected_size})"
            )
        return bv, flat
