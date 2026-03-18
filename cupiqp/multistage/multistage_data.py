from typing import Optional
import cupy as cp
import warp as wp

from ..data import Data
from ..typedef import PIQP_INF
from .multistage_utils import BlockTridiagMat, BlockBidiagMat, BlockVec


class MultistageData(Data):
    """
    Multistage QP data with block-structured matrices.

    Formulation::

        min  0.5 * x^T P x + c^T x
        s.t. A x = b
             h_l <= G x <= h_u
             x_l <= x <= x_u

    Where P is block-tridiagonal, A and G are block-bidiagonal,
    and all vectors are block-wise (BlockVec).

    Flat CuPy arrays (_c, _b, _h_l, …) are zero-copy DLPack views of the
    underlying Warp buffers.  Updating the Warp buffer (e.g. via ``set_c``)
    automatically makes the flat view reflect the new values — no explicit
    sync step is needed.
    """

    def __init__(
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
        # NOTE: intentionally NOT calling super().__init__() — base Data
        # expects dense/sparse 2-D matrices; we store block structures instead.

        _block_size = P.diag_blocks.data.shape[1]
        _num_blocks = P.num_diag_blocks
        _n = _num_blocks * _block_size

        # ---- P (block-tridiagonal) ----
        self._P = P

        # ---- c (linear cost) ----
        self._c_blk = c
        self._c = self._block_vec_to_flat(c)
        if self._c.shape[0] != _n:
            raise ValueError(
                f"c has {self._c.shape[0]} elements, expected {_n}"
            )

        # ---- A, b (equality constraints) ----
        if A is not None and b is not None:
            self._validate_bidiag(A, _block_size, _num_blocks, "A")
            self._A = A
            _p = (A.N + 1) * A.rows_of_blocks
            self._b_blk, self._b = self._init_block_vec(b, _p, "b")
        elif A is None and b is None:
            self._A = None
            self._b_blk = None
            self._b = cp.zeros(0, dtype=cp.float64)
        else:
            raise ValueError("A and b must both be provided or both be None")

        # ---- G, h_u, h_l (inequality constraints) ----
        if G is not None:
            self._validate_bidiag(G, _block_size, _num_blocks, "G")
            if h_u is None and h_l is None:
                raise ValueError(
                    "Either h_l or h_u must be provided when G is given"
                )
            self._G = G
            _m = (G.N + 1) * G.rows_of_blocks
            self._h_u_blk, self._h_u = self._init_block_vec(h_u, _m, "h_u")
            self._h_l_blk, self._h_l = self._init_block_vec(h_l, _m, "h_l")
        else:
            if h_u is not None or h_l is not None:
                raise ValueError("h_u and h_l must be None when G is None")
            self._G = None
            self._h_u_blk = self._h_l_blk = None
            self._h_u = self._h_l = None

        # ---- x_u, x_l (box constraints) ----
        if x_u is not None:
            self._x_u_block = x_u
            self._x_u = self._block_vec_to_flat(x_u).astype(cp.float64)
            if self._x_u.shape[0] != _n:
                raise ValueError(
                    f"x_u has {self._x_u.shape[0]} elements, expected {_n}"
                )
        else:
            self._x_u_block = None
            self._x_u = None

        if x_l is not None:
            self._x_l_block = x_l
            self._x_l = self._block_vec_to_flat(x_l).astype(cp.float64)
            if self._x_l.shape[0] != _n:
                raise ValueError(
                    f"x_l has {self._x_l.shape[0]} elements, expected {_n}"
                )
        else:
            self._x_l_block = None
            self._x_l = None

        # preprocessing (reuses inherited _init_h_l / _init_h_u / _init_x_l / _init_x_u)
        self._preprocess()
        self._constraints_rhs_inf_norm = cp.empty(1, dtype=cp.float64)
        self._compute_constraints_rhs_inf_norm()

        # cache flat dlpack views for allocation-free update() path, 
        # points directly into the warp block data buffers.
        self._c_flat_view = self._block_vec_to_flat(self._c_blk)
        if self._b_blk is not None:
            self._b_flat_view = self._block_vec_to_flat(self._b_blk)
        if self._h_l_blk is not None:
            self._h_l_flat_view = self._block_vec_to_flat(self._h_l_blk)
        if self._h_u_blk is not None:
            self._h_u_flat_view = self._block_vec_to_flat(self._h_u_blk)
        if self._x_l_block is not None:
            self._x_l_flat_view = self._block_vec_to_flat(self._x_l_block)
        if self._x_u_block is not None:
            self._x_u_flat_view = self._block_vec_to_flat(self._x_u_block)

    @property
    def n(self):
        return self._P.num_diag_blocks * self._P.diag_blocks.data.shape[1]

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
        return self._P.diag_blocks.data.shape[1]

    @property
    def num_blocks(self):
        return self._P.num_diag_blocks

    def disable_inf_constraints(self):
        """Zero out rows of G where both h_l and h_u are infinite."""
        if self._G is None:
            return

        inf_mask = (self._h_l <= -PIQP_INF) & (self._h_u >= PIQP_INF)
        if not inf_mask.any():
            return

        N = self._G.N
        r = self._G.rows_of_blocks

        # cupy views of the warp block arrays (zero-copy)
        D_cp = cp.from_dlpack(wp.to_dlpack(self._G.D))  # (N, r, c)
        E_cp = cp.from_dlpack(wp.to_dlpack(self._G.E))  # (N, r, c)

        inf_mask_2d = inf_mask.reshape(N + 1, r)

        # D blocks occupy block rows 0..N-1
        D_mask = inf_mask_2d[:N]
        if D_mask.any():
            D_cp[D_mask] = 0.0

        # E blocks occupy block rows 1..N (stored as E[0..N-1])
        E_mask = inf_mask_2d[1:]
        if E_mask.any():
            E_cp[E_mask] = 0.0

        self._h_l[inf_mask] = -1.0
        self._h_u[inf_mask] = 1.0

    def extract_P_diag(self, diag_P: cp.ndarray):
        raise NotImplementedError

    def set_P(self, value: BlockTridiagMat, check: bool = True):
        if check:
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
            self._check_same_block_vec(self._c_blk, value, "c")
        wp.copy(self._c_blk.data, value.data)
        self._c[:] = self._c_flat_view

    def set_A(self, value: BlockBidiagMat, check: bool = True):
        if check:
            self._check_same_bidiag(self._A, value, "A")
        wp.copy(self._A.D, value.D)
        wp.copy(self._A.E, value.E)

    def set_b(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._b_blk, value, "b")
        wp.copy(self._b_blk.data, value.data)
        self._b[:] = self._b_flat_view

    def set_G(self, value: BlockBidiagMat, check: bool = True):
        if check:
            self._check_same_bidiag(self._G, value, "G")
        wp.copy(self._G.D, value.D)
        wp.copy(self._G.E, value.E)

    def set_h_l(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._h_l_blk, value, "h_l")
        wp.copy(self._h_l_blk.data, value.data)
        self._h_l[:] = self._h_l_flat_view

    def set_h_u(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._h_u_blk, value, "h_u")
        wp.copy(self._h_u_blk.data, value.data)
        self._h_u[:] = self._h_u_flat_view

    def set_x_l(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._x_l_block, value, "x_l")
        wp.copy(self._x_l_block.data, value.data)
        self._x_l[:] = self._x_l_flat_view

    def set_x_u(self, value: BlockVec, check: bool = True):
        if check:
            self._check_same_block_vec(self._x_u_block, value, "x_u")
        wp.copy(self._x_u_block.data, value.data)
        self._x_u[:] = self._x_u_flat_view

    @staticmethod
    def _block_vec_to_flat(bv: BlockVec) -> cp.ndarray:
        """Zero-copy flat cupy view of a BlockVec's warp data."""
        return cp.from_dlpack(wp.to_dlpack(bv.data)).reshape(-1)

    @staticmethod
    def _validate_bidiag(mat, block_size, num_blocks, name):
        if mat.cols_of_blocks != block_size:
            raise ValueError(
                f"{name} column block size ({mat.cols_of_blocks}) != P block size ({block_size})"
            )
        if mat.N != num_blocks:
            raise ValueError(
                f"{name} column block count ({mat.N}) != P block count ({num_blocks})"
            )

    @staticmethod
    def _check_same_bidiag(old, new, name):
        if new.N != old.N:
            raise ValueError(f"{name} N mismatch: got {new.N}, expected {old.N}")
        if new.rows_of_blocks != old.rows_of_blocks:
            raise ValueError(
                f"{name} rows_of_blocks mismatch: got {new.rows_of_blocks}, expected {old.rows_of_blocks}"
            )
        if new.cols_of_blocks != old.cols_of_blocks:
            raise ValueError(
                f"{name} cols_of_blocks mismatch: got {new.cols_of_blocks}, expected {old.cols_of_blocks}"
            )

    @staticmethod
    def _check_same_block_vec(old, new, name):
        if new.data.shape != old.data.shape:
            raise ValueError(
                f"{name} shape mismatch: got {tuple(new.data.shape)}, expected {tuple(old.data.shape)}"
            )

    def _init_block_vec(self, bv, expected_size, name):
        if bv is None:
            return None, None
        flat = self._block_vec_to_flat(bv)
        if flat.shape[0] != expected_size:
            raise ValueError(
                f"{name} has {flat.shape[0]} elements, expected {expected_size}"
            )
        return bv, flat
