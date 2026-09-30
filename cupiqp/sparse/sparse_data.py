from typing import Any, Optional
import warp as wp

from ..data import Data
from ..utils import batch_broadcast_view
from .batched_csr import UniformBatchedCsrMatrix


# Type alias for the accepted matrix input forms.
# - For batched (B > 1): UniformBatchedCsrMatrix, 3-D torch.sparse_csr_tensor, or List[cupy csr_matrix].
# - For single (B = 1): cupy csr_matrix or 2-D torch.sparse_csr_tensor.
SparseMatrixInput = Any


class SparseData(Data):
    """Sparse data structure for batched QP problems.

    Matrices ``P``, ``A``, ``G`` are stored internally as
    :class:`UniformBatchedCsrMatrix` instances -- one shared ``indptr``/``indices``
    pair plus a packed ``(B, nnz)`` Warp values buffer. Callers may pass any of
    the following for each matrix:

    * :class:`UniformBatchedCsrMatrix` (already has a uniform structure)
    * ``torch.sparse_csr_tensor`` -- 3-D ``(B, M, N)`` for batched,
      2-D ``(M, N)`` for single
    * ``list[cupy csr_matrix]`` sharing the same sparsity pattern
    * a single cupy ``csr_matrix`` (B = 1)

    Whatever the input, the normalized storage is a ``UniformBatchedCsrMatrix``
    accessible via ``self.P`` / ``self.A`` / ``self.G``. Dense vectors
    (``c``, ``b``, ``h_l``, ``h_u``, ``x_l``, ``x_u``) are Warp arrays with a
    leading batch dimension ``(B, k)``.
    """

    def __init__(self, dtype="float64", device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)

    def init(
        self,
        P: SparseMatrixInput,
        c,
        A: Optional[SparseMatrixInput] = None,
        b=None,
        G: Optional[SparseMatrixInput] = None,
        h_u=None,
        h_l=None,
        x_u=None,
        x_l=None
    ):
        # -- P (determines B and n) -------------------------------------
        self._P = UniformBatchedCsrMatrix.from_input(
            P, dtype=self._dtype, device=self._device, validate_shared_sparsity=True
        )
        B = self._P.batch_size
        if self._P.rows != self._P.cols:
            raise ValueError("P must be square.")
        n = self._P.rows
        self._batch_size = B
        self._n = n

        # -- c -----------------------------------------------------------
        self._c = self._init_vec(c, n, "c", B)

        # -- A, b --------------------------------------------------------
        if (A is None) != (b is None):
            raise ValueError("A and b must either both be provided or both be None.")
        if A is not None and b is not None:
            self._A = UniformBatchedCsrMatrix.from_input(
                A, dtype=self._dtype, device=self._device, validate_shared_sparsity=True
            )
            if self._A.batch_size != B:
                raise ValueError(
                    f"A batch size ({self._A.batch_size}) != P batch size ({B})"
                )
            if self._A.cols != n:
                raise ValueError(
                    f"A.cols ({self._A.cols}) != n ({n})"
                )
            self._b = self._init_vec(b, self._A.rows, "b", B)
        else:
            self._A = UniformBatchedCsrMatrix.empty(B, 0, n, dtype=self._dtype, device=self._device)
            self._b = wp.zeros((B, 0), dtype=self._dtype, device=self._device)

        # -- G, h_u, h_l ------------------------------------------------
        if G is not None:
            if h_l is None and h_u is None:
                raise ValueError("Either h_l or h_u must be provided when G is given.")
            self._G = UniformBatchedCsrMatrix.from_input(
                G, dtype=self._dtype, device=self._device, validate_shared_sparsity=True
            )
            if self._G.batch_size != B:
                raise ValueError(
                    f"G batch size ({self._G.batch_size}) != P batch size ({B})"
                )
            if self._G.cols != n:
                raise ValueError(
                    f"G.cols ({self._G.cols}) != n ({n})"
                )
        else:
            if h_u is not None or h_l is not None:
                raise ValueError("h_l and h_u must be None when G is None.")
            self._G = UniformBatchedCsrMatrix.empty(B, 0, n, dtype=self._dtype, device=self._device)

        m = self._G.rows
        self._has_h_l = h_l is not None
        self._has_h_u = h_u is not None
        self._h_u = self._init_vec(h_u, m, "h_u", B) if h_u is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)
        self._h_l = self._init_vec(h_l, m, "h_l", B) if h_l is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)

        self._has_x_l = x_l is not None
        self._has_x_u = x_u is not None
        self._x_u = self._init_vec(x_u, n, "x_u", B) if x_u is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)
        self._x_l = self._init_vec(x_l, n, "x_l", B) if x_l is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)

        self._finalize()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def p(self) -> int:
        return self._A.rows

    @property
    def m(self) -> int:
        return self._G.rows

    # ------------------------------------------------------------------
    # In-place setters
    # ------------------------------------------------------------------

    def _set_matrix_values(
        self,
        target: UniformBatchedCsrMatrix,
        value: wp.array,
        name: str
    ):
        """Common helper for set_P / set_A / set_G.

        ``value`` holds only the nonzero values, laid out against the CSR
        pattern fixed at setup(): a Warp array of shape ``(B, nnz)``, or
        ``(nnz,)`` to broadcast one set of values over the whole batch. The
        pattern itself cannot change, so no structural comparison (and no
        device-to-host sync) is needed. The shape check only reads host-side
        metadata and is therefore always performed.
        """
        shape = tuple(value.shape)
        if shape == (target.nnz,):
            value = batch_broadcast_view(value, target.batch_size)
        elif shape != (target.batch_size, target.nnz):
            raise ValueError(
                f"{name} values shape mismatch: expected "
                f"({target.batch_size}, {target.nnz}) or ({target.nnz},), "
                f"got {shape}."
            )
        # In-place copy keeps the buffer address stable for cuDSS / cuSPARSE.
        if target.nnz > 0:
            wp.copy(target.data, value)

    def set_P(self, value, check: bool = True):
        self._set_matrix_values(self._P, value, "P")

    def set_c(self, value, check: bool = True):
        self._write(self._c, value, "c")

    def set_A(self, value, check: bool = True):
        self._set_matrix_values(self._A, value, "A")

    def set_b(self, value, check: bool = True):
        self._write(self._b, value, "b")

    def set_G(self, value, check: bool = True):
        self._set_matrix_values(self._G, value, "G")

    def set_h_l(self, value, check: bool = True):
        self._set_bound(self._h_l, self._has_h_l, value, "h_l", check)

    def set_h_u(self, value, check: bool = True):
        self._set_bound(self._h_u, self._has_h_u, value, "h_u", check)

    def set_x_l(self, value, check: bool = True):
        self._set_bound(self._x_l, self._has_x_l, value, "x_l", check)

    def set_x_u(self, value, check: bool = True):
        self._set_bound(self._x_u, self._has_x_u, value, "x_u", check)
