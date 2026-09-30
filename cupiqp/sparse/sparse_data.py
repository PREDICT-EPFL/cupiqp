from typing import Optional, Tuple
import warp as wp

from ..data import Data
from ..typedef import CudaArray
from ..utils import batch_broadcast_view
from .batched_csr import UniformBatchedCsrMatrix


# A sparse matrix as a CSR triple (indptr, indices, values) of GPU arrays,
# values (nnz,) shared by every problem or (B, nnz). SparseSolver.setup takes
# any GPU arrays; SparseData.init receives Warp arrays.
CsrTriple = Tuple[CudaArray, CudaArray, CudaArray]


class SparseData(Data):
    """Sparse data for a batch of QPs sharing one sparsity pattern per matrix.

    Matrices ``P``, ``A``, ``G`` are stored as :class:`UniformBatchedCsrMatrix`
    instances - one shared ``indptr`` / ``indices`` pair plus a packed
    ``(B, nnz)`` Warp values buffer - accessible via ``self.P`` / ``self.A`` /
    ``self.G``. Dense vectors (``c``, ``b``, ``h_l``, ``h_u``, ``x_l``,
    ``x_u``) are Warp arrays with a leading batch dimension ``(B, k)``.
    """

    def __init__(self, dtype="float64", device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)

    def init(
        self,
        P: CsrTriple,
        c: wp.array,
        A: Optional[CsrTriple] = None,
        b: Optional[wp.array] = None,
        G: Optional[CsrTriple] = None,
        h_u: Optional[wp.array] = None,
        h_l: Optional[wp.array] = None,
        x_u: Optional[wp.array] = None,
        x_l: Optional[wp.array] = None
    ):
        """Allocate and populate the batched buffers.

        Matrices are CSR triples ``(indptr, indices, values)`` of Warp arrays
        with a valid pattern (``SparseSolver`` checks it at its public
        boundary): ``P`` is square with ``len(indptr) - 1`` rows, ``A`` and
        ``G`` have as many columns as ``P``. Every value array and vector is
        either **shared** (``(nnz,)``, ``(k,)``, copied into every problem) or
        **batched** (``(B, nnz)``, ``(B, k)``). ``B`` is read from the batched
        arrays, which must agree; with none it is ``1``.
        """
        if (A is None) != (b is None):
            raise ValueError("A and b must either both be provided or both be None.")
        if G is not None and h_l is None and h_u is None:
            raise ValueError("Either h_l or h_u must be provided when G is given.")
        if G is None and (h_u is not None or h_l is not None):
            raise ValueError("h_l and h_u must be None when G is None.")

        batch_dims = {}
        for name, M in (("P", P), ("A", A), ("G", G)):
            if M is not None:
                batch_dims[name] = int(M[2].shape[0]) if M[2].ndim == 2 else None
        for name, v in (("c", c), ("b", b), ("h_u", h_u), ("h_l", h_l), ("x_u", x_u), ("x_l", x_l)):
            if v is not None:
                batch_dims[name] = int(v.shape[0]) if v.ndim == 2 else None
        B = self._resolve_batch_size(batch_dims)
        n = int(P[0].shape[0]) - 1
        self._batch_size = B
        self._n = n

        def csr(M: Optional[CsrTriple]) -> UniformBatchedCsrMatrix:
            if M is None:
                return UniformBatchedCsrMatrix.empty(B, 0, n, dtype=self._dtype, device=self._device)
            indptr, indices, values = M
            return UniformBatchedCsrMatrix(
                B, indptr, indices, values, shape=(int(indptr.shape[0]) - 1, n),
                dtype=self._dtype, device=self._device
            )

        self._P = csr(P)
        self._A = csr(A)
        self._G = csr(G)
        p, m = self._A.rows, self._G.rows

        self._c = self._init_vec(c, n, "c", B)
        self._b = self._init_vec(b, p, "b", B) if b is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)

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
