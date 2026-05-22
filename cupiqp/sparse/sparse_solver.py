import cupy as cp
import warp as wp

from ..results import Variables
from ..settings import Settings
from ..solver import SolverBase
from ..utils import is_cuda_array
from .batched_csr import BatchedCsrMatrix
from .sparse_data import SparseData
from .sparse_preconditioner import SparseRuizEquilibration
from .sparse_solver_kernels import create_sparse_data_gradients_kernel



def _is_sparse_csr(m) -> bool:
    """True iff ``m`` is a **GPU CSR** sparse matrix (or list of them).

    Strict CSR contract — cupiqp's sparse LDL^T backend operates on CSR,
    so any other layout (CSC, BSR, BSC, COO, scipy.sparse anything,
    CPU torch sparse) is rejected to avoid a silent format conversion at
    setup. Lazy imports of ``cupy`` and ``torch`` so users with only one
    framework installed don't pay an import-time cost.
    """
    try:
        from cupyx.scipy.sparse import csr_matrix as cp_csr
        if isinstance(m, cp_csr):
            return True
    except ImportError:
        pass

    try:
        import torch
        # Strict: only torch CSR layout on CUDA. CSC / BSR / BSC / COO
        # would require a silent format conversion to CSR at setup —
        # explicit user conversion is required instead.
        if (isinstance(m, torch.Tensor)
                and m.layout == torch.sparse_csr
                and m.is_cuda):
            return True
    except ImportError:
        pass

    try:
        from .batched_csr import BatchedCsrMatrix
        if isinstance(m, BatchedCsrMatrix):
            return True
    except ImportError:
        pass

    if isinstance(m, (list, tuple)) and len(m) > 0:
        return _is_sparse_csr(m[0])

    return False


def _check_sparse(name: str, m) -> None:
    """Validate that ``m`` is a GPU CSR sparse matrix (skip if ``None``)."""
    if m is None:
        return
    if not _is_sparse_csr(m):
        raise TypeError(
            f"SparseSolver requires {name} to be a GPU CSR sparse matrix "
            f"(cupyx.scipy.sparse.csr_matrix, torch.sparse_csr_tensor on "
            f"CUDA, list of these, or cupiqp.BatchedCsrMatrix); "
            f"got {type(m).__name__}. "
            f"cupiqp is GPU-only and CSR-only — convert scipy.sparse via "
            f"cupyx.scipy.sparse.csr_matrix({name}) first, and convert "
            f"non-CSR sparse layouts with .tocsr() before passing."
        )


def _check_dense_vector(name: str, m) -> None:
    """Validate that vector ``m`` (c, b, h_u, h_l, x_u, x_l) is a GPU
    dense array (skip if ``None``). cupiqp's vectors are always dense
    on the GPU regardless of the matrix backend, so this is identical
    to the dense matrix check — only the error message differs."""
    if m is None:
        return
    if not is_cuda_array(m):
        raise TypeError(
            f"SparseSolver requires the vector {name} to be a GPU dense "
            f"array (any object exposing __cuda_array_interface__: "
            f"cupy.ndarray, dense CUDA torch.Tensor, JAX CUDA array, etc.); "
            f"got {type(m).__name__}. "
        )



class SparseSolver(SolverBase):
    """Concrete :class:`SolverBase` subclass for the **sparse LDL^T**
    KKT backend.

    ``SparseSolver`` is the type-strict, user-facing entry point for
    solving QPs whose problem data are **GPU-resident sparse matrices
    in CSR layout**. Vectors (``c, b, h_*, x_*``) must be GPU dense.
    Anything else — non-CSR sparse layouts, CPU sparse, dense matrices,
    block-structured matrices — is rejected with a clear, actionable
    :class:`TypeError`.

    Accepts ``P, A, G`` as **GPU CSR** sparse:

    * :class:`cupyx.scipy.sparse.csr_matrix`
    * :class:`torch.sparse_csr_tensor` on a CUDA device
    * ``list`` / ``tuple`` of any of the above (one per batch element)
    * :class:`cupiqp.sparse.batched_csr.BatchedCsrMatrix`

    cupiqp is GPU-only **and** CSR-only:

    * CPU sparse formats (:class:`scipy.sparse.csr_matrix`,
      CPU :class:`torch.sparse_csr_tensor`) are **rejected**. Convert
      with :class:`cupyx.scipy.sparse.csr_matrix` first.
    * Non-CSR sparse layouts (CSC, BSR, BSC, COO) are **rejected**.
      The KKT backend operates on CSR; accepting another layout would
      require a silent format conversion at setup. Convert explicitly
      with :meth:`tocsr` first.

    Examples
    --------
    >>> import cupy as cp
    >>> import scipy.sparse as sp
    >>> import numpy as np
    >>> from cupyx.scipy.sparse import csr_matrix as cp_csr
    >>> from cupiqp import SparseSolver
    >>> P = cp_csr(sp.csr_matrix(np.eye(4)))   # lift scipy onto the GPU
    >>> c = cp.zeros(4)
    >>> s = SparseSolver()
    >>> s.setup(P=P, c=c)
    >>> s.solve()
    """

    def __init__(self):
        super().__init__()
        self._settings.kkt_solver = "sparse_ldlt"

    @SolverBase.settings.setter
    def settings(self, value: Settings) -> None:
        # TODO: here we have to set the kkt solver back. That's pretty ugly. Should be improved in the future
        value.kkt_solver = "sparse_ldlt"
        self._settings = value


    def _init_data(self, P, c, A, b, G, h_u, h_l, x_u, x_l):
        # Matrices: CSR-style sparse. Vectors: GPU dense (always).
        _check_sparse("P", P)
        _check_sparse("A", A)
        _check_sparse("G", G)
        _check_dense_vector("c", c)
        _check_dense_vector("b", b)
        _check_dense_vector("h_u", h_u)
        _check_dense_vector("h_l", h_l)
        _check_dense_vector("x_u", x_u)
        _check_dense_vector("x_l", x_l)
        return SparseData(P, c, A, b, G, h_u, h_l, x_u, x_l)

    def _init_preconditioner(self):
        return SparseRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            self._data.idx_xl, self._data.idx_xu,
            self._data.idx_hl, self._data.idx_hu,
            use_warp_tile_kernels=True,
        )

    def setup(self, P, c, A=None, b=None, G=None,
              h_u=None, h_l=None, x_u=None, x_l=None):
        super().setup(P, c, A, b, G, h_u, h_l, x_u, x_l)
        # Cache CSR row decompressions for the backward-pass gather.
        # Sparsity patterns are fixed at setup, so this is done once.
        if self.settings.enable_grad:
            d = self._data
            B = d.batch_size

            # Row indices for each nnz position (CSR-to-COO). All
            # row/col index arrays are cast to int32 to match the
            # warp kernel's index type (consistent with idx_hu etc.).
            P_csr = d._P
            nnz_P = int(P_csr.nnz)
            self._p_rows = (cp.searchsorted(
                P_csr.indptr,
                cp.arange(nnz_P, dtype=P_csr.indptr.dtype),
                side="right",
            ) - 1).astype(cp.int32)
            p_indices_for_kernel = P_csr.indices.astype(cp.int32)

            if d.p > 0:
                A_csr = d._A
                nnz_A = int(A_csr.nnz)
                self._a_rows = (cp.searchsorted(
                    A_csr.indptr,
                    cp.arange(nnz_A, dtype=A_csr.indptr.dtype),
                    side="right",
                ) - 1).astype(cp.int32)
                a_indices_for_kernel = A_csr.indices.astype(cp.int32)
            else:
                nnz_A = 0
                self._a_rows = cp.empty(0, dtype=cp.int32)
                a_indices_for_kernel = cp.empty(0, dtype=cp.int32)

            if d.m > 0:
                G_csr = d._G
                nnz_G = int(G_csr.nnz)
                self._g_rows = (cp.searchsorted(
                    G_csr.indptr,
                    cp.arange(nnz_G, dtype=G_csr.indptr.dtype),
                    side="right",
                ) - 1).astype(cp.int32)
                g_indices_for_kernel = G_csr.indices.astype(cp.int32)
            else:
                nnz_G = 0
                self._g_rows = cp.empty(0, dtype=cp.int32)
                g_indices_for_kernel = cp.empty(0, dtype=cp.int32)

            # Stash kernel-input arrays so _compute_data_gradients can
            # find them without recomputing per call.
            self._p_indices_arr = p_indices_for_kernel
            self._a_indices_arr = a_indices_for_kernel
            self._g_indices_arr = g_indices_for_kernel

            # Eager-compile the fused sparse data-gradients kernel.
            self._sparse_data_gradients_kernel = create_sparse_data_gradients_kernel(
                nnz_P, nnz_A, nnz_G, d.p, d.m, d.n,
            )

            # Pre-allocate output buffers. Shape-zero rows are valid
            # cupy arrays; the kernel's corresponding dispatch range
            # collapses to 0 threads.
            self._dP_values_buf = cp.empty((B, nnz_P), dtype=cp.float64)
            self._dA_values_buf = cp.empty((B, nnz_A), dtype=cp.float64)
            self._dG_values_buf = cp.empty((B, nnz_G), dtype=cp.float64)
            self._db_buf        = cp.empty((B, d.p),  dtype=cp.float64)
            self._dh_u_buf      = cp.empty((B, d.m),  dtype=cp.float64)
            self._dx_u_buf      = cp.empty((B, d.n),  dtype=cp.float64)

    def _compute_data_gradients(self, adjoint_vector: Variables) -> SparseData:
        r"""Build a :class:`SparseData` populated with user-space
        gradients on the same sparsity pattern as the original
        problem matrices, via a single fused warp launch.

        Matrix gradients are gathered directly at each structural
        nonzero (``O(B · nnz)``) rather than materialising the full
        outer product. ``dP, dA, dG, db, dh_u, dx_u`` are written to
        pre-allocated buffers (see :meth:`setup`); ``dc, dh_l, dx_l``
        are direct aliases of ``sol_adj.x``, ``self._lam_zl_full``,
        ``self._lam_zbl_full``. All inputs are user-space.
        """
        data = self._data
        B    = data.batch_size
        total = (
            self._dP_values_buf.shape[1]
            + self._dA_values_buf.shape[1]
            + self._dG_values_buf.shape[1]
            + data.p + data.m + data.n
        )
        if total > 0:
            wp.launch(
                kernel=self._sparse_data_gradients_kernel,
                dim=(B, total),
                inputs=[
                    adjoint_vector.x, adjoint_vector.y,
                    self._lam_zu_full, self._lam_zl_full,
                    self._lam_zbu_full,
                    self._zu_full, self._zl_full,
                    self._result.x, self._result.y,
                    self._p_rows, self._p_indices_arr,
                    self._a_rows, self._a_indices_arr,
                    self._g_rows, self._g_indices_arr,
                    self._dP_values_buf, self._dA_values_buf, self._dG_values_buf,
                    self._db_buf, self._dh_u_buf, self._dx_u_buf,
                ],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )

        # Wrap the matrix-grad value buffers in BatchedCsrMatrix on
        # the original sparsity patterns. Vector grads are aliases or
        # the negated buffers written by the kernel.
        P_csr = data._P
        dP = BatchedCsrMatrix(
            B, P_csr.indices, P_csr.indptr, self._dP_values_buf,
            shape=(P_csr.rows, P_csr.cols),
        )
        if data.p > 0:
            A_csr = data._A
            dA = BatchedCsrMatrix(
                B, A_csr.indices, A_csr.indptr, self._dA_values_buf,
                shape=(A_csr.rows, A_csr.cols),
            )
        else:
            dA = None
        if data.m > 0:
            G_csr = data._G
            dG = BatchedCsrMatrix(
                B, G_csr.indices, G_csr.indptr, self._dG_values_buf,
                shape=(G_csr.rows, G_csr.cols),
            )
        else:
            dG = None

        dc   = adjoint_vector.x
        db   = self._db_buf         if data.p      > 0 else None
        dh_u = self._dh_u_buf       if data.num_hu > 0 else None
        dh_l = self._lam_zl_full    if data.num_hl > 0 else None
        dx_u = self._dx_u_buf       if data.num_xu > 0 else None
        dx_l = self._lam_zbl_full   if data.num_xl > 0 else None

        return SparseData(
            P=dP, c=dc,
            A=dA, b=db,
            G=dG, h_u=dh_u, h_l=dh_l,
            x_u=dx_u, x_l=dx_l,
        )
