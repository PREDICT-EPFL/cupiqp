from ..settings import Settings
from ..solver import SolverBase
from ..utils import is_cuda_array
from .sparse_data import SparseData
from .sparse_preconditioner import SparseRuizEquilibration



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
