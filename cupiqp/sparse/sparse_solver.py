import cupy as cp
import warp as wp
from cupyx.scipy.sparse import csr_matrix

from ..results import Variables
from typing import Literal, Optional, Sequence, Union

from ..settings import Settings
from ..solver import SolverBase
from ..typedef import CudaArray
from ..utils import is_cuda_array
from .batched_csr import UniformBatchedCsrMatrix
from .sparse_data import SparseData
from .sparse_preconditioner import SparseRuizEquilibration
from .sparse_solver_kernels import create_sparse_data_gradients_kernel


# GPU CSR matrix inputs accepted by SparseSolver for P / A / G: a single GPU
# CSR matrix (a cupy csr_matrix or a CUDA torch sparse-CSR tensor), cuPIQP's
# own batched-CSR container, or a same-pattern sequence of cupy CSR matrices
CsrMatrixInput = Union[csr_matrix, "torch.Tensor", UniformBatchedCsrMatrix, Sequence[csr_matrix]]



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
        from .batched_csr import UniformBatchedCsrMatrix
        if isinstance(m, UniformBatchedCsrMatrix):
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
            f"CUDA, list of these, or UniformBatchedCsrMatrix); "
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
    r"""GPU solver for general sparse convex quadratic programs that
    solves a QP - or a whole batch of QPs - of the form

    $$
    \begin{aligned}
    \min_{x}\quad & \tfrac{1}{2}\, x^\top P x + c^\top x \\
    \text{s.t.}\quad & A x = b, \\
    & h_l \le G x \le h_u, \\
    & x_l \le x \le x_u,
    \end{aligned}
    $$

    using the proximal interior-point method with a sparse LDL^T
    factorization, running entirely on the GPU. It is built for large,
    structurally sparse ``P`` / ``A`` / ``G``.

    **Inputs - CSR matrices, dense vectors.** ``P``, ``A``, ``G`` must be
    **GPU sparse matrices in CSR layout**; the vectors (``c``, ``b``,
    ``h_l``, ``h_u``, ``x_l``, ``x_u``) are dense GPU arrays. Accepted
    matrix types are:

    * ``cupyx.scipy.sparse.csr_matrix`` (a GPU CSR matrix),
    * a CUDA ``torch.sparse_csr_tensor``, or
    * a [``UniformBatchedCsrMatrix``][cupiqp.UniformBatchedCsrMatrix] -
      cuPIQP's own container holding a batch of CSR matrices that share one
      sparsity pattern (the efficient way to pass a batch).

    A ``list`` / ``tuple`` of ``cupyx.scipy.sparse.csr_matrix`` (one per batch
    element, all sharing the same sparsity pattern) is also accepted, but
    **discouraged**: separate matrix objects cannot be organized with the
    uniform stride that batched linear-algebra routines require, so cuPIQP
    must copy them into a single contiguous
    [``UniformBatchedCsrMatrix``][cupiqp.UniformBatchedCsrMatrix] at
    ``setup``. Build and pass one of those yourself to avoid the copy.

    cuPIQP is **GPU-only and CSR-only**, and never converts formats behind
    your back:

    * CPU sparse matrices (``scipy.sparse.*``, a CPU
      ``torch.sparse_csr_tensor``) are rejected - lift them onto the GPU
      first, e.g. ``cupyx.scipy.sparse.csr_matrix(P_scipy)``.
    * Non-CSR GPU layouts (CSC, BSR, BSC, COO) are rejected - convert with
      ``.tocsr()`` before passing.

    **Batching.** Solve ``B`` independent QPs in a single GPU call by passing
    ``P``, ``A``, ``G`` as a
    [``UniformBatchedCsrMatrix``][cupiqp.UniformBatchedCsrMatrix] (the
    preferred, fastest batched input), with the vectors stacked along a
    leading batch dimension (``c`` of shape ``(B, n)``, etc.). A single
    problem is simply ``B = 1``, and ``solver.result.x`` then has shape
    ``(B, n)``. (A ``list`` of per-element CSR matrices also works but is
    slower - see above.)

    Parameters
    ----------
    dtype : {"float64", "float32"}, default: "float64"
        Floating-point precision used throughout the solve. ``"float32"``
        is faster and uses less memory but converges to looser tolerances;
        the default convergence tolerances are chosen to match the dtype.

    Examples
    --------
    A sparse problem assembled on the host with SciPy, then lifted to the
    GPU:

    ```python
    import scipy.sparse as sp
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    from cupiqp import SparseSolver

    P = csr_matrix(sp.eye(4, format="csr"))     # lift scipy -> GPU CSR
    c = cp.zeros(4)

    solver = SparseSolver()
    solver.setup(P=P, c=c)
    solver.solve()

    print(solver.result.info.status[0].name)    # CUPIQP_SOLVED
    ```

    See Also
    --------
    DenseSolver: solver for general dense problems.

    MultistageSolver: structure-exploiting solver for multistage optimization
        (e.g. optimal-control) problems.

    Notes
    -----
    The problem *structure* - array shapes, the **sparsity pattern** of
    each matrix, which constraint blocks are present, and which bounds are
    finite vs. ``+/-inf`` - is fixed by ``setup`` and can only be set up
    once per instance. To re-solve after changing only the *numerical
    values* (keeping the same sparsity pattern), call ``update`` and then
    ``solve`` again, which reuses all GPU allocations; for a different
    structure, create a new ``SparseSolver``. Solver behaviour (tolerances,
    verbosity, iteration cap, ...) is configured through ``solver.settings``.
    """

    def __init__(self, dtype: Literal["float32", "float64"] = "float64"):
        super().__init__(dtype=dtype)
        self._settings.kkt_solver = "sparse_ldlt"

    @SolverBase.settings.setter
    def settings(self, value: Settings) -> None:
        # TODO: here we have to set the kkt solver back. That's pretty ugly. Should be improved in the future
        value.kkt_solver = "sparse_ldlt"
        self._settings = value


    def _init_data(
        self,
        P: CsrMatrixInput,
        c: CudaArray,
        A: Optional[CsrMatrixInput],
        b: Optional[CudaArray],
        G: Optional[CsrMatrixInput],
        h_u: Optional[CudaArray],
        h_l: Optional[CudaArray],
        x_u: Optional[CudaArray],
        x_l: Optional[CudaArray],
    ) -> SparseData:
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
        data = SparseData(dtype=self.settings.dtype, device=self.settings.device)
        data.init(P, c, A, b, G, h_u, h_l, x_u, x_l)
        return data

    def _init_preconditioner(self) -> SparseRuizEquilibration:
        return SparseRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            self._data.idx_xl, self._data.idx_xu,
            self._data.idx_hl, self._data.idx_hu,
            use_warp_tile_kernels=True,
            dtype=self._data.dtype,
        )

    def setup(
        self,
        P: CsrMatrixInput,
        c: CudaArray,
        A: Optional[CsrMatrixInput] = None,
        b: Optional[CudaArray] = None,
        G: Optional[CsrMatrixInput] = None,
        h_u: Optional[CudaArray] = None,
        h_l: Optional[CudaArray] = None,
        x_u: Optional[CudaArray] = None,
        x_l: Optional[CudaArray] = None,
    ) -> None:
        super().setup(P, c, A, b, G, h_u, h_l, x_u, x_l)
        # Cache CSR row decompressions for the backward-pass gather.
        # Sparsity patterns are fixed at setup, so this is done once.
        if self.settings.enable_grad:
            d = self._data
            B = d.batch_size
            dtype = d.dtype

            # Row indices for each nnz position (CSR-to-COO). All row/col
            # index arrays are cast to int32 to match the warp kernel's
            # index type (consistent with idx_hu etc.).
            P_csr = d._P
            nnz_P = int(P_csr.nnz)
            self._p_rows = (cp.searchsorted(
                P_csr.indptr,
                cp.arange(nnz_P, dtype=P_csr.indptr.dtype),
                side="right",
            ) - 1).astype(cp.int32)
            self._p_indices_arr = P_csr.indices.astype(cp.int32)

            if d.p > 0:
                A_csr = d._A
                nnz_A = int(A_csr.nnz)
                self._a_rows = (cp.searchsorted(
                    A_csr.indptr,
                    cp.arange(nnz_A, dtype=A_csr.indptr.dtype),
                    side="right",
                ) - 1).astype(cp.int32)
                self._a_indices_arr = A_csr.indices.astype(cp.int32)
            else:
                nnz_A = 0
                self._a_rows = cp.empty(0, dtype=cp.int32)
                self._a_indices_arr = cp.empty(0, dtype=cp.int32)

            if d.m > 0:
                G_csr = d._G
                nnz_G = int(G_csr.nnz)
                self._g_rows = (cp.searchsorted(
                    G_csr.indptr,
                    cp.arange(nnz_G, dtype=G_csr.indptr.dtype),
                    side="right",
                ) - 1).astype(cp.int32)
                self._g_indices_arr = G_csr.indices.astype(cp.int32)
            else:
                nnz_G = 0
                self._g_rows = cp.empty(0, dtype=cp.int32)
                self._g_indices_arr = cp.empty(0, dtype=cp.int32)

            # Eager-compile the fused sparse data-gradients kernel.
            self._sparse_data_gradients_kernel = create_sparse_data_gradients_kernel(
                nnz_P, nnz_A, nnz_G, d.p, d.m, d.n, dtype=dtype)

            # Pre-allocate the gradient SparseData. The matrix
            # UniformBatchedCsrMatrix views share the forward sparsity (same
            # indices/indptr); their values buffers become the kernel-
            # output targets. Vector grads (c, h_l, x_l) are filled via
            # slice-assign in :meth:`_compute_data_gradients`.
            P_grad_csr = UniformBatchedCsrMatrix(
                B, P_csr.indices, P_csr.indptr, cp.zeros((B, nnz_P), dtype=dtype),
                shape=(P_csr.rows, P_csr.cols), dtype=dtype,
            )
            A_grad_csr = (UniformBatchedCsrMatrix(
                B, A_csr.indices, A_csr.indptr, cp.zeros((B, nnz_A), dtype=dtype),
                shape=(A_csr.rows, A_csr.cols), dtype=dtype,
            ) if d.p > 0 else None)
            G_grad_csr = (UniformBatchedCsrMatrix(
                B, G_csr.indices, G_csr.indptr, cp.zeros((B, nnz_G), dtype=dtype),
                shape=(G_csr.rows, G_csr.cols), dtype=dtype,
            ) if d.m > 0 else None)
            self._grad_data = SparseData(dtype=dtype, device=self.settings.device)
            self._grad_data.init(
                P=P_grad_csr,
                c=cp.zeros((B, d.n), dtype=dtype),
                A=A_grad_csr,
                b=cp.zeros((B, d.p), dtype=dtype) if d.p > 0 else None,
                G=G_grad_csr,
                h_u=cp.zeros((B, d.m), dtype=dtype) if d.num_hu > 0 else None,
                h_l=cp.zeros((B, d.m), dtype=dtype) if d.num_hl > 0 else None,
                x_u=cp.zeros((B, d.n), dtype=dtype) if d.num_xu > 0 else None,
                x_l=cp.zeros((B, d.n), dtype=dtype) if d.num_xl > 0 else None,
            )
            # Kernel value-buffer inputs. SparseData._A / _G are always
            # allocated (empty BatchedCsr placeholders when the
            # corresponding block is absent), so their ``.data`` is always
            # a (B, nnz_*) array matching the compiled kernel signature.
            self._grad_P_values = self._grad_data._P.data
            self._grad_A_values = self._grad_data._A.data
            self._grad_G_values = self._grad_data._G.data

    def _compute_data_gradients(self, adjoint_vector: Variables) -> SparseData:
        r"""Populate ``self._grad_data`` in place and return it.

        Matrix gradients are gathered directly at each structural nonzero
        (``O(B · nnz)``) rather than materialising the full outer product
        — written into ``self._grad_data._P/_A/_G.data``. Vector grads
        ``c``, ``h_l``, ``x_l`` are copies of ``adjoint_vector.x``,
        ``self._lam_zl_full``, ``self._lam_zbl_full``.

        Returns the same instance on every call; its buffers are
        overwritten by the next backward.
        """
        data = self._data
        grad_data = self._grad_data
        B = data.batch_size
        total = (
            grad_data._P.nnz + grad_data._A.nnz + grad_data._G.nnz
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
                    self._grad_P_values, self._grad_A_values, self._grad_G_values,
                    grad_data._b, grad_data._h_u, grad_data._x_u,
                ],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )

        grad_data._c[:] = adjoint_vector.x
        if data.num_hl > 0:
            grad_data._h_l[:] = self._lam_zl_full
        if data.num_xl > 0:
            grad_data._x_l[:] = self._lam_zbl_full

        return grad_data
