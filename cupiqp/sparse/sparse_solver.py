import cupy as cp
import numpy as np
import warp as wp
from cupyx.scipy.sparse import csr_matrix

from ..results import Variables
from typing import Literal, Optional, Union

from ..settings import Settings
from ..solver import SolverBase
from ..typedef import CudaArray
from ..utils import is_cuda_array
from .batched_csr import UniformBatchedCsrMatrix
from .sparse_data import SparseData
from .sparse_preconditioner import SparseRuizEquilibration
from .sparse_solver_kernels import create_sparse_data_gradients_kernel


# GPU CSR matrix accepted by SparseSolver.setup for P / A / G: a single 2-D GPU
# CSR matrix (a cupy csr_matrix or a 2-D CUDA torch sparse-CSR tensor).
CsrMatrixInput = Union[csr_matrix, "torch.Tensor"]


def _is_2d_gpu_csr(m) -> bool:
    """True iff ``m`` is a single 2-D **GPU CSR** matrix.

    Strict CSR contract: cupiqp's sparse LDL^T backend operates on CSR, so any
    other layout (CSC, BSR, COO, scipy.sparse, CPU torch sparse) or container
    (list, batched tensor) is rejected rather than silently converted. The
    ``torch`` import is lazy so users without torch pay no import cost.
    """
    if isinstance(m, csr_matrix):
        return True
    try:
        import torch
    except ImportError:
        return False
    return (isinstance(m, torch.Tensor) and m.layout == torch.sparse_csr
            and m.is_cuda and m.dim() == 2)


def _check_sparse(name: str, m) -> None:
    """Validate that ``m`` is a single 2-D GPU CSR matrix (skip if ``None``)."""
    if m is None:
        return
    if not _is_2d_gpu_csr(m):
        raise TypeError(
            f"SparseSolver requires {name} to be a single 2-D GPU CSR "
            f"matrix (cupyx.scipy.sparse.csr_matrix or a 2-D CUDA "
            f"torch.sparse_csr_tensor) describing one problem; got "
            f"{type(m).__name__}. The batch size is passed separately. "
            f"Convert scipy.sparse via cupyx.scipy.sparse.csr_matrix({name}) "
            f"and other sparse layouts with .tocsr() first."
        )


def _check_dense_vector(name: str, m) -> None:
    """Validate that ``m`` is a 1-D GPU dense array (skip if ``None``)."""
    if m is None:
        return
    if not is_cuda_array(m):
        raise TypeError(
            f"SparseSolver requires the vector {name} to be a GPU dense "
            f"array (any object exposing __cuda_array_interface__: "
            f"cupy.ndarray, dense CUDA torch.Tensor, JAX CUDA array, etc.); "
            f"got {type(m).__name__}."
        )
    ndim = cp.asarray(m).ndim
    if ndim != 1:
        raise ValueError(
            f"SparseSolver.setup requires {name} to be a 1-D vector describing "
            f"one problem; got a {ndim}-D array. The batch size is passed "
            f"separately; set per-problem values with update()."
        )


def _tile_csr(m, batch_size: int, dtype) -> UniformBatchedCsrMatrix:
    """Replicate one 2-D GPU CSR matrix into a batch of ``batch_size`` copies."""
    if isinstance(m, csr_matrix):
        return UniformBatchedCsrMatrix.from_cupy_csr_matrix(m, batch_size=batch_size, dtype=dtype)
    single = UniformBatchedCsrMatrix.from_torch_sparse_csr_tensor(m, dtype=dtype)
    return UniformBatchedCsrMatrix(
        batch_size, single.indices, single.indptr,
        cp.broadcast_to(single.data, (batch_size, single.nnz)),
        shape=(single.rows, single.cols), dtype=dtype,
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

    **Setup from one problem, then per-problem values.** ``setup`` takes the
    batch size and **one** template problem: ``P``, ``A``, ``G`` as single 2-D
    GPU CSR matrices (``cupyx.scipy.sparse.csr_matrix`` or a 2-D CUDA
    ``torch.sparse_csr_tensor``) and 1-D GPU vectors. The template fixes the
    structure shared by all ``B`` problems - sparsity patterns, which
    constraint blocks and bound sides exist - and its values are copied into
    every problem. ``update`` then sets per-problem values: the nonzero values
    of each matrix as a dense ``(B, nnz)`` array in the CSR order of the
    template, and the vectors as ``(B, k)`` arrays (or ``(nnz,)`` / ``(k,)`` to
    share one value across the batch). The solution ``solver.result.x`` has
    shape ``(B, n)``.

    cuPIQP is **GPU-only and CSR-only** and never converts formats behind
    your back: CPU inputs (``scipy.sparse``, ``numpy``, CPU torch) and
    non-CSR sparse layouts are rejected - lift / convert them first, e.g.
    ``cupyx.scipy.sparse.csr_matrix(P_scipy)`` or ``.tocsr()``.

    Parameters
    ----------
    dtype : {"float64", "float32"}, default: "float64"
        Floating-point precision used throughout the solve. ``"float32"``
        is faster and uses less memory but converges to looser tolerances;
        the default convergence tolerances are chosen to match the dtype.

    Examples
    --------
    ```python
    import scipy.sparse as sp
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    from cupiqp import SparseSolver

    B, n = 8, 4
    P = csr_matrix(sp.eye(n, format="csr"))     # lift scipy -> GPU CSR
    c = cp.zeros(n)

    solver = SparseSolver()
    solver.setup(B, P, c)                       # template problem, B copies
    solver.update(
        P=cp.random.uniform(1.0, 2.0, (B, P.nnz)),  # per-problem nonzeros
        c=cp.random.standard_normal((B, n)),
    )
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
    ``setup`` can be called only once per instance; for a different
    structure (patterns, blocks, bound sides), create a new
    ``SparseSolver``. ``update`` reuses all GPU allocations. Solver behaviour
    (tolerances, verbosity, iteration cap, ...) is configured through
    ``solver.settings``.
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
        # Replicate the single-problem template over the batch: matrices are
        # tiled here, 1-D vectors are broadcast by SparseData.init.
        B = self._setup_batch_size
        dtype = self.settings.dtype
        data = SparseData(dtype=dtype, device=self.settings.device)
        data.init(
            _tile_csr(P, B, dtype), c,
            None if A is None else _tile_csr(A, B, dtype), b,
            None if G is None else _tile_csr(G, B, dtype),
            h_u, h_l, x_u, x_l,
        )
        return data

    def _init_preconditioner(self) -> SparseRuizEquilibration:
        return SparseRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            has_h_l=self._data.has_h_l, has_h_u=self._data.has_h_u,
            has_x_l=self._data.has_x_l, has_x_u=self._data.has_x_u,
            active_x_bound=self._data.active_x_bound,
            use_warp_tile_kernels=(self._kernel_strategy == "warp_tile"),
            enable_cuda_graph=self.settings.enable_cuda_graph,
            dtype=self._data.dtype,
        )

    def setup(
        self,
        batch_size: int,
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
        """Fix the problem structure from one template problem and allocate
        all GPU memory for a batch of ``batch_size`` problems.

        You describe a **single** problem here. Its sparsity patterns, and
        which constraint blocks and bound sides are present, become the
        structure shared by all ``batch_size`` problems; its values are copied
        into every problem of the batch. Afterwards, give each problem its own
        values with :meth:`update`, then call :meth:`solve`. Calling
        :meth:`solve` directly after ``setup()`` solves ``batch_size`` copies of
        the template. Call ``setup()`` once per solver instance.

        Parameters
        ----------
        batch_size : int
            Number of QPs ``B`` solved together.
        P : GPU CSR matrix
            Quadratic cost of one problem, ``(n, n)``, as a
            ``cupyx.scipy.sparse.csr_matrix`` or a 2-D CUDA
            ``torch.sparse_csr_tensor``. Must be symmetric positive
            semidefinite; store the full matrix (both triangles). Required.
        c : GPU array
            Linear cost, 1-D of shape ``(n,)``. Required.
        A, b : GPU CSR matrix and GPU array, optional
            Equality constraints ``A x = b``: ``A`` of shape ``(p, n)``, ``b``
            1-D of shape ``(p,)``. Provide both or neither.
        G, h_l, h_u : GPU CSR matrix and GPU arrays, optional
            Inequalities ``h_l <= G x <= h_u``: ``G`` of shape ``(m, n)``,
            bounds 1-D of shape ``(m,)``. At least one bound is required when
            ``G`` is given; an omitted side is absent for the lifetime of the
            solver. Use ``-inf`` / ``+inf`` entries for one-sided rows.
        x_l, x_u : GPU array, optional
            Box bounds ``x_l <= x <= x_u``, 1-D of shape ``(n,)``. An omitted
            side is absent for the lifetime of the solver.

        Every matrix stored here fixes a sparsity pattern: an entry that may
        become nonzero in *any* problem of the batch must be stored
        (explicit zeros are fine). The number of stored entries, e.g.
        ``P.nnz``, is the length of the values passed to :meth:`update`.

        Raises
        ------
        RuntimeError
            If ``setup()`` has already been called on this instance.
        TypeError
            If a matrix is not a single 2-D GPU CSR matrix, or a vector is
            not a GPU array.
        ValueError
            If ``batch_size`` is not a positive integer or a vector is not 1-D.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer; got {batch_size!r}.")
        _check_sparse("P", P)
        _check_sparse("A", A)
        _check_sparse("G", G)
        for name, v in (("c", c), ("b", b), ("h_u", h_u), ("h_l", h_l),
                        ("x_u", x_u), ("x_l", x_l)):
            _check_dense_vector(name, v)
        self._setup_batch_size = int(batch_size)
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
                nnz_P, nnz_A, nnz_G, d.p, d.m, d.n, d.num_hu, d.num_xu, dtype=dtype)

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

    def update(
        self,
        P: Optional[CudaArray] = None,
        c: Optional[CudaArray] = None,
        A: Optional[CudaArray] = None,
        b: Optional[CudaArray] = None,
        G: Optional[CudaArray] = None,
        h_u: Optional[CudaArray] = None,
        h_l: Optional[CudaArray] = None,
        x_u: Optional[CudaArray] = None,
        x_l: Optional[CudaArray] = None,
        check_validity: bool = False,
    ) -> None:
        """Set new numerical data, per problem or shared, then ``solve()``.

        The sparsity patterns of ``P``, ``A`` and ``G`` are fixed by
        ``setup()``, so this method takes only their **nonzero values**, not
        sparse matrices. Any argument left as ``None`` keeps its current
        value.

        Parameters
        ----------
        P, A, G : GPU array, optional
            Nonzero values of the matrix, as a dense GPU array of shape
            ``(B, nnz)`` (one row per problem) or ``(nnz,)`` (the same values
            for every problem). Entry ``[k, j]`` is the ``j``-th stored entry
            of problem ``k``, in the CSR order of the matrix given to
            ``setup()`` - for example ``M.data`` of a ``csr_matrix`` ``M`` that
            has the setup pattern. ``solver.data.P.indices`` /
            ``solver.data.P.indptr`` (likewise for ``A`` / ``G``) expose that
            pattern.
        c, b, h_u, h_l, x_u, x_l : GPU array, optional
            New vectors, shape ``(B, k)`` (one row per problem) or ``(k,)``
            (shared). Bound values may switch between finite and ``+/-inf``,
            but only bound sides given at ``setup()`` can be updated.
        check_validity : bool, default: False
            If ``True``, also validate the vector shapes. The shapes of the
            matrix values are always checked (this costs no GPU sync).

        Examples
        --------
        ```python
        solver.setup(B, P, c)                       # one template problem
        solver.update(P=P_values, c=c_batch)        # (B, P.nnz), (B, n)
        solver.solve()
        ```
        """
        super().update(
            P=P, c=c, A=A, b=b, G=G, h_u=h_u, h_l=h_l, x_u=x_u, x_l=x_l,
            check_validity=check_validity,
        )

    def _compute_data_gradients(self, adjoint_vector: Variables, linearization_point: Variables) -> SparseData:
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
            + data.p + data.num_hu + data.num_xu
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
                    linearization_point.x, linearization_point.y,
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
