import cupy as cp
import numpy as np
import warp as wp
from cupyx.scipy.sparse import csr_matrix

from ..results import Variables
from typing import Literal, Optional, Union

from ..settings import Settings
from ..solver import SolverBase
from ..typedef import CudaArray
from ..utils import is_cuda_array, as_warp_array
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


def _check_dense_vector(name: str, m, dtype) -> Optional[wp.array]:
    """Warp view of the 1-D GPU array ``m`` in the solver ``dtype`` (``None`` passes through)."""
    if m is None:
        return None
    if not is_cuda_array(m):
        raise TypeError(
            f"SparseSolver requires the vector {name} to be a GPU dense "
            f"array (any object exposing __cuda_array_interface__: "
            f"cupy.ndarray, dense CUDA torch.Tensor, JAX CUDA array, etc.); "
            f"got {type(m).__name__}."
        )
    arr = as_warp_array(m, name, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(
            f"SparseSolver.setup requires {name} to be a 1-D vector describing "
            f"one problem; got a {arr.ndim}-D array. The batch size is passed "
            f"separately; set per-problem values with update()."
        )
    return arr


def _tile_csr(m, batch_size: int, dtype, device: str) -> UniformBatchedCsrMatrix:
    """Replicate one 2-D GPU CSR matrix into a batch of ``batch_size`` copies."""
    if isinstance(m, csr_matrix):
        return UniformBatchedCsrMatrix.from_cupy_csr_matrix(m, batch_size=batch_size, dtype=dtype, device=device)
    return UniformBatchedCsrMatrix.from_torch_sparse_csr_tensor(m, batch_size=batch_size, dtype=dtype, device=device)



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
    share one value across the batch).

    **Results are Warp arrays.** ``solver.result.x`` is a ``(B, n)``
    ``warp.array`` on the GPU; use ``.numpy()`` for a host copy, or view it
    zero-copy from another framework (``cupy.asarray(x)``,
    ``torch.from_dlpack(x)``, ``jax.dlpack.from_dlpack(x)``).

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
    stream : warp.Stream, optional
        CUDA stream to run on. By default the solver creates and owns one;
        pass a ``warp.Stream`` to run on it instead (see the ``stream``
        property).

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

    def __init__(self, dtype: Literal["float32", "float64"] = "float64", stream=None):
        super().__init__(dtype=dtype, stream=stream)
        self._settings.kkt_solver = "sparse_ldlt"
        # Non-owning cupy view of the solver stream. cupy is used only at
        # setup, to build CSR patterns and index maps; running those ops on
        # the solver stream orders them with the Warp copies that consume
        # their results. The Warp stream owns the handle.
        self._cupy_stream = cp.cuda.ExternalStream(self._stream.cuda_stream)

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
        dtype, device = self._dtype, self.settings.device
        data = SparseData(dtype=dtype, device=device)
        data.init(
            _tile_csr(P, B, dtype, device), c,
            None if A is None else _tile_csr(A, B, dtype, device), b,
            None if G is None else _tile_csr(G, B, dtype, device),
            h_u, h_l, x_u, x_l,
        )
        return data

    def _init_preconditioner(self) -> SparseRuizEquilibration:
        return SparseRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            has_h_l=self._data.has_h_l, has_h_u=self._data.has_h_u,
            has_x_l=self._data.has_x_l, has_x_u=self._data.has_x_u,
            active_x_bound=self._data.active_x_bound,
            enable_cuda_graph=self.settings.enable_cuda_graph,
            dtype=self._data.dtype,
            device=self._data.device
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
        c, b, h_u, h_l, x_u, x_l = (
            _check_dense_vector(name, v, self._dtype)
            for name, v in (("c", c), ("b", b), ("h_u", h_u), ("h_l", h_l),
                            ("x_u", x_u), ("x_l", x_l))
        )
        self._setup_batch_size = int(batch_size)
        with wp.ScopedStream(self._stream), self._cupy_stream:
            self._setup_impl(P, c, A, b, G, h_u, h_l, x_u, x_l)
            if self.settings.enable_grad:
                self._init_grad_data()

    def _init_grad_data(self) -> None:
        """Allocate the gradient storage and cache the CSR row decompressions
        used by the backward-pass gather (sparsity patterns are fixed at setup)."""
        d = self._data
        B = d.batch_size
        # COO row indices and column indices of every stored entry, used by
        # the backward-pass gather (the patterns are fixed at setup).
        P_csr, A_csr, G_csr = d.P, d.A, d.G
        nnz_P, nnz_A, nnz_G = P_csr.nnz, A_csr.nnz, G_csr.nnz
        self._p_rows, self._p_indices_arr = P_csr.row_indices, P_csr.indices
        self._a_rows, self._a_indices_arr = A_csr.row_indices, A_csr.indices
        self._g_rows, self._g_indices_arr = G_csr.row_indices, G_csr.indices

        # Eager-compile the fused sparse data-gradients kernel.
        self._sparse_data_gradients_kernel = create_sparse_data_gradients_kernel(
            nnz_P, nnz_A, nnz_G, d.p, d.m, d.n, d.num_hu, d.num_xu, dtype=d.dtype)

        # Pre-allocate the gradient SparseData: the matrices share the forward
        # sparsity and their (B, nnz) value buffers are the kernel outputs.
        # Vector grads (c, h_l, x_l) are copied in by _compute_data_gradients.
        def grad_csr(M):
            return UniformBatchedCsrMatrix(
                B, M.indptr, M.indices, wp.zeros((B, M.nnz), dtype=d.dtype, device=d.device),
                shape=(M.rows, M.cols), dtype=d.dtype, device=d.device)

        self._grad_data = SparseData(dtype=d.dtype, device=d.device)
        self._grad_data.init(
            P=grad_csr(P_csr),
            c=wp.zeros((B, d.n), dtype=d.dtype, device=d.device),
            A=grad_csr(A_csr) if d.p > 0 else None,
            b=wp.zeros((B, d.p), dtype=d.dtype, device=d.device) if d.p > 0 else None,
            G=grad_csr(G_csr) if d.m > 0 else None,
            h_u=wp.zeros((B, d.m), dtype=d.dtype, device=d.device) if d.num_hu > 0 else None,
            h_l=wp.zeros((B, d.m), dtype=d.dtype, device=d.device) if d.num_hl > 0 else None,
            x_u=wp.zeros((B, d.n), dtype=d.dtype, device=d.device) if d.num_xu > 0 else None,
            x_l=wp.zeros((B, d.n), dtype=d.dtype, device=d.device) if d.num_xl > 0 else None,
        )
        # SparseData always allocates A / G (empty placeholders when the block
        # is absent), so their values are (B, nnz_*) arrays matching the kernel.
        self._grad_P_values = self._grad_data.P.data
        self._grad_A_values = self._grad_data.A.data
        self._grad_G_values = self._grad_data.G.data

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
        for name, v in (("P", P), ("A", A), ("G", G)):
            if v is not None and not is_cuda_array(v):
                raise TypeError(
                    f"{name} must be a dense GPU array holding the nonzero values, "
                    f"of shape (B, nnz) or (nnz,), in the CSR order of the matrix "
                    f"passed to setup(); got {type(v).__name__}. The sparsity "
                    "pattern is fixed at setup() and cannot be updated."
                )
        P, c, A, b, G, h_u, h_l, x_u, x_l = (
            as_warp_array(v, name, self._dtype)
            for name, v in (("P", P), ("c", c), ("A", A), ("b", b), ("G", G),
                            ("h_u", h_u), ("h_l", h_l), ("x_u", x_u), ("x_l", x_l))
        )
        with wp.ScopedStream(self._stream):
            self._update_impl(P, c, A, b, G, h_u, h_l, x_u, x_l, check_validity)

    def _compute_data_gradients(self, adjoint_vector: Variables, linearization_point: Variables) -> SparseData:
        r"""Populate ``self._grad_data`` in place and return it.

        Matrix gradients are gathered directly at each structural nonzero
        (``O(B * nnz)``) rather than materialising the full outer product,
        written into ``self._grad_data.P/A/G.data``. Vector grads
        ``c``, ``h_l``, ``x_l`` are copies of ``adjoint_vector.x``,
        ``self._lam_zl_full``, ``self._lam_zbl_full``.

        Returns the same instance on every call; its buffers are
        overwritten by the next backward.
        """
        data = self._data
        grad_data = self._grad_data
        B = data.batch_size
        total = (
            grad_data.P.nnz + grad_data.A.nnz + grad_data.G.nnz
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
                    grad_data.b, grad_data.h_u, grad_data.x_u,
                ],
                device=self._device
            )

        wp.copy(grad_data.c, adjoint_vector.x)
        if data.num_hl > 0:
            wp.copy(grad_data.h_l, self._lam_zl_full)
        if data.num_xl > 0:
            wp.copy(grad_data.x_l, self._lam_zbl_full)

        return grad_data
