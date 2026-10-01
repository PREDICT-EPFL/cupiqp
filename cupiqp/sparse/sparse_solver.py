import cupy as cp
import numpy as np
import warp as wp

from ..results import Variables
from typing import Optional, Tuple, Union

from ..solver import SolverBase
from ..typedef import CudaArray
from ..utils import is_cuda_array, as_warp_array
from .sparse_data import SparseData, CsrTriple
from .sparse_preconditioner import SparseRuizEquilibration
from .sparse_solver_kernels import create_sparse_data_gradients_kernel


def _to_host(arr) -> np.ndarray:
    """NumPy copy of a host or device array."""
    return as_warp_array(arr).numpy() if is_cuda_array(arr) else np.ascontiguousarray(arr)


def _check_csr(name: str, value, dtype, device, cols: Optional[int] = None) -> Optional[Tuple[wp.array, wp.array, wp.array]]:
    """Device Warp arrays of a CSR triple ``(indptr, indices, values)`` (``None`` passes through).

    ``indptr`` and ``indices`` are 1-D integer arrays on the host or the device;
    they are checked on the host and copied to ``device`` once. ``values`` is a
    device array of the stored entries in the solver ``dtype``, ``(nnz,)``
    shared by every problem or ``(B, nnz)`` per problem. The matrix has
    ``len(indptr) - 1`` rows and ``cols`` columns (as many as rows when ``cols``
    is None).
    """
    if value is None:
        return None
    if not (isinstance(value, (tuple, list)) and len(value) == 3):
        raise TypeError(
            f"SparseSolver requires {name} as a CSR triple (indptr, indices, values); "
            f"got {type(value).__name__}. For a scipy or cupyx csr_matrix M pass "
            f"(M.indptr, M.indices, M.data) with M.data on the device; for a CUDA torch "
            f"CSR tensor T pass (T.crow_indices(), T.col_indices(), T.values())."
        )
    indptr, indices, values = value

    # The values are numerical data: they must already be on the device.
    if not is_cuda_array(values):
        raise TypeError(
            f"SparseSolver requires {name} values to be a GPU array; got "
            f"{type(values).__name__}. The pattern (indptr, indices) may be on the "
            f"host, but the values must be on the device."
        )
    values = as_warp_array(values, f"{name} values", dtype)

    # The pattern is checked on the host.
    indptr_host, indices_host = _to_host(indptr), _to_host(indices)
    for part, arr in (("indptr", indptr_host), ("indices", indices_host)):
        if arr.dtype not in (np.int32, np.int64):
            raise TypeError(f"{name} {part} must be int32 or int64; got {arr.dtype}.")
        if arr.ndim != 1:
            raise ValueError(f"{name} {part} must be 1-D; got shape {arr.shape}.")
    rows, nnz = len(indptr_host) - 1, len(indices_host)
    cols = rows if cols is None else cols
    if rows < 0:
        raise ValueError(f"{name} indptr must have at least one entry.")
    if nnz >= 2 ** 31:
        raise ValueError(f"{name} has {nnz} stored entries; at most 2**31 - 1 are supported.")
    if values.ndim not in (1, 2) or values.shape[-1] != nnz:
        raise ValueError(
            f"{name} values must have shape ({nnz},) or (B, {nnz}) to match indices; "
            f"got {values.shape}."
        )
    if indptr_host[0] != 0 or indptr_host[-1] != nnz or np.any(np.diff(indptr_host) < 0):
        raise ValueError(
            f"{name} indptr must start at 0, never decrease, and end at nnz = {nnz}."
        )
    if np.any(indices_host < 0) or np.any(indices_host >= cols):
        raise ValueError(f"{name} column indices must lie in [0, {cols}).")
    row = np.repeat(np.arange(rows), np.diff(indptr_host))     # row of every stored entry
    same_row = row[1:] == row[:-1]
    if np.any(np.diff(indices_host)[same_row] <= 0):
        raise ValueError(
            f"{name} column indices must be strictly increasing within each row "
            f"(sorted, no duplicates). Products and stacks of scipy sparse matrices "
            f"can leave them unsorted; call sort_indices() on the scipy matrix "
            f"before passing its pattern."
        )

    return wp.array(indptr_host, device=device), wp.array(indices_host, device=device), values


def _check_dense_vector(name: str, m, dtype) -> Optional[wp.array]:
    """Warp view of the GPU vector ``m`` in the solver ``dtype``, ``(k,)`` or
    ``(B, k)`` (``None`` passes through)."""
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
    if arr.ndim not in (1, 2):
        raise ValueError(
            f"SparseSolver.setup requires {name} to be a vector: 1-D for one "
            f"problem shared by the whole batch, or 2-D with a leading batch "
            f"axis; got a {arr.ndim}-D array."
        )
    return arr


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

    **Matrices are CSR triples.** ``P``, ``A`` and ``G`` are given to
    ``setup`` as ``(indptr, indices, values)``. The pattern (``indptr``,
    ``indices``) may be on the host or the device; it is shared by every problem
    of the batch and fixed by ``setup``. The ``values`` must be on the device;
    ``update`` then takes the ``values`` alone.

    **Batching.** Every value array of ``setup`` and ``update`` is either
    **batched**, with a leading batch axis (matrix values ``(B, nnz)``,
    vectors ``(B, k)``), one value per problem, or **shared** (``(nnz,)``,
    ``(k,)``), one value for every problem. ``setup`` reads ``B`` from the
    batched arrays.

    **Results are Warp arrays.** ``solver.result.x`` is a ``(B, n)``
    ``warp.array`` on the GPU; use ``.numpy()`` for a host copy, or view it
    zero-copy from another framework (``cupy.asarray(x)``,
    ``torch.from_dlpack(x)``, ``jax.dlpack.from_dlpack(x)``).

    cuPIQP is **GPU-only and CSR-only** and never converts formats behind
    your back: host arrays are rejected - move them to the GPU first, and
    convert other sparse layouts to CSR (e.g. with ``.tocsr()``).

    Parameters
    ----------
    dtype : {wp.float64, wp.float32}, default: wp.float64
        Floating-point precision used throughout the solve. ``wp.float32``
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

    solver = SparseSolver()
    solver.setup(
        P=(P.indptr, P.indices, cp.random.uniform(1.0, 2.0, (B, P.nnz))),  # per-problem values
        c=cp.zeros(n),                          # shared by all B problems
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

    def __init__(self, dtype: Union[type[wp.float32], type[wp.float64]] = wp.float64, stream=None):
        super().__init__(dtype=dtype, stream=stream)
        # Non-owning cupy view of the solver stream. cupy is used only at
        # setup, to build CSR patterns and index maps; running those ops on
        # the solver stream orders them with the Warp copies that consume
        # their results. The Warp stream owns the handle.
        self._cupy_stream = cp.cuda.ExternalStream(self._stream.cuda_stream)

    def _print_problem_size(self):
        d = self._data
        print("sparse backend:")
        print(f"batch size B = {d.batch_size}")
        print(f"variables n = {d.n}, nnz(P) = {d.P.nnz}")
        print(f"equality constraints p = {d.p}, nnz(A) = {d.A.nnz}")
        print(f"inequality constraints m = {d.m}, nnz(G) = {d.G.nnz}")

    def _init_data(
        self,
        P: CsrTriple,
        c: wp.array,
        A: Optional[CsrTriple],
        b: Optional[wp.array],
        G: Optional[CsrTriple],
        h_u: Optional[wp.array],
        h_l: Optional[wp.array],
        x_u: Optional[wp.array],
        x_l: Optional[wp.array]
    ) -> SparseData:
        # SparseData.init reads the batch size from the batched inputs and
        # copies shared inputs into every problem.
        data = SparseData(dtype=self._dtype, device=self._device)
        data.init(P, c, A, b, G, h_u, h_l, x_u, x_l)
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
        P: CsrTriple,
        c: CudaArray,
        A: Optional[CsrTriple] = None,
        b: Optional[CudaArray] = None,
        G: Optional[CsrTriple] = None,
        h_u: Optional[CudaArray] = None,
        h_l: Optional[CudaArray] = None,
        x_u: Optional[CudaArray] = None,
        x_l: Optional[CudaArray] = None,
    ) -> None:
        """Fix the problem structure, load the data of every problem of the
        batch, and allocate all GPU memory.

        Each matrix is a CSR triple ``(indptr, indices, values)``: the
        pattern, shared by every problem, and its stored values. Every value
        array - the matrix values and the vectors - is either **shared**, in
        the shape of one problem, or **batched**, with a leading batch axis.
        The batch size ``B`` is read from the batched arrays, which must
        agree; if every array is shared, ``B = 1``. The patterns and which
        constraint blocks and bound sides are present become the structure
        of the solver. Call :meth:`solve` right away, or change values with
        :meth:`update` first. Call ``setup()`` once per solver instance.

        Parameters
        ----------
        P : CSR triple
            Quadratic cost, ``(n, n)`` with ``n = len(indptr) - 1``; values
            ``(nnz,)`` or ``(B, nnz)``. Must be symmetric positive
            semidefinite; store the full matrix (both triangles). Required.
        c : GPU array
            Linear cost, ``(n,)`` or ``(B, n)``. Required.
        A, b : CSR triple and GPU array, optional
            Equality constraints ``A x = b``: ``A`` with ``p = len(indptr) - 1``
            rows and ``n`` columns, ``b`` of shape ``(p,)`` or ``(B, p)``.
            Provide both or neither.
        G, h_l, h_u : CSR triple and GPU arrays, optional
            Inequalities ``h_l <= G x <= h_u``: ``G`` with ``m`` rows and
            ``n`` columns, bounds ``(m,)`` or ``(B, m)``. At least one bound
            is required when ``G`` is given; an omitted side is absent for the
            lifetime of the solver. Use ``-inf`` / ``+inf`` entries for
            one-sided rows.
        x_l, x_u : GPU array, optional
            Box bounds ``x_l <= x <= x_u``, ``(n,)`` or ``(B, n)``. An omitted
            side is absent for the lifetime of the solver.

        ``indptr`` and ``indices`` are 1-D ``int32`` or ``int64`` arrays on the
        host (e.g. numpy) or the device; a host pattern is copied to the device
        once. The values must be device arrays of the solver dtype. Column
        indices must be sorted within each row, without duplicates. Every pattern fixes
        which entries are stored: an entry that may become nonzero in *any*
        problem must be stored (explicit zeros are fine). The number of stored
        entries ``nnz`` is the length of the values passed to :meth:`update`.
        For a scipy ``csr_matrix`` ``M`` pass ``(M.indptr, M.indices, values)``
        with ``values`` on the device; for a cupyx ``csr_matrix`` ``M``,
        ``(M.indptr, M.indices, M.data)``; for a CUDA torch CSR tensor ``T``,
        ``(T.crow_indices(), T.col_indices(), T.values())``. To set up ``B``
        equal problems, give at least one value array its batch axis, e.g.
        ``cupy.broadcast_to(c, (B, n))``.

        Raises
        ------
        RuntimeError
            If ``setup()`` has already been called on this instance.
        TypeError
            If a matrix is not a CSR triple, an index array is not integer,
            or a value array is not a GPU array of the solver dtype.
        ValueError
            If a pattern is invalid, a value array has the wrong shape, or
            the batched arrays disagree on the batch size.
        """
        if P is None:
            raise TypeError("SparseSolver.setup requires P.")
        P = _check_csr("P", P, self._dtype, self._device)
        n = int(P[0].shape[0]) - 1
        A = _check_csr("A", A, self._dtype, self._device, cols=n)
        G = _check_csr("G", G, self._dtype, self._device, cols=n)
        c, b, h_u, h_l, x_u, x_l = (
            _check_dense_vector(name, v, self._dtype)
            for name, v in (("c", c), ("b", b), ("h_u", h_u), ("h_l", h_l),
                            ("x_u", x_u), ("x_l", x_l))
        )
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
            return (M.indptr, M.indices, wp.zeros((B, M.nnz), dtype=d.dtype, device=d.device))

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
            of problem ``k``, in the order of the ``indices`` given to
            ``setup()``. ``solver.data.P.indices`` / ``solver.data.P.indptr``
            (likewise for ``A`` / ``G``) expose that pattern.
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
        solver.setup(P=(indptr, indices, P_values), c=c)   # pattern fixed here
        solver.solve()
        solver.update(P=P_values_new, c=c_new)            # (B, nnz), (B, n)
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
