import warp as wp

from ..results import Variables
from typing import Optional

from ..solver import SolverBase
from ..typedef import CudaArray
from ..utils import is_cuda_array, as_warp_array
from .dense_data import DenseData
from .dense_preconditioner import DenseRuizEquilibration
from .dense_solver_kernels import create_dense_data_gradients_kernel


def _check_dense(name: str, m, ndim: int, dtype) -> Optional[wp.array]:
    """Warp view of the GPU array ``m`` in the solver ``dtype`` (``None``
    passes through).

    Used for **all** P / c / A / b / G / h_* / x_* inputs of
    :meth:`DenseSolver.setup`. ``ndim`` is the single-problem rank (2 for a
    matrix, 1 for a vector); one extra leading batch axis is allowed. Every
    argument that's not ``None`` must be a GPU array (Warp, cupy, CUDA
    torch, JAX, ...) of the solver dtype; this is the one place that checks
    it.
    """
    if m is None:
        return None
    if not is_cuda_array(m):
        raise TypeError(
            f"DenseSolver requires {name} to be a GPU dense array "
            f"(a warp.array, cupy.ndarray, dense CUDA torch.Tensor, JAX CUDA "
            f"array, etc.); got {type(m).__name__}."
        )
    arr = as_warp_array(m, name, dtype=dtype)
    if arr.ndim not in (ndim, ndim + 1):
        kind = "matrix" if ndim == 2 else "vector"
        raise ValueError(
            f"DenseSolver.setup requires {name} to be a {kind}: {ndim}-D for one "
            f"problem shared by the whole batch, or {ndim + 1}-D with a leading "
            f"batch axis; got a {arr.ndim}-D array."
        )
    return arr


class DenseSolver(SolverBase):
    r"""GPU solver for general dense convex quadratic programs that
    solves a QP - or a whole batch of QPs - of the form

    $$
    \begin{aligned}
    \min_{x}\quad & \tfrac{1}{2}\, x^\top P x + c^\top x \\
    \text{s.t.}\quad & A x = b, \\
    & h_l \le G x \le h_u, \\
    & x_l \le x \le x_u,
    \end{aligned}
    $$

    using the proximal interior-point method, running entirely on the GPU.

    **Inputs.** ``P``, ``A``, ``G`` and every vector (``c``, ``b``, ``h_l``,
    ``h_u``, ``x_l``, ``x_u``) must be **dense arrays that live on the GPU**:
    a ``warp.array``, a ``cupy.ndarray``, a CUDA ``torch.Tensor``, a CUDA
    JAX array, or any other object exposing DLPack or the
    ``__cuda_array_interface__`` protocol. Values are copied into the
    solver's own Warp buffers and cast to the solver dtype.

    cuPIQP is **GPU-only**: CPU data (``numpy.ndarray``, CPU torch tensors,
    CPU JAX arrays) is rejected with a ``TypeError`` rather than copied to
    the device behind your back.

    **Batching.** ``DenseSolver`` solves ``B`` independent QPs in a single
    GPU call. Every argument of ``setup`` and ``update`` is either
    **batched**, with a leading batch axis (``P`` of shape ``(B, n, n)``,
    ``c`` of shape ``(B, n)``, ...), one value per problem, or **shared**,
    in the single-problem shape (``(n, n)``, ``(n,)``, ...), one value for
    every problem. ``setup`` reads ``B`` from the batched arguments and fixes
    the structure shared by all problems - the dimensions and which
    constraint blocks and bound sides exist.

    **Results are Warp arrays.** ``solver.result.x`` is a ``(B, n)``
    ``warp.array`` on the GPU; use ``.numpy()`` for a host copy, or view it
    zero-copy from another framework (``cupy.asarray(x)``,
    ``torch.from_dlpack(x)``, ``jax.dlpack.from_dlpack(x)``).

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
    A small inequality-constrained QP (one row is one-sided via ``-inf``):

    ```python
    import cupy as cp
    from cupiqp import DenseSolver, Status

    P = cp.eye(2)
    c = cp.array([-1.0, -4.0])
    G = cp.array([[1.0, 1.0]])      # constrain x1 + x2
    h_l = cp.array([-cp.inf])       # no lower bound on the row
    h_u = cp.array([1.0])           # x1 + x2 <= 1

    solver = DenseSolver()
    solver.setup(P=P, c=c, G=G, h_l=h_l, h_u=h_u)   # all shared: one problem
    solver.solve()

    print(Status(solver.result.info.status.numpy()[0]).name)    # CUPIQP_SOLVED
    x = solver.result.x.numpy()[0]              # bring the solution to the host
    ```

    A batch of 8 problems that differ only in ``c``:

    ```python
    solver = DenseSolver()
    solver.setup(P=P, c=cp.random.standard_normal((8, 2)),   # batched c: B = 8
                 G=G, h_l=h_l, h_u=h_u)                       # shared by all 8
    solver.solve()
    solver.update(c=cp.random.standard_normal((8, 2)))      # new values, same structure
    solver.solve()
    ```

    See Also
    --------
    SparseSolver: solver for general sparse problems.

    MultistageSolver: structure-exploiting solver for multistage optimization
        (e.g. optimal-control) problems.

    Notes
    -----
    ``setup`` can be called only once per instance; for a different
    structure (dimensions, constraint blocks, bound sides), create a new
    ``DenseSolver``. ``update`` reuses all GPU allocations; bound values
    may switch between finite and ``+/-inf``. Solver behaviour (tolerances,
    verbosity, iteration cap, ...) is configured through
    ``solver.settings``.
    """

    def _print_problem_size(self):
        d = self._data
        print("dense backend:")
        print(f"batch size B = {d.batch_size}")
        print(f"variables n = {d.n}")
        print(f"equality constraints p = {d.p}")
        print(f"inequality constraints m = {d.m}")

    def _init_data(
        self,
        P: wp.array,
        c: wp.array,
        A: Optional[wp.array],
        b: Optional[wp.array],
        G: Optional[wp.array],
        h_u: Optional[wp.array],
        h_l: Optional[wp.array],
        x_u: Optional[wp.array],
        x_l: Optional[wp.array]
    ) -> DenseData:
        # DenseData.init reads the batch size from the batched inputs and
        # copies shared inputs into every problem.
        data = DenseData(dtype=self.settings.dtype, device=self._device)
        data.init(P, c, A, b, G, h_u, h_l, x_u, x_l)
        return data

    def _init_preconditioner(self) -> DenseRuizEquilibration:
        return DenseRuizEquilibration(
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
        P: CudaArray,
        c: CudaArray,
        A: Optional[CudaArray] = None,
        b: Optional[CudaArray] = None,
        G: Optional[CudaArray] = None,
        h_u: Optional[CudaArray] = None,
        h_l: Optional[CudaArray] = None,
        x_u: Optional[CudaArray] = None,
        x_l: Optional[CudaArray] = None
    ) -> None:
        """Fix the problem structure, load the data of every problem of the
        batch, and allocate all GPU memory.

        Each argument is either **shared** - the shape of a single problem,
        e.g. ``P`` of shape ``(n, n)`` or ``c`` of shape ``(n,)``, used by
        every problem - or **batched**, with a leading batch axis, e.g. ``P``
        of shape ``(B, n, n)`` or ``c`` of shape ``(B, n)``. The batch size
        ``B`` is read from the batched arguments, which must agree; if every
        argument is shared, ``B = 1``. The dimensions and which constraint
        blocks and bound sides are present become the structure of the
        solver. Call :meth:`solve` right away, or change values with
        :meth:`update` first. Call ``setup()`` once per solver instance.

        Parameters
        ----------
        P : GPU array
            Quadratic cost, ``(n, n)`` or ``(B, n, n)``. Must be symmetric
            positive semidefinite. Required.
        c : GPU array
            Linear cost, ``(n,)`` or ``(B, n)``. Required.
        A, b : GPU array, optional
            Equality constraints ``A x = b``: ``A`` of shape ``(p, n)`` or
            ``(B, p, n)``, ``b`` of shape ``(p,)`` or ``(B, p)``. Provide both
            or neither.
        G, h_l, h_u : GPU array, optional
            Inequalities ``h_l <= G x <= h_u``: ``G`` of shape ``(m, n)`` or
            ``(B, m, n)``, bounds ``(m,)`` or ``(B, m)``. At least one bound
            is required when ``G`` is given; an omitted side is absent for the
            lifetime of the solver. Use ``-inf`` / ``+inf`` entries for
            one-sided rows.
        x_l, x_u : GPU array, optional
            Box bounds ``x_l <= x <= x_u``, ``(n,)`` or ``(B, n)``. An omitted
            side is absent for the lifetime of the solver.

        To set up ``B`` problems that are all equal for now, give at least one
        argument its batch axis, e.g. ``cupy.broadcast_to(c, (B, n))``.

        Raises
        ------
        RuntimeError
            If ``setup()`` has already been called on this instance.
        TypeError
            If an input is not a GPU dense array of the solver dtype.
        ValueError
            If an input has the wrong rank or shape, the batched inputs
            disagree on the batch size, or only one of ``A`` / ``b`` is given.
        """
        # Every non-None input must be a GPU dense array. cupiqp does not
        # silently do H2D copies.
        P, c, A, b, G, h_u, h_l, x_u, x_l = (
            _check_dense(name, v, ndim, self._dtype)
            for name, v, ndim in (("P", P, 2), ("c", c, 1), ("A", A, 2), ("b", b, 1),
                                  ("G", G, 2), ("h_u", h_u, 1), ("h_l", h_l, 1),
                                  ("x_u", x_u, 1), ("x_l", x_l, 1))
        )
        if (A is None) != (b is None):
            raise ValueError("A and b must either both be provided or both be None.")
        with wp.ScopedStream(self._stream, sync_enter=not self._stream.is_capturing):
            self._setup_impl(P, c, A, b, G, h_u, h_l, x_u, x_l)
            if self.settings.enable_grad:
                self._init_grad_data()

    def _init_grad_data(self) -> None:
        """Allocate the zero-initialized gradient storage (forward structure)."""
        d = self._data
        B = d.batch_size
        self._dense_data_gradients_kernel = create_dense_data_gradients_kernel(
            d.n, d.p, d.m, d.num_hu, d.num_xu, dtype=d.dtype)
        # Zero-initialized (B, ...) gradient storage with the forward structure.
        self._grad_data = DenseData(dtype=d.dtype, device=d.device)
        self._grad_data.init(
            P=wp.zeros((B, d.n, d.n), dtype=d.dtype, device=d.device),
            c=wp.zeros((B, d.n), dtype=d.dtype, device=d.device),
            A=wp.zeros((B, d.p, d.n), dtype=d.dtype, device=d.device) if d.p > 0 else None,
            b=wp.zeros((B, d.p), dtype=d.dtype, device=d.device) if d.p > 0 else None,
            G=wp.zeros((B, d.m, d.n), dtype=d.dtype, device=d.device) if d.m > 0 else None,
            h_u=wp.zeros((B, d.m), dtype=d.dtype, device=d.device) if d.num_hu > 0 else None,
            h_l=wp.zeros((B, d.m), dtype=d.dtype, device=d.device) if d.num_hl > 0 else None,
            x_u=wp.zeros((B, d.n), dtype=d.dtype, device=d.device) if d.num_xu > 0 else None,
            x_l=wp.zeros((B, d.n), dtype=d.dtype, device=d.device) if d.num_xl > 0 else None,
        )

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
        check_validity: bool = False
    ) -> None:
        """Set new numerical data, per problem or shared, then ``solve()``.

        Any argument left as ``None`` keeps its current value. Each argument
        is either **batched** - a leading batch axis, e.g. ``P`` of shape
        ``(B, n, n)`` or ``c`` of shape ``(B, n)``, one entry per problem - or
        **unbatched** - the single-problem shape, e.g. ``(n, n)`` or ``(n,)``,
        in which case the same value is used for every problem. Bound values
        may switch between finite and ``+/-inf``, but only bound sides given
        at ``setup()`` can be updated.

        Parameters
        ----------
        P, c, A, b, G, h_u, h_l, x_u, x_l : GPU array, optional
            New values for the corresponding problem block.
        check_validity : bool, default: False
            Unused by this backend: shapes are always checked (metadata only,
            no GPU sync).
        """
        P, c, A, b, G, h_u, h_l, x_u, x_l = (
            as_warp_array(v, name, self._dtype)
            for name, v in (("P", P), ("c", c), ("A", A), ("b", b), ("G", G),
                            ("h_u", h_u), ("h_l", h_l), ("x_u", x_u), ("x_l", x_l))
        )
        with wp.ScopedStream(self._stream, sync_enter=not self._stream.is_capturing):
            self._update_impl(P, c, A, b, G, h_u, h_l, x_u, x_l, check_validity)

    def _compute_data_gradients(self, adjoint_vector: Variables, linearization_point: Variables) -> DenseData:
        """Populate ``self._grad_data`` in place and return it.

        The returned instance is the same on every call; its buffers are
        overwritten by the next backward. Copy fields if you need to keep
        them across calls.
        """
        data = self._data
        grad_data = self._grad_data
        B = data.batch_size
        total = (data.n * data.n + data.p * data.n + data.m * data.n
                 + data.p + data.num_hu + data.num_xu)
        if total > 0:
            wp.launch(
                kernel=self._dense_data_gradients_kernel,
                dim=(B, total),
                inputs=[
                    adjoint_vector.x, adjoint_vector.y,
                    self._lam_zu_full, self._lam_zl_full,
                    self._lam_zbu_full,
                    self._zu_full, self._zl_full,
                    linearization_point.x, linearization_point.y,
                    grad_data.P, grad_data.A, grad_data.G,
                    grad_data.b, grad_data.h_u, grad_data.x_u,
                ],
                device=self._device
            )

        # Vector grads that are aliases of solver-internal buffers -- copy in.
        wp.copy(grad_data.c, adjoint_vector.x)
        if data.num_hl > 0:
            wp.copy(grad_data.h_l, self._lam_zl_full)
        if data.num_xl > 0:
            wp.copy(grad_data.x_l, self._lam_zbl_full)

        return grad_data
