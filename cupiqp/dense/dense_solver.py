import cupy as cp
import warp as wp

from ..results import Variables
from typing import Literal, Optional

from ..settings import Settings
from ..solver import SolverBase
from ..typedef import CudaArray
from ..utils import is_cuda_array
from .dense_data import DenseData
from .dense_preconditioner import DenseRuizEquilibration
from .dense_solver_kernels import create_dense_data_gradients_kernel


def _check_dense(name: str, m) -> None:
    """Validate that ``m`` is a GPU dense array (skip if ``None``).

    Used for **all** P / c / A / b / G / h_* / x_* inputs in
    :class:`DenseSolver` — every ``setup`` argument that's not ``None``
    must be a GPU array exposing the CUDA Array Interface protocol.
    See :func:`cupiqp.utils.is_cuda_array`.
    """
    if m is None:
        return
    if not is_cuda_array(m):
        raise TypeError(
            f"DenseSolver requires {name} to be a GPU dense array "
            f"(any object exposing __cuda_array_interface__: "
            f"cupy.ndarray, dense CUDA torch.Tensor, JAX CUDA array, etc.); "
            f"got {type(m).__name__}."
        )


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
    ``h_u``, ``x_l``, ``x_u``) must be **dense arrays that live on the GPU**.
    Any object exposing the ``__cuda_array_interface__`` protocol is
    accepted - a ``cupy.ndarray``, a CUDA ``torch.Tensor``, a CUDA JAX
    array, a Numba device array, and so on.

    cuPIQP is **GPU-only**: CPU data (``numpy.ndarray``, CPU torch tensors,
    CPU JAX arrays) is rejected with a ``TypeError`` rather than copied to
    the device behind your back.

    **Batching.** ``DenseSolver`` is natively batched: solve ``B``
    independent QPs in a single GPU call by giving every array a leading
    batch dimension - ``P`` of shape ``(B, n, n)``, ``c`` of shape
    ``(B, n)``, and so on. A single problem is simply ``B = 1``, and
    ``solver.result.x`` then has shape ``(B, n)``.

    Parameters
    ----------
    dtype : {"float64", "float32"}, default: "float64"
        Floating-point precision used throughout the solve. ``"float32"``
        is faster and uses less memory but converges to looser tolerances;
        the default convergence tolerances are chosen to match the dtype.

    Examples
    --------
    A small inequality-constrained QP (one row is one-sided via ``-inf``):

    ```python
    import cupy as cp
    from cupiqp import DenseSolver

    P = cp.eye(2)
    c = cp.array([-1.0, -4.0])
    G = cp.array([[1.0, 1.0]])      # constrain x1 + x2
    h_l = cp.array([-cp.inf])       # no lower bound on the row
    h_u = cp.array([1.0])           # x1 + x2 <= 1

    solver = DenseSolver()
    solver.setup(P=P, c=c, G=G, h_l=h_l, h_u=h_u)
    solver.solve()

    print(solver.result.info.status[0].name)    # CUPIQP_SOLVED
    x = solver.result.x.get()[0]                 # bring the solution to the host
    ```

    See Also
    --------
    SparseSolver: solver for general sparse problems.

    MultistageSolver: structure-exploiting solver for multistage optimization
        (e.g. optimal-control) problems.

    Notes
    -----
    The problem *structure* - array shapes, which constraint blocks are
    present, and which bounds are finite vs. ``+/-inf`` - is fixed by
    ``setup`` and can only be set up once per instance. To re-solve with new
    *numerical* values of the same structure (e.g. a moving target ``b`` in
    receding-horizon control), call ``update`` and then ``solve`` again,
    which reuses all GPU allocations; for a different structure, create a
    new ``DenseSolver``. Solver behaviour (tolerances, verbosity, iteration
    cap, ...) is configured through ``solver.settings``.
    """

    def __init__(self, dtype: Literal["float32", "float64"] = "float64"):
        super().__init__(dtype=dtype)
        self._settings.kkt_solver = "dense_cholesky"

    @SolverBase.settings.setter
    def settings(self, value: Settings) -> None:
        # TODO: here we have to set the kkt solver back. That's pretty ugly. Should be improved in the future
        value.kkt_solver = "dense_cholesky"
        self._settings = value


    def _init_data(
        self,
        P: CudaArray,
        c: CudaArray,
        A: Optional[CudaArray],
        b: Optional[CudaArray],
        G: Optional[CudaArray],
        h_u: Optional[CudaArray],
        h_l: Optional[CudaArray],
        x_u: Optional[CudaArray],
        x_l: Optional[CudaArray],
    ) -> DenseData:
        # Every non-None input must be a GPU dense array — matrices and
        # vectors alike. cupiqp does not silently do H2D copies.
        _check_dense("P", P)
        _check_dense("c", c)
        _check_dense("A", A)
        _check_dense("b", b)
        _check_dense("G", G)
        _check_dense("h_u", h_u)
        _check_dense("h_l", h_l)
        _check_dense("x_u", x_u)
        _check_dense("x_l", x_l)
        data = DenseData(dtype=self.settings.dtype, device=self.settings.device)
        data.init(P, c, A, b, G, h_u, h_l, x_u, x_l)
        return data

    def _init_preconditioner(self) -> DenseRuizEquilibration:
        return DenseRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            has_h_l=self._data.has_h_l, has_h_u=self._data.has_h_u,
            has_x_l=self._data.has_x_l, has_x_u=self._data.has_x_u,
            active_x_bound=self._data.active_x_bound,
            use_warp_tile_kernels=(self._kernel_strategy == "warp_tile"),
            dtype=self._data.dtype,
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
        x_l: Optional[CudaArray] = None,
    ) -> None:
        super().setup(P, c, A, b, G, h_u, h_l, x_u, x_l)
        if self.settings.enable_grad:
            d = self._data
            B = d.batch_size
            dtype = d.dtype
            self._dense_data_gradients_kernel = create_dense_data_gradients_kernel(
                d.n, d.p, d.m, d.num_hu, d.num_xu, dtype=dtype)
            self._grad_data = DenseData(dtype=dtype, device=self.settings.device)
            self._grad_data.init(
                P=cp.zeros((B, d.n, d.n), dtype=dtype),
                c=cp.zeros((B, d.n), dtype=dtype),
                A=cp.zeros((B, d.p, d.n), dtype=dtype) if d.p > 0 else None,
                b=cp.zeros((B, d.p), dtype=dtype) if d.p > 0 else None,
                G=cp.zeros((B, d.m, d.n), dtype=dtype) if d.m > 0 else None,
                h_u=cp.zeros((B, d.m), dtype=dtype) if d.num_hu > 0 else None,
                h_l=cp.zeros((B, d.m), dtype=dtype) if d.num_hl > 0 else None,
                x_u=cp.zeros((B, d.n), dtype=dtype) if d.num_xu > 0 else None,
                x_l=cp.zeros((B, d.n), dtype=dtype) if d.num_xl > 0 else None,
            )

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
                    grad_data._P, grad_data._A, grad_data._G,
                    grad_data._b, grad_data._h_u, grad_data._x_u,
                ],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )

        # Vector grads that are aliases of solver-internal buffers — copy in.
        grad_data._c[:] = adjoint_vector.x
        if data.num_hl > 0:
            grad_data._h_l[:] = self._lam_zl_full
        if data.num_xl > 0:
            grad_data._x_l[:] = self._lam_zbl_full

        return grad_data
