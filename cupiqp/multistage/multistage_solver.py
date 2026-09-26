import cupy as cp
import numpy as np
import warp as wp

from ..results import Variables
from typing import Literal, Optional, Tuple

from ..settings import Settings
from ..solver import SolverBase
from .multistage_data import MultistageData
from .multistage_preconditioner import MultistageRuizEquilibration
from .multistage_solver_kernels import create_multistage_data_gradients_kernel
from ..typedef import CudaArray
from ..utils import to_warp_dtype


# A block-structured matrix input: a (diag, offdiag) pair of GPU arrays.
BlockPair = Tuple[Optional[CudaArray], Optional[CudaArray]]


class MultistageSolver(SolverBase):
    r"""GPU solver for multistage (block-structured) convex quadratic
    programs that solves a QP - or a whole batch of QPs - of the form

    $$
    \begin{aligned}
    \min_{x}\quad & \tfrac{1}{2}\, x^\top P x + c^\top x \\
    \text{s.t.}\quad & A x = b, \\
    & h_l \le G x \le h_u, \\
    & x_l \le x \le x_u,
    \end{aligned}
    $$

    using the proximal interior-point method with a block-Cholesky
    factorization that exploits block-tridiagonal / block-tridiagonal-arrow
    KKT structure, running entirely on the GPU. It is built for multistage
    problems such as optimal control (OCPs / MPC), where that structure
    arises from the stage-by-stage dynamics and costs.

    **Inputs - plain GPU arrays, block by block.** A block-structured matrix
    is passed as a ``(diag, offdiag)`` tuple of GPU arrays (Warp, cupy, CUDA
    torch, JAX, ... - anything supporting DLPack) and vectors are GPU arrays,
    all of the solver dtype (``float64`` by default). With ``N`` stages of
    size ``d`` (``n = N*d``):

    * ``P = (P_diag, P_offdiag)``: symmetric block-tridiagonal, with diagonal
      blocks ``(N, d, d)`` and lower off-diagonal blocks ``(N-1, d, d)``
      (block ``(k+1, k)``; the upper ones are their transposes).
    * ``A = (A_diag, A_offdiag)`` and ``G = (G_diag, G_offdiag)``: block
      lower-bidiagonal with ``N + 1`` block rows of ``r`` rows each. Block
      row ``k`` is ``[.. A_offdiag[k-1]  A_diag[k] ..]``, i.e. both arrays
      have shape ``(N, r, d)``; the last block row holds only
      ``A_offdiag[N-1]`` (acting on the last stage).
    * ``c``, ``x_l``, ``x_u``: ``(N, d)`` or flat ``(n,)``; ``b``, ``h_l``,
      ``h_u``: ``(N + 1, r)`` or flat ``((N + 1) * r,)``.

    Only ``P`` and ``c`` are required. As with the other backends, ``+/-inf``
    entries in the bounds mark one-sided or free bounds.

    **Batching - setup from one problem, then per-problem values.**
    ``setup`` takes the batch size and **one** template problem (the shapes
    above); its values are copied into all ``B`` problems. ``update`` then
    sets per-problem values with a leading batch axis (e.g. ``P_diag`` of
    shape ``(B, N, d, d)``), or with the unbatched shape to share one value
    across the batch. Either entry of a ``(diag, offdiag)`` tuple may be
    ``None`` to leave it unchanged. Every given array is copied once, in
    place, into the solver's Warp buffers - no allocation, no GPU-CPU sync.
    ``solver.result.x`` has shape ``(B, n)``.

    Parameters
    ----------
    dtype : {"float64", "float32"}, default: "float64"
        Floating-point precision used throughout the solve. ``"float32"``
        is faster and uses less memory but converges to looser tolerances;
        the default convergence tolerances are chosen to match the dtype.

    Examples
    --------
    ```python
    import cupy as cp
    from cupiqp import MultistageSolver

    N, d = 10, 3
    P_diag = cp.broadcast_to(2.0 * cp.eye(d), (N, d, d))     # (N, d, d)
    P_offdiag = cp.broadcast_to(-0.5 * cp.eye(d), (N - 1, d, d))
    c = cp.ones((N, d))

    s = MultistageSolver()
    s.setup(8, P=(P_diag, P_offdiag), c=c,                  # template, 8 copies
            x_l=-cp.ones((N, d)), x_u=cp.ones((N, d)))
    s.update(c=cp.random.standard_normal((8, N, d)))        # per-problem c
    s.solve()
    ```

    See Also
    --------
    DenseSolver: solver for general dense problems.

    SparseSolver: solver for general sparse problems.

    Notes
    -----
    Requires the ``socu`` block-solver package (install the ``multistage``
    extra: ``pip install ".[cuda13,multistage]"``). ``setup`` can be called
    only once per instance; for a different structure (block sizes, which
    blocks and bound sides are present), create a new ``MultistageSolver``.
    Solver behaviour (tolerances, verbosity, iteration cap, ...) is
    configured through ``solver.settings``.
    """

    def __init__(self, dtype: Literal["float32", "float64"] = "float64"):
        super().__init__(dtype=dtype)
        self._settings.kkt_solver = "multistage_block_cholesky"

    @SolverBase.settings.setter
    def settings(self, value: Settings) -> None:
        # TODO: here we have to set the kkt solver back. That's pretty ugly. Should be improved in the future
        value.kkt_solver = "multistage_block_cholesky"
        self._settings = value


    def _init_data(self, P, c, A, b, G, h_u, h_l, x_u, x_l) -> MultistageData:
        data = MultistageData(dtype=self.settings.dtype, device=self.settings.device)
        data.init(self._setup_batch_size, P, c, A, b, G, h_u, h_l, x_u, x_l)
        return data

    def _init_preconditioner(self) -> MultistageRuizEquilibration:
        return MultistageRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            has_h_l=self._data.has_h_l, has_h_u=self._data.has_h_u,
            has_x_l=self._data.has_x_l, has_x_u=self._data.has_x_u,
            active_x_bound=self._data.active_x_bound,
            data=self._data,
            use_warp_tile_kernels=(self._kernel_strategy == "warp_tile"),
            enable_cuda_graph=self.settings.enable_cuda_graph,
            dtype=self._data.dtype,
        )

    def setup(
        self,
        batch_size: int,
        P: BlockPair,
        c: CudaArray,
        A: Optional[BlockPair] = None,
        b: Optional[CudaArray] = None,
        G: Optional[BlockPair] = None,
        h_u: Optional[CudaArray] = None,
        h_l: Optional[CudaArray] = None,
        x_u: Optional[CudaArray] = None,
        x_l: Optional[CudaArray] = None,
    ) -> None:
        """Fix the problem structure from one template problem and allocate
        all GPU memory for a batch of ``batch_size`` problems.

        You describe a **single** problem here (no batch axis); its block
        sizes, and which constraint blocks and bound sides are present,
        become the structure shared by all ``batch_size`` problems, and its
        values are copied into every problem. Afterwards, give each problem
        its own values with :meth:`update`, then call :meth:`solve`. Call
        ``setup()`` once per solver instance.

        Parameters
        ----------
        batch_size : int
            Number of QPs ``B`` solved together.
        P : tuple of GPU arrays
            ``(P_diag, P_offdiag)``: diagonal blocks ``(N, d, d)`` and lower
            off-diagonal blocks ``(N-1, d, d)`` of the symmetric
            block-tridiagonal cost matrix. ``P_offdiag`` may be ``None``
            (zero). Required.
        c : GPU array
            Linear cost, ``(N, d)`` or flat ``(N*d,)``. Required.
        A, b : optional
            Equality constraints ``A x = b``: ``A = (A_diag, A_offdiag)``,
            both ``(N, r, d)`` (``A_offdiag`` may be ``None``), and ``b`` of
            shape ``(N+1, r)`` or flat. Provide both or neither.
        G, h_l, h_u : optional
            Inequalities ``h_l <= G x <= h_u``: ``G = (G_diag, G_offdiag)``,
            both ``(N, r, d)``, and bounds ``(N+1, r)`` or flat. At least one
            bound is required when ``G`` is given; an omitted side is absent
            for the lifetime of the solver.
        x_l, x_u : GPU array, optional
            Box bounds, ``(N, d)`` or flat ``(N*d,)``. An omitted side is
            absent for the lifetime of the solver.

        Raises
        ------
        RuntimeError
            If ``setup()`` has already been called on this instance.
        TypeError
            If a matrix is not a ``(diag, offdiag)`` tuple or an input is not
            a GPU array.
        ValueError
            If ``batch_size`` is not a positive integer or the block shapes
            are inconsistent.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer; got {batch_size!r}.")
        self._setup_batch_size = int(batch_size)
        super().setup(P, c, A, b, G, h_u, h_l, x_u, x_l)
        if self.settings.enable_grad:
            d = self._data
            B = d.batch_size
            N = d.num_blocks
            d_sz = d.block_size
            dtype = d.dtype
            wp_dtype = to_warp_dtype(dtype)

            r_a, N_a = (d.A_rows, N) if d.p > 0 else (0, 0)
            r_g, N_g = (d.G_rows, N) if d.m > 0 else (0, 0)

            # Zero-initialized gradient storage with the forward structure: a
            # one-problem zero template, tiled to the batch by init(). Blocks and
            # bound sides exist exactly when they do in the forward problem.
            z = lambda *shape: cp.zeros(shape, dtype=dtype)
            has_A, has_G = d.p > 0, d.m > 0
            self._grad_data = MultistageData(dtype=dtype, device=self.settings.device)
            self._grad_data.init(
                B,
                P=(z(N, d_sz, d_sz), None), c=z(N, d_sz),
                A=(z(N_a, r_a, d_sz), None) if has_A else None,
                b=z(N_a + 1, r_a) if has_A else None,
                G=(z(N_g, r_g, d_sz), None) if has_G else None,
                h_u=z(N_g + 1, r_g) if has_G and d.has_h_u else None,
                h_l=z(N_g + 1, r_g) if has_G and d.has_h_l else None,
                x_u=z(N, d_sz) if d.has_x_u else None,
                x_l=z(N, d_sz) if d.has_x_l else None,
            )

            # Empty placeholder warp buffers for when A or G are absent —
            # the kernel still needs valid array arguments even though its
            # corresponding dispatch sub-range collapses to size 0.
            empty_blocks = wp.zeros((B, 0, 0, 0), dtype=wp_dtype, device="cuda")
            g = self._grad_data
            self._grad_dA_D = g.A_diag if g.A_diag is not None else empty_blocks
            self._grad_dA_E = g.A_offdiag if g.A_diag is not None else empty_blocks
            self._grad_dG_D = g.G_diag if g.G_diag is not None else empty_blocks
            self._grad_dG_E = g.G_offdiag if g.G_diag is not None else empty_blocks

            # Eager-compile the fused multistage data-gradients kernel.
            self._multistage_data_gradients_kernel = create_multistage_data_gradients_kernel(
                N, d_sz, N_a, r_a, N_g, r_g, d.p, d.m, d.n, d.num_hu, d.num_hl, d.num_xu, d.num_xl, dtype=dtype)

    def update(
        self,
        P: Optional[BlockPair] = None,
        c: Optional[CudaArray] = None,
        A: Optional[BlockPair] = None,
        b: Optional[CudaArray] = None,
        G: Optional[BlockPair] = None,
        h_u: Optional[CudaArray] = None,
        h_l: Optional[CudaArray] = None,
        x_u: Optional[CudaArray] = None,
        x_l: Optional[CudaArray] = None,
        check_validity: bool = False,
    ) -> None:
        """Set new numerical data, per problem or shared, then ``solve()``.

        Any argument left as ``None`` keeps its current value; so does a
        ``None`` entry of a ``(diag, offdiag)`` tuple, e.g.
        ``update(P=(None, P_offdiag))`` changes only the coupling blocks.
        Each array is either **batched** - a leading batch axis, e.g.
        ``P_diag`` of shape ``(B, N, d, d)`` or ``c`` of shape ``(B, N, d)`` /
        ``(B, N*d)`` - or **unbatched** (the ``setup()`` shape), in which case
        the same value is used for every problem. Each given array is copied
        once, in place, into the solver's buffers.

        Parameters
        ----------
        P, A, G : tuple of GPU arrays, optional
            New ``(diag, offdiag)`` blocks; see :meth:`setup` for the layout.
        c, b, h_u, h_l, x_u, x_l : GPU array, optional
            New vectors, block or flat layout. Bound values may switch between
            finite and ``+/-inf``, but only bound sides given at ``setup()``
            can be updated.
        check_validity : bool, default: False
            Unused by this backend: shapes are always checked (metadata only,
            no GPU sync).
        """
        super().update(
            P=P, c=c, A=A, b=b, G=G, h_u=h_u, h_l=h_l, x_u=x_u, x_l=x_l,
            check_validity=check_validity,
        )

    def _compute_data_gradients(self, adjoint_vector: Variables, linearization_point: Variables) -> MultistageData:
        r"""Populate ``self._grad_data`` in place and return it.

        Matrix gradients are written into the block arrays of
        ``self._grad_data`` (``P_diag`` / ``P_offdiag``, ...). Vector grads ``c``, ``h_l``, ``x_l`` are
        copies of ``adjoint_vector.x``, ``self._lam_zl_full``,
        ``self._lam_zbl_full``.

        Returns the same instance on every call; its buffers are
        overwritten by the next backward.
        """
        data = self._data
        grad_data = self._grad_data
        B = data.batch_size
        N = data.num_blocks
        d_sz = data.block_size

        N_off = max(N - 1, 0)
        r_a, N_a = (data.A_rows, N) if data.p > 0 else (0, 0)
        r_g, N_g = (data.G_rows, N) if data.m > 0 else (0, 0)
        total = (
            N * d_sz * d_sz
            + N_off * d_sz * d_sz
            + 2 * N_a * r_a * d_sz
            + 2 * N_g * r_g * d_sz
            + data.n + data.p + data.num_hu + data.num_hl + data.num_xu + data.num_xl
        )
        if total > 0:
            wp.launch(
                kernel=self._multistage_data_gradients_kernel,
                dim=(B, total),
                inputs=[
                    adjoint_vector.x, adjoint_vector.y,
                    self._lam_zu_full, self._lam_zl_full,
                    self._lam_zbu_full, self._lam_zbl_full,
                    self._zu_full, self._zl_full,
                    linearization_point.x, linearization_point.y,
                    grad_data.P_diag,
                    grad_data.P_offdiag,
                    self._grad_dA_D, self._grad_dA_E,
                    self._grad_dG_D, self._grad_dG_E,
                    grad_data._c, grad_data._b,
                    grad_data._h_u, grad_data._h_l,
                    grad_data._x_u, grad_data._x_l,
                ],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )

        return grad_data
