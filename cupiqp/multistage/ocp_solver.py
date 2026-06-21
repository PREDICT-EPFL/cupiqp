from typing import Literal, Sequence, Union, List

import cupy as cp

from ..typedef import CudaArray
from ..results import Status
from .multistage_solver import MultistageSolver
from .ocp_data import OcpData


class OcpSolver(MultistageSolver):
    r"""High-level optimal-control-problem (OCP) interface for the multistage
    backend, in the spirit of HPIPM.

    Instead of building the block-structured matrices by hand, you declare the
    problem **dimensions** once with :meth:`setup`, then fill in the numerical
    data field by field with :meth:`set`, in the style of HPIPM's
    ``d_ocp_qp_set(field, stage, value, ...)``. The problem data lives in an
    :class:`OcpData` (accessible as ``solver.ocp_data``); this solver owns the
    solve lifecycle and pushes that data into its scaled internal solver state.

    The problem solved over stages ``k = 0, ..., N`` uses stage variables
    ``y_k = [x_k; u_k]``. The terminal ``u_N`` is padding required by the
    uniform block layout and carries no cost or constraint:

    $$
    \begin{aligned}
    \min_{x, u}\ & \sum_{k=0}^{N-1}
        \left(
        \tfrac{1}{2}
        \begin{bmatrix} x_k \\ u_k \end{bmatrix}^\top
        \begin{bmatrix} Q_k & S_k^\top \\ S_k & R_k \end{bmatrix}
        \begin{bmatrix} x_k \\ u_k \end{bmatrix}
        + q_k^\top x_k + r_k^\top u_k
        \right)
        + \tfrac{1}{2} x_N^\top Q_N x_N + q_N^\top x_N \\
    \text{s.t.}\ & x_0 = x_0^{\text{init}}, \\
                 & E_k\, x_{k+1} = A_k\, x_k + B_k\, u_k + b_k,
                   \quad k = 0, \dots, N-1, \\
                 & l^g_k \le C_k\, x_k + D_k\, u_k \le u^g_k,
                   \quad k = 0, \dots, N-1, \\
                 & l^g_N \le C_N\, x_N \le u^g_N, \\
                 & x^l_k \le x_k \le x^u_k, \quad k = 0, \dots, N, \\
                 & u^l_k \le u_k \le u^u_k, \quad k = 0, \dots, N-1.
    \end{aligned}
    $$

    The row ``E_k x_{k+1} = A_k x_k + B_k u_k + b_k`` is treated as a **general
    linear equality constraint** coupling two consecutive stages. ``E_k``
    defaults to the identity, recovering the usual explicit form
    ``x_{k+1} = A_k x_k + B_k u_k + b_k``; setting ``E_k`` to a non-identity
    matrix (for example a mass / descriptor matrix) lets you express implicit
    integrators and descriptor systems directly, without inverting ``E_k``
    yourself.

    Fields accepted by :meth:`set` (HPIPM names):

    * equality coupling (``k = 0, ..., N-1``): ``'A'`` ``(nx, nx)``,
      ``'B'`` ``(nx, nu)``, ``'E'`` ``(nx, nx)``, ``'b'`` ``(nx,)``;
    * initial condition: ``'x0'`` ``(nx,)``;
    * state cost (``k = 0, ..., N``): ``'Q'`` ``(nx, nx)``, ``'q'``
      ``(nx,)``; input cost (``k = 0, ..., N-1``): ``'R'`` ``(nu, nu)``,
      ``'S'`` ``(nu, nx)``, ``'r'`` ``(nu,)``;
    * general inequality (needs ``ng > 0``): ``'C'``, ``'lg'``, and ``'ug'``
      for ``k = 0, ..., N``; ``'D'`` only for ``k = 0, ..., N-1``;
    * state bounds ``'lbx'`` / ``'ubx'`` for ``k = 0, ..., N``, and input
      bounds ``'lbu'`` / ``'ubu'`` for ``k = 0, ..., N-1``.

    Each value may be passed unbatched (broadcast across the whole batch) or
    with a leading batch axis ``(B, ...)`` for per-problem data (for example a
    different ``'x0'`` per batch element). Read the solution back with
    :meth:`get` (``'x'`` / ``'u'``) or the :attr:`x_traj` / :attr:`u_traj`
    properties; the solve status is in ``solver.result.info.status``.

    Parameters
    ----------
    dtype : {"float64", "float32"}, default: "float64"
        Floating-point precision used throughout the solve.

    Examples
    --------
    ```python
    from cupiqp import OcpSolver

    s = OcpSolver()
    s.setup(N=20, nx=2, nu=1, idxbu=[0])    # dimensions only; bound input 0
    for k in range(20):
        s.set("A", k, A); s.set("B", k, B)  # E_k defaults to identity
        s.set("Q", k, Q); s.set("R", k, R)
        s.set("lbu", k, [-u_max]); s.set("ubu", k, [u_max])
    s.set("Q", 20, Qf)                      # terminal cost
    s.set("x0", 0, x0)
    s.solve()
    x_traj = s.x_traj          # (B, N+1, nx)
    ```

    Notes
    -----
    The multistage backend uses a single block size, so ``nx``, ``nu`` and
    ``ng`` are uniform across stages (unlike HPIPM's time-varying dimensions).
    The problem *structure* (the dimensions and which bounds are present) is
    fixed by :meth:`setup`; afterwards :meth:`set` may change only numerical
    values, keeping every declared bound finite (to drop a bound use a large
    value, not infinity). The next-input coupling ``F_k u_{k+1}`` is not
    implemented but the block layout leaves room for a future ``'F'`` field.

    **Not differentiable (yet).** ``OcpSolver`` does not support gradients.
    For differentiable QPs use ``DenseSolver``, ``SparseSolver`` or ``MultistageSolver``.
    """

    def __init__(self, dtype: Literal["float32", "float64"] = "float64") -> None:
        super().__init__(dtype=dtype)
        self._ocp_data = None
        self._ocp_ready = False

        self._update_P = False
        self._update_A = False
        self._update_G = False
        self._solution_available = False

    @property
    def ocp_data(self) -> OcpData:
        """The :class:`OcpData` holding the raw OCP problem data (after setup)."""
        return self._ocp_data

    def setup(
        self,
        N: int,
        nx: int,
        nu: int,
        ng: int = 0,
        idxbx: Union[int, Sequence[int], None] = None,
        idxbu: Union[int, Sequence[int], None] = None,
        batch_size: int = 1,
    ) -> None:
        """Declare the OCP dimensions and allocate all GPU memory.

        Builds an :class:`OcpData` (zeroed, with sensible defaults), then hands
        its blocks to the underlying multistage solver to allocate buffers and
        compile kernels. Fill in the data with :meth:`set` afterwards, then call
        :meth:`solve`.

        Parameters
        ----------
        N : int
            Horizon -- the number of equality-coupling steps. There are
            ``N + 1`` stages ``k = 0, ..., N``.
        nx, nu : int
            State and input dimensions (uniform across stages).
        ng : int, default: 0
            Number of general inequality constraints per stage. ``0`` means no
            general inequalities (no ``G`` block is allocated).
        idxbx, idxbu : int or sequence of int, optional
            Indices of the box-bounded state / input components (e.g. ``[0, 2]``
            or a single ``2``), in ``[0, nx)`` / ``[0, nu)``. ``None`` (default)
            means no box bounds for that category. State bounds apply at
            stages ``0..N``; input bounds apply at stages ``0..N-1``.
        batch_size : int, default: 1
            Number of OCPs solved together (leading batch axis).
        """
        if self._setup_done:
            raise RuntimeError(
                "setup() may only be called once per solver instance; "
                "create a new solver instance to set up a different problem."
            )

        if self.settings.enable_grad:
            raise NotImplementedError(
                "OcpSolver does not yet support differentiation: settings.enable_grad "
                "To differentiate a QP, use DenseSolver, SparseSolver or MultistageSolver."
            )

        ocp_data = OcpData(
            N, nx, nu, ng=ng, idxbx=idxbx, idxbu=idxbu,
            dtype=self.settings.dtype, device=self.settings.device,
            batch_size=batch_size,
        )
        MultistageSolver.setup(self, **ocp_data.blocks)

        self._ocp_data = ocp_data
        self._solution_available = False
        self._ocp_ready = True

    def set(self, field: str, stage: int, value: CudaArray) -> None:
        """Set one block of OCP data at a given stage (delegates to :class:`OcpData`).

        See the class docstring for the accepted ``field`` names and shapes.
        ``value`` must be a CUDA array; cuPIQP is GPU-only and never silently copies host
        data to the device, so passing a host array raises ``TypeError``.

        ``value`` is accepted in one of two shapes:

        * **unbatched** -- the field's plain per-stage shape (e.g. ``(nx, nx)``
          for ``'A'``). The same value is **broadcast across the whole batch**:
          every one of the ``B`` problems is set to this value.
        * **batched** -- with a leading batch axis ``(B, ...)``, to set a
          different value per problem (e.g. a distinct ``'x0'`` per element).
        """
        if not self._ocp_ready:
            raise RuntimeError("Call setup() before set().")

        if field in ("Q", "R", "S"):
            self._update_P = True
        elif field in ("A", "B", "E"):
            self._update_A = True
        elif field in ("C", "D"):
            self._update_G = True

        self._ocp_data.set_field(field, stage, value)

    def solve(self) -> List[Status]:
        """Flush any pending :meth:`set` updates into the solver, then solve.

        Returns the solve status (a single ``Status`` for ``batch_size == 1``,
        otherwise a list of one ``Status`` per problem). The full solution is
        available through :meth:`get`, :attr:`x_traj`, :attr:`u_traj`, and
        ``solver.result``.
        """
        if not self._ocp_ready:
            raise RuntimeError("Call setup() before solve().")

        blocks = self._ocp_data.blocks
        changed = {
            group: block
            for group, block in blocks.items()
            if block is not None and group not in ("P", "A", "G")
        }
        if self._update_P:
            changed["P"] = blocks["P"]
        if self._update_A:
            changed["A"] = blocks["A"]
        if self._update_G:
            changed["G"] = blocks["G"]

        MultistageSolver.update(self, **changed)

        # set back to false to prepare for next solver update
        self._update_P = self._update_A = self._update_G = False

        status = MultistageSolver.solve(self)
        self._solution_available = True
        return status

    def get(self, field: str, stage: int) -> cp.ndarray:
        """Read part of the solution.

        ``field='x'`` returns the stage state ``(B, nx)`` for ``stage = 0..N``;
        ``field='u'`` returns the stage input ``(B, nu)`` for ``stage = 0..N-1``
        (``u_N`` is a dummy).
        """
        self._require_solution()
        if field == "x":
            return self._ocp_data.state(self._result.x, stage)
        if field == "u":
            return self._ocp_data.input(self._result.x, stage)
        raise ValueError(f"Unknown solution field {field!r}; use 'x' or 'u'.")

    @property
    def x_traj(self) -> cp.ndarray:
        """State trajectory, shape ``(B, N+1, nx)``."""
        self._require_solution()
        return self._ocp_data.state_traj(self._result.x)

    @property
    def u_traj(self) -> cp.ndarray:
        """Input trajectory, shape ``(B, N, nu)`` (the dummy ``u_N`` is dropped)."""
        self._require_solution()
        return self._ocp_data.input_traj(self._result.x)

    def _require_solution(self) -> None:
        if not self._solution_available:
            raise RuntimeError("No solution available; call solve() first.")
