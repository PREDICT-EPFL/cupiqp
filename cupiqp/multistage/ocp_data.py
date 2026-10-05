from types import MappingProxyType
from typing import Sequence, Union

import numpy as np
import warp as wp

from ..utils import to_warp_dtype, as_warp_array, batch_broadcast_view, strided_view
from .ocp_data_kernels import create_ocp_data_kernels


# Finite placeholder magnitude for a declared-but-not-yet-set bound. It must be
# strictly below PIQP_INF so the backend keeps the row/column (does not treat it
# as infinite and drop it), while being large enough to be numerically inactive.
# Declared bounds should be overwritten by ``set_field`` before solving.
# Any left unset behaves as an effectively unbounded constraint.
_BOUND_SENTINEL = 1e12


def _normalize_idx(spec, dim: int, name: str) -> np.ndarray:
    """Normalize a box-bound index set into an array of component indices.

    ``spec`` is ``None`` (no bounds) or one or more component indices in
    ``[0, dim)`` -- a single int (e.g. ``2``) or a sequence (e.g. ``[0, 2]``).
    """
    if spec is None:
        return np.zeros(0, dtype=np.int64)
    if isinstance(spec, (bool, np.bool_)):
        raise TypeError(f"{name} indices must be integers; got bool.")
    raw = np.asarray(spec)
    if raw.size == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} indices must be integers; got dtype {raw.dtype}.")
    idx = raw.astype(np.int64, copy=False).reshape(-1)
    if idx.size and (idx.min() < 0 or idx.max() >= dim):
        raise ValueError(f"{name} indices must lie in [0, {dim}); got {idx}.")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{name} indices must be unique; got {idx}.")
    return idx


class OcpData:
    r"""Block-structured data of an optimal-control QP, with HPIPM-style access.

    ``OcpData`` owns the multistage block storage for a single OCP structure
    (over stages ``k = 0, ..., N`` with stage variable ``y_k = [x_k; u_k]``) and
    lets you fill them field by field with :meth:`set_field`, using HPIPM field
    names.
    It is a *pure data holder*: it does not solve, scale, or clone -- the solver
    (:class:`OcpSolver`) reads :attr:`arrays` and keeps its own scaled copy. This
    is what keeps the raw values you set separate from the Ruiz-scaled data the
    interior-point method factorizes.

    The blocks map the OCP onto the condensed multistage QP form
    (``min 1/2 z^T P z + c^T z`` s.t. ``A z = b``, ``h_l <= G z <= h_u``,
    ``x_l <= z <= x_u``) with ``z = (y_0, ..., y_N)``:

    * the equality constraints carry the initial condition ``x_0 = x0`` (row 0)
      and the stage-coupling rows ``E_k x_{k+1} = A_k x_k + B_k u_k + b_k``
      (``E_k`` defaults to the identity);
    * running cost blocks are ``[[Q_k, S_k^T], [S_k, R_k]]``; the terminal
      block contains only ``Q_N`` and ``q_N``;
    * ``G`` carries the per-stage general inequalities ``l^g_k <= C_k x_k + D_k u_k <= u^g_k``
      (only when ``ng > 0``);
    * the box bounds act on the state components in ``idxbx`` and input components
      in ``idxbu``.

    All storage is Warp arrays with a leading batch axis.

    Parameters
    ----------
    N : int
        Horizon -- number of stage-coupling steps. There are ``N + 1`` stages.
    nx, nu : int
        State and input dimensions (uniform across stages).
    ng : int, default: 0
        Number of general inequality constraints per stage (``0`` -> no ``G`` block).
    idxbx, idxbu : int or sequence of int, optional
        Indices of the box-bounded state / input components (e.g. ``[0, 2]`` or
        a single ``2``), in ``[0, nx)`` / ``[0, nu)``. ``None`` (default) means no
        box bounds for that category. State bounds apply at stages ``0..N``;
        input bounds apply at control stages ``0..N-1``.
    dtype : {wp.float64, wp.float32}, default: wp.float64
    device : str, default: "cuda"
    batch_size : int, default: 1
        Number of OCPs stored together (leading batch axis).
    """

    def __init__(self, N: int, nx: int, nu: int, ng: int = 0,
                 idxbx: Union[int, Sequence[int], None] = None,
                 idxbu: Union[int, Sequence[int], None] = None,
                 dtype: Union[type[wp.float32], type[wp.float64]] = wp.float64,
                 device: str = "cuda",
                 batch_size: int = 1) -> None:
        assert isinstance(N, (int, np.integer)) and N >= 1, "N must be an integer >= 1."
        assert isinstance(nx, (int, np.integer)) and nx >= 1, "nx must be an integer >= 1."
        assert isinstance(nu, (int, np.integer)) and nu >= 1, "nu must be an integer >= 1."
        assert isinstance(ng, (int, np.integer)) and ng >= 0, "ng must be an integer >= 0."
        assert isinstance(batch_size, (int, np.integer)) and batch_size >= 1, "batch_size must be an integer >= 1."
        self._N, self._nx, self._nu = int(N), int(nx), int(nu)
        self._ng, self._batch_size = int(ng), int(batch_size)
        self._d = self._nx + self._nu
        self._idxbx = _normalize_idx(idxbx, self._nx, "idxbx")
        self._idxbu = _normalize_idx(idxbu, self._nu, "idxbu")
        self._dtype = to_warp_dtype(dtype)
        self._device = device
        self._kernels = create_ocp_data_kernels(self._dtype)

        nx, nu, ng, N, B = self._nx, self._nu, self._ng, self._N, self._batch_size
        _base = {
            "x0": ((nx,), 0, 0),
            "A": ((nx, nx), 0, N - 1),
            "B": ((nx, nu), 0, N - 1),
            "E": ((nx, nx), 0, N - 1),
            "b": ((nx,), 0, N - 1),
            "Q": ((nx, nx), 0, N),
            "R": ((nu, nu), 0, N - 1),
            "S": ((nu, nx), 0, N - 1),
            "q": ((nx,), 0, N),
            "r": ((nu,), 0, N - 1),
            "C": ((ng, nx), 0, N),
            "D": ((ng, nu), 0, N - 1),
            "lg": ((ng,), 0, N),
            "ug": ((ng,), 0, N),
            "lbx": ((self._idxbx.size,), 0, N),
            "ubx": ((self._idxbx.size,), 0, N),
            "lbu": ((self._idxbu.size,), 0, N - 1),
            "ubu": ((self._idxbu.size,), 0, N - 1),
        }
        self._field_specific_info = {
            f: ((B,) + shape, lo, hi) for f, (shape, lo, hi) in _base.items()
        }

        blk_size, N_blk = self._d, self._N + 1

        # ---- allocate the block storage (Warp arrays, batch axis first) ----
        self._P_diag = wp.zeros((B, N_blk, blk_size, blk_size), dtype=self._dtype, device=device)
        self._P_offdiag = wp.zeros((B, N_blk - 1, blk_size, blk_size), dtype=self._dtype, device=device)
        self._c = wp.zeros((B, N_blk, blk_size), dtype=self._dtype, device=device)
        self._A_diag = wp.zeros((B, N_blk, self._nx, blk_size), dtype=self._dtype, device=device)
        self._A_offdiag = wp.zeros((B, N_blk, self._nx, blk_size), dtype=self._dtype, device=device)
        self._b = wp.zeros((B, N_blk + 1, self._nx), dtype=self._dtype, device=device)
        self._G_diag = wp.zeros((B, N_blk, self._ng, blk_size), dtype=self._dtype, device=device)
        self._G_offdiag = wp.zeros((B, N_blk, self._ng, blk_size), dtype=self._dtype, device=device)
        self._h_l = wp.zeros((B, N_blk + 1, self._ng), dtype=self._dtype, device=device)
        self._h_u = wp.zeros((B, N_blk + 1, self._ng), dtype=self._dtype, device=device)
        self._x_l = wp.zeros((B, N_blk, blk_size), dtype=self._dtype, device=device)
        self._x_u = wp.zeros((B, N_blk, blk_size), dtype=self._dtype, device=device)

        # Box-bounded columns of the stage variable y_k = [x_k; u_k].
        self._cols_bx = wp.array(self._idxbx.astype(np.int32), dtype=wp.int32, device=device)
        self._cols_bu = wp.array((self._nx + self._idxbu).astype(np.int32), dtype=wp.int32, device=device)

        # ---- structural defaults ----
        # initial-condition row 0:  [I, 0] y_0 = x0
        # stage-coupling rows 1..N: default descriptor E_k = I -> D[k+1] = [-I, 0]
        wp.launch(self._kernels["init_coupling_diag"], dim=(B, N_blk, self._nx),
                  inputs=[self._A_diag], device=device)

        # box bounds: everything free, declared components set to a finite sentinel
        self._x_l.fill_(-float("inf"))
        self._x_u.fill_(float("inf"))
        if self._idxbx.size:
            wp.launch(self._kernels["fill_cols"], dim=(B, N_blk, self._idxbx.size),
                      inputs=[self._x_l, self._cols_bx, self._dtype(-_BOUND_SENTINEL)], device=device)
            wp.launch(self._kernels["fill_cols"], dim=(B, N_blk, self._idxbx.size),
                      inputs=[self._x_u, self._cols_bx, self._dtype(_BOUND_SENTINEL)], device=device)
        if self._idxbu.size:
            # u_N is padding used only to keep a uniform stage block size.
            wp.launch(self._kernels["fill_cols"], dim=(B, self._N, self._idxbu.size),
                      inputs=[self._x_l, self._cols_bu, self._dtype(-_BOUND_SENTINEL)], device=device)
            wp.launch(self._kernels["fill_cols"], dim=(B, self._N, self._idxbu.size),
                      inputs=[self._x_u, self._cols_bu, self._dtype(_BOUND_SENTINEL)], device=device)

        if self._ng > 0:
            # all ng rows present at stages 0..N (finite sentinel); the trailing
            # padding row stays infinite so the backend drops it.
            self._h_l.fill_(-float("inf"))
            self._h_u.fill_(float("inf"))
            self._h_l[:, :N_blk, :].fill_(-_BOUND_SENTINEL)
            self._h_u[:, :N_blk, :].fill_(_BOUND_SENTINEL)


    @property
    def N(self) -> int:
        """Horizon: number of stage-coupling steps; there are ``N + 1`` stages (0..N)."""
        return self._N

    @property
    def nx(self) -> int:
        """State dimension (uniform across stages)."""
        return self._nx

    @property
    def nu(self) -> int:
        """Input dimension (uniform across stages)."""
        return self._nu

    @property
    def ng(self) -> int:
        """Number of general inequality constraints per stage (0 if none)."""
        return self._ng

    @property
    def d(self) -> int:
        """Stage block size ``nx + nu`` (length of the stacked variable ``y_k = [x_k; u_k]``)."""
        return self._d

    @property
    def batch_size(self) -> int:
        """Number of OCPs stored together (the leading batch axis)."""
        return self._batch_size

    @property
    def idxbx(self) -> np.ndarray:
        """Indices of the box-bounded state components (empty if none)."""
        return self._idxbx

    @property
    def idxbu(self) -> np.ndarray:
        """Indices of the box-bounded input components (empty if none)."""
        return self._idxbu

    @property
    def arrays(self):
        """Read-only mapping of the problem data as MultistageSolver inputs.

        Keys ``P, c, A, b, G, h_l, h_u, x_l, x_u``; matrices are
        ``(diag, offdiag)`` pairs and vectors ``(B, num_blocks, rows)`` arrays,
        all the Warp buffers owned by this object. Modify them only through
        :meth:`set_field`.
        """
        return MappingProxyType({
            "P": (self._P_diag, self._P_offdiag),
            "c": self._c,
            "A": (self._A_diag, self._A_offdiag),
            "b": self._b,
            "G": (self._G_diag, self._G_offdiag),
            "h_l": self._h_l,
            "h_u": self._h_u,
            "x_l": self._x_l,
            "x_u": self._x_u,
        })

    def set_field(self, field: str, stage: int, value: wp.array) -> None:
        """Set one block of OCP data at a given stage.

        ``field`` is an HPIPM field name (``A``, ``B``, ``E``, ``b``, ``x0``,
        ``Q``, ``R``, ``S``, ``q``, ``r``, ``C``, ``D``, ``lg``, ``ug``,
        ``lbx``, ``ubx``, ``lbu``, ``ubu``). ``value`` is a Warp array of this
        object's dtype (``OcpSolver.set`` converts user arrays).

        ``value`` must match exactly one of two shapes:

        * **unbatched** -- the field's plain per-stage shape. The same value is
          **broadcast across the whole batch**: every one of the ``B`` problems
          is set to this value.
        * **batched** -- with a leading batch axis ``(B, ...)``, to set a
          different value per problem.
        """
        nx, ng = self._nx, self._ng

        if field not in self._field_specific_info:
            raise ValueError(
                f"Unknown field {field!r}. Valid fields: {sorted(self._field_specific_info)}."
            )
        if field in ("C", "D", "lg", "ug") and ng == 0:
            raise ValueError(
                f"field {field!r} requires ng > 0; OcpData was built with ng=0."
            )
        if field in ("lbx", "ubx") and self._idxbx.size == 0:
            raise ValueError(f"field {field!r} requires a non-empty idxbx.")
        if field in ("lbu", "ubu") and self._idxbu.size == 0:
            raise ValueError(f"field {field!r} requires a non-empty idxbu.")

        expected_shape, lo, hi = self._field_specific_info[field]
        k = self._check_stage(stage, lo, hi, field)
        val = self._batched_value(field, value, expected_shape)
        kernels = self._kernels
        one = self._dtype(1.0)
        minus_one = self._dtype(-1.0)
        B = self._batch_size

        def block(dst, k, row0, col0, src, scale=one, transpose=0):
            rows = int(src.shape[2]) if transpose else int(src.shape[1])
            cols = int(src.shape[1]) if transpose else int(src.shape[2])
            if rows == 0 or cols == 0:
                return
            wp.launch(kernels["set_block"], dim=(B, rows, cols),
                      inputs=[dst, wp.int32(k), wp.int32(row0), wp.int32(col0), src, scale, wp.int32(transpose)],
                      device=self._device)

        def vec(dst, k, col0, src, scale=one):
            if int(src.shape[1]) == 0:
                return
            wp.launch(kernels["set_vec"], dim=(B, int(src.shape[1])),
                      inputs=[dst, wp.int32(k), wp.int32(col0), src, scale],
                      device=self._device)

        if field == "x0":
            vec(self._b, 0, 0, val)
        elif field == "A":
            block(self._A_offdiag, k, 0, 0, val)
        elif field == "B":
            block(self._A_offdiag, k, 0, nx, val)
        elif field == "E":
            block(self._A_diag, k + 1, 0, 0, val, scale=minus_one)
        elif field == "b":
            vec(self._b, k + 1, 0, val, scale=minus_one)
        elif field == "Q":
            block(self._P_diag, k, 0, 0, val)
        elif field == "R":
            block(self._P_diag, k, nx, nx, val)
        elif field == "S":
            block(self._P_diag, k, nx, 0, val)
            block(self._P_diag, k, 0, nx, val, transpose=1)
        elif field == "q":
            vec(self._c, k, 0, val)
        elif field == "r":
            vec(self._c, k, nx, val)
        elif field == "C":
            block(self._G_diag, k, 0, 0, val)
        elif field == "D":
            block(self._G_diag, k, 0, nx, val)
        elif field == "lg":
            vec(self._h_l, k, 0, val)
        elif field == "ug":
            vec(self._h_u, k, 0, val)
        else:
            cols = self._cols_bx if field in ("lbx", "ubx") else self._cols_bu
            dst = self._x_l if field in ("lbx", "lbu") else self._x_u
            wp.launch(kernels["set_vec_cols"], dim=(B, int(cols.shape[0])),
                      inputs=[dst, wp.int32(k), cols, val],
                      device=self._device)

    def _batched_value(self, field: str, value: wp.array, expected_shape: tuple) -> wp.array:
        """``(B, ...)`` view of a field value: the value itself when batched,
        or a zero-stride batch view of an unbatched value."""
        shape = tuple(value.shape)
        if shape == expected_shape:
            return value
        if shape == expected_shape[1:]:
            return batch_broadcast_view(value, self._batch_size)
        raise ValueError(
            f"field {field!r} has shape {shape}; expected {expected_shape} "
            f"or {expected_shape[1:]} (unbatched, broadcast across the batch)."
        )

    def _staged(self, x_flat: wp.array) -> wp.array:
        """``(B, N+1, d)`` view of a flat ``(B, n)`` primal solution (zero-copy)."""
        x_flat = as_warp_array(x_flat, "x")
        itemsize = wp.types.type_size_in_bytes(x_flat.dtype)
        return strided_view(
            x_flat, (self._batch_size, self._N + 1, self._d),
            (x_flat.strides[0], self._d * itemsize, itemsize)
        )

    def state_traj(self, x_flat) -> wp.array:
        """State trajectory ``(B, N+1, nx)`` from a flat primal solution."""
        return self._staged(x_flat)[:, :, :self._nx]

    def input_traj(self, x_flat) -> wp.array:
        """Input trajectory ``(B, N, nu)`` (the dummy ``u_N`` is dropped)."""
        return self._staged(x_flat)[:, :self._N, self._nx:]

    def state(self, x_flat, stage: int) -> wp.array:
        """Stage state ``(B, nx)`` for ``stage = 0..N``."""
        k = self._check_stage(stage, 0, self._N, "x")
        return self._staged(x_flat)[:, k, :self._nx]

    def input(self, x_flat, stage: int) -> wp.array:
        """Stage input ``(B, nu)`` for ``stage = 0..N-1`` (``u_N`` is a dummy)."""
        k = self._check_stage(stage, 0, self._N - 1, "u")
        return self._staged(x_flat)[:, k, self._nx:]

    @staticmethod
    def _check_stage(stage: int, lb: int, ub: int, field: str) -> int:
        if not isinstance(stage, (int, np.integer)):
            raise TypeError(
                f"stage must be an integer for field {field!r}; "
                f"got {type(stage).__name__}."
            )
        if stage < lb or stage > ub:
            raise ValueError(
                f"stage {stage} out of range for field {field!r}; expected {lb}..{ub}."
            )
        return int(stage)
