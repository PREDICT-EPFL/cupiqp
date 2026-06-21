from types import MappingProxyType
from typing import Sequence, Union, Literal

import numpy as np
import cupy as cp
import warp as wp

from ..typedef import CudaArray
from ..utils import to_warp_dtype, is_cuda_array
from .multistage_utils import BlockTridiagMat, BlockBidiagMat, BlockVec


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

    ``OcpData`` owns the multistage block containers for a single OCP structure
    (over stages ``k = 0, ..., N`` with stage variable ``y_k = [x_k; u_k]``) and
    lets you fill them field by field with :meth:`set_field`, using HPIPM field
    names.
    It is a *pure data holder*: it does not solve, scale, or clone -- the solver
    (:class:`OcpSolver`) reads :attr:`blocks` and keeps its own scaled copy. This
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
    dtype : {"float64", "float32"}, default: "float64"
    device : str, default: "cuda"
    batch_size : int, default: 1
        Number of OCPs stored together (leading batch axis).
    """

    def __init__(self, N: int, nx: int, nu: int, ng: int = 0,
                 idxbx: Union[int, Sequence[int], None] = None,
                 idxbu: Union[int, Sequence[int], None] = None,
                 dtype: Literal["float32", "float64"] = "float64",
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
        self._cp_dtype = cp.dtype(dtype)

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

        blk_size, N_blk, B = self._d, self._N + 1, self._batch_size
        wp_dtype = to_warp_dtype(dtype)

        # ---- allocate the block containers ----
        self._P_raw = BlockTridiagMat(num_diag_blocks=N_blk, block_size=blk_size,
                            batch_size=B, dtype=wp_dtype, device=device)
        self._c_raw = BlockVec(num_blocks=N_blk, rows=blk_size, batch_size=B, dtype=wp_dtype, device=device)
        self._A_raw = BlockBidiagMat(rows_of_blocks=self._nx, cols_of_blocks=blk_size, N=N_blk,
                           batch_size=B, dtype=wp_dtype, device=device)
        self._b_raw = BlockVec(num_blocks=N_blk + 1, rows=self._nx, batch_size=B, dtype=wp_dtype, device=device)

        self._G_raw = BlockBidiagMat(rows_of_blocks=self._ng, cols_of_blocks=blk_size, N=N_blk,
                            batch_size=B, dtype=wp_dtype, device=device)
        self._h_l_raw = BlockVec(num_blocks=N_blk + 1, rows=self._ng, batch_size=B, dtype=wp_dtype, device=device)
        self._h_u_raw = BlockVec(num_blocks=N_blk + 1, rows=self._ng, batch_size=B, dtype=wp_dtype, device=device)

        self._x_l_raw = BlockVec(num_blocks=N_blk, rows=blk_size, batch_size=B, dtype=wp_dtype, device=device)
        self._x_u_raw = BlockVec(num_blocks=N_blk, rows=blk_size, batch_size=B, dtype=wp_dtype, device=device)

        self._blocks = {
            "P": self._P_raw,
            "c": self._c_raw,
            "A": self._A_raw,
            "b": self._b_raw,
            "G": self._G_raw,
            "h_l": self._h_l_raw,
            "h_u": self._h_u_raw,
            "x_l": self._x_l_raw,
            "x_u": self._x_u_raw,
        }
        self._blocks_view = MappingProxyType(self._blocks)

        # ---- structural defaults ----
        A_D = cp.from_dlpack(wp.to_dlpack(self._A_raw.D))
        eye_nx = cp.eye(self._nx, dtype=self._cp_dtype)
        # initial-condition row 0:  [I, 0] y_0 = x0
        A_D[:, 0, :, :self._nx] = eye_nx
        # stage-coupling rows 1..N: default descriptor E_k = I -> D[k+1] = [-I, 0]
        A_D[:, 1:, :, :self._nx] = -eye_nx

        # box bounds: everything free, declared components set to a finite sentinel
        xl = cp.from_dlpack(wp.to_dlpack(self._x_l_raw.data))
        xu = cp.from_dlpack(wp.to_dlpack(self._x_u_raw.data))
        xl[:] = -cp.inf
        xu[:] = cp.inf
        if self._idxbx.size:
            jx = cp.asarray(self._idxbx)
            xl[:, :, jx] = -_BOUND_SENTINEL
            xu[:, :, jx] = _BOUND_SENTINEL
        if self._idxbu.size:
            ju = cp.asarray(self._nx + self._idxbu)
            # u_N is padding used only to keep a uniform stage block size.
            xl[:, :self._N, ju] = -_BOUND_SENTINEL
            xu[:, :self._N, ju] = _BOUND_SENTINEL

        if self._ng > 0:
            # all ng rows present at stages 0..N (finite sentinel); the trailing
            # padding row stays infinite so the backend drops it.
            hl = cp.from_dlpack(wp.to_dlpack(self._h_l_raw.data))
            hu = cp.from_dlpack(wp.to_dlpack(self._h_u_raw.data))
            hl[:] = -cp.inf
            hu[:] = cp.inf
            hl[:, :N_blk, :] = -_BOUND_SENTINEL
            hu[:, :N_blk, :] = _BOUND_SENTINEL


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
    def blocks(self):
        """Read-only mapping from solver block-group names to containers.

        The mapping structure is immutable. The contained GPU buffers are owned
        by this object and should be modified only through :meth:`set_field`.
        """
        return self._blocks_view

    def set_field(self, field: str, stage: int, value: CudaArray) -> None:
        """Set one block of OCP data at a given stage.

        ``field`` is an HPIPM field name (``A``, ``B``, ``E``, ``b``, ``x0``,
        ``Q``, ``R``, ``S``, ``q``, ``r``, ``C``, ``D``, ``lg``, ``ug``,
        ``lbx``, ``ubx``, ``lbu``, ``ubu``). ``value`` must be a CUDA array;
        cuPIQP is GPU-only and never silently copies host data to the device.

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
        if not is_cuda_array(value):
            raise TypeError(
                f"field {field!r}: value must be a CUDA array; got {type(value)}."
            )
        val = cp.asarray(value)
        self._check_value_shape(field, val, expected_shape)

        if field == "x0":
            cp.from_dlpack(wp.to_dlpack(self._b_raw.data))[:, 0, :] = val
        elif field in ("A", "B", "E", "b"):
            if field == "A":
                cp.from_dlpack(wp.to_dlpack(self._A_raw.E))[:, k, :, :nx] = val
            elif field == "B":
                cp.from_dlpack(wp.to_dlpack(self._A_raw.E))[:, k, :, nx:] = val
            elif field == "E":
                cp.from_dlpack(wp.to_dlpack(self._A_raw.D))[:, k + 1, :, :nx] = -val
            else:
                cp.from_dlpack(wp.to_dlpack(self._b_raw.data))[:, k + 1, :] = -val
        elif field in ("Q", "R", "S"):
            P = cp.from_dlpack(wp.to_dlpack(self._P_raw.D))
            if field == "Q":
                P[:, k, :nx, :nx] = val
            elif field == "R":
                P[:, k, nx:, nx:] = val
            else:
                P[:, k, nx:, :nx] = val
                P[:, k, :nx, nx:] = cp.swapaxes(val, -1, -2)
        elif field in ("q", "r"):
            c = cp.from_dlpack(wp.to_dlpack(self._c_raw.data))
            if field == "q":
                c[:, k, :nx] = val
            else:
                c[:, k, nx:] = val
        elif field in ("C", "D", "lg", "ug"):
            if field == "C":
                cp.from_dlpack(wp.to_dlpack(self._G_raw.D))[:, k, :, :nx] = val
            elif field == "D":
                cp.from_dlpack(wp.to_dlpack(self._G_raw.D))[:, k, :, nx:] = val
            elif field == "lg":
                cp.from_dlpack(wp.to_dlpack(self._h_l_raw.data))[:, k, :] = val
            else:
                cp.from_dlpack(wp.to_dlpack(self._h_u_raw.data))[:, k, :] = val
        else:
            if field in ("lbx", "ubx"):
                cols = cp.asarray(self._idxbx)
            else:
                cols = cp.asarray(nx + self._idxbu)
            if field in ("lbx", "lbu"):
                cp.from_dlpack(wp.to_dlpack(self._x_l_raw.data))[:, k, cols] = val
            else:
                cp.from_dlpack(wp.to_dlpack(self._x_u_raw.data))[:, k, cols] = val

    def _staged(self, x_flat: cp.ndarray) -> cp.ndarray:
        return x_flat.reshape(self._batch_size, self._N + 1, self._d)

    def state_traj(self, x_flat: cp.ndarray) -> cp.ndarray:
        """State trajectory ``(B, N+1, nx)`` from a flat primal solution."""
        return self._staged(x_flat)[:, :, :self._nx]

    def input_traj(self, x_flat: cp.ndarray) -> cp.ndarray:
        """Input trajectory ``(B, N, nu)`` (the dummy ``u_N`` is dropped)."""
        return self._staged(x_flat)[:, :self._N, self._nx:]

    def state(self, x_flat: cp.ndarray, stage: int) -> cp.ndarray:
        """Stage state ``(B, nx)`` for ``stage = 0..N``."""
        k = self._check_stage(stage, 0, self._N, "x")
        return self._staged(x_flat)[:, k, :self._nx]

    def input(self, x_flat: cp.ndarray, stage: int) -> cp.ndarray:
        """Stage input ``(B, nu)`` for ``stage = 0..N-1`` (``u_N`` is a dummy)."""
        k = self._check_stage(stage, 0, self._N - 1, "u")
        return self._staged(x_flat)[:, k, self._nx:]

    def _check_value_shape(self, field: str, value: cp.ndarray, expected_shape: tuple) -> None:
        shape = value.shape
        if shape != expected_shape and shape != expected_shape[1:]:
            raise ValueError(
                f"field {field!r} has shape {shape}; expected {expected_shape} "
                f"or {expected_shape[1:]} (unbatched, broadcast across the batch)."
            )

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
