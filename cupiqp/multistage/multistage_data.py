from typing import Optional
import cupy as cp
import warp as wp

from ..data import Data
from ..typedef import PIQP_INF
from ..utils import to_warp_dtype, is_cuda_array
from .multistage_data_kernels import create_broadcast_copy_kernels


class MultistageData(Data):
    """
    Multistage QP data with block-structured matrices, natively batched.

    Stored in condensed QP form::

        min  0.5 * x^T P x + c^T x
        s.t. A x = b
             h_l <= G x <= h_u
             x_l <= x <= x_u

    where P is block-tridiagonal, and A and G are block-bidiagonal. This
    structure comes from an underlying multistage (optimal-control) problem
    over stages ``i = 0, ..., N`` -- equation (1) of Schwan et al.,
    "Exploiting Multistage Optimization Structure in Proximal Solvers"
    (arXiv:2503.12664), specialized to the case with no global variables::

        min   sum_{i=0}^{N-1} l_i(x_i, x_{i+1})  +  l_N(x_N)
        s.t.  A_i x_i + B_i x_{i+1} = b_i,     i = 0, ..., N-1
              A_N x_N = b_N
              C_i x_i + D_i x_{i+1} <= h_i,    i = 0, ..., N-1
              C_N x_N <= h_N

    with per-stage costs::

        l_i(x_i, x_{i+1}) = 0.5 x_i^T Q_i x_i + x_{i+1}^T S_i x_i + c_i^T x_i
        l_N(x_N)          = 0.5 x_N^T Q_N x_N + c_N^T x_N

    Stacking the stage variables ``x = (x_0, ..., x_N)`` recovers the
    condensed QP above: the stage Hessians ``Q_i`` become the diagonal
    blocks of P and the coupling terms ``S_i`` its lower off-diagonal blocks
    (so P is block-tridiagonal); the ``c_i`` stack into c; and the stage
    constraint matrices form the block-bidiagonal A (diagonal blocks ``A_i``,
    lower off-diagonal blocks ``B_i``) and G (diagonal blocks ``C_i``, lower
    off-diagonal blocks ``D_i``). The solver additionally supports two-sided
    inequality bounds ``h_l <= G x <= h_u`` and box bounds
    ``x_l <= x <= x_u`` on top of the one-sided form above.

    A batch of such QPs is stored together: every matrix and vector carries
    a leading batch axis, and the same formulation is solved independently
    for each batch entry. ``batch_size`` defaults to 1 so the single-QP API
    is unchanged at the user level.

    All flat CuPy views (``_c``, ``_b``, ``_h_l``, ...) are zero-copy DLPack
    views of the underlying Warp buffers and have shape ``(B, k)``. Updating
    the Warp buffer (e.g. via ``set_c``) automatically makes the flat view
    reflect the new values.
    """

    def __init__(self, dtype=cp.float64, device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)

    def _adopt_storage(self, P_diag, P_offdiag, c, A_diag, A_offdiag, b,
                       G_diag, G_offdiag, h_u, h_l, x_u, x_l):
        """Adopt freshly allocated Warp buffers as this object's storage.

        Only :meth:`init` calls this, with buffers it allocated itself; the
        solver later modifies them in place (e.g. the preconditioner scales
        them). Matrix blocks are ``(B, N, rows, cols)``; vectors are
        ``(B, num_blocks, rows)`` and also get flat ``(B, k)`` cupy views
        (zero-copy) for the shared IPM code.
        """
        B, N, d = (int(v) for v in P_diag.shape[:3])
        self._batch_size = B
        self._n = N * d
        self._N, self._d = N, d

        self._P_diag, self._P_offdiag = P_diag, P_offdiag
        self._c_wp = c
        self._c = self._flat(c)

        self._A_diag, self._A_offdiag = A_diag, A_offdiag
        self._rows_A = int(A_diag.shape[2]) if A_diag is not None else 0
        self._b_wp = b
        self._b = self._flat(b) if b is not None else cp.zeros((B, 0), dtype=self._dtype)

        self._G_diag, self._G_offdiag = G_diag, G_offdiag
        self._rows_G = int(G_diag.shape[2]) if G_diag is not None else 0
        self._has_h_l, self._has_h_u = h_l is not None, h_u is not None
        self._h_l_wp, self._h_u_wp = h_l, h_u
        self._h_l = self._flat(h_l) if h_l is not None else cp.zeros((B, 0), dtype=self._dtype)
        self._h_u = self._flat(h_u) if h_u is not None else cp.zeros((B, 0), dtype=self._dtype)

        # Box-block presence is structural and fixed here: an omitted bound
        # gets no storage (empty (B, 0)); a provided one is a full (B, n) block.
        self._has_x_l, self._has_x_u = x_l is not None, x_u is not None
        self._x_l_wp, self._x_u_wp = x_l, x_u
        self._x_l = self._flat(x_l) if x_l is not None else cp.zeros((B, 0), dtype=self._dtype)
        self._x_u = self._flat(x_u) if x_u is not None else cp.zeros((B, 0), dtype=self._dtype)

        # Shared post-init: bound masks etc., all on the flat (B, k) views.
        self._finalize()
        self._x_b_scaling = cp.ones((B, self._n), dtype=self._dtype)

    @staticmethod
    def _flat(vec: wp.array) -> cp.ndarray:
        """``(B, num_blocks, rows)`` Warp -> ``(B, num_blocks*rows)`` cupy (zero-copy)."""
        return cp.from_dlpack(wp.to_dlpack(vec)).reshape(vec.shape[0], -1)

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def n(self):
        return self._n

    @property
    def p(self):
        return (self._N + 1) * self._rows_A

    @property
    def m(self):
        return (self._N + 1) * self._rows_G

    @property
    def block_size(self):
        """Stage variable size ``d``."""
        return self._d

    @property
    def num_blocks(self):
        """Number of stages ``N`` (diagonal blocks of P)."""
        return self._N

    @property
    def A_rows(self):
        """Rows per block row of A (``0`` if there are no equalities)."""
        return self._rows_A

    @property
    def G_rows(self):
        """Rows per block row of G (``0`` if there are no inequalities)."""
        return self._rows_G

    def extract_P_diag(self, diag_P: cp.ndarray):
        """Extract the diagonal of every batch's P into ``diag_P`` of shape ``(B, n)``."""
        B = self._batch_size
        d = self.block_size
        N = self.num_blocks
        # (B, N, d, d) Warp buffer, viewed with cupy.
        P_D = cp.from_dlpack(wp.to_dlpack(self._P_diag))
        # cp.diagonal over axes 2, 3 gives (B, N, d) -> reshape to (B, N*d).
        diag_P[:] = cp.diagonal(P_D, axis1=2, axis2=3).reshape(B, N * d)

    # ------------------------------------------------------------------
    # Construction from arrays
    # ------------------------------------------------------------------

    def init(
        self,
        batch_size: int,
        P,
        c,
        A=None,
        b=None,
        G=None,
        h_u=None,
        h_l=None,
        x_u=None,
        x_l=None,
    ):
        """Allocate the batched block storage and fill it from ONE template problem.

        Matrices are ``(diag, offdiag)`` tuples of GPU arrays and vectors are
        GPU arrays, all describing a single problem (no batch axis):

        * ``P``: ``(N, d, d)`` diagonal blocks and ``(N-1, d, d)`` lower
          off-diagonal blocks (``offdiag`` may be ``None`` for zero).
        * ``A`` / ``G``: ``(N, r, d)`` diagonal and ``(N, r, d)`` lower
          off-diagonal blocks, i.e. ``N+1`` block rows (``offdiag`` may be
          ``None`` for zero).
        * ``c``, ``x_l``, ``x_u``: ``(N, d)`` or flat ``(N*d,)``;
          ``b``, ``h_l``, ``h_u``: ``(N+1, r)`` or flat ``((N+1)*r,)``.

        Any array supporting DLPack (Warp, cupy, CUDA torch, JAX, ...) of this
        object's dtype is accepted. The storage is Warp arrays owned by this
        object; the template values are copied into all ``batch_size``
        problems, and the given arrays are only read, never adopted.
        """
        B = int(batch_size)
        wp_dtype = self._wp_dtype
        # Compiled here, eagerly: init() runs inside solver setup().
        self._broadcast_3d_to_4d, self._broadcast_2d_to_3d = create_broadcast_copy_kernels(wp_dtype)

        P_diag, P_off = self._split_pair(P, "P")
        D = self._as_warp(P_diag, "P diag")
        if D.ndim != 3 or D.shape[1] != D.shape[2]:
            raise ValueError(f"P diag must have shape (N, d, d); got {tuple(D.shape)}.")
        N, d = int(D.shape[0]), int(D.shape[1])
        if N < 1:
            raise ValueError("P diag must contain at least one block.")
        zeros = lambda *shape: wp.zeros(shape, dtype=wp_dtype, device=self._device)

        P_diag_wp = zeros(B, N, d, d)
        P_off_wp = zeros(B, N - 1, d, d)
        self._write_blocks(P_diag_wp, D, "P diag", allow_batched=False)
        if P_off is not None:
            self._write_blocks(P_off_wp, P_off, "P offdiag", allow_batched=False)

        c_wp = zeros(B, N, d)
        self._write_vec(c_wp, c, "c", allow_batched=False)

        def bidiag(M, name):
            """(diag, offdiag) of a block lower-bidiagonal matrix -> two (B, N, r, d) buffers."""
            M_diag, M_off = self._split_pair(M, name)
            Md = self._as_warp(M_diag, f"{name} diag")
            if Md.ndim != 3 or Md.shape[0] != N or Md.shape[2] != d:
                raise ValueError(
                    f"{name} diag must have shape ({N}, r, {d}); got {tuple(Md.shape)}."
                )
            r = int(Md.shape[1])
            diag_wp, off_wp = zeros(B, N, r, d), zeros(B, N, r, d)
            self._write_blocks(diag_wp, Md, f"{name} diag", allow_batched=False)
            if M_off is not None:
                self._write_blocks(off_wp, M_off, f"{name} offdiag", allow_batched=False)
            return diag_wp, off_wp

        def vec(v, name, num_blocks, rows):
            if v is None:
                return None
            out = zeros(B, num_blocks, rows)
            self._write_vec(out, v, name, allow_batched=False)
            return out

        if (A is None) != (b is None):
            raise ValueError("A and b must both be provided or both be None")
        A_diag_wp, A_off_wp = bidiag(A, "A") if A is not None else (None, None)
        r_a = int(A_diag_wp.shape[2]) if A_diag_wp is not None else 0
        G_diag_wp, G_off_wp = bidiag(G, "G") if G is not None else (None, None)
        r_g = int(G_diag_wp.shape[2]) if G_diag_wp is not None else 0
        if G is None and (h_u is not None or h_l is not None):
            raise ValueError("h_u and h_l must be None when G is None")
        if G is not None and h_u is None and h_l is None:
            raise ValueError("Either h_l or h_u must be provided when G is given")

        self._adopt_storage(
            P_diag_wp, P_off_wp, c_wp,
            A_diag_wp, A_off_wp, vec(b, "b", N + 1, r_a),
            G_diag_wp, G_off_wp, vec(h_u, "h_u", N + 1, r_g), vec(h_l, "h_l", N + 1, r_g),
            vec(x_u, "x_u", N, d), vec(x_l, "x_l", N, d),
        )

    # ------------------------------------------------------------------
    # Public block storage (Warp arrays, owned by this object)
    # ------------------------------------------------------------------

    @property
    def P_diag(self) -> wp.array:
        """Diagonal blocks of P, a ``(B, N, d, d)`` Warp array."""
        return self._P_diag

    @property
    def P_offdiag(self) -> wp.array:
        """Lower off-diagonal blocks of P, a ``(B, N-1, d, d)`` Warp array."""
        return self._P_offdiag

    @property
    def A_diag(self) -> Optional[wp.array]:
        """Diagonal blocks of A, a ``(B, N, r, d)`` Warp array (``None`` if absent)."""
        return self._A_diag

    @property
    def A_offdiag(self) -> Optional[wp.array]:
        """Lower off-diagonal blocks of A, a ``(B, N, r, d)`` Warp array (``None`` if absent)."""
        return self._A_offdiag

    @property
    def G_diag(self) -> Optional[wp.array]:
        """Diagonal blocks of G, a ``(B, N, r, d)`` Warp array (``None`` if absent)."""
        return self._G_diag

    @property
    def G_offdiag(self) -> Optional[wp.array]:
        """Lower off-diagonal blocks of G, a ``(B, N, r, d)`` Warp array (``None`` if absent)."""
        return self._G_offdiag

    @property
    def P(self):
        """``(P_diag, P_offdiag)``, the same layout as the solver inputs."""
        return (self._P_diag, self._P_offdiag)

    @property
    def A(self):
        """``(A_diag, A_offdiag)``, or ``None`` if there are no equalities."""
        return (self._A_diag, self._A_offdiag) if self._A_diag is not None else None

    @property
    def G(self):
        """``(G_diag, G_offdiag)``, or ``None`` if there are no inequalities."""
        return (self._G_diag, self._G_offdiag) if self._G_diag is not None else None

    # ------------------------------------------------------------------
    # In-place setters: (diag, offdiag) tuples for matrices, arrays for
    # vectors. One device-to-device copy per given array, no allocation.
    # ------------------------------------------------------------------

    def set_P(self, value, check: bool = True):
        diag, off = self._split_pair(value, "P")
        if diag is not None:
            self._write_blocks(self._P_diag, diag, "P diag")
        if off is not None:
            self._write_blocks(self._P_offdiag, off, "P offdiag")

    def set_c(self, value, check: bool = True):
        self._write_vec(self._c_wp, value, "c")

    def set_A(self, value, check: bool = True):
        if self._A_diag is None:
            raise ValueError("Cannot set A: no equality block was provided at setup().")
        diag, off = self._split_pair(value, "A")
        if diag is not None:
            self._write_blocks(self._A_diag, diag, "A diag")
        if off is not None:
            self._write_blocks(self._A_offdiag, off, "A offdiag")

    def set_b(self, value, check: bool = True):
        if self._b_wp is None:
            raise ValueError("Cannot set b: no equality block was provided at setup().")
        self._write_vec(self._b_wp, value, "b")

    def set_G(self, value, check: bool = True):
        if self._G_diag is None:
            raise ValueError("Cannot set G: no inequality block was provided at setup().")
        diag, off = self._split_pair(value, "G")
        if diag is not None:
            self._write_blocks(self._G_diag, diag, "G diag")
        if off is not None:
            self._write_blocks(self._G_offdiag, off, "G offdiag")

    def set_h_l(self, value, check: bool = True):
        if not self._has_h_l:
            raise ValueError(
                "Cannot set h_l: no lower-inequality block was provided at setup(). "
                "Adding an inequality block requires a new setup()."
            )
        self._write_vec(self._h_l_wp, value, "h_l")
        self._update_finite_bound_masks()

    def set_h_u(self, value, check: bool = True):
        if not self._has_h_u:
            raise ValueError(
                "Cannot set h_u: no upper-inequality block was provided at setup(). "
                "Adding an inequality block requires a new setup()."
            )
        self._write_vec(self._h_u_wp, value, "h_u")
        self._update_finite_bound_masks()

    def set_x_l(self, value, check: bool = True):
        if not self._has_x_l:
            raise ValueError(
                "Cannot set x_l: no lower box-bound block was provided at setup(). "
                "Adding a box-bound block requires a new setup()."
            )
        self._write_vec(self._x_l_wp, value, "x_l")
        self._update_finite_bound_masks()

    def set_x_u(self, value, check: bool = True):
        if not self._has_x_u:
            raise ValueError(
                "Cannot set x_u: no upper box-bound block was provided at setup(). "
                "Adding a box-bound block requires a new setup()."
            )
        self._write_vec(self._x_u_wp, value, "x_u")
        self._update_finite_bound_masks()

    # ------------------------------------------------------------------
    # Write helpers (Warp only)
    # ------------------------------------------------------------------

    @property
    def _wp_dtype(self):
        return to_warp_dtype(self._dtype)

    @staticmethod
    def _stream() -> wp.Stream:
        # All solver work runs on the current CuPy stream; write there too.
        return wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)

    @staticmethod
    def _split_pair(value, name: str):
        """Unpack a ``(diag, offdiag)`` matrix tuple."""
        if not (isinstance(value, (tuple, list)) and len(value) == 2):
            raise TypeError(
                f"{name} must be a (diag, offdiag) tuple of GPU arrays; either "
                f"entry may be None (zero at setup(), unchanged in update()). "
                f"Got {type(value).__name__}."
            )
        return value[0], value[1]

    def _as_warp(self, value, name: str) -> wp.array:
        """Zero-copy Warp view of a GPU array of this object's dtype."""
        if isinstance(value, wp.array):
            arr = value
        else:
            if not is_cuda_array(value):
                raise TypeError(
                    f"{name} must be a GPU array (Warp, cupy, CUDA torch, JAX, ...); "
                    f"got {type(value).__name__}. Host arrays are not copied implicitly."
                )
            try:
                arr = wp.from_dlpack(value)
            except Exception as e:
                raise TypeError(
                    f"{name} must support DLPack to be read by the multistage "
                    f"backend; got {type(value).__name__}."
                ) from e
        if arr.dtype != self._wp_dtype:
            raise TypeError(
                f"{name} has dtype {arr.dtype.__name__}; this solver uses "
                f"{self._wp_dtype.__name__}. Pass arrays of the solver dtype."
            )
        return arr

    def _write_blocks(self, dst: wp.array, value, name: str, allow_batched: bool = True):
        """Copy ``value`` into the ``(B, N, rows, cols)`` block buffer in place.

        ``value`` is ``(N, rows, cols)`` (shared by every problem) or, when
        ``allow_batched``, ``(B, N, rows, cols)``.
        """
        src = self._as_warp(value, name)
        full, single = tuple(dst.shape), tuple(dst.shape[1:])
        if allow_batched and tuple(src.shape) == full:
            wp.copy(dst, src, stream=self._stream())
        elif tuple(src.shape) == single:
            if dst.size > 0:
                wp.launch(self._broadcast_3d_to_4d, dim=full, inputs=[src, dst],
                          device=dst.device, stream=self._stream())
        else:
            shapes = (full, single) if allow_batched else (single,)
            raise ValueError(
                f"{name} must have shape {' or '.join(str(s) for s in shapes)}; "
                f"got {tuple(src.shape)}."
            )

    def _write_vec(self, dst: wp.array, value, name: str, allow_batched: bool = True):
        """Copy ``value`` into the ``(B, num_blocks, rows)`` vector buffer in place.

        ``value`` is block ``(num_blocks, rows)`` or flat ``(num_blocks*rows,)``
        (shared by every problem) or, when ``allow_batched``, the same with a
        leading batch axis.
        """
        src = self._as_warp(value, name)
        B, nb, rows = (int(s) for s in dst.shape)
        K = nb * rows
        shape = tuple(src.shape)
        if shape in ((K,), (nb, rows)):
            if shape == (K,):
                src = src.contiguous().reshape((nb, rows))
            if dst.size > 0:
                wp.launch(self._broadcast_2d_to_3d, dim=(B, nb, rows), inputs=[src, dst],
                          device=dst.device, stream=self._stream())
        elif allow_batched and shape in ((B, K), (B, nb, rows)):
            if shape == (B, K):
                src = src.contiguous().reshape((B, nb, rows))
            wp.copy(dst, src, stream=self._stream())
        else:
            allowed = [(nb, rows), (K,)]
            if allow_batched:
                allowed += [(B, nb, rows), (B, K)]
            raise ValueError(
                f"{name} must have shape {' or '.join(str(s) for s in allowed)}; "
                f"got {shape}."
            )
