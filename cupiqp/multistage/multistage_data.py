from typing import Optional, Tuple
import warp as wp

from ..data import Data
from ..typedef import CudaArray


# A block-structured matrix: a (diag, offdiag) pair of GPU arrays, either of
# which may be None (zero at setup, unchanged in update). MultistageSolver
# takes any GPU arrays; MultistageData.init receives Warp arrays.
BlockPair = Tuple[Optional[CudaArray], Optional[CudaArray]]


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
    for each batch entry.

    All storage is Warp arrays owned by this object. Block vectors are
    ``(B, num_blocks, rows)`` buffers; the flat ``(B, k)`` views (``c``,
    ``b``, ``h_l``, ...) used by the interior-point iteration alias them.
    """

    def __init__(self, dtype=wp.float64, device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)

    def _adopt_storage(self, P_diag: wp.array, P_offdiag: wp.array, c: wp.array,
                       A_diag: Optional[wp.array], A_offdiag: Optional[wp.array], b: Optional[wp.array],
                       G_diag: Optional[wp.array], G_offdiag: Optional[wp.array],
                       h_u: Optional[wp.array], h_l: Optional[wp.array],
                       x_u: Optional[wp.array], x_l: Optional[wp.array]):
        """Adopt freshly allocated Warp buffers as this object's storage.

        Only :meth:`init` calls this, with buffers it allocated itself; the
        solver later modifies them in place (e.g. the preconditioner scales
        them). Matrix blocks are ``(B, N, rows, cols)``; vectors are
        ``(B, num_blocks, rows)`` and also get flat ``(B, k)`` views
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
        self._b = self._flat(b) if b is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)

        self._G_diag, self._G_offdiag = G_diag, G_offdiag
        self._rows_G = int(G_diag.shape[2]) if G_diag is not None else 0
        self._has_h_l, self._has_h_u = h_l is not None, h_u is not None
        self._h_l_wp, self._h_u_wp = h_l, h_u
        self._h_l = self._flat(h_l) if h_l is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)
        self._h_u = self._flat(h_u) if h_u is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)

        # Box-block presence is structural and fixed here: an omitted bound
        # gets no storage (empty (B, 0)); a provided one is a full (B, n) block.
        self._has_x_l, self._has_x_u = x_l is not None, x_u is not None
        self._x_l_wp, self._x_u_wp = x_l, x_u
        self._x_l = self._flat(x_l) if x_l is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)
        self._x_u = self._flat(x_u) if x_u is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)

        # Shared post-init: bound masks etc., all on the flat (B, k) views.
        self._finalize()

    @staticmethod
    def _flat(vec: wp.array) -> wp.array:
        """``(B, num_blocks, rows)`` -> ``(B, num_blocks*rows)`` view (zero-copy)."""
        B = int(vec.shape[0])
        return vec.reshape((B, int(vec.shape[1]) * int(vec.shape[2])))

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self._n

    @property
    def p(self) -> int:
        return (self._N + 1) * self._rows_A

    @property
    def m(self) -> int:
        return (self._N + 1) * self._rows_G

    @property
    def block_size(self) -> int:
        """Stage variable size ``d``."""
        return self._d

    @property
    def num_blocks(self) -> int:
        """Number of stages ``N`` (diagonal blocks of P)."""
        return self._N

    @property
    def A_rows(self) -> int:
        """Rows per block row of A (``0`` if there are no equalities)."""
        return self._rows_A

    @property
    def G_rows(self) -> int:
        """Rows per block row of G (``0`` if there are no inequalities)."""
        return self._rows_G

    # ------------------------------------------------------------------
    # Construction from arrays
    # ------------------------------------------------------------------

    def init(
        self,
        P: BlockPair,
        c: wp.array,
        A: Optional[BlockPair] = None,
        b: Optional[wp.array] = None,
        G: Optional[BlockPair] = None,
        h_u: Optional[wp.array] = None,
        h_l: Optional[wp.array] = None,
        x_u: Optional[wp.array] = None,
        x_l: Optional[wp.array] = None
    ):
        """Allocate the batched block storage and fill it.

        Matrices are ``(diag, offdiag)`` tuples of GPU arrays and vectors are
        GPU arrays. For one problem the shapes are:

        * ``P``: ``(N, d, d)`` diagonal blocks and ``(N-1, d, d)`` lower
          off-diagonal blocks (``offdiag`` may be ``None`` for zero).
        * ``A`` / ``G``: ``(N, r, d)`` diagonal and ``(N, r, d)`` lower
          off-diagonal blocks, i.e. ``N+1`` block rows (``offdiag`` may be
          ``None`` for zero).
        * ``c``, ``x_l``, ``x_u``: ``(N, d)`` or flat ``(N*d,)``;
          ``b``, ``h_l``, ``h_u``: ``(N+1, r)`` or flat ``((N+1)*r,)``.

        Every array is either **shared** (the shape above, copied into every
        problem) or **batched** (the same with a leading batch axis). ``B`` is
        read from the batched arrays, which must agree. Inputs are Warp arrays of
        this object's dtype (``MultistageSolver`` converts user arrays at its
        public boundary); they are only read, never adopted.
        """
        P_diag, P_off = self._split_pair(P, "P")
        if P_diag.ndim not in (3, 4) or P_diag.shape[-1] != P_diag.shape[-2]:
            raise ValueError(f"P diag must have shape (N, d, d) or (B, N, d, d); got {tuple(P_diag.shape)}.")
        N, d = int(P_diag.shape[-3]), int(P_diag.shape[-1])
        if N < 1:
            raise ValueError("P diag must contain at least one block.")

        pairs = {"P": (P_diag, P_off)}
        rows = {}
        for name, M in (("A", A), ("G", G)):
            if M is None:
                rows[name] = 0
                continue
            M_diag, M_off = self._split_pair(M, name)
            if M_diag.ndim not in (3, 4) or M_diag.shape[-3] != N or M_diag.shape[-1] != d:
                raise ValueError(
                    f"{name} diag must have shape ({N}, r, {d}) or (B, {N}, r, {d}); got {tuple(M_diag.shape)}."
                )
            pairs[name] = (M_diag, M_off)
            rows[name] = int(M_diag.shape[-2])

        # Batch axis of each input: blocks are batched when 4-D; a vector
        # when 3-D, or 2-D other than its block shape (flat and batched).
        vectors = {"c": (c, N, d), "x_u": (x_u, N, d), "x_l": (x_l, N, d),
                   "b": (b, N + 1, rows["A"]), "h_u": (h_u, N + 1, rows["G"]), "h_l": (h_l, N + 1, rows["G"])}
        batch_dims = {}
        for name, (M_diag, M_off) in pairs.items():
            for part, arr in (("diag", M_diag), ("offdiag", M_off)):
                if arr is not None:
                    batch_dims[f"{name} {part}"] = int(arr.shape[0]) if arr.ndim == 4 else None
        for name, (v, nb, r) in vectors.items():
            if v is not None:
                batched = v.ndim == 3 or (v.ndim == 2 and tuple(v.shape) != (nb, r))
                batch_dims[name] = int(v.shape[0]) if batched else None
        B = self._resolve_batch_size(batch_dims)

        def blocks(M_diag, M_off, name, num_off):
            """(diag, offdiag) of a block matrix -> two (B, N, rows, cols) buffers."""
            r, cols = int(M_diag.shape[-2]), int(M_diag.shape[-1])
            diag_wp = wp.zeros((B, N, r, cols), dtype=self._dtype, device=self._device)
            off_wp = wp.zeros((B, num_off, r, cols), dtype=self._dtype, device=self._device)
            self._write_blocks(diag_wp, M_diag, f"{name} diag")
            if M_off is not None:
                self._write_blocks(off_wp, M_off, f"{name} offdiag")
            return diag_wp, off_wp

        def vec(v: Optional[wp.array], name: str, num_blocks: int, r: int) -> Optional[wp.array]:
            if v is None:
                return None
            out = wp.zeros((B, num_blocks, r), dtype=self._dtype, device=self._device)
            self._write_vec(out, v, name)
            return out

        if (A is None) != (b is None):
            raise ValueError("A and b must both be provided or both be None")
        if G is None and (h_u is not None or h_l is not None):
            raise ValueError("h_u and h_l must be None when G is None")
        if G is not None and h_u is None and h_l is None:
            raise ValueError("Either h_l or h_u must be provided when G is given")

        P_diag_wp, P_off_wp = blocks(P_diag, P_off, "P", N - 1)
        A_diag_wp, A_off_wp = blocks(*pairs["A"], "A", N) if A is not None else (None, None)
        G_diag_wp, G_off_wp = blocks(*pairs["G"], "G", N) if G is not None else (None, None)
        self._adopt_storage(
            P_diag_wp, P_off_wp, vec(c, "c", N, d),
            A_diag_wp, A_off_wp, vec(b, "b", N + 1, rows["A"]),
            G_diag_wp, G_off_wp, vec(h_u, "h_u", N + 1, rows["G"]), vec(h_l, "h_l", N + 1, rows["G"]),
            vec(x_u, "x_u", N, d), vec(x_l, "x_l", N, d)
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
    def P(self) -> Tuple[wp.array, wp.array]:
        """``(P_diag, P_offdiag)``, the same layout as the solver inputs."""
        return (self._P_diag, self._P_offdiag)

    @property
    def A(self) -> Optional[Tuple[wp.array, wp.array]]:
        """``(A_diag, A_offdiag)``, or ``None`` if there are no equalities."""
        return (self._A_diag, self._A_offdiag) if self._A_diag is not None else None

    @property
    def G(self) -> Optional[Tuple[wp.array, wp.array]]:
        """``(G_diag, G_offdiag)``, or ``None`` if there are no inequalities."""
        return (self._G_diag, self._G_offdiag) if self._G_diag is not None else None

    # ------------------------------------------------------------------
    # In-place setters: (diag, offdiag) tuples for matrices, arrays for
    # vectors. One device-to-device copy per given array, no allocation.
    # ------------------------------------------------------------------

    def set_P(self, value: BlockPair, check: bool = True):
        diag, off = self._split_pair(value, "P")
        if diag is not None:
            self._write_blocks(self._P_diag, diag, "P diag")
        if off is not None:
            self._write_blocks(self._P_offdiag, off, "P offdiag")

    def set_c(self, value: wp.array, check: bool = True):
        self._write_vec(self._c_wp, value, "c")

    def set_A(self, value: BlockPair, check: bool = True):
        if self._A_diag is None:
            raise ValueError("Cannot set A: no equality block was provided at setup().")
        diag, off = self._split_pair(value, "A")
        if diag is not None:
            self._write_blocks(self._A_diag, diag, "A diag")
        if off is not None:
            self._write_blocks(self._A_offdiag, off, "A offdiag")

    def set_b(self, value: wp.array, check: bool = True):
        if self._b_wp is None:
            raise ValueError("Cannot set b: no equality block was provided at setup().")
        self._write_vec(self._b_wp, value, "b")

    def set_G(self, value: BlockPair, check: bool = True):
        if self._G_diag is None:
            raise ValueError("Cannot set G: no inequality block was provided at setup().")
        diag, off = self._split_pair(value, "G")
        if diag is not None:
            self._write_blocks(self._G_diag, diag, "G diag")
        if off is not None:
            self._write_blocks(self._G_offdiag, off, "G offdiag")

    def set_h_l(self, value: wp.array, check: bool = True):
        if not self._has_h_l:
            raise ValueError(
                "Cannot set h_l: no lower-inequality block was provided at setup(). "
                "Adding an inequality block requires a new setup()."
            )
        self._write_vec(self._h_l_wp, value, "h_l")
        self._update_finite_bound_masks()

    def set_h_u(self, value: wp.array, check: bool = True):
        if not self._has_h_u:
            raise ValueError(
                "Cannot set h_u: no upper-inequality block was provided at setup(). "
                "Adding an inequality block requires a new setup()."
            )
        self._write_vec(self._h_u_wp, value, "h_u")
        self._update_finite_bound_masks()

    def set_x_l(self, value: wp.array, check: bool = True):
        if not self._has_x_l:
            raise ValueError(
                "Cannot set x_l: no lower box-bound block was provided at setup(). "
                "Adding a box-bound block requires a new setup()."
            )
        self._write_vec(self._x_l_wp, value, "x_l")
        self._update_finite_bound_masks()

    def set_x_u(self, value: wp.array, check: bool = True):
        if not self._has_x_u:
            raise ValueError(
                "Cannot set x_u: no upper box-bound block was provided at setup(). "
                "Adding a box-bound block requires a new setup()."
            )
        self._write_vec(self._x_u_wp, value, "x_u")
        self._update_finite_bound_masks()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_pair(value: BlockPair, name: str) -> BlockPair:
        """Unpack a ``(diag, offdiag)`` matrix tuple."""
        if not (isinstance(value, (tuple, list)) and len(value) == 2):
            raise TypeError(
                f"{name} must be a (diag, offdiag) tuple of GPU arrays; either "
                f"entry may be None (zero at setup(), unchanged in update()). "
                f"Got {type(value).__name__}."
            )
        return value[0], value[1]

    def _write_blocks(self, dst: wp.array, value: wp.array, name: str, allow_batched: bool = True):
        """Copy ``value`` into the ``(B, N, rows, cols)`` block buffer in place.

        ``value`` is ``(N, rows, cols)`` (shared by every problem) or, when
        ``allow_batched``, ``(B, N, rows, cols)``.
        """
        self._write(dst, value, name, allow_batched=allow_batched)

    def _write_vec(self, dst: wp.array, value: wp.array, name: str, allow_batched: bool = True):
        """Copy ``value`` into the ``(B, num_blocks, rows)`` vector buffer in place.

        ``value`` is block ``(num_blocks, rows)`` or flat ``(num_blocks*rows,)``
        (shared by every problem) or, when ``allow_batched``, the same with a
        leading batch axis.
        """
        src = value
        B, nb, rows = (int(s) for s in dst.shape)
        K = nb * rows
        shape = tuple(src.shape)
        if shape == (K,):
            src = src.contiguous().reshape((nb, rows))
        elif allow_batched and shape == (B, K):
            src = src.contiguous().reshape((B, nb, rows))
        elif shape not in ((nb, rows), (B, nb, rows)):
            allowed = [(nb, rows), (K,)]
            if allow_batched:
                allowed += [(B, nb, rows), (B, K)]
            raise ValueError(
                f"{name} must have shape {' or '.join(str(s) for s in allowed)}; "
                f"got {shape}."
            )
        self._write(dst, src, name, allow_batched=allow_batched)
