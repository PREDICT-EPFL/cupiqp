from typing import Optional

import warp as wp

from ..data import Data


class DenseData(Data):
    """Dense data for one or more QPs with identical dimensions and bound structure.

    Storage is Warp arrays owned by this object: matrices ``(B, rows, cols)``
    and vectors ``(B, k)``. Inputs are Warp arrays of this object's dtype
    (``DenseSolver`` converts user arrays at its public boundary) and are
    copied in.
    """
    def __init__(self, dtype="float64", device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)

    def init(self,
             P: wp.array,
             c: wp.array,
             A: Optional[wp.array] = None,
             b: Optional[wp.array] = None,
             G: Optional[wp.array] = None,
             h_u: Optional[wp.array] = None,
             h_l: Optional[wp.array] = None,
             x_u: Optional[wp.array] = None,
             x_l: Optional[wp.array] = None):
        """Allocate and populate the batched buffers.

        Every input is either **shared** - the single-problem shape (``P`` of
        shape ``(n, n)``, vectors ``(k,)``), copied into every problem - or
        **batched**, with a leading batch axis (``(B, n, n)``, ``(B, k)``).
        ``B`` is read from the batched inputs, which must agree.
        """
        if P.ndim not in (2, 3):
            raise ValueError(f"P must have shape (n, n) or (B, n, n), got {tuple(P.shape)}")
        n = int(P.shape[-1])
        if P.shape[-2] != n:
            raise ValueError(f"P must be square, got {tuple(P.shape)}")
        # A matrix is batched when 3-D, a vector when 2-D.
        batch_dims = {}
        for name, v, single_ndim in (("P", P, 2), ("c", c, 1), ("A", A, 2), ("b", b, 1), ("G", G, 2),
                                     ("h_u", h_u, 1), ("h_l", h_l, 1), ("x_u", x_u, 1), ("x_l", x_l, 1)):
            if v is not None:
                batch_dims[name] = int(v.shape[0]) if v.ndim == single_ndim + 1 else None
        B = self._resolve_batch_size(batch_dims)

        self._batch_size = B
        self._n = n

        self._P = wp.zeros((B, n, n), dtype=self._dtype, device=self._device)
        self._write(self._P, P, "P")
        self._c = self._init_vec(c, n, "c", B)

        # --- equality constraints ---
        if (A is None) != (b is None):
            raise ValueError("A and b must either both be provided or both be None.")
        if A is not None:
            if A.ndim not in (2, 3):
                raise ValueError(f"A must have shape (p, n) or (B, p, n), got {tuple(A.shape)}")
            if A.shape[-1] != n:
                raise ValueError("Column mismatch between A and P.")
            p = int(A.shape[-2])
            self._A = wp.zeros((B, p, n), dtype=self._dtype, device=self._device)
            self._write(self._A, A, "A")
            self._b = self._init_vec(b, p, "b", B)
        else:
            self._A = wp.zeros((B, 0, n), dtype=self._dtype, device=self._device)
            self._b = wp.zeros((B, 0), dtype=self._dtype, device=self._device)

        # --- inequality constraints ---
        if G is not None:
            if G.ndim not in (2, 3):
                raise ValueError(f"G must have shape (m, n) or (B, m, n), got {tuple(G.shape)}")
            if G.shape[-1] != n:
                raise ValueError("Shape mismatch in G.")
            if h_l is None and h_u is None:
                raise ValueError("Either h_l or h_u must be provided when G is given.")
            m = int(G.shape[-2])
            self._G = wp.zeros((B, m, n), dtype=self._dtype, device=self._device)
            self._write(self._G, G, "G")
        else:
            if h_u is not None or h_l is not None:
                raise ValueError("h_l and h_u must be None when G is None.")
            m = 0
            self._G = wp.zeros((B, 0, n), dtype=self._dtype, device=self._device)

        # Inequality-block presence is structural and fixed here: an omitted
        # side gets no storage (empty (B, 0)); a provided one is a full (B, m)
        # block. Both omitted is only valid when G is absent (handled above).
        self._has_h_l = h_l is not None
        self._has_h_u = h_u is not None
        self._h_u = self._init_vec(h_u, m, "h_u", B) if h_u is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)
        self._h_l = self._init_vec(h_l, m, "h_l", B) if h_l is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)

        # --- variable bounds ---
        # Box-block presence is structural and fixed here: an omitted bound
        # gets no storage (empty (B, 0)); a provided one is a full (B, n) block.
        self._has_x_l = x_l is not None
        self._has_x_u = x_u is not None
        self._x_u = self._init_vec(x_u, n, "x_u", B) if x_u is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)
        self._x_l = self._init_vec(x_l, n, "x_l", B) if x_l is not None else wp.zeros((B, 0), dtype=self._dtype, device=self._device)

        self._finalize()

    @property
    def P(self) -> wp.array:
        """Quadratic cost, a ``(B, n, n)`` Warp array."""
        return self._P

    @property
    def A(self) -> wp.array:
        """Equality matrix, a ``(B, p, n)`` Warp array (``(B, 0, n)`` without equalities)."""
        return self._A

    @property
    def G(self) -> wp.array:
        """Inequality matrix, a ``(B, m, n)`` Warp array (``(B, 0, n)`` without inequalities)."""
        return self._G

    # ------------------------------------------------------------------
    # In-place setters
    # ------------------------------------------------------------------

    def set_P(self, value: wp.array, check: bool = True):
        self._write(self._P, value, "P")

    def set_c(self, value: wp.array, check: bool = True):
        self._write(self._c, value, "c")

    def set_A(self, value: wp.array, check: bool = True):
        self._write(self._A, value, "A")

    def set_b(self, value: wp.array, check: bool = True):
        self._write(self._b, value, "b")

    def set_G(self, value: wp.array, check: bool = True):
        self._write(self._G, value, "G")

    def set_h_l(self, value: wp.array, check: bool = True):
        self._set_bound(self._h_l, self._has_h_l, value, "h_l", check)

    def set_h_u(self, value: wp.array, check: bool = True):
        self._set_bound(self._h_u, self._has_h_u, value, "h_u", check)

    def set_x_l(self, value: wp.array, check: bool = True):
        self._set_bound(self._x_l, self._has_x_l, value, "x_l", check)

    def set_x_u(self, value: wp.array, check: bool = True):
        self._set_bound(self._x_u, self._has_x_u, value, "x_u", check)
