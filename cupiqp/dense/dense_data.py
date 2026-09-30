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
             x_l: Optional[wp.array] = None,
             batch_size: Optional[int] = None):
        """Allocate and populate the batched buffers.

        Every input is either **one problem** (``P`` of shape ``(n, n)``,
        vectors ``(k,)``) or a **batch** with a leading batch axis
        (``(B, n, n)``, ``(B, k)``). Single-problem inputs are copied into
        every one of ``batch_size`` problems (``1`` when not given); batched
        inputs must all share the batch size.
        """
        P_src = P
        c_src = c
        if P_src.ndim == 2:
            B = 1 if batch_size is None else int(batch_size)
        elif P_src.ndim == 3:
            B = int(P_src.shape[0])
            if batch_size is not None and int(batch_size) != B:
                raise ValueError(f"P has batch size {B}, but batch_size={batch_size} was given.")
        else:
            raise ValueError(f"P must have shape (n, n) or (B, n, n), got {tuple(P_src.shape)}")
        n = int(P_src.shape[-1])
        if P_src.shape[-2] != n:
            raise ValueError(f"P must be square, got {tuple(P_src.shape)}")
        if c_src.ndim not in (1, 2) or c_src.shape[-1] != n:
            raise ValueError(f"c must have shape ({n},) or ({B}, {n}), got {tuple(c_src.shape)}")
        if c_src.ndim == 2 and c_src.shape[0] != B:
            raise ValueError("Batch size mismatch between P and c.")

        self._batch_size = B
        self._n = n

        self._P = wp.zeros((B, n, n), dtype=self._dtype, device=self._device)
        self._write(self._P, P_src, "P")
        self._c = self._init_vec(c_src, n, "c", B)

        # --- equality constraints ---
        if (A is None) != (b is None):
            raise ValueError("A and b must either both be provided or both be None.")
        if A is not None:
            A_src = A
            if A_src.ndim not in (2, 3):
                raise ValueError(f"A must have shape (p, n) or (B, p, n), got {tuple(A_src.shape)}")
            if A_src.shape[-1] != n:
                raise ValueError("Column mismatch between A and P.")
            p = int(A_src.shape[-2])
            self._A = wp.zeros((B, p, n), dtype=self._dtype, device=self._device)
            self._write(self._A, A_src, "A")
            self._b = self._init_vec(b, p, "b", B)
        else:
            self._A = wp.zeros((B, 0, n), dtype=self._dtype, device=self._device)
            self._b = wp.zeros((B, 0), dtype=self._dtype, device=self._device)

        # --- inequality constraints ---
        if G is not None:
            G_src = G
            if G_src.ndim not in (2, 3):
                raise ValueError(f"G must have shape (m, n) or (B, m, n), got {tuple(G_src.shape)}")
            if G_src.shape[-1] != n:
                raise ValueError("Shape mismatch in G.")
            if h_l is None and h_u is None:
                raise ValueError("Either h_l or h_u must be provided when G is given.")
            m = int(G_src.shape[-2])
            self._G = wp.zeros((B, m, n), dtype=self._dtype, device=self._device)
            self._write(self._G, G_src, "G")
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
