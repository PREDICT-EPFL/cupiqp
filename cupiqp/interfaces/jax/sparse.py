"""The sparse QP solver of the JAX interface."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import jax
import numpy as np
import warp as wp

from ...sparse.sparse_solver import SparseSolver as _CoreSparseSolver
from ...utils import is_cuda_array
from .base import JaxSolverBase, check_constraint_blocks
from .types import QPSolution


def normalize_csr_index_array(name: str, part: str, value: Any) -> np.ndarray:
    """``value`` as a 1-D int32 NumPy array.

    Patterns are static structure read once on the host, so host arrays
    (NumPy, lists) are expected. A concrete JAX array is copied to the host
    by JAX; other GPU arrays (CuPy, Warp, torch) are rejected rather than
    converted through a provider-specific path."""
    if isinstance(value, jax.core.Tracer):
        raise TypeError(
            f"{name} {part} is a traced JAX value. Sparsity patterns are static "
            "structure: create the solver outside jax.jit, from concrete arrays."
        )
    if not isinstance(value, jax.Array) and is_cuda_array(value):
        raise TypeError(
            f"{name} {part} must be a host array (e.g. NumPy); got {type(value).__name__}. "
            "Copy device patterns to the host first, e.g. cupy.asnumpy(M.indptr)."
        )
    array = np.asarray(value)
    if array.dtype not in (np.int32, np.int64):
        raise TypeError(f"{name} {part} must be int32 or int64; got {array.dtype}.")
    if array.ndim != 1:
        raise ValueError(f"{name} {part} must be 1-D; got shape {array.shape}.")
    if array.dtype == np.int64 and array.size and array.max() >= 2 ** 31:
        raise ValueError(f"{name} {part} holds values beyond int32.")
    return np.ascontiguousarray(array, dtype=np.int32)


def validate_csr_pattern(name: str, value: Any, cols: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Validated CSR pattern of ``value``, a pair ``(indptr, indices)`` of
    1-D integer arrays. ``cols`` is the number of columns (``None``: as
    many as rows)."""
    if not (isinstance(value, (tuple, list)) and len(value) == 2):
        raise TypeError(
            f"{name} must be a CSR pattern (indptr, indices); got {type(value).__name__}. "
            f"For a SciPy csr_matrix M pass (M.indptr, M.indices)."
        )
    indptr, indices = value
    indptr = normalize_csr_index_array(name, "indptr", indptr)
    indices = normalize_csr_index_array(name, "indices", indices)
    rows, nnz = len(indptr) - 1, len(indices)
    cols = rows if cols is None else cols
    if rows < 0:
        raise ValueError(f"{name} indptr must have at least one entry.")
    if indptr[0] != 0 or indptr[-1] != nnz or np.any(np.diff(indptr) < 0):
        raise ValueError(f"{name} indptr must start at 0, never decrease, and end at nnz = {nnz}.")
    if nnz and (indices.min() < 0 or indices.max() >= cols):
        raise ValueError(f"{name} column indices must lie in [0, {cols}).")
    row = np.repeat(np.arange(rows), np.diff(indptr))
    same_row = row[1:] == row[:-1]
    if np.any(np.diff(indices)[same_row] <= 0):
        raise ValueError(
            f"{name} column indices must be strictly increasing within each row "
            "(sorted, no duplicates); call sort_indices() on a SciPy matrix first."
        )
    return indptr, indices


class SparseSolver(JaxSolverBase):
    r"""A sparse convex QP as a pure JAX function, solved on the GPU by the
    core :class:`cupiqp.SparseSolver`. This is ``cupiqp.jax.SparseSolver``:
    it takes and returns JAX arrays, and :meth:`solve` is a pure function.

    It solves the same QP as :class:`DenseSolver`,

    $$
    \begin{aligned}
    \min_{x}\quad & \tfrac{1}{2}\, x^\top P x + c^\top x \\
    \text{s.t.}\quad & A x = b, \\
    & h_l \le G x \le h_u, \\
    & x_l \le x \le x_u,
    \end{aligned}
    $$

    with ``P``, ``A`` and ``G`` stored in CSR form. The sparsity patterns
    are part of the structure: they are given once, to the constructor, as
    the ``(indptr, indices)`` of each matrix, and every call of
    :meth:`solve` takes only the **stored values** of each matrix, in the
    order of its pattern. This splits the core's CSR triple
    ``(indptr, indices, values)`` into what is fixed and what changes. Changing a pattern means creating
    a new solver, which also means a new symbolic analysis of the KKT system
    in the core, so build the solver outside your loop.

    Everything else is as for :class:`DenseSolver`: values are shared or
    batched (``(nnz,)`` or ``(B, nnz)`` for a matrix, ``(k,)`` or ``(B, k)``
    for a vector), the dtype and :attr:`settings` are held by the solver
    object, :meth:`solve` works eagerly and under ``jax.jit`` and
    ``jax.vmap``, failures are reported in ``solution.status``, and
    gradients flow to every array argument (for a matrix, to its stored
    values).

    Parameters
    ----------
    P : (indptr, indices)
        CSR pattern of the ``(n, n)`` quadratic cost: two 1-D ``int32`` or
        ``int64`` host arrays, e.g. NumPy (for a SciPy ``csr_matrix`` ``M``,
        ``(M.indptr, M.indices)``). Concrete JAX arrays are accepted too;
        other GPU arrays are not. Store the full
        symmetric matrix (both triangles). Column indices must be sorted within each row, without
        duplicates. Every entry that may be nonzero in *any* problem must be
        stored; explicit zeros are fine. Required.
    A : (indptr, indices), optional
        CSR pattern of the ``(p, n)`` equality constraint matrix. Give it when
        :meth:`solve` will take ``A`` and ``b``.
    G : (indptr, indices), optional
        CSR pattern of the ``(m, n)`` inequality constraint matrix. Give it when
        :meth:`solve` will take ``G`` and at least one of ``h_l``, ``h_u``.
    dtype : jax.numpy.float64 or jax.numpy.float32, default: float64
        Floating-point precision of the solve; see :class:`DenseSolver`.
    settings : cupiqp.Settings, optional
        Solver configuration; see :class:`DenseSolver`.

    Examples
    --------
    ```python
    import jax, jax.numpy as jnp
    import scipy.sparse as sp
    jax.config.update("jax_enable_x64", True)
    from cupiqp.jax import SparseSolver

    P = sp.csr_matrix(sp.diags([2.0, 1.0, 2.0]))     # (3, 3) with 3 stored entries
    A = sp.csr_matrix([[1.0, 1.0, 1.0]])             # (1, 3) with 3 stored entries
    solver = SparseSolver(P=(P.indptr, P.indices),   # patterns fixed here
                          A=(A.indptr, A.indices))

    sol = solver.solve(jnp.asarray(P.data), jnp.array([1.0, -1.0, 0.0]),
                       A=jnp.asarray(A.data), b=jnp.array([1.0]))

    @jax.jit
    def solve_many(P_values, cs):                    # (B, 3) and (B, 3)
        return solver.solve(P_values, cs, A=jnp.asarray(A.data), b=jnp.array([1.0])).x
    ```

    See Also
    --------
    DenseSolver : the same interface for dense problems, with the full
        description of batching, settings, caching and memory.
    """

    _core_class = _CoreSparseSolver

    def __init__(
        self,
        P: Any,
        A: Any | None = None,
        G: Any | None = None,
        dtype: Any = jax.numpy.float64,
        settings: Any = None,
    ) -> None:
        super().__init__(dtype, settings)
        # The validated patterns as int32 host arrays, and a content hash
        # that identifies them in the signature: two solvers built from
        # equal patterns share one compiled solve and core solver.
        self._csr = {"P": validate_csr_pattern("P", P, None)}
        self._n = len(self._csr["P"][0]) - 1
        self._p = self._m = 0
        if A is not None:
            self._csr["A"] = validate_csr_pattern("A", A, self._n)
            self._p = len(self._csr["A"][0]) - 1
        if G is not None:
            self._csr["G"] = validate_csr_pattern("G", G, self._n)
            self._m = len(self._csr["G"][0]) - 1
        digest = hashlib.sha256()
        for name, (indptr, indices) in self._csr.items():
            digest.update(name.encode())
            digest.update(indptr.tobytes())
            digest.update(indices.tobytes())
        self._pattern_key = (tuple(self._csr), digest.hexdigest())

    def __repr__(self) -> str:
        nnz = ", ".join(f"nnz_{name}={len(indices)}" for name, (_, indices) in self._csr.items())
        return f"SparseSolver(dtype={self.dtype}, n={self._n}, p={self._p}, m={self._m}, {nnz})"

    def _input_rank(self, name: str) -> int:
        return 1                          # matrices are their stored values

    def _setup_arguments(self, inputs: Mapping[str, wp.array]) -> dict[str, Any]:
        """``setup()`` takes each matrix as a CSR triple: the pattern fixed
        at construction plus the values of this call."""
        arguments = dict(inputs)
        for name, (indptr, indices) in self._csr.items():
            arguments[name] = (indptr, indices, inputs[name])
        return arguments

    def _gradient_arrays(self, gradients: Any, inputs: tuple[str, ...]) -> dict[str, wp.array]:
        """Matrix gradients are the gradients of the stored values, in the
        order of the pattern."""
        return {name: (getattr(gradients, name).data if name in self._csr else getattr(gradients, name))
                for name in inputs}

    def solve(
        self,
        P: jax.Array,
        c: jax.Array,
        A: jax.Array | None = None,
        b: jax.Array | None = None,
        G: jax.Array | None = None,
        h_u: jax.Array | None = None,
        h_l: jax.Array | None = None,
        x_u: jax.Array | None = None,
        x_l: jax.Array | None = None,
    ) -> QPSolution:
        """Solve the QP (or the batch of QPs) given by the arrays.

        Parameters
        ----------
        P : jax.Array
            Stored values of the quadratic cost, ``(nnz_P,)`` or
            ``(B, nnz_P)``, in the order of the pattern given to the
            constructor. Required.
        c : jax.Array
            Linear cost, ``(n,)`` or ``(B, n)``. Required.
        A, b : jax.Array, optional
            Stored values of ``A``, ``(nnz_A,)`` or ``(B, nnz_A)``, and
            ``b`` of shape ``(p,)`` or ``(B, p)``. Give both exactly when the
            solver has a pattern for ``A``.
        G, h_l, h_u : jax.Array, optional
            Stored values of ``G``, ``(nnz_G,)`` or ``(B, nnz_G)``, and the
            bounds ``(m,)`` or ``(B, m)``. Give ``G`` exactly when the
            solver has a pattern for it, with at least one bound. Use
            ``-inf`` / ``+inf`` entries for one-sided rows.
        x_l, x_u : jax.Array, optional
            Box bounds ``x_l <= x <= x_u``, ``(n,)`` or ``(B, n)``.

        Returns
        -------
        QPSolution
            The solution, with a leading batch axis if any argument was
            batched. Differentiable with respect to every array argument;
            matrix gradients are the gradients of the stored values.

        Raises
        ------
        TypeError
            If an argument does not have the solver dtype.
        ValueError
            If an argument has the wrong rank or shape (a value array whose
            length is not the pattern's ``nnz``), the batched arguments
            disagree on ``B``, or the given matrices do not match the
            patterns of the solver.
        """
        arguments = dict(P=P, c=c, A=A, b=b, G=G, h_u=h_u, h_l=h_l, x_u=x_u, x_l=x_l)
        given = {name: value for name, value in arguments.items() if value is not None}
        for name in ("P", "A", "G"):
            if name in self._csr and name not in given:
                raise ValueError(f"the solver has a pattern for {name}, so {name} values must be given.")
            if name in given and name not in self._csr:
                raise ValueError(f"the solver has no pattern for {name}, so {name} values must not be given.")
        check_constraint_blocks(A is not None, b is not None, G is not None, h_l is not None, h_u is not None)
        matrix_shapes = {name: (len(indices),) for name, (_, indices) in self._csr.items()}
        return self._compiled(given, self._n, self._p, self._m, matrix_shapes, self._pattern_key).function(*given.values())
