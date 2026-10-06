"""Setup-time pattern algebra on host (NumPy / SciPy) CSR matrices.

Only the sparsity pattern (``indptr`` / ``indices``) is read; the results are
int32 index arrays that the sparse backend uploads to the device once.
"""
import numpy as np
from scipy.sparse import csr_matrix


def csr_row_indices(indptr: np.ndarray) -> np.ndarray:
    """Row index of every stored entry of a CSR pattern (the COO row array).

    Row ``r`` holds the entries ``indptr[r]:indptr[r + 1]``, so each row index
    is repeated as many times as the row has entries.

    Example
    -------
    ::

        K = [ 5  0  8 ]   indptr  = [0, 2, 3, 5]
            [ 0  3  0 ]   indices = [0, 2, 1, 0, 2]
            [ 2  0  4 ]   data    = [5, 8, 3, 2, 4]   (nnz = 5)

        row_indices = [0, 0, 1, 2, 2]

    i.e. entries 0-1 are in row 0, entry 2 in row 1, entries 3-4
    in row 2.
    """
    indptr = np.asarray(indptr)
    return np.repeat(np.arange(len(indptr) - 1, dtype=np.int32), np.diff(indptr)).astype(np.int32)


def csr_diag_indices(mat: csr_matrix) -> np.ndarray:
    """Positions in ``mat.data`` of the diagonal entries of a square CSR matrix.

    Returns an int32 array of length ``rows``; entry ``i`` is the position of
    ``mat[i, i]``, or -1 when the diagonal entry is not stored.
    """
    assert isinstance(mat, csr_matrix)
    assert mat.shape[0] == mat.shape[1], f"The provided csr_matrix is not square. Got shape: {mat.shape}"
    rows = csr_row_indices(mat.indptr)
    positions = np.flatnonzero(rows == mat.indices)
    diag_idx = np.full(mat.shape[0], -1, dtype=np.int32)
    diag_idx[rows[positions]] = positions
    return diag_idx


def csr_subblock_indices(
    A: csr_matrix,
    B: csr_matrix,
    row_offset: int,
    col_offset: int,
    transa: bool = False
) -> np.ndarray:
    """Return the positions in ``B.data`` of every non-zero of ``A`` (or ``A^T``).

    ``A`` is a CSR matrix placed as a sub-block inside the larger CSR matrix
    ``B`` at offset ``(row_offset, col_offset)``:

    * ``transa=False``: ``A[i, j]`` lives at ``B[row_offset + i, col_offset + j]``.
    * ``transa=True`` : ``A[i, j]`` lives at ``B[row_offset + j, col_offset + i]``
      — i.e. ``A^T`` is placed into ``B``, but we iterate over ``A``'s own
      storage so no transpose is materialized.

    Returns an ``int32`` array of length ``A.nnz`` whose k-th entry is
    the position in ``B.data`` that holds the k-th value of ``A.data``.
    Callers use it for vectorized scatter, e.g.::

        B.data[:, csr_subblock_indices(P, B, 0, 0)]         = P.data
        B.data[:, csr_subblock_indices(A, B, n, 0)]         = A.data          # below diag
        B.data[:, csr_subblock_indices(A, B, 0, n, True)]   = A.data          # A^T above diag

    The algorithm
    -------------
    Every CSR entry at ``(row, col)`` gets a single int64 "fingerprint"::

        fp = row * ncols + col

    Because CSR is row-major with sorted columns within each row, the
    fingerprints of ``B.data`` are **strictly ascending** — so one
    ``np.searchsorted`` finds every target fingerprint in ``O(log nnz)``
    time, vectorized across all of ``A``'s entries. The only part that
    differs between the two modes is how the target fingerprint is
    assembled::

        transa=False:  A_fp = A.indices + col_offset + (row_offset + A_rows)    * ncols
        transa=True :  A_fp = A_rows    + col_offset + (row_offset + A.indices) * ncols

    (A's row and column indices swap roles in the transpose case.)

    Worked example — ``transa=False``
    ---------------------------------
    Let::

        B = [ 5  8  0 ]      B.indices = [0, 1, 1, 0, 2]
            [ 0  3  0 ]      B.indptr  = [0, 2, 3, 5]
            [ 2  0  4 ]      B.data    = [5, 8, 3, 2, 4]

        A = [ 0  3 ]         A.indices = [1, 0]
            [ 2  0 ]         A.indptr  = [0, 1, 2]
                             A.data    = [3, 2]

    placed at ``row_offset = 1, col_offset = 0``. With ``ncols = 3``::

        B_fp = B.indices + B_rows * 3 = [0, 1, 4, 6, 8]         # strictly increasing
        A_fp = A.indices + col_offset + (A_rows + row_offset) * 3
             = [1 + 3, 0 + 6] = [4, 6]

    ``np.searchsorted(B_fp, A_fp) = [2, 3]`` — the positions in ``B.data``
    of ``A``'s two non-zeros (values 3 and 2).

    Worked example — ``transa=True``
    --------------------------------
    Consider a KKT-like ``B`` containing the constraint block both below
    the diagonal and ``A^T`` above it::

        B = [ 5  0  2 ]      B.indices = [0, 2, 1, 2, 0, 1, 2]
            [ 0  3  4 ]      B.indptr  = [0, 2, 4, 7]
            [ 2  4 -1 ]      B.data    = [5, 2, 3, 4, 2, 4, -1]

        A = [ 2  4 ]         A.indices = [0, 1]
                             A.indptr  = [0, 2]
                             A.data    = [2, 4]

    With ``A^T`` placed at ``row_offset = 0, col_offset = 2`` (the
    upper-right block of ``B``)::

        B_fp = [0, 2, 4, 5, 6, 7, 8]                         # same formula
        A_fp = A_rows + col_offset + (row_offset + A.indices) * 3
             = [0, 0] + 2          + (0 + [0, 1])            * 3
             = [2, 5]

    ``np.searchsorted(B_fp, A_fp) = [1, 3]`` — the positions in
    ``B.data`` where ``A[0,0]=2`` and ``A[0,1]=4`` appear transposed
    (as ``B[0,2]`` and ``B[1,2]``).

    Complexity: ``O(A.nnz log B.nnz)`` in vectorized NumPy ops — no Python
    loop over entries.
    """
    if A.nnz == 0:
        return np.empty(0, dtype=np.int32)

    # int64 fingerprints: row * ncols + col can exceed the int32 range.
    A_col_indices = A.indices.astype(np.int64)
    A_row_indices = csr_row_indices(A.indptr).astype(np.int64)
    B_col_indices = B.indices.astype(np.int64)
    B_row_indices = csr_row_indices(B.indptr).astype(np.int64)
    B_ncols = B.shape[1]

    B_fingerprint = B_col_indices + B_row_indices * B_ncols

    if transa:
        # Transposed placement: A's row feeds the KKT column, A's col feeds
        # the KKT row.
        A_fingerprint = A_row_indices + col_offset + (row_offset + A_col_indices) * B_ncols
    else:
        A_fingerprint = A_col_indices + col_offset + (row_offset + A_row_indices) * B_ncols

    return np.searchsorted(B_fingerprint, A_fingerprint).astype(np.int32)