import cupy as cp
from cupyx.scipy.sparse import csr_matrix


def csr_diag_indices(mat: csr_matrix) -> cp.ndarray:
    """Find indicies of diagonal entries within a CSR matrix's data array.

    Returns a cupy int32 array of length ``min(rows, cols)``.
    """
    assert isinstance(mat, csr_matrix)
    assert mat.shape[0] == mat.shape[1], "The provided csr_matrix is not square. Got shape: {mat.shape}"
    indptr = mat.indptr.get()
    indices = mat.indices.get()
    n = mat.shape[0]
    diag_idx = cp.empty(n, dtype=cp.int32)
    for i in range(n):
        for k in range(indptr[i], indptr[i + 1]):
            if indices[k] == i:
                diag_idx[i] = k
                break
    return diag_idx


def csr_row_indices(mat: csr_matrix) -> cp.ndarray:
    """Find row index of every non-zero in a CSR matrix, derived from ``indptr``.

    For entry ``k`` in ``[0, nnz)``, its row is the number of
    row-end markers (``indptr[1:]``) that are ``<=k``. That's
    exactly ``cp.searchsorted(indptr[1:], arange(nnz), 'right')``.

    Example
    -------
    ::

        K = [ 5  0  8 ]   indptr  = [0, 2, 3, 5]
            [ 0  3  0 ]   indices = [0, 2, 1, 0, 2]
            [ 2  0  4 ]   data    = [5, 8, 3, 2, 4]   (nnz = 5)

        row_indices = cp.searchsorted([2, 3, 5], arange(5), 'right')
                    = [0, 0, 1, 2, 2]

    i.e. entries 0-1 are in row 0, entry 2 in row 1, entries 3-4
    in row 2.
    """
    assert isinstance(mat, csr_matrix)
    nnz = mat.nnz
    if nnz == 0:
        return cp.empty(0, dtype=mat.indptr.dtype)
    return cp.searchsorted(mat.indptr[1:], cp.arange(nnz), side='right')


def csr_subblock_indices(
    A: csr_matrix,
    B: csr_matrix,
    row_offset: int,
    col_offset: int,
    transa: bool = False,
) -> cp.ndarray:
    """Return the positions in ``B.data`` of every non-zero of ``A`` (or ``A^T``).

    ``A`` is a CSR matrix placed as a sub-block inside the larger CSR matrix
    ``B`` at offset ``(row_offset, col_offset)``:

    * ``transa=False``: ``A[i, j]`` lives at ``B[row_offset + i, col_offset + j]``.
    * ``transa=True`` : ``A[i, j]`` lives at ``B[row_offset + j, col_offset + i]``
      — i.e. ``A^T`` is placed into ``B``, but we iterate over ``A``'s own
      storage so no transpose is materialized.

    Returns a cupy ``int32`` array of length ``A.nnz`` whose k-th entry is
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
    ``cp.searchsorted`` finds every target fingerprint in ``O(log nnz)``
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

    ``cp.searchsorted(B_fp, A_fp) = [2, 3]`` — the positions in ``B.data``
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

    ``cp.searchsorted(B_fp, A_fp) = [1, 3]`` — the positions in
    ``B.data`` where ``A[0,0]=2`` and ``A[0,1]=4`` appear transposed
    (as ``B[0,2]`` and ``B[1,2]``).

    Complexity: ``O(A.nnz + B.nnz)`` in vectorized cupy ops — no Python
    loop over entries.
    """
    if A.nnz == 0:
        return cp.empty(0, dtype=cp.int32)

    A_col_indices = A.indices
    A_row_indices = csr_row_indices(A)
    B_col_indices = B.indices
    B_row_indices = csr_row_indices(B)
    B_ncols = B.shape[1]

    B_fingerprint = B_col_indices + B_row_indices * B_ncols

    if transa:
        # Transposed placement: A's row feeds the KKT column, A's col feeds
        # the KKT row.
        A_fingerprint = A_row_indices + col_offset + (row_offset + A_col_indices) * B_ncols
    else:
        A_fingerprint = A_col_indices + col_offset + (row_offset + A_row_indices) * B_ncols

    return cp.searchsorted(B_fingerprint, A_fingerprint).astype(cp.int32)