from typing import Any, Optional
import numpy as np
import warp as wp


def to_warp_dtype(dtype: Any):
    """Warp scalar type for a numpy / cupy dtype (or dtype name). Warp types pass through."""
    if isinstance(dtype, type) and issubclass(dtype, (wp.float32, wp.float64, wp.int32, wp.int64, wp.bool)):
        return dtype
    try:
        return wp.dtype_from_numpy(np.dtype(dtype))
    except Exception:
        return dtype


def is_cuda_array(m) -> bool:
    """True iff ``m`` exposes the ``__cuda_array_interface__`` protocol.

    A single, framework-agnostic test for "GPU-resident dense ndarray".
    All of these are accepted:

    * :class:`warp.array` on a CUDA device
    * :class:`cupy.ndarray`
    * dense CUDA :class:`torch.Tensor` (``layout == torch.strided``)
    * JAX CUDA array
    * :class:`numba.cuda.devicearray.DeviceNDArray`

    Anything CPU-only (numpy, CPU torch, CPU JAX) doesn't expose
    ``__cuda_array_interface__`` and is rejected — cupiqp is GPU-only
    and never silently does a host-to-device copy.

    Two robustness guards make this safe to call on user inputs:

    * :class:`torch.sparse_csr_tensor` *defines* ``__cuda_array_interface__``
      as a property that **raises** :class:`RuntimeError` on access (an
      ATen quirk), so we exclude non-strided torch tensors before probing.
    * The ``try``/``except`` around the property access catches errors
      other than :class:`AttributeError` — torch sparse CSR is the
      concrete case we've observed, but any library could plausibly
      define ``__cuda_array_interface__`` as a lazy property that
      raises (e.g., :class:`NotImplementedError` on a CPU backend, or
      :class:`RuntimeError` on a buggy implementation). Treating those
      as "not a CUDA array" is the right behavior.
    """
    if isinstance(m, wp.array):
        return m.device.is_cuda
    try:
        import torch
        if isinstance(m, torch.Tensor) and m.layout != torch.strided:
            return False
    except ImportError:
        pass
    try:
        return m.__cuda_array_interface__ is not None
    except (AttributeError, RuntimeError, NotImplementedError):
        return False


def as_warp_array(value, name: str = "array", dtype=None) -> wp.array:
    """Zero-copy ``warp.array`` view of a GPU array.

    Accepts a ``warp.array`` (returned as is) or any GPU array exposing the
    CUDA Array Interface or DLPack (cupy, CUDA torch, JAX, numba, ...). The
    view shares the caller's memory: it is never copied and its dtype is not
    changed. Host arrays raise ``TypeError``; cupiqp never copies host data to
    the device implicitly.

    The CUDA Array Interface is preferred: it is a plain description of the
    buffer, whereas a DLPack exchange may record stream events, which is not
    allowed while a CUDA graph is being captured.

    With ``dtype`` (a Warp scalar type), an array of any other dtype raises
    ``TypeError``: cupiqp never casts dtypes implicitly. This is the one
    dtype check on the way into a solver; everything below trusts it.
    ``None`` is returned as is, so optional inputs can be passed through
    unconditionally.
    """
    if value is None:
        return None
    if isinstance(value, wp.array):
        if not value.device.is_cuda:
            raise TypeError(f"{name} must be a GPU array; got a warp array on {value.device}.")
        arr = value
    elif is_cuda_array(value):
        arr = wp.array(value, copy=False)
    elif hasattr(value, "__dlpack__"):
        try:
            arr = wp.from_dlpack(value)
        except Exception as e:
            raise TypeError(f"{name} could not be imported through DLPack: {e}") from e
        if not arr.device.is_cuda:
            raise TypeError(f"{name} must be a GPU array; got a {arr.device} array.")
    else:
        raise TypeError(
            f"{name} must be a GPU array (warp, cupy, CUDA torch, JAX, ... - "
            f"any object exposing __cuda_array_interface__ or DLPack); got "
            f"{type(value).__name__}. Host arrays are not copied implicitly."
        )
    
    if dtype is not None and arr.dtype != dtype:
        raise TypeError(
            f"{name} has dtype {arr.dtype.__name__}; this solver uses "
            f"{dtype.__name__}. cuPIQP does not cast dtypes implicitly; "
            f"convert {name} to {dtype.__name__} first."
        )
    return arr


def device_ptr(a) -> int:
    """Device address of a Warp array or of any CUDA Array Interface object."""
    if isinstance(a, wp.array):
        return a.ptr
    return int(a.__cuda_array_interface__["data"][0])


def strided_view(src: wp.array, shape: tuple, strides: tuple) -> wp.array:
    """A non-owning Warp view of ``src`` with an explicit shape and byte strides.

    Used to re-block a flat ``(B, k)`` vector into ``(B, num_blocks, rows)``
    without a copy (also when the vector is a strided column view of a wider
    buffer), and to broadcast one problem over the batch with a zero batch
    stride. The caller keeps ``src`` alive while the view is in use.
    """
    return wp.array(
        ptr=src.ptr, dtype=src.dtype, shape=tuple(int(s) for s in shape),
        strides=tuple(int(s) for s in strides), device=src.device, copy=False,
    )


def batch_broadcast_view(src: wp.array, batch_size: int) -> wp.array:
    """``(B, *src.shape)`` view of ``src`` whose batch stride is zero.

    Every batch entry of the view aliases the same single-problem data, so a
    plain element-wise copy kernel replicates ``src`` into a batched buffer.
    """
    return strided_view(src, (batch_size,) + tuple(src.shape), (0,) + tuple(src.strides))


def print_matlab_format(arr, name=None):
    """
    Print a numpy array in MATLAB format.

    Args:
        arr: numpy array (1D or 2D)
        name: optional name for the array
    """
    if name:
        print(f"{name} = ", end="")

    if arr.ndim == 1:
        # 1D array
        print("[", end="")
        print("; ".join(f"{x:.6f}" for x in arr), end="")
        print("];")
    elif arr.ndim == 2:
        # 2D array
        print("[", end="")
        rows = []
        for i in range(arr.shape[0]):
            row = " ".join(f"{x:.6f}" for x in arr[i])
            rows.append(row)
        print("; \n".join(rows), end="")
        print("];")
    else:
        print("Error: Only 1D and 2D arrays are supported")

def column_slice(arr: wp.array, start: int, stop: Optional[int] = None) -> wp.array:
    """``arr[:, start:stop]`` for a 2-D Warp array, allowing an empty range.

    Warp rejects slices that start at the end of a dimension (and any slice
    of a zero-width dimension); an empty range therefore returns a fresh
    ``(B, 0)`` array of the same dtype instead, which every kernel accepts
    as an absent block.
    """
    B, width = int(arr.shape[0]), int(arr.shape[1])
    stop = width if stop is None else int(stop)
    start = int(start)
    if width == 0 or stop <= start:
        return wp.empty((B, 0), dtype=arr.dtype, device=arr.device)
    return arr[:, start:stop]
