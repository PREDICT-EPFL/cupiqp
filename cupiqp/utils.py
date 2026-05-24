from typing import Any, Callable, Optional
import functools
import cupy as cp
import numpy as np
import warp as wp


def to_warp_dtype(dtype: Any):
    try:
        return wp.dtype_from_numpy(np.dtype(dtype))
    except Exception:
        return dtype


def is_cuda_array(m) -> bool:
    """True iff ``m`` exposes the ``__cuda_array_interface__`` protocol.

    A single, framework-agnostic test for "GPU-resident dense ndarray".
    All of these are accepted:

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


def cuda_graph_capture(key: Optional[Callable] = None, enable: Optional[Callable] = None):
    """Decorator that caches a method's GPU operations as a CUDA graph.

    On first call (per unique key), captures all GPU operations inside the
    decorated method into a CUDA graph. On subsequent calls with the same key,
    replays the cached graph instead of re-executing the operations.

    Args:
        key: A callable ``(self, *args, **kwargs) -> hashable`` that computes
             the cache key from the method's arguments. Different key values
             produce separate cached graphs.
        enable: A callable ``(self) -> bool`` that determines whether CUDA
             graph capture is enabled at runtime. When it returns False, the
             decorated method is called directly without graph capture/replay.
             Defaults to None (always enabled).

    Example::

        @cuda_graph_capture(key=lambda self: (self._result.buffer_ptr,))
        def _calculate_sigma(self):
            self._result.info.sigma[:] = 0.
            self._result.info.sigma += cp.dot(...)
            ...
    """
    def decorator(fn):
        cache_attr = f'_cuda_graphs_{fn.__name__}'

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if enable is not None and not enable(self):
                return fn(self, *args, **kwargs)

            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, {})

            cache = getattr(self, cache_attr)
            k = key(self, *args, **kwargs) if key is not None else None

            if k not in cache:
                stream = cp.cuda.Stream(non_blocking=True)
                stream.begin_capture()
                with stream:
                    fn(self, *args, **kwargs)
                cache[k] = stream.end_capture()

            cache[k].launch()

        return wrapper

    return decorator


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