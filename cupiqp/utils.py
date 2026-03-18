import functools
import cupy as cp
import warp as wp


def cuda_graph_capture(contains_warp_kernels=False):
    """Decorator that caches a method's GPU operations as a CUDA graph.

    On first call (per unique key), captures all GPU operations inside the
    decorated method into a CUDA graph. On subsequent calls with the same key,
    replays the cached graph instead of re-executing the operations.

    Key convention:
        The cache key is read from ``self._key_{method_name}`` at each call.
        The caller must set this attribute (a hashable tuple) before invoking
        the decorated method. Different key values produce separate cached
        graphs, allowing the same method to handle varying buffer layouts or
        control-flow branches.

    Args:
        contains_warp_kernels: if True, creates a warp Stream wrapping the
            capture stream and passes it to the method as the
            ``stream_wp_capture`` keyword argument, so that ``wp.launch()``
            calls inside the method are recorded into the same CUDA graph.

    Example::

        # In the class body:
        @cuda_graph_capture(contains_warp_kernels=False)
        def _calculate_sigma(self):
            self._result.info.sigma[:] = 0.
            self._result.info.sigma += cp.dot(...)
            ...

        # Before calling (e.g. in setup or the caller):
        self._key__calculate_sigma = (self._result.buffer_ptr,)
    """
    def decorator(fn):
        cache_attr = f'_cuda_graph_{fn.__name__}'
        count_attr = f'_cuda_graph_{fn.__name__}_count'

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, {})
                setattr(self, count_attr, 0)

            cache = getattr(self, cache_attr)

            key = getattr(self, f'_key_{fn.__name__}')

            if key not in cache:
                setattr(self, count_attr, getattr(self, count_attr) + 1)
                stream_cp_capture = cp.cuda.Stream(non_blocking=True)
                if contains_warp_kernels:
                    kwargs['stream_wp_capture'] = wp.Stream(cuda_stream=stream_cp_capture.ptr)
                stream_cp_capture.begin_capture()
                with stream_cp_capture:
                    fn(self, *args, **kwargs)
                cache[key] = stream_cp_capture.end_capture()

            cache[key].launch()

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