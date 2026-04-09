from typing import Callable, Optional
import functools
import torch


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
            self._result.info.sigma += torch.dot(...)
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
                # PyTorch CUDA graph capture requires a warm-up run
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    fn(self, *args, **kwargs)
                torch.cuda.current_stream().wait_stream(stream)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    fn(self, *args, **kwargs)
                cache[k] = graph

            cache[k].replay()

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
