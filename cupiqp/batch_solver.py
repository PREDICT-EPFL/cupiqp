import copy
import threading
from typing import Optional

import numpy as np
import torch
from cuda.core import Device as CudaCoreDevice

from .solver import SolverBase
from .settings import Settings
from .results import Status


class BatchSolver:
    """Solve N independent QPs concurrently on separate CUDA streams.

    Each sub-solver is a fully independent ``SolverBase`` that runs on its
    own non-blocking CUDA stream and owns its own mutable workspace.  QPs in
    the batch may have different dimensions, sparsity patterns, and KKT
    backends.

    Uses persistent worker threads (one per solver) so that cuDSS
    thread-local state created during ``setup()`` is available during
    ``solve()``.

    CUDA graph capture is disabled for all sub-solvers (graphs replay on
    their capture stream, which would prevent true multi-stream concurrency).
    """

    def __init__(self, num_solvers: int, settings: Optional[Settings] = None):
        self._num_solvers = num_solvers
        self._device_id = torch.cuda.current_device()
        self._streams = [torch.cuda.Stream() for _ in range(num_solvers)]
        self._solvers = []
        base_settings = settings or Settings()
        for stream in self._streams:
            solver = SolverBase(stream=stream)
            solver.settings = copy.deepcopy(base_settings)
            solver.settings.enable_cuda_graph = False
            self._solvers.append(solver)

        # Persistent worker threads — one per solver.
        # Guarantees thread affinity: setup() and solve() for solver[i]
        # always run on the same thread, preserving cuDSS thread-local state.
        self._task = [None] * num_solvers          # callable to execute
        self._task_ready = [threading.Event() for _ in range(num_solvers)]
        self._task_done = [threading.Event() for _ in range(num_solvers)]
        self._results = [None] * num_solvers       # return values
        self._shutdown = False
        self._workers = []
        for i in range(num_solvers):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self._workers.append(t)

    # -- persistent worker loop ------------------------------------------------

    def _worker(self, idx: int):
        """Worker loop for solver *idx*: init device, wait for tasks."""
        torch.cuda.set_device(self._device_id)
        CudaCoreDevice(self._device_id).set_current()
        # Workaround for nvmath bug: _tls.size_written is only initialized
        # on the thread that first imports cudss_data_ifc.  Replicate it here
        # so that cuDSS factorize/solve work in worker threads.
        from nvmath.sparse._internal import cudss_data_ifc
        if not hasattr(cudss_data_ifc._tls, "size_written"):
            cudss_data_ifc._tls.size_written = np.empty((1,), dtype=np.uint64)
        while True:
            self._task_ready[idx].wait()
            self._task_ready[idx].clear()
            if self._shutdown:
                return
            fn = self._task[idx]
            if fn is not None:
                self._results[idx] = fn()
            self._task_done[idx].set()

    def _submit(self, idx: int, fn):
        """Submit *fn* to solver *idx*'s worker and wait for completion."""
        self._task[idx] = fn
        self._task_done[idx].clear()
        self._task_ready[idx].set()
        self._task_done[idx].wait()
        return self._results[idx]

    def _submit_all(self, fns: list):
        """Submit one callable per solver in parallel, wait for all."""
        for i, fn in enumerate(fns):
            self._task[i] = fn
            self._task_done[i].clear()
        for i in range(len(fns)):
            self._task_ready[i].set()
        for i in range(len(fns)):
            self._task_done[i].wait()
        return [self._results[i] for i in range(len(fns))]

    # -- public API ------------------------------------------------------------

    def setup(self, i, P, c, A, b, G, h_u, h_l, x_u, x_l):
        """Set up the *i*-th QP on its dedicated stream and worker thread."""
        self._submit(i, lambda: self._solvers[i].setup(P, c, A, b, G, h_u, h_l, x_u, x_l))

    def update(self, i, **kwargs):
        """Update problem data for the *i*-th QP on its dedicated stream."""
        self._submit(i, lambda: self._solvers[i].update(**kwargs))

    def solve(self):
        """Solve all QPs concurrently and return a list of :class:`Status`.

        Each solve runs on its own non-blocking CUDA stream in a persistent
        worker thread.
        """
        fns = [lambda i=i: self._solvers[i].solve() for i in range(self._num_solvers)]
        statuses = self._submit_all(fns)
        for s in self._streams:
            s.synchronize()
        return statuses

    @property
    def results(self):
        """List of :class:`Result` objects, one per sub-solver."""
        return [s.result for s in self._solvers]

    @property
    def solvers(self):
        """Direct access to the underlying ``SolverBase`` instances."""
        return self._solvers

    @property
    def streams(self):
        """Dedicated CUDA streams, one per sub-solver."""
        return self._streams

    # -- resource cleanup ------------------------------------------------------

    def close(self):
        """Shut down worker threads."""
        self._shutdown = True
        for event in self._task_ready:
            event.set()
        for t in self._workers:
            t.join(timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        self.close()
