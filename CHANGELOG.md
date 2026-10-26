# Changelog

All notable changes to cuPIQP are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - Unreleased

First public release.

### Added

- `DenseSolver`, `SparseSolver` and `MultistageSolver`: GPU proximal
  interior-point QP solvers based on PIQP, for dense QPs, sparse QPs and
  multistage problems such as optimal control problems.
- Native batched solving: `B` independent QPs solved in parallel from one solver
  instance, with the batch as the leading dimension of every input and result.
- Warp-native data: results are `warp.array` objects on the GPU, and inputs can
  be any CUDA array (Warp, CuPy, PyTorch, JAX) through the CUDA Array Interface
  or DLPack.
- Fully GPU-resident, asynchronous solves, run as a single CUDA graph for the
  dense and multistage backends. `update()` and `solve()` can be recorded into
  your own CUDA graph.
- `float64` and `float32` precision.
- Differentiation through the solution with `solver.backward()`.
- PyTorch layers in `cupiqp.torch`: `DenseSolverLayer`, `SparseSolverLayer` and
  `MultistageSolverLayer`.
- JAX interface in `cupiqp.jax`, usable under `jax.jit`, `jax.vmap` and
  `jax.grad`.
- `cuda12` and `cuda13` extras that install the matching NVIDIA runtime
  libraries through nvmath-python, plus `torch` and `jax` extras.

[Unreleased]: https://github.com/PREDICT-EPFL/cupiqp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PREDICT-EPFL/cupiqp/releases/tag/v0.1.0
