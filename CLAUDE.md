# CUPIQP development guidelines

## Project Overview

This repository implements **cuPIQP**, a GPU-accelerated quadratic programming solver based on the PIQP proximal interior-point method. The project focuses on high-performance QP solving on NVIDIA GPUs, with support for sparse QPs, batched QPs, and structure-exploiting multistage optimal control problems.

The main design goal is to keep the solver architecture close to PIQP while introducing GPU-native implementation, with natively designed batched API (batch size is always the leading dimension in memory layouts).


---

## Core Development Principles

- Preserve numerical correctness before optimizing performance.
- Do not change stopping criteria, tolerances, regularization, or scaling behavior unless explicitly requested.
- Avoid unnecessary CPU-GPU synchronization.
- Avoid unnecessary host-device memory transfers.
- Keep data on the GPU whenever possible.
- Keep the CPU and GPU solver paths conceptually aligned when possible.
- Only call the public APIs of a class when it's called from outside of the class. The internal ones (the ones that starts with an underline _*) should only be used within the class.

---

## Solver Architecture

The solver is organized around an interior-point method loop. The algorithmic layer should remain independent of the concrete linear algebra backend.

Typical backend categories:

- Sparse GPU backend using cuDSS / cuSPARSE.
- Dense or batched dense GPU backend using cuBLAS / cuSOLVER / custom kernels.
- Multistage backend exploiting block-tridiagonal or block-tridiagonal-arrow KKT structure.
- CPU backend for reference behavior and numerical comparison.

When modifying the solver, keep the distinction clear between:

- Algorithm-level IPM logic.
- Data layout and variable storage.
- KKT matrix assembly/update.
- KKT factorization and solve.
- Residual and merit computation.
- Iterative refinement.
- Scaling and equilibration.
- CUDA stream and CUDA graph management.

---

## GPU Programming Rules

- Use CuPy for GPU array storage and lightweight NumPy-like operations.
- Use NVIDIA Warp for custom GPU kernels, especially batched small dense block operations.
- Use cuDSS for sparse direct factorization when appropriate.
- Use cuBLAS/cuSOLVER for dense linear algebra where possible.
- Avoid implicit synchronization, including accidental `.get()`, `.item()`, `cp.asnumpy()`, or printing GPU arrays inside hot paths.
- Be careful with operations that allocate memory inside the IPM loop.
- Reuse allocated buffers whenever possible.
- Keep CUDA stream semantics explicit.
- If adding a new GPU operation, make sure it runs on the intended stream.
- Avoid mixing default stream behavior with custom streams unless explicitly handled.
- Be careful when combining CuPy streams and Warp streams.
- CUDA Graph capture should avoid unsupported operations such as certain cuSPARSE calls.
- Do not assume a library call is graph-capturable unless verified.

---

## CUDA Graph Rules

The solver may capture repeated IPM operations with CUDA Graphs to reduce kernel launch overhead.

When modifying graph-captured code:

- Avoid CPU-side branching that changes the sequence of GPU calls after capture.
- Avoid dynamic allocation during capture.
- Avoid unsupported library calls during capture.
- Keep shapes, sparsity patterns, and memory addresses stable across replays.
- Do not introduce operations that synchronize the stream unless necessary.

## Git
- Never commit and push by yourself. Leave it to me!