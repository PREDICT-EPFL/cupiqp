"""
CuClarabel benchmark for the chain mass OCP problem.
Called from benchmark_cupiqp_vs_cuclarabel.py with problem data passed via .npz files.

Usage:
    julia --project=<julia_env> bench_cuclarabel.jl <problem.npz> <result.npz>
"""

using Clarabel, SparseArrays, CUDA, CUDA.CUSPARSE
using NPZ

function main()
    problem_file = ARGS[1]
    result_file = ARGS[2]

    data = npzread(problem_file)

    P_colptr = Vector{Int32}(data["P_colptr"])
    P_rowval = Vector{Int32}(data["P_rowval"])
    P_nzval  = Vector{Float64}(data["P_nzval"])
    P_m      = Int(data["P_m"])
    P_n      = Int(data["P_n"])

    A_colptr = Vector{Int32}(data["A_colptr"])
    A_rowval = Vector{Int32}(data["A_rowval"])
    A_nzval  = Vector{Float64}(data["A_nzval"])
    A_m      = Int(data["A_m"])
    A_n      = Int(data["A_n"])

    q = Vector{Float64}(data["q"])
    b = Vector{Float64}(data["b"])

    n_eq   = Int(data["n_eq"])
    n_ineq = Int(data["n_ineq"])
    n_runs = Int(data["n_runs"])
    verbose = Bool(data["verbose"])

    # Build CPU sparse matrices (1-indexed)
    P_csc = SparseMatrixCSC(P_m, P_n, P_colptr, P_rowval, P_nzval)
    A_csc = SparseMatrixCSC(A_m, A_n, A_colptr, A_rowval, A_nzval)

    # Transfer to GPU
    P_gpu = CuSparseMatrixCSR(P_csc)
    A_gpu = CuSparseMatrixCSR(A_csc)
    q_gpu = CuVector{Float64}(q)
    b_gpu = CuVector{Float64}(b)

    # Cone specification
    cones = Dict{String,Any}()
    if n_eq > 0
        cones["f"] = n_eq
    end
    if n_ineq > 0
        cones["l"] = n_ineq
    end

    settings = Clarabel.Settings(
        direct_solve_method = :cudss,
        verbose = verbose,
        iterative_refinement_enable = false
    )

    solver = Clarabel.Solver(P_gpu, q_gpu, A_gpu, b_gpu, cones, settings)

    # Warmup solve
    Clarabel.solve!(solver)

    # Timed runs
    times = Float64[]
    for _ in 1:n_runs
        CUDA.synchronize()
        t = CUDA.@elapsed begin
            Clarabel.solve!(solver)
        end
        push!(times, t * 1e3)  # seconds → ms
    end

    obj = solver.solution.obj_val
    iters = solver.solution.iterations
    status = string(solver.solution.status)

    # Write results (NPZ cannot serialize strings, so write status separately)
    npzwrite(result_file, Dict(
        "times" => times,
        "obj" => [obj],
        "iters" => [iters],
    ))
    status_file = replace(result_file, "_result.npz" => "_status.txt")
    open(status_file, "w") do f
        write(f, status)
    end

    println("CuClarabel done: status=$status, iters=$iters, obj=$obj")
    println("Times (ms): ", times)
end

main()
