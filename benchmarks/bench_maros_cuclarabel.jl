"""
Solve a Maros-Meszaros QP with CuClarabel (GPU-accelerated Clarabel).

The .mat files use PIQP format:
    min  0.5 x'Px + c'x
    s.t. Ax = b
         h_l <= Gx <= h_u
         x_l <= x  <= x_u

This script converts to Clarabel's cone form:
    min  0.5 x'Px + q'x
    s.t. Ax + s = b,  s ∈ K

Usage:
    julia bench_maros_cuclarabel.jl <problem_name> <n_runs>

Example:
    julia bench_maros_cuclarabel.jl CVXQP1_S 10
"""

using MAT, Clarabel, SparseArrays, LinearAlgebra, CUDA, Statistics

function load_piqp_mat(path::String)
    data = matread(path)
    P  = sparse(data["P"])
    c  = vec(data["c"])
    A  = sparse(data["A"])
    b  = vec(data["b"])
    G  = sparse(data["G"])
    h_l = vec(data["h_l"])
    h_u = vec(data["h_u"])
    x_l = vec(data["x_l"])
    x_u = vec(data["x_u"])
    return P, c, A, b, G, h_l, h_u, x_l, x_u
end

function piqp_to_clarabel(P, c, A_eq, b_eq, G, h_l, h_u, x_l, x_u)
    n = size(P, 1)
    n_eq = length(b_eq)

    # --- Build inequality rows from G, h_l, h_u ---
    # Upper bounds: Gx <= h_u  (where h_u is finite)
    upper_finite = isfinite.(h_u)
    # Lower bounds: Gx >= h_l  →  -Gx <= -h_l  (where h_l is finite)
    lower_finite = isfinite.(h_l)

    # --- Build bound rows from x_l, x_u ---
    # x <= x_u  →  Ix <= x_u  (where x_u is finite)
    xu_finite = isfinite.(x_u)
    # x >= x_l  →  -Ix <= -x_l  (where x_l is finite)
    xl_finite = isfinite.(x_l)

    I_n = sparse(1.0I, n, n)

    # Stack: [A_eq; G_upper; -G_lower; I_xu; -I_xl]
    rows = SparseMatrixCSC{Float64,Int}[]
    b_parts = Vector{Float64}[]

    # Equalities
    if n_eq > 0
        push!(rows, A_eq)
        push!(b_parts, b_eq)
    end

    # G upper
    n_gu = count(upper_finite)
    if n_gu > 0
        push!(rows, G[upper_finite, :])
        push!(b_parts, h_u[upper_finite])
    end

    # G lower (negated)
    n_gl = count(lower_finite)
    if n_gl > 0
        push!(rows, -G[lower_finite, :])
        push!(b_parts, -h_l[lower_finite])
    end

    # x upper bounds
    n_xu = count(xu_finite)
    if n_xu > 0
        push!(rows, I_n[xu_finite, :])
        push!(b_parts, x_u[xu_finite])
    end

    # x lower bounds (negated)
    n_xl = count(xl_finite)
    if n_xl > 0
        push!(rows, -I_n[xl_finite, :])
        push!(b_parts, -x_l[xl_finite])
    end

    n_ineq = n_gu + n_gl + n_xu + n_xl

    A_clar = vcat(rows...)
    b_clar = vcat(b_parts...)

    # Cones
    cones = Clarabel.SupportedCone[]
    if n_eq > 0
        push!(cones, Clarabel.ZeroConeT(n_eq))
    end
    if n_ineq > 0
        push!(cones, Clarabel.NonnegativeConeT(n_ineq))
    end

    # Clarabel wants upper-triangular P
    P_upper = sparse(triu(P))

    return P_upper, c, A_clar, b_clar, cones
end

function main(; problem_name::String = "", n_runs::Int = 10)
    if isempty(problem_name)
        if length(ARGS) >= 1
            problem_name = ARGS[1]
        else
            println("Usage: julia bench_maros_cuclarabel.jl <problem_name> [n_runs]")
            println("   or: set problem_name in main(problem_name=\"CVXQP1_S\", n_runs=10)")
            return
        end
    end
    if length(ARGS) >= 2
        n_runs = parse(Int, ARGS[2])
    end
    mat_path = joinpath(@__DIR__, "..", "tests", "data", "maros_meszaros", problem_name * ".mat")

    if !isfile(mat_path)
        error("File not found: $mat_path")
    end

    println("Loading $problem_name ...")
    P, c, A_eq, b_eq, G, h_l, h_u, x_l, x_u = load_piqp_mat(mat_path)
    n = size(P, 1)
    println("  n=$n, n_eq=$(length(b_eq)), n_ineq=$(size(G,1)), n_vars_bounded=$(count(isfinite.(x_l) .| isfinite.(x_u)))")

    P_cl, q_cl, A_cl, b_cl, cones = piqp_to_clarabel(P, c, A_eq, b_eq, G, h_l, h_u, x_l, x_u)
    println("  Clarabel: A is $(size(A_cl,1))x$(size(A_cl,2)), cones: $cones")

    # Pass CPU data; Clarabel transfers to GPU internally when cudss is used
    warmup_settings = Clarabel.Settings(
        direct_solve_method = :cudss,
        verbose = false,
    )

    solver = Clarabel.Solver(P_cl, q_cl, A_cl, b_cl, cones, warmup_settings)

    # Warmup solve
    Clarabel.solve!(solver)
    println("Warmup solve finished.")

    # Timed solves
    solver.settings.verbose = false
    times_ms = Float64[]
    for i in 1:n_runs
        CUDA.synchronize()
        t_start = time()
        Clarabel.solve!(solver)
        CUDA.synchronize()
        t_end = time()
        push!(times_ms, (t_end - t_start) * 1e3)
    end

    status = solver.solution.status
    obj = solver.solution.obj_val
    iters = solver.solution.iterations

    println("\n===== Results =====")
    println("Problem:    $problem_name")
    println("Status:     $status")
    println("Objective:  $obj")
    println("Iterations: $iters")
    println("Runs:       $n_runs")
    println("Solve time (ms):")
    println("  Median:   $(round(median(times_ms), digits=3))")
    println("  Mean:     $(round(mean(times_ms), digits=3))")
    println("  Min:      $(round(minimum(times_ms), digits=3))")
    println("  Max:      $(round(maximum(times_ms), digits=3))")
    println("  Std:      $(round(std(times_ms), digits=3))")
end

main()
