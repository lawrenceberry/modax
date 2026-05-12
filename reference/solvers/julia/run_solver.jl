using CUDA
using DiffEqGPU
using JSON
using OrdinaryDiffEq
using SciMLBase
using StaticArrays

include("common.jl")
include("../../systems/julia/registry.jl")

# Why we don't IMEX on Julia EnsembleGPUArray:
#
# DiffEqGPU 3.13's `EnsembleGPUArray` only ships a `generate_problem`
# method for ODEFunction (not SplitFunction), so passing a
# SplitODEProblem crashes immediately with
# `FieldError: type SplitFunction has no field f`.
#
# A type-piracy override of `generate_problem` that wraps `f.f1`/`f.f2`
# in separate GPU kernels gets further but trips deeper integration
# issues: OrdinaryDiffEq's W-matrix / Jacobian setup paths in
# OrdinaryDiffEqDifferentiation still reach into `f.f1.f` and `f.f2.f`
# expecting the *original* Julia closures (for autodiff / linearity
# detection / Wfact fallback), so even when both halves are GPU-wrapped
# the JVPCache/derivative paths fail compiling for Dual-typed `t` on the
# GPU.
#
# Patching this end-to-end would require overriding several
# OrdinaryDiffEqDifferentiation entry points too — out of scope.
#
# Practical fallback: for `kencarp5 + EnsembleGPUArray`, pass the
# combined full ODEProblem instead of the split problem.  Julia's
# KenCarp5 then treats the whole RHS implicitly (no IMEX advantage),
# which is *slower* than the IMEX path my JAX solver uses but is the
# only thing DiffEqGPU 3.13 actually runs.  The comparison is therefore
# labelled `julia kencarp5 (fully-implicit)` in scripts 11/12/14.

function parse_first_step(value::String)
    return value == "none" ? nothing : parse(Float64, value)
end

function make_solver_algorithm(solver_name::String, ensemble_backend::String)
    if ensemble_backend == "EnsembleGPUArray"
        if solver_name == "tsit5"
            return Tsit5()
        elseif solver_name == "kencarp5"
            return KenCarp5()
        elseif solver_name == "rodas5"
            return Rodas5()
        elseif solver_name == "kvaerno5"
            return Kvaerno5()
        end
    elseif ensemble_backend == "EnsembleGPUKernel"
        if solver_name == "tsit5"
            return GPUTsit5()
        elseif solver_name == "rodas5"
            return GPURodas5P()
        elseif solver_name == "kvaerno5"
            return GPUKvaerno5()
        end
    end
    error("Unsupported solver/backend combination: $(solver_name) + $(ensemble_backend)")
end

function make_ensemble_algorithm(ensemble_backend::String)
    if ensemble_backend == "EnsembleGPUArray"
        return EnsembleGPUArray(CUDA.CUDABackend(), 0.0)
    elseif ensemble_backend == "EnsembleGPUKernel"
        return EnsembleGPUKernel(CUDA.CUDABackend(), 0.0)
    end
    error("Unknown ensemble backend '$ensemble_backend'")
end

function solve_ensemble(ensemble_prob, alg, ensemble_alg, solve_kwargs, first_step)
    if first_step === nothing
        return SciMLBase.solve(ensemble_prob, alg, ensemble_alg; solve_kwargs...)
    end
    return SciMLBase.solve(ensemble_prob, alg, ensemble_alg; solve_kwargs..., dt=first_step)
end

const SOLVE_WARMUP_RUNS = 2
const SOLVE_TIMED_RUNS = 3

function make_problem(spec::ReferenceSystemSpec, solver_name::String, ensemble_backend::String, y0, tspan, p0)
    if ensemble_backend == "EnsembleGPUArray"
        # See the "Why we don't IMEX on Julia EnsembleGPUArray" comment
        # at the top of this file. kencarp5 + EnsembleGPUArray is run on
        # the *full* (non-split) ODEProblem; Julia treats the RHS as
        # fully implicit. The comparison in scripts/11,12,14 labels this
        # row accordingly.
        return spec.build_array_full_problem(y0, tspan, p0)
    end
    return spec.build_kernel_full_problem(y0, tspan, p0)
end

function remake_param(params, row_idx::Int, ensemble_backend::String)
    if ensemble_backend == "EnsembleGPUKernel"
        return matrix_row_to_svector(params, row_idx)
    end
    return copy(vec(params[row_idx, :]))
end

function remake_u0(y0_raw, row_idx::Int, ensemble_backend::String)
    if ensemble_backend == "EnsembleGPUKernel"
        return matrix_row_to_svector(y0_raw, row_idx)
    end
    return copy(vec(y0_raw[row_idx, :]))
end

function collect_solution(sol, n_trajectories::Int, n_times::Int, n_vars::Int)
    ys = Array{Float64}(undef, n_trajectories, n_times, n_vars)
    for traj_idx in 1:n_trajectories
        traj = sol.u[traj_idx]
        for time_idx in 1:n_times
            ys[traj_idx, time_idx, :] .= Array(traj.u[time_idx])
        end
    end
    return ys
end

function main(args)
    if length(args) != 16
        error(
            "Usage: run_solver.jl SOLVER SYSTEM BACKEND RTOL ATOL FIRST_STEP MAX_STEPS " *
            "SYSTEM_CONFIG_JSON Y0_BIN Y0_META PARAMS_BIN PARAMS_META TSPAN_BIN TSPAN_META " *
            "YS_BIN YS_META"
        )
    end

    solver_name = args[1]
    system_name = args[2]
    ensemble_backend = args[3]
    rtol = parse(Float64, args[4])
    atol = parse(Float64, args[5])
    first_step = parse_first_step(args[6])
    max_steps = parse(Int, args[7])
    system_config_path = args[8]
    y0_bin = args[9]
    y0_meta = args[10]
    params_bin = args[11]
    params_meta = args[12]
    t_span_bin = args[13]
    t_span_meta = args[14]
    ys_bin = args[15]
    ys_meta = args[16]

    CUDA.allowscalar(false)
    CUDA.functional(true)

    system_config = parse_config(system_config_path)
    y0_raw = read_c_order_array(y0_bin, y0_meta)
    y0_batched = ndims(y0_raw) == 2
    y0 = y0_batched ? vec(y0_raw[1, :]) : vec(y0_raw)
    params = read_c_order_array(params_bin, params_meta)
    t_span = vec(read_c_order_array(t_span_bin, t_span_meta))
    n_trajectories = size(params, 1)
    spec = make_system_spec(system_name, system_config)
    tspan = (Float64(t_span[1]), Float64(t_span[end]))
    p0 = remake_param(params, 1, ensemble_backend)
    prob = make_problem(spec, solver_name, ensemble_backend, y0, tspan, p0)
    ensemble_prob = SciMLBase.EnsembleProblem(
        prob;
        prob_func=(prob, i, repeat) -> SciMLBase.remake(
            prob,
            p=remake_param(params, i, ensemble_backend),
            u0=y0_batched ? remake_u0(y0_raw, i, ensemble_backend) : prob.u0,
        ),
        safetycopy=false,
    )

    solve_kwargs = (
        trajectories=n_trajectories,
        saveat=collect(Float64.(t_span)),
        reltol=rtol,
        abstol=atol,
        maxiters=max_steps,
    )
    alg = make_solver_algorithm(solver_name, ensemble_backend)
    ensemble_alg = make_ensemble_algorithm(ensemble_backend)

    # Burn startup/JIT/first-launch overhead before timing the actual solve.
    # Drop each warmup solution and reclaim CUDA pool memory before the next
    # solve allocates — otherwise the per-stage GPUArray scratch buffers and
    # solution objects from prior iterations stay pinned in the CUDA pool and
    # OOM at high dimensions / large ensembles.
    for _ in 1:SOLVE_WARMUP_RUNS
        warmup_sol = solve_ensemble(
            ensemble_prob, alg, ensemble_alg, solve_kwargs, first_step
        )
        CUDA.synchronize()
        warmup_sol = nothing
        GC.gc()
        CUDA.reclaim()
    end

    solve_time_samples_s = Float64[]
    sol = nothing
    for _ in 1:SOLVE_TIMED_RUNS
        # Free the previous timed solution before allocating the next one so
        # the CUDA pool footprint stays at one solution's worth, not three.
        if sol !== nothing
            sol = nothing
            GC.gc()
            CUDA.reclaim()
        end
        solve_start_ns = time_ns()
        sol = solve_ensemble(ensemble_prob, alg, ensemble_alg, solve_kwargs, first_step)
        CUDA.synchronize()
        push!(solve_time_samples_s, (time_ns() - solve_start_ns) / 1e9)
    end
    solve_time_s = minimum(solve_time_samples_s)

    ys = collect_solution(sol, n_trajectories, length(t_span), length(y0))
    sol = nothing
    GC.gc()
    CUDA.reclaim()
    write_c_order_array(ys_bin, ys_meta, ys)
    JSON.print(
        stdout,
        Dict(
            "status" => "ok",
            "shape" => collect(size(ys)),
            "solver" => solver_name,
            "system" => system_name,
            "backend" => ensemble_backend,
            "solve_time_s" => solve_time_s,
            "solve_time_samples_s" => solve_time_samples_s,
        ),
    )
    return nothing
end

main(ARGS)
