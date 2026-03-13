"""
Julia GPU ensemble benchmark for Robertson equation (paper baseline).

Reproduces the benchmark flow from GPUODEBenchmarks:
- DiffEqGPU.vectorized_solve (fixed dt)
- DiffEqGPU.vectorized_asolve (adaptive dt)

Usage:
    julia benchmarks/robertson_julia.jl [N] [rtol] [atol]

N: number of trajectories (default 8192)
rtol: relative tolerance (default 1e-6)
atol: absolute tolerance (default 1e-8)

Outputs JSON to stdout with minimum benchmark times in milliseconds.
"""

using DiffEqGPU, StaticArrays
using CUDA
using JSON
using Random

numberOfParameters = length(ARGS) >= 1 ? parse(Int64, ARGS[1]) : 8192
const RTOL = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-6
const ATOL = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-8

const SOLVER = GPURosenbrock23()
const SOLVER_NAME = "GPURosenbrock23"

function robertson(u, p, t)
    y1, y2, y3 = u
    k1, k2, k3 = p
    du1 = -k1 * y1 + k3 * y2 * y3
    du2 =  k1 * y1 - k2 * y2^2 - k3 * y2 * y3
    du3 =  k2 * y2^2
    return @SVector [du1, du2, du3]
end

u0 = @SVector [1.0, 0.0, 0.0]
tspan = (0.0, 1e5)
p = @SVector [0.04, 3e7, 1e4]

robertsonProblem = DiffEqGPU.ODEProblem(robertson, u0, tspan, p)
fixed_dt = (tspan[2] - tspan[1]) / 1000  # 1000 steps, matching the Julia paper's fixed dt

rng = MersenneTwister(42)
base = [0.04, 3e7, 1e4]
parameterList = [SVector{3,Float64}(base .* (1.0 .+ 0.1 .* (2.0 .* rand(rng, Float64, 3) .- 1.0))) for _ in 1:numberOfParameters]

prob_func = (prob, i, repeat) -> DiffEqGPU.remake(prob, p = parameterList[i])
ensembleProb = DiffEqGPU.EnsembleProblem(robertsonProblem, prob_func = prob_func)

function min_time_ms(f; repeats = 5)
    best = Inf
    for _ in 1:repeats
        CUDA.synchronize()
        t0 = time_ns()
        f()
        CUDA.synchronize()
        dt_ms = (time_ns() - t0) / 1e6
        best = min(best, dt_ms)
    end
    return best
end

fixed_min_ms = 0.0
adaptive_min_ms = 0.0

# Strict paper path: build per-trajectory problems and use vectorized APIs.
I = 1:numberOfParameters
if ensembleProb.safetycopy
    probs = map(I) do i
        p = ensembleProb.prob_func(deepcopy(ensembleProb.prob), i, 1)
        convert(DiffEqGPU.ImmutableODEProblem, p)
    end
else
    probs = map(I) do i
        p = ensembleProb.prob_func(ensembleProb.prob, i, 1)
        convert(DiffEqGPU.ImmutableODEProblem, p)
    end
end

probs = cu(probs)

fixed_min_ms = min_time_ms() do
    CUDA.@sync DiffEqGPU.vectorized_solve(probs, ensembleProb.prob,
                                          SOLVER;
                                          save_everystep = false,
                                          dt = fixed_dt)
end
adaptive_min_ms = min_time_ms() do
    CUDA.@sync DiffEqGPU.vectorized_asolve(probs, ensembleProb.prob,
                                           SOLVER;
                                           dt = 1e-3,
                                           reltol = RTOL,
                                           abstol = ATOL)
end

result = Dict(
    "solver" => SOLVER_NAME,
    "n_trajectories" => numberOfParameters,
    "fixed_dt" => Dict(
        "dt" => fixed_dt,
        "min_time_ms" => fixed_min_ms,
    ),
    "adaptive_dt" => Dict(
        "dt" => 1e-3,
        "reltol" => RTOL,
        "abstol" => ATOL,
        "min_time_ms" => adaptive_min_ms,
    ),
)
println(JSON.json(result))
