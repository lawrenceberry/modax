"""
Julia GPU ensemble benchmark for Lorenz equation (paper baseline).

Reproduces the benchmark flow from GPUODEBenchmarks:
- DiffEqGPU.vectorized_solve (fixed dt)
- DiffEqGPU.vectorized_asolve (adaptive dt)

Usage:
    julia benchmarks/lorenz_julia.jl [N] [rtol] [atol]

N: number of trajectories (default 8192)
rtol: relative tolerance (default 1e-8)
atol: absolute tolerance (default 1e-8)

Outputs JSON to stdout with minimum benchmark times in milliseconds.
"""

using DiffEqGPU, StaticArrays
using CUDA
using JSON
using SimpleDiffEq

numberOfParameters = length(ARGS) >= 1 ? parse(Int64, ARGS[1]) : 8192
const RTOL = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-8
const ATOL = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-8

const SOLVER = GPUTsit5()
const SOLVER_NAME = "GPUTsit5"

function lorenz(u, p, t)
    du1 = 10.0f0 * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - 2.666f0 * u[3]
    return @SVector [du1, du2, du3]
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 1.0f0)
p = @SArray [21.0f0]

lorenzProblem = DiffEqGPU.ODEProblem(lorenz, u0, tspan, p)
fixed_dt = (tspan[2] - tspan[1]) / 1000  # 1000 steps, matching the Julia paper's fixed dt
parameterList = range(0.0f0, stop = 21.0f0, length = numberOfParameters)
prob_func = (prob, i, repeat) -> DiffEqGPU.remake(prob, p = @SArray [parameterList[i]])
ensembleProb = DiffEqGPU.EnsembleProblem(lorenzProblem, prob_func = prob_func)

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
                                           dt = fixed_dt,
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
        "dt" => fixed_dt,
        "reltol" => RTOL,
        "abstol" => ATOL,
        "min_time_ms" => adaptive_min_ms,
    ),
)
println(JSON.json(result))
