"""
Julia GPU ensemble benchmark for 72D CMB-inspired Boltzmann system.

72-dimensional ODE with time-varying stiffness:
- y[1]: clock (dy/dt = 1)
- y[2:37]: 18 non-stiff oscillator pairs (constant damping 0.01)
- y[38:71]: 17 stiff oscillator pairs (damping = p[1] * clock)
- y[72]: single stiff decay mode (rate = p[2] * clock)

Usage:
    julia benchmarks/boltzmann_72d_julia.jl [N] [rtol] [atol]

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

function boltzmann_72d(u, p, t)
    clock = u[1]
    return SVector{72, Float64}(ntuple(Val(72)) do i
        if i == 1
            return one(eltype(u))
        elseif i == 72
            return -p[2] * clock * u[72]
        else
            pair = div(i - 2, 2)  # 0-indexed pair (0..34)
            omega = 2 * π * (pair + 1)
            if pair < 18
                damping = eltype(u)(0.01)
            else
                damping = p[1] * clock
            end
            if iseven(i)
                return -damping * u[i] + omega * u[i + 1]
            else
                return -omega * u[i - 1] - damping * u[i]
            end
        end
    end)
end

u0 = SVector{72, Float64}(ntuple(Val(72)) do i
    if i == 1
        0.0       # clock
    elseif i == 72
        1.0       # decay mode
    elseif iseven(i)
        1.0       # first in pair
    else
        0.0       # second in pair
    end
end)

tspan = (0.0, 10.0)
p = SVector{2, Float64}(10.0, 50.0)

prob = DiffEqGPU.ODEProblem(boltzmann_72d, u0, tspan, p)

rng = MersenneTwister(42)
base = [10.0, 50.0]
parameterList = [SVector{2,Float64}(base .* (1.0 .+ 0.1 .* (2.0 .* rand(rng, Float64, 2) .- 1.0))) for _ in 1:numberOfParameters]

prob_func = (prob, i, repeat) -> DiffEqGPU.remake(prob, p = parameterList[i])
ensembleProb = DiffEqGPU.EnsembleProblem(prob, prob_func = prob_func)

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

adaptive_min_ms = min_time_ms() do
    CUDA.@sync DiffEqGPU.vectorized_asolve(probs, ensembleProb.prob,
                                           SOLVER;
                                           dt = 1e-4,
                                           reltol = RTOL,
                                           abstol = ATOL)
end

result = Dict(
    "solver" => SOLVER_NAME,
    "n_trajectories" => numberOfParameters,
    "adaptive_dt" => Dict(
        "dt" => 1e-4,
        "reltol" => RTOL,
        "abstol" => ATOL,
        "min_time_ms" => adaptive_min_ms,
    ),
)
println(JSON.json(result))
