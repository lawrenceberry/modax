function make_stiff_scalar_spec(config)
    function ode!(du, u, p, t)
        du[1] = -p[1] * u[1]
        return nothing
    end

    function implicit_ode!(du, u, p, t)
        return ode!(du, u, p, t)
    end

    function explicit_ode!(du, u, p, t)
        du[1] = 0.0
        return nothing
    end

    function jac!(J, u, p, t)
        J[1, 1] = -p[1]
        return nothing
    end

    function kernel_ode(u, p, t)
        return SVector{1, Float64}(-p[1] * u[1])
    end

    function kernel_jac(u, p, t)
        return @SMatrix [-p[1]]
    end

    function kernel_tgrad(u, p, t)
        return SVector{1, eltype(u)}(0.0)
    end

    return ReferenceSystemSpec(
        build_array_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(ode!; jac=jac!, tgrad=zero_tgrad!),
            copy(y0),
            tspan,
            copy(p0),
        ),
        build_array_split_problem=(y0, tspan, p0) -> SciMLBase.SplitODEProblem(
            SciMLBase.ODEFunction(implicit_ode!; jac=jac!, tgrad=zero_tgrad!),
            SciMLBase.ODEFunction(explicit_ode!; jac=make_zero_jac!(1), tgrad=zero_tgrad!),
            copy(y0),
            tspan,
            copy(p0),
        ),
        build_kernel_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem{false}(
            SciMLBase.ODEFunction(kernel_ode; jac=kernel_jac, tgrad=kernel_tgrad),
            vector_to_svector(y0),
            tspan,
            vector_to_svector(p0),
        ),
    )
end
