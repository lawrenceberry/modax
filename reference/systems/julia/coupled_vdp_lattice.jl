function make_coupled_vdp_lattice_spec(config)
    n_osc = require_config_int(config, "n_osc")
    MU = 100.0
    D = 10.0
    OMEGA_SQ = 1.0

    function ode!(du, u, p, t)
        scale = p[1]
        for i in 1:n_osc
            xi = u[2i-1]
            vi = u[2i]
            il = mod(i - 2, n_osc) + 1
            ir = mod(i, n_osc) + 1
            laplacian_i = u[2ir-1] - 2.0 * xi + u[2il-1]
            du[2i-1] = vi
            du[2i] = scale * MU * (1.0 - xi * xi) * vi - OMEGA_SQ * xi + D * laplacian_i
        end
        return nothing
    end

    function jac!(J, u, p, t)
        scale = p[1]
        fill!(J, 0.0)
        for i in 1:n_osc
            xi = u[2i-1]
            vi = u[2i]
            il = mod(i - 2, n_osc) + 1
            ir = mod(i, n_osc) + 1
            # d(dx_i/dt)/dv_i = 1
            J[2i-1, 2i] = 1.0
            # d(dv_i/dt)/dx_i = -2*scale*MU*xi*vi - OMEGA_SQ + D*(-2)
            J[2i, 2i-1] = -2.0 * scale * MU * xi * vi - OMEGA_SQ - 2.0 * D
            # d(dv_i/dt)/dv_i = scale*MU*(1 - xi^2)
            J[2i, 2i] = scale * MU * (1.0 - xi * xi)
            # ring-Laplacian neighbours: d(dv_i/dt)/d(x_ir) = D, d(dv_i/dt)/d(x_il) = D
            J[2i, 2*ir-1] += D
            J[2i, 2*il-1] += D
        end
        return nothing
    end

    function kernel_ode(u::SVector{N, T}, p, t) where {N, T}
        scale = p[1]
        n_osc_local = N ÷ 2
        du = MVector{N, T}(undef)
        for i in 1:n_osc_local
            xi = u[2i-1]
            vi = u[2i]
            il = mod(i - 2, n_osc_local) + 1
            ir = mod(i, n_osc_local) + 1
            laplacian_i = u[2ir-1] - 2.0 * xi + u[2il-1]
            du[2i-1] = vi
            du[2i] = scale * 100.0 * (1.0 - xi * xi) * vi - xi + 10.0 * laplacian_i
        end
        return SVector(du)
    end

    return ReferenceSystemSpec(
        build_array_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(ode!; jac=jac!, tgrad=zero_tgrad!),
            copy(y0),
            tspan,
            copy(p0),
        ),
        build_kernel_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem{false}(
            kernel_ode,
            vector_to_svector(y0),
            tspan,
            vector_to_svector(p0),
        ),
    )
end
