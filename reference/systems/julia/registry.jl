include("coupled_vdp_lattice.jl")
include("heat_equation.jl")
include("bateman.jl")
include("lorenz.jl")
include("robertson.jl")
include("kaps.jl")
include("brusselator.jl")

function make_system_spec(system_name::String, config)
    if system_name == "coupled_vdp_lattice"
        return make_coupled_vdp_lattice_spec(config)
    elseif system_name == "heat_equation"
        return make_heat_equation_spec(config)
    elseif system_name == "bateman"
        return make_bateman_spec(config)
    elseif system_name == "lorenz"
        return make_lorenz_spec(config)
    elseif system_name == "robertson"
        return make_robertson_spec(config)
    elseif system_name == "kaps"
        return make_kaps_spec(config)
    elseif system_name == "brusselator"
        return make_brusselator_spec(config)
    end
    error("Unknown reference ODE system '$system_name'")
end
