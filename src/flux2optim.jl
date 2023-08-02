"Aims at replacing FluxOptTools.jl which does not handle having complex and real parameters at the same time. `Flux.destructure` is an alternative but promotes everything to complex data type.
"
module Flux2Optim

import Flux
using TestItems

export params2vec, params2vec!, vec2params!

function params2vec(params)
    U = reduce(promote_type, real(eltype(p)) for p ∈ params)
    v = zeros(U, sum(length(reinterpret(U, p)) for p ∈ params))
    params2vec!(v, params)
    v
end

function params2vec!(v, params)
    i = 1
    for p ∈ params
        s = length(reinterpret(eltype(v), p))
        v[i:i+s-1] = reinterpret(eltype(v), p)
        i += s
    end
end

function vec2params!(params, v)
    i = 1
    for p ∈ params
        s = length(reinterpret(eltype(v), p))
        p .= reshape(reinterpret(eltype(p), v[i:i+s-1]), size(p))
        i += s
    end
end

@testitem "Params ↔ Vec" begin
    using Flux

    struct Model
        a
        b
    end
    Flux.@functor Model

    model = Model(randn(Complex{Float32}, 3, 4), randn(Float32, 2, 5))
    params = Flux.params(model)
    params_copied = deepcopy(params)
    v = params2vec(params)
    map(x -> x .= 0, params)
    vec2params!(params, v)

    @test all(params_copied .== params)
end

end  # Flux2Optim