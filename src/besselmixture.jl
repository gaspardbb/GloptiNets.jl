struct BesselMixtureTrigo{T,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: ObjFunc{T,ℤ}
    constant::T
    coefficients::V  # (n, )
    anchors::M       # (d, n)
    scale::T         # Same variance in all direction

    function BesselMixtureTrigo(constant, coefficients, anchors, scale)
        @assert ndims(coefficients) == 1 && ndims(anchors) == 2
        @assert length(coefficients) == size(anchors, 2)

        new{eltype(coefficients),typeof(coefficients),typeof(anchors)}(constant, coefficients, anchors, scale)
    end
end

constant(f::BesselMixtureTrigo) = f.constant
coefficients(f::BesselMixtureTrigo) = f.coefficients
anchors(f::BesselMixtureTrigo) = f.anchors
scale(f::BesselMixtureTrigo) = f.scale
ncoeffs(f::BesselMixtureTrigo) = size(anchors(f), 2)
dim(f::BesselMixtureTrigo) = size(anchors(f), 1)

Base.convert(::Type{Any}, f::BesselMixtureTrigo) = f
Base.convert(::Type{T}, f::BesselMixtureTrigo) where {T} = BesselMixtureTrigo(
    convert(T, constant(f)),
    convert(Vector{T}, coefficients(f)),
    convert(Matrix{T}, anchors(f)),
    convert(T, scale(f))
)
gpu(f::BesselMixtureTrigo) = BesselMixtureTrigo(
    constant(f),
    CuArray(coefficients(f)),
    CuArray(anchors(f)),
    scale(f)
)
cpu(f::BesselMixtureTrigo) = BesselMixtureTrigo(
    constant(f),
    collect(coefficients(f)),
    collect(anchors(f)),
    scale(f)
)

save(f::BesselMixtureTrigo, filename) = jldsave("$filename.jld2"; constant=constant(f), coefficients=coefficients(f), anchors=anchors(f), scale=scale(f)
)
load(::Type{BesselMixtureTrigo}, filename) = begin
    file = jldopen("$filename.jld2", "r+")
    f = BesselMixtureTrigo(file["constant"], file["coefficients"], file["anchors"], file["scale"])
    close(file)
    f
end

_bessel_kernel(X, Y, s) = exp.(dropdims(sum(
        (cospi.(2(X' .- reshape(
            Y, 1, size(Y)...)
        )) .- one(s)) * s,  # (p, d, n)
        dims=2), dims=2))  # (p, n)

function evaluate!(r::AbstractVector{T}, f::BesselMixtureTrigo{T}, X::AbstractMatrix{T}) where {T}
    # a bit absurd to have r changed inplace while allocating embedding but compliant with poly api. to change. 
    embedding = _bessel_kernel(X, anchors(f), scale(f))
    sum!(reshape(r, 1, :), coefficients(f) .* embedding')
    r .+= constant(f)
    nothing
end
function (f::BesselMixtureTrigo)(X::AbstractMatrix)
    r = zeros(size(X, 2))
    isgpu(f) && (r = CuArray(r))
    evaluate!(r, f, X)
    r
end
(f::BesselMixtureTrigo)(x::AbstractVector) = f([x;;])[1]

function fourier(f::BesselMixtureTrigo{T}, ωs::AbstractMatrix{U}, proba::ApproxBesselSampler{T,ℤ}) where {T,U}
    @assert allequal(proba.scale)
    @assert proba.scale[1] == f.scale
    @assert dim(f) == size(ωs, 1)
    rs = zeros(Complex{T}, size(ωs, 2))

    for (i, ω) ∈ enumerate(eachcol(ωs))
        rs[i] = sum(coefficients(f) .* sum(cispi.(-2anchors(f)' * ω), dims=2)) * prod(proba.weights[j][abs(ωₖ)+1] / _ι.(ℤ, abs(ωₖ)) for (j, ωₖ) ∈ enumerate(ω))

        iszero(ω) && (rs[i] += constant(f))
    end

    rs
end

fourier(f::BesselMixtureTrigo, ωs::AbstractMatrix) = fourier(f, ωs, ApproxBesselSampler(ℤ, scale(f) * ones(dim(f)); tol=1e-10))

@testitem "Fourier Bessel Mixture" begin
    const GN = GloptiNets
    using HCubature

    _approx_fourier(f, ω) = begin
        hcubature(x -> f(x) * cispi(-2ω' * x), zeros(2), ones(2); rtol=1e-5, atol=1e-5)[1]
    end

    f = GloptiNets.BesselMixtureTrigo(randn(), randn(3), randn(2, 3), 2.0)
    ωs = [
        0 1 -1 2 3
        0 0 0 1 -2
    ]
    fourier_comput = fourier(f, ωs)
    fourier_approx = _approx_fourier.(Ref(f), eachcol(ωs))
    @test ≈(fourier_comput, fourier_approx; rtol=1e-5, atol=1e-5)
end

function Hnorm2(f::BesselMixtureTrigo{T}, proba) where {T}
    K = _bessel_kernel(anchors(f), anchors(f), scale(f))
    dot(coefficients(f), K, coefficients(f)) + 2constant(f) * sum(coefficients(f)) + abs2(f.constant) / prod(w[1] for w in proba.weights)
end
Hnorm2(f) = Hnorm2(f, ApproxBesselSampler(ℤ, scale(f) * ones(dim(f)); tol=1e-10))

@testitem "Positive Hnorm2" begin
    proba = ApproxBesselSampler(ℤ, 2.1ones(2); tol=1e-8)
    for _ in 1:10
        f = random_besselmixture(10, 2, 2.1)
        @test Hnorm2(f) > 0
    end
end

function withgradient(f::BesselMixtureTrigo{T}, x::AbstractVector{T}) where {T}
    pw = 2 * (x .- anchors(f))
    embed = exp.(dropdims(sum((cospi.(pw) .- 1) * scale(f), dims=1), dims=1))

    val = constant(f) + embed' * coefficients(f)
    grads = dropdims(sum(-2π * scale(f) * sinpi.(pw) .* embed' .* coefficients(f)', dims=2), dims=2)
    (val=val, grad=(grads,))
end

@testitem "Gradient Bessel Mixture" begin
    f = GloptiNets.BesselMixtureTrigo(randn(), randn(3), randn(2, 3), 2.0)
    xs = randn(2, 3)
    ϵ = 1e-6
    ϵ1, ϵ2 = ϵ * [1, 0], ϵ * [0, 1]

    for x ∈ eachcol(xs)
        (; val, grad) = GloptiNets.withgradient(f, x)
        @test val ≈ f(x)
        @test ≈(grad[1], [(f(x + ϵ1) - f(x - ϵ1)) / 2ϵ, (f(x + ϵ2) - f(x - ϵ2)) / 2ϵ]; atol=1e-4, rtol=1e-4)
    end
end

# TODO: move all the following in a specific abstraction ObjFunc{ℤ} :> AbstractPoly{ℤ, U}
random_besselmixture(ncoeffs, dim, s, ::Type{T}=Float64) where {T} = BesselMixtureTrigo(randn(T,), randn(T, ncoeffs), randn(T, dim, ncoeffs), s)
random_pos_besselmixture(ncoeffs, dim, s, hnorm, ::Type{T}=Float64; ntries_min=100) where {T} =
    let f = random_besselmixture(ncoeffs, dim, s, T)
        f★, _ = estimate_min(f, ntries_min)
        fpos = BesselMixtureTrigo(constant(f) - f★, coefficients(f), anchors(f), scale(f))
        C = √Hnorm2(fpos) / hnorm
        BesselMixtureTrigo(constant(fpos) / C, coefficients(fpos) / C, anchors(fpos), scale(fpos))
    end

"A lower bound on `f` corresponding to the lower bound hierarchy when `g=0`, with the RKHS norm."
function lowerbound(f::BesselMixtureTrigo)
    constant(f) - √dot(coefficients(f), _bessel_kernel(anchors(f), anchors(f), scale(f)), coefficients(f))
end

_candidate_position(f::BesselMixtureTrigo{T}) where {T} = rand(T, dim(f))
_constructor(f::BesselMixtureTrigo, ps...) = BesselMixtureTrigo(ps...)
isgpu(f::BesselMixtureTrigo) = typeof(coefficients(f)) <: CuArray

Base.show(io::IO, f::BesselMixtureTrigo) = print(io,
    """
    $(nameof(typeof(f)))(\
    nc=$(ncoeffs(f)),\
    d=$(dim(f)))\
    """
)