abstract type AbstractBlockPSDModel{T,D<:Domain} end
Base.eltype(g::AbstractBlockPSDModel{T}) where {T} = T

"""
# Notations 
- `b` is the number of blocks 
- `s` is the size of a block
- `r` is the rank of a block
- `d` is the dimension 

# Remarks
The anchors are stored without constraints. Accessing them with `anchors` puts them in the range `(0, 1/2)`.
"""
struct PSDBlockBesselCheby{T<:AbstractFloat,M<:AbstractArray{T,3},V<:AbstractVector{T}} <: AbstractBlockPSDModel{T,ℕ}
    uncons_anchors::M  # d, s, b
    coefficients::M    # s, r, b
    variances::V        # d
    regularization::T

    function PSDBlockBesselCheby(uncons_anchors, coefficients, variances; reg=1e-6)
        @assert size(uncons_anchors, 2) === size(coefficients, 1)
        @assert size(uncons_anchors, 3) === size(coefficients, 3)
        @assert size(uncons_anchors, 1) == size(variances, 1)

        new{eltype(uncons_anchors),typeof(uncons_anchors),typeof(variances)}(uncons_anchors, coefficients, variances, reg)
    end
end
uncons_anchors(g::PSDBlockBesselCheby) = g.uncons_anchors
coefficients(g::PSDBlockBesselCheby) = g.coefficients
variances(g::PSDBlockBesselCheby) = g.variances
regularization(g::PSDBlockBesselCheby) = g.regularization
coefficients_scaled(g::PSDBlockBesselCheby) = reshape(reduce(hcat,  # Use cholesky instead of cholesky! for Zygote support; speed up could be obtained by avoiding new realloc at each call
        cholesky((_K(i, i, g) + regularization(g) * I)).U \ view(coefficients(g), :, :, i) for i ∈ 1:nblocks(g)  # (s, s) \ (s, r) -> (s, r)
    ), blocksize(g), rank(g), nblocks(g))  # s, r, b
blocksize(g::PSDBlockBesselCheby) = size(coefficients(g), 1)
rank(g::PSDBlockBesselCheby) = size(coefficients(g), 2)
nblocks(g::PSDBlockBesselCheby) = size(coefficients(g), 3)
dim(g::PSDBlockBesselCheby) = size(variances(g), 1)

"""
Moves the model to the GPU. 

# Remarks
Only implemented for CUDA. Does not use Flux' @functor, as it mixes trainable parameters and parameter which can be moved to GPU. Easier to do it ourselves.
"""
function gpu(g::AbstractBlockPSDModel) end
gpu(g::PSDBlockBesselCheby) = PSDBlockBesselCheby(
    CuArray(uncons_anchors(g)),
    CuArray(coefficients(g)),
    CuArray(variances(g)),
    reg=regularization(g)
)
cpu(g::PSDBlockBesselCheby) = PSDBlockBesselCheby(
    collect(uncons_anchors(g)),
    collect(coefficients(g)),
    collect(variances(g)),
    reg=regularization(g)
)
trainable_params(g::PSDBlockBesselCheby) = (uncons_anchors(g), coefficients(g))

"""
    anchors(g)

Returns the anchors of the model, in ``(0, 1/2)``.

# Remark
The function ``x ↦ |x - round(x)|`` is equal to ``x ↦ 1/(2π) acos(cos(2π x))``.
"""  # TODO: make custom chain rule?  # PERF: calling anchors(g)[i,j,k] computes all the anchors...
anchors(g::PSDBlockBesselCheby) = abs.(uncons_anchors(g) .- round.(uncons_anchors(g)))
bijacos(x) = abs(x - round(x))

"""
    besselkernel_cheby_cos(X, Y; γ)

Compute the Bessel kernel in the Chebychev basis. It is a r.k on (-1, 1). Evaluates for arguments `x, y` in the torus, i.e. corresponds to `cos 2πx, cos 2πy`, with `x, y ∈ (0, ½)`. Only requirement is that `X, Y, γ` have the same size for the first dimension.

# Arguments
- `X` and `Y` must be in `(0, ½)`. If not, use [`bijacos`](@ref). `γ` must be positive.
"""
function besselkernel_cheby_cos(X::AbstractMatrix{T}, Y::AbstractMatrix{T}; γ::AbstractVector{T}) where {T}
    @assert size(X, 1) == size(Y, 1) == size(γ, 1)
    # @assert all(0 .< X .< 0.5) && all(0 .< Y .< 0.5)
    d, n, m = size(X, 1), size(X, 2), size(Y, 2)
    dropdims(prod((
                exp.(γ .* (cospi.(2(  # + part
                    reshape(X, d, n, 1) .+ reshape(Y, d, 1, m)  # d, n, m
                )) .- 1)) +
                exp.(γ .* (cospi.(2(  # - part 
                    reshape(X, d, n, 1) .- reshape(Y, d, 1, m)  # d, n, m
                )) .- 1))
            ) / 2, dims=1), dims=1)  # n, m
end

function besselkernel_cheby_cos(X::AbstractArray{T}, Y::AbstractArray{T}; γ) where {T}
    @assert ndims(X) ≥ 2 && ndims(Y) ≥ 2
    d = size(X, 1)
    size_X, size_Y = size(X)[2:end], size(Y)[2:end]
    reshape(besselkernel_cheby_cos(
            reshape(X, d, prod(size_X)), reshape(Y, d, prod(size_Y)); γ=γ
        ), size_X..., size_Y...)
end

"""
    evaluate_cos(g, U)

Evaluate the model when composed with cosine, i.e. we require ``U ∈ (0, 1/2)``, and we compute 
```
    g.(cos(2π U))
```
"""
function evaluate_cos(g::PSDBlockBesselCheby{T}, U::AbstractMatrix{T}) where {T}
    @assert size(U, 1) == dim(g)
    # @assert all(0 .<= U .<= 1 / 2)

    embedding = besselkernel_cheby_cos(U, anchors(g); γ=variances(g))  # n, s, b
    dropdims(sum(abs2.(
                batched_mul(embedding, coefficients_scaled(g))  # n, r, b
            ), dims=(2, 3)), dims=(2, 3))  # n,
end

"""
    evaluate(g, X)

Evaluate the model for input ``X ∈ (-1, 1)``.
"""
function evaluate(g::PSDBlockBesselCheby{T}, X::AbstractMatrix{T}) where {T}
    @assert size(X, 1) == dim(g)
    @assert all(-1 .<= X .<= 1)

    evaluate_cos(g, acos.(X) / (2π))
end

# Baseline: 1.218 s (854 allocations: 3.44 MiB)
function cheby_baseline(g::PSDBlockBesselCheby{T}, ωs::AbstractMatrix{U}) where {T,U}
    @assert dim(g) == size(ωs, 1)

    cs = coefficients_scaled(g)
    psdcoeffs = batched_mul(cs, permutedims(cs, (2, 1, 3)))  # s, s, b
    result = zeros(T, size(ωs, 2))
    # Tricky thing is that they are 5 dimensions:
    # - the number of blocks b
    # - the block size in each block: s, s
    # - the dimension d
    # - the number of frequencies to predict n 
    # Certainly a way to make this much faster
    # - use besseli! on a range of values 
    # - make h the inner loop and not the outer one 
    @inbounds for ind ∈ axes(ωs, 2)
        for i ∈ 1:nblocks(g)
            for j ∈ 1:blocksize(g), k ∈ j:blocksize(g)
                z = one(T)
                for l ∈ 1:dim(g)
                    sₗ = variances(g)[l]
                    ω = ωs[l, ind]
                    # Don't call anchors(g), it computes all the anchors 
                    m₊ = (bijacos(g.uncons_anchors[l, j, i]) + bijacos(g.uncons_anchors[l, k, i])) / 2
                    m₋ = (bijacos(g.uncons_anchors[l, j, i]) - bijacos(g.uncons_anchors[l, k, i])) / 2

                    z *= exp(-2sₗ) / 2 * (ω ≠ 0 ? 2 : 1) * (
                             besseli(ω, 2sₗ * cospi(2m₊)) * cospi(2ω * m₋) +
                             besseli(ω, 2sₗ * cospi(2m₋)) * cospi(2ω * m₊)
                         )
                end
                result[ind] += psdcoeffs[j, k, i] * z * (
                                   # The sum is on all the (j, k) ∈ 1:s so we count the outer diagonal twice    
                                   j == k ? one(T) : 2one(T)
                               )
            end
        end
    end
    result
end

function cheby(
    g::PSDBlockBesselCheby{T}, ωs::AbstractMatrix{U}) where {T,U<:Integer}
    out = Vector{T}(undef, size(ωs, 2))
    cheby!(out, g, ωs)
    out
end

#  40.967 ms (786 allocations: 2.80 MiB)
"""
Computes the Chebychev coefficient in-place. For each `(i, j, k, l) ∈ (1:nblocks, 1:blocksize, 1:blocksize, 1:dim)`, precomputes all the possible Bessel function and cosine values, in order to scale well with the number of frequencies `N`.
"""
function cheby!(
    out::Vector{T}, g::PSDBlockBesselCheby{T}, ωs::AbstractMatrix{U}) where {T,U<:Integer}
    @assert dim(g) == size(ωs, 1)

    d, s, N = dim(g), blocksize(g), size(ωs, 2)
    g_mat = Matrix{T}(undef, s, s)
    r_scaled = Matrix{T}(undef, s, rank(g))
    out .= zero(T)

    ωs_max = maximum(ωs, dims=2)
    out_prod = Vector{T}(undef, N)
    out_bessel₋ = Vector{T}(undef, maximum(ωs_max) + 1)
    out_bessel₊ = Vector{T}(undef, maximum(ωs_max) + 1)
    out_cospi₋ = Vector{T}(undef, maximum(ωs_max) + 1)
    out_cospi₊ = Vector{T}(undef, maximum(ωs_max) + 1)

    for i = 1:nblocks(g)
        # g_mat ≈ coefficients_scaled(g)[:, :, i] * coefficients_scaled(g)[:, :, i]'
        ldiv!(r_scaled, cholesky((_K(i, i, g) + regularization(g) * I)).U, view(coefficients(g), :, :, i))
        mul!(g_mat, r_scaled, r_scaled')
        for j ∈ 1:blocksize(g), k ∈ j:blocksize(g)
            # Nothing in this loop allocates
            out_prod .= one(T)
            for l ∈ 1:dim(g)
                sₗ = g.variances[l]
                m₊ = (bijacos(g.uncons_anchors[l, j, i]) + bijacos(g.uncons_anchors[l, k, i]))
                m₋ = (bijacos(g.uncons_anchors[l, j, i]) - bijacos(g.uncons_anchors[l, k, i]))

                @views besseli!(out_bessel₋[1:ωs_max[l]+1], 0:ωs_max[l], 2sₗ * cospi(m₋))
                @views besseli!(out_bessel₊[1:ωs_max[l]+1], 0:ωs_max[l], 2sₗ * cospi(m₊))

                @views out_cospi₊[1:ωs_max[l]+1] .= cospi.(-(0:ωs_max[l]) .* m₊)
                @views out_cospi₋[1:ωs_max[l]+1] .= cospi.(-(0:ωs_max[l]) .* m₋)

                v = exp(-2sₗ) / 2
                for h ∈ eachindex(out_prod)
                    idx = ωs[l, h] + 1
                    out_prod[h] *= (
                        v * (
                            out_bessel₊[idx] * out_cospi₋[idx] +
                            out_bessel₋[idx] * out_cospi₊[idx]
                        ) * (idx ≠ 1 ? 2one(T) : one(T))
                    )
                end
            end
            v = g_mat[j, k] * (j ≠ k ? 2one(T) : one(T))
            @. out = muladd(out_prod, v, out)
        end
    end
end

@testitem "Cheby coefficient" begin
    using QuadGK

    h(x, s) = exp(s * (cos(2π * x) - 1))
    u(x) = acos(x) / 2π
    K(x, y, s) = (h(u(x) + u(y), s) + h(u(x) - u(y), s)) / 2

    "Computes Chebychev coefficient with numerical integration"
    function _cheby(g, ω)
        r = 0
        cs = coefficients_scaled(g)
        for i ∈ 1:nblocks(g)
            for j ∈ 1:blocksize(g), k ∈ 1:blocksize(g)
                z = one(eltype(cs))
                for l ∈ 1:dim(g)
                    z *= quadgk(xx -> (
                            K(cospi(2xx), cospi(2g.uncons_anchors[l, j, i]), g.variances[l]) *
                            K(cospi(2xx), cospi(2g.uncons_anchors[l, k, i]), g.variances[l]) *
                            cospi(2xx * ω[l])
                        ), 0, 1)[1] * (ω[l] ≠ 0 ? 2 : 1)

                    # htruecos = quadgk(
                    # xx -> K(cos(2π * xx), y, s) * K(cos(2π * xx), z, s) * cos(2π * ω * xx),
                    # 0, 1)[1] * (ω ≠ 0 ? 2 : 1)
                end
                r += @views z * cs[j, :, i]' * cs[k, :, i]
            end
        end
        r
    end

    d, r, s, b = 2, 3, 4, 5

    g = PSDBlockBesselCheby(
        randn(d, s, b),
        randn(s, r, b),
        1.0 .+ rand(d) / 10
    )

    ωs = [
        0 1 0 2
        0 0 1 3
    ]

    ĝ_c = _cheby.(Ref(g), eachcol(ωs))
    ĝ = cheby(g, ωs)
    @test ĝ_c ≈ ĝ
end

struct PSDBlockBesselFourier{T<:AbstractFloat,M<:AbstractArray{T,3},V<:AbstractVector{T}} <: AbstractBlockPSDModel{T,ℤ}
    anchors::M       # d, s, b 
    coefficients::M  # s, r, b 
    variances::V      # d 

    regularization::T

    function PSDBlockBesselFourier(anchors, coefficients, variances; reg=1e-8)
        @assert size(anchors, 2) === size(coefficients, 1)
        @assert size(anchors, 3) === size(coefficients, 3)
        @assert size(anchors, 1) == size(variances, 1)

        new{eltype(anchors),typeof(coefficients),typeof(variances)}(anchors, coefficients, variances, reg)
    end
end
"Anchors of the model, of shape `dim(g), blocksize(g), nblocks(g)`."
anchors(g::PSDBlockBesselFourier) = g.anchors
"Coefficients of the model, of shape `blocksize(g), rank(g), nblocks(g)`."
coefficients(g::PSDBlockBesselFourier) = g.coefficients
variances(g::PSDBlockBesselFourier) = g.variances
regularization(g::PSDBlockBesselFourier) = g.regularization
coefficients_scaled(g::PSDBlockBesselFourier) = reshape(reduce(hcat,
        cholesky((_K(i, i, g) + regularization(g) * blocksize(g) * I)).U \ view(coefficients(g), :, :, i) for i ∈ 1:nblocks(g)  # (s, s) \ (s, r) -> (s, r)
    ), blocksize(g), rank(g), nblocks(g))  # s, r, b
blocksize(g::PSDBlockBesselFourier) = size(coefficients(g), 1)
rank(g::PSDBlockBesselFourier) = size(coefficients(g), 2)
nblocks(g::PSDBlockBesselFourier) = size(coefficients(g), 3)
dim(g::PSDBlockBesselFourier) = size(anchors(g), 1)

trainable_params(g::PSDBlockBesselFourier) = (coefficients(g), anchors(g))
gpu(g::PSDBlockBesselFourier) = PSDBlockBesselFourier(
    CuArray(anchors(g)),
    CuArray(coefficients(g)),
    CuArray(variances(g));
    reg=regularization(g)
)
cpu(g::PSDBlockBesselFourier) = PSDBlockBesselFourier(
    collect(anchors(g)),
    collect(coefficients(g)),
    collect(variances(g));
    reg=regularization(g)
)

"""
Bessel kernel defined on the torus, with 
```
    K(x, y) = exp(s (cos(2π (x - y)) - 1))
```
"""
function besselkernel_fourier(X::AbstractMatrix{T}, Y::AbstractMatrix{T}; γ::AbstractVector{T}) where {T}
    @assert size(X, 1) == size(Y, 1)
    d = size(X, 1)
    exp.(dropdims(sum(
            (cospi.(
                2(reshape(X, d, :, 1) .- reshape(Y, d, 1, :))
            ) .- one(T)) .* γ, dims=1
        ), dims=1)
    )
end

function besselkernel_fourier(X::AbstractArray{T}, Y::AbstractArray{T}; γ::AbstractVector{T}) where {T}
    @assert ndims(X) ≥ 2 && ndims(Y) ≥ 2
    d = size(X, 1)
    size_X, size_Y = size(X)[2:end], size(Y)[2:end]
    reshape(besselkernel_fourier(
            reshape(X, d, prod(size_X)), reshape(Y, d, prod(size_Y)); γ=γ
        ), size_X..., size_Y...)
end

function (g::PSDBlockBesselFourier{T})(X::AbstractMatrix{T}) where {T}
    @assert dim(g) == size(X, 1)
    embeddings = besselkernel_fourier(X, anchors(g); γ=g.variances)  # n, s, b
    dropdims(sum(abs2.(
                batched_mul(
                    embeddings,               # n, s, b
                    coefficients_scaled(g)    # s, r, b 
                ),                            # n, r, b
            ), dims=(2, 3)), dims=(2, 3))
end

# 781.994 ms (406 allocations: 1.87 MiB)
"""
Fourier coefficient of `g`. Simple loop, kept for test purposes.
"""
function fourier_baseline(g::PSDBlockBesselFourier{T}, ωs::AbstractMatrix{U}) where {T,U<:Integer}
    @assert dim(g) == size(ωs, 1)

    cs = coefficients_scaled(g)
    psdcoeffs = batched_mul(cs, permutedims(cs, (2, 1, 3)))  # s, s, b
    result = zeros(Complex{T}, size(ωs, 2))
    # See comments in cheby
    @inbounds for h ∈ axes(ωs, 2)
        for i ∈ 1:nblocks(g)
            for j ∈ 1:blocksize(g), k ∈ j:blocksize(g)
                z = one(Complex{T})
                for l ∈ 1:dim(g)
                    sₗ = g.variances[l]
                    z *= exp(-2sₗ) * besseli(abs(ωs[l, h]), 2sₗ * cospi(anchors(g)[l, j, i] - anchors(g)[l, k, i])) * cispi(-ωs[l, h] * (anchors(g)[l, j, i] + anchors(g)[l, k, i]))
                end
                result[h] += psdcoeffs[j, k, i] * z * (j == k ? one(T) : 2one(T))
            end
        end
    end
    result
end

function fourier(g::PSDBlockBesselFourier{T}, ωs::AbstractMatrix{U}) where {T,U<:Integer}
    result = Vector{Complex{T}}(undef, size(ωs, 2))
    fourier!(result, g, ωs)
    result
end

#   30.904 ms (338 allocations: 1.25 MiB)
function fourier!(
    out::Vector{Complex{T}}, g::PSDBlockBesselFourier{T}, ωs::AbstractMatrix{U}) where {T,U<:Integer}
    @assert dim(g) == size(ωs, 1)

    d, s, N = dim(g), blocksize(g), size(ωs, 2)
    g_mat = Matrix{T}(undef, s, s)
    r_scaled = Matrix{T}(undef, s, rank(g))
    out .= zero(Complex{T})

    ωs_max = maximum(abs.(ωs), dims=2)
    out_prod = Vector{Complex{T}}(undef, N)
    out_bessel = Vector{T}(undef, maximum(ωs_max) + 1)
    out_cispi = Vector{Complex{T}}(undef, maximum(ωs_max) + 1)

    for i = 1:nblocks(g)
        # g_mat ≈ coefficients_scaled(g)[:, :, i] * coefficients_scaled(g)[:, :, i]'
        ldiv!(r_scaled, cholesky((_K(i, i, g) + regularization(g) * I)).U, view(coefficients(g), :, :, i))
        mul!(g_mat, r_scaled, r_scaled')
        for j ∈ 1:blocksize(g), k ∈ j:blocksize(g)
            # Nothing in this loop allocates
            out_prod .= one(Complex{T})
            for l ∈ 1:dim(g)
                sₗ = g.variances[l]
                @views besseli!(out_bessel[1:ωs_max[l]+1], 0:ωs_max[l], 2sₗ * cospi(anchors(g)[l, j, i] - anchors(g)[l, k, i]))
                @views out_cispi[1:ωs_max[l]+1] .= cispi.(-(0:ωs_max[l]) .* (anchors(g)[l, j, i] + anchors(g)[l, k, i]))

                v = exp(-2sₗ)
                for h ∈ eachindex(out_prod)
                    out_prod[h] *= (
                        v *
                        out_bessel[abs(ωs[l, h])+1] * (
                            ωs[l, h] ≥ 0 ?
                            out_cispi[ωs[l, h]+1] :
                            conj(out_cispi[-ωs[l, h]+1]))
                    )
                end
            end
            v = g_mat[j, k] * (j ≠ k ? 2one(Complex{T}) : one(Complex{T}))
            @. out = muladd(out_prod, v, out)
        end
    end
end

"Returns a random model **whose coefficients are normalized to 1 by default**."
random_model(C::Type{U}, ::Type{T}, rank, blocksize, nblocks, γ; init_scale) where {T,U<:Union{PSDBlockBesselCheby,PSDBlockBesselFourier}} =
    C(rand(T, length(γ), blocksize, nblocks),
        begin
            coeffs = randn(T, blocksize, rank, nblocks)
            coeffs .* √(init_scale / sum(abs2.(coeffs)))
        end,
        γ)
random_model(C, rank, blocksize, nblocks, γ; init_scale=1.0) = random_model(C, Float64, rank, blocksize, nblocks, γ; init_scale=init_scale)

nparams(g::AbstractBlockPSDModel) = (rank(g) + dim(g)) * blocksize(g) * nblocks(g)

Base.show(io::IO, ::MIME"text/plain", g::AbstractBlockPSDModel) = print(io,
    """
    $(nameof(typeof(g))):
    • blocksize: $(blocksize(g))
    • rank: $(rank(g))
    • nblocks: $(nblocks(g))
    • dim: $(dim(g))
    • nparams: $(nparams(g))\
    """
)
Base.show(io::IO, g::AbstractBlockPSDModel) = print(io,
    """
    $(nameof(typeof(g)))(\
    bs=$(blocksize(g)), \
    rk=$(rank(g)), \
    nb=$(nblocks(g)), \
    d=$(dim(g)))\
    """
)

"Kernel of `g` applied to the anchors `i` and `j` of `g`."
function _K(i, j, g) end
function _K(i, j, g::PSDBlockBesselCheby)
    besselkernel_cheby_cos(
        bijacos.(view(g.uncons_anchors, :, :, i)),
        bijacos.(view(g.uncons_anchors, :, :, j));
        γ=variances(g))
end
function _K(i, j, g::PSDBlockBesselFourier)
    besselkernel_fourier(
        view(anchors(g), :, :, i),
        view(anchors(g), :, :, j);
        γ=variances(g))
end

function HSnorm2(g::AbstractBlockPSDModel{T}) where {T}
    r = zero(T)
    cs = coefficients_scaled(g)

    for i ∈ 1:nblocks(g), j ∈ i:(nblocks(g))
        r += sum(abs2.(view(cs, :, :, i)' * _K(i, j, g) * view(cs, :, :, j))) * (i ≠ j ? 2one(T) : one(T))
    end
    r
end

"""
An upper bound on the HS norm of `g` which relies on the approximation 
```
    ‖g‖² = ∑ᵢⱼ Tr Gᵢ Eᵢᵀ Eⱼ Gⱼ Eⱼ Eᵢ ≤ ∑ᵢⱼ ‖Gᵢ‖ ‖Eᵢᵀ Eⱼ Gⱼ Eⱼ Eᵢ‖ ≤ (∑ᵢ ‖Gᵢ‖)².
```
"""
function HSnorm2_upper(g::AbstractBlockPSDModel)
    psdcoeffs = batched_mul(permutedims(coefficients(g), (2, 1, 3)), coefficients(g))  # r, r, b
    hsnorms = sqrt.(dropdims(sum(abs2.(psdcoeffs), dims=(1, 2)), dims=(1, 2)))  # b
    sum(hsnorms)^2
end

function HSnorm2_proxy(g::AbstractBlockPSDModel)
    psdcoeffs = batched_mul(permutedims(coefficients(g), (2, 1, 3)), coefficients(g))  # r, r, b
    hsnorms = dropdims(sum(abs2.(psdcoeffs), dims=(1, 2)), dims=(1, 2))  # b
    sum(hsnorms)
end

function isgpu(g::AbstractBlockPSDModel) end
isgpu(g::PSDBlockBesselCheby) = typeof(coefficients(g)) <: CuArray
isgpu(g::PSDBlockBesselFourier) = typeof(coefficients(g)) <: CuArray