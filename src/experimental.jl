"""
Experimental code snippets which may be useful.
"""

"""
For a model `G` defined on `S(Hz)`, first projects it on `H̄z`, then projects it back on `S(Hz)`.

# Arguments
- `R`: low-rank matrix corresponding to the psd matrix `G = R Rᵀ`.
- `T`: cholesky decomposition of the kernel matrix `K`.
- `K̄`: element-wise square of the kernel matrix, `K̄ = K .^ 2`.  

# Issues
The approximation of the first projection is correct, but the second one is not (checked experimentally by evaluating the intermediate quantities). 

The lowest eigenvalues `vals` are hugely negative (~-1e8) instead of being approximately close to 0, which is expected from a linear model which is approximately positive, because the model it's projected from is positive. 
"""
function roundtrip_Hx(R, T, K̄)
    r = size(R, 2)
    # Evaluation of the model `g` on z₁, ..., zₙ
    eval_z = dropdims(sum(abs2.(T.L * R), dims=2), dims=2)
    # PERF: see if we can have the cholesky of K̄ given cholesky T of K
    coeffs = K̄ \ eval_z
    # Projection on H̄z
    proj_H̄z_G̃ = T.U * diagm(coeffs) * T.U'
    # PERF: could only keep the first `r` eigenvectors with arpack
    vals, vecs = eigen(proj_H̄z_G̃)
    Rnew = @views vecs[:, end-r+1:end] * diagm(sqrt.(max.(zero(eltype(R)), vals[end-r+1:end])))
    Rnew
end

using Bessels

"""
Performs a roundtrip of every function, eventhough it should consider all the model at once, but not done in practice because of the `O((bs)³)` huge computation cost. 
"""
function roundtrip_Hx(g::PSDBlockBesselCheby)
    newcoeffs = zero(coefficients(g))
    for i ∈ 1:nblocks(g)
        K = _K(i, i, g)
        T = cholesky(K + g.regularization * I)
        @views newcoeffs[:, :, i] = roundtrip_Hx(coefficients(g)[:, :, i], T, K .^ 2)
    end
    PSDBlockBesselCheby(g.uncons_anchors, newcoeffs, g.variances)
end


# Baseline:   590.998 ms (405 allocations: 1.86 MiB)
using Bessels: besseli!

#   123.358 ms (339 allocations: 1.26 MiB)
# If all the frequency are positive and we can compute the cispi once (and not for all frequency), this drops to 
#   66.239 ms (336 allocations: 1.23 MiB)
# So this approach is possible for Chebychev polynomials (where the frequencies are positive); see commit db60947

# This looks for the indices b s² times instead of once
# function fourier!_toomanyindicesfind(
#     r::AbstractVector{Complex{T}}, g::PSDBlockBesselFourier{T}, ωs::AbstractMatrix{U}) where {T,U<:Integer}
#     @assert dim(g) == size(ωs, 1)

#     d, s, N = dim(g), blocksize(g), size(ωs, 2)

#     ωs .= abs.(ωs)
#     ωs_max = maximum(ωs, dims=2)
#     _anchors_g = collect(anchors(g))

#     # ωs_max = collect(ωs_max)
#     out_gmat = Matrix{T}(undef, s, s)
#     out_scaled = Matrix{T}(undef, s, rank(g))
#     out_prod = Array{Complex{T}}(undef, N)
#     out_bessel = zeros(T, maximum(ωs_max) + 1)
#     out_cispi = zeros(Complex{T}, maximum(ωs_max) + 1)
#     out_bool = similar(out_prod, Bool)

#     for i = 1:nblocks(g)
#         # out_gmat ≈ coefficients_scaled(g)[:, :, i] * coefficients_scaled(g)[:, :, i]'
#         chol_out = cholesky((_K(i, i, g) + regularization(g) * I))
#         ldiv!(out_scaled, chol_out.U, view(coefficients(g), :, :, i))
#         mul!(out_gmat, out_scaled, out_scaled')
#         # copy!(g_mat_cpu, out_gmat)
#         for j ∈ 1:blocksize(g), k ∈ j:blocksize(g)
#             out_prod .= one(Complex{T})
#             for l ∈ 1:dim(g)
#                 sₗ = g.variances[l]
#                 @views besseli!(out_bessel[1:ωs_max[l]+1], 0:ωs_max[l], 2sₗ * cospi(_anchors_g[l, j, i] - _anchors_g[l, k, i]))
#                 # out_bessel[1:ωs_max[l]+1] .= besseli.(0:ωs_max[l], 2sₗ * cospi(_anchors_g[l, j, i] - _anchors_g[l, k, i]))
#                 # @views besseli!(out_bessel[1:ωs_max[l]+1], 0:ωs_max[l], 2sₗ * cospi(anchors(g)[l, j, i] - anchors(g)[l, k, i]))
#                 @views out_cispi[1:ωs_max[l]+1] .= cispi.(-(0:ωs_max[l]) .* (_anchors_g[l, j, i] + _anchors_g[l, k, i]))

#                 for ω in 0:ωs_max[l]
#                     out_bool .= view(ωs, l, :) .== ω
#                     v = exp(-2sₗ) * out_bessel[ω+1] * out_cispi[ω+1]
#                     # Does `out_prod[out_bool] .= v` but without allocating (as `out_prod[out_bool]` creates a view)
#                     out_prod .*= ifelse.(out_bool, v, one(Complex{T}))
#                 end
#             end
#             v = out_gmat[j, k] * (j != k ? 2one(Complex{T}) : one(Complex{T}))
#             @. r += out_prod * v
#         end
#     end
# end


"""
Returns the masks associated to the frequencies `ωs`. 
`μ[h, n, l]` is `true` if the `h`-th frequency has `n` is its `l`-th dimension.
"""
function get_masks(ωs::AbstractMatrix{U}, ω̄::U) where {U<:Integer}
    d, N = size(ωs)
    μ = similar(ωs, Bool, N, ω̄ + 1, d)
    for ω ∈ 0:ω̄, l ∈ 1:d
        @views μ[:, ω+1, l] .= ωs[l, :] .== ω
    end
    μ
end

"""
Term inside the product computed for all `ω` in `0:ω̄` and all dimension `l` in `1:dim(s)`.
"""
function get_prod_values!(out::AbstractMatrix{Complex{T}}, s::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}, ω̄::U) where {T<:AbstractFloat,U<:Integer}
    # Method 1: good for GPU
    # @. out = exp(-2s') * besseli(0:ω̄, 2s' * cospi(x' - y')) * cispi(-(0:ω̄) * (x' + y'))
    # Method 2: uses besseli!
    out .= one(Complex{T})
    tmp = Vector{T}(undef, ω̄ + 1)
    for l ∈ axes(out, 2)
        besseli!(tmp, 0:ω̄, 2s[l] * cospi(x[l] - y[l]))
        view(out, :, l) .*= tmp .* exp(-2s[l]) .* cispi.(-(0:ω̄) .* (x[l] + y[l]))
    end
end

"""
For each frequency `h`, computes `out[h] = ∏ₗ coefs[ωs[h, l], l]`. But instead of passing `ωs` directly, we pass the mask `μ`, and we consider all the possible values of `ωs[h, l]`.
"""
function apply_masks!(out::AbstractVector{Complex{T}}, coefs::AbstractMatrix{Complex{T}}, μ::AbstractArray{Bool,3}) where {T<:AbstractFloat}
    out .= one(Complex{T})
    for l ∈ axes(coefs, 2)
        for ω ∈ axes(coefs, 1)  # 1:ω̄+1
            out .*= ifelse.(view(μ, :, ω, l), coefs[ω, l], one(Complex{T}))
        end
    end
end

"""
Approach with masks: we compute once the position of each frequency in `ωs`, then we apply it for each combination `(i, j, k) ∈ (1:nblocks(g), 1:blocksize(g), 1:blocksize(g)).`

ERROR here. Do not use as is. It changes the frequencies to consider only positive frequencies. Wrong as the term in `cispi.(- ωᵀ (zⱼ + zₖ))` depends on the sign of `ω`.
"""
function fourier!_wmasks(
    out::AbstractVector{Complex{T}}, g::PSDBlockBesselFourier{T}, ωs::AbstractMatrix{U}) where {T,U<:Integer}
    @assert dim(g) == size(ωs, 1)
    d, s, N = dim(g), blocksize(g), size(ωs, 2)

    ωs .= abs.(ωs)
    ω̄ = maximum(ωs)
    μ = get_masks(ωs, ω̄)
    out .= zero(Complex{T})

    anchors_g = anchors(g)
    variances_g = variances(g)
    out_scaled = similar(coefficients(g), s, rank(g))  # Scaled coefficients of block i
    out_gmat = similar(coefficients(g), s, s)  # PSD matrix of block i
    out_prod_in = similar(out, ω̄ + 1, d)
    out_prod_out = similar(out)

    for i = 1:nblocks(g)
        # out_gmat contains 
        # coefficients_scaled(g)[:, :, i] * coefficients_scaled(g)[:, :, i]'
        chol_out = cholesky((_K(i, i, g) + regularization(g) * I))  # TODO preallocate? 
        ldiv!(out_scaled, chol_out.U, view(coefficients(g), :, :, i))
        mul!(out_gmat, out_scaled, out_scaled')

        for j ∈ 1:blocksize(g), k ∈ j:blocksize(g)
            get_prod_values!(out_prod_in, variances_g, view(anchors_g, :, j, i), view(anchors_g, :, k, i), ω̄)
            apply_masks!(out_prod_out, out_prod_in, μ)

            v = out_gmat[j, k] * (j != k ? 2one(Complex{T}) : one(Complex{T}))
            @. out = muladd(out_prod_out, v, out)
        end
    end
end

"""
Returns the `n`-th Fourier coefficient of the Bessel function of order `ω` composed with cosine, as in 
```
    exp(-2s) × besseli(ω, 2s cos(2π z)).
```
"""
function fourier_besselcos(ω::Integer, n::Integer, s::AbstractFloat)
    ω % 2 ≠ n % 2 && return zero(eltype(s))

    q = (n - ω) ÷ 2
    p = max(0, q)
    out = val_p = (s / 2)^(2p + ω) / (factorial(p) * factorial(p + ω)) * binomial(2p + ω, p - q)

    C = (s / 2)^2
    maxiter = 1000

    for _ ∈ 1:maxiter
        val_p *= C * (2p + 1 + ω) * (2p + 2 + ω) / ((p + 1) * (p + 1 - q) * (p + 1 + ω) * (p + 1 + q + ω))
        out += val_p
        abs(val_p) < eps(eltype(s)) * abs(out) && break
        p += 1
    end
    exp(-2s) * out
end

@testitem "Fourier Bessel ∘ cos" begin
    using GloptiNets: fourier_besselcos
    using QuadGK: quadgk
    using Bessels: besseli

    s = 1.2
    for (ω, n) ∈ Iterators.product((0, 1, 2), (0, 1, 2))
        val = fourier_besselcos(ω, n, s)
        val_num = quadgk(z -> exp(-2s) * besseli(ω, 2s * cospi(2z)) * cospi(2n * z), 0, 1)[1]
        @test ≈(val, val_num; atol=1e-8)
    end
end

"""
Returns an array of dimension 3, s.t. `out[l, ω, n]` contains the `n`-th _non-zero_ Fourier coefficient of the Bessel function composed with cosine of parameter `s[l], ω` as defined in [fourier_besselcos](@ref).
"""
function all_besselcos_fourier(s::AbstractVector{T}, ωmax::Integer, nvals::Integer) where {T}
    out = Array{T}(undef, length(s), nvals, ωmax + 1)
    for ω ∈ 0:ωmax
        for (l, sₗ) ∈ enumerate(s)
            @views out[l, :, ω+1] .= fourier_besselcos.(ω, (ω%2):2:(2nvals-1), sₗ)  # PERF: probably a faster way to compute the range
        end
    end
    out
end

"""
Returns the indices of the Hyperbolic Cross, i.e. the subset of ``ℕᵈ`` defined as
``
    {ω ∈ ℕᵈ; prod(max(1, ωₗ) for ωₗ ∈ ω) ≤ n}.
``
"""
function hyperboliccross(d::U, n::U) where {U<:Integer}
    @assert d ≥ 1 && n ≥ 1 "Dimension `d` and `n` must be positive"

    S = Set((collect ∘ tuple).(0:n))
    for _ ∈ 2:d
        newS = Set{Vector{U}}()
        for ω ∈ S
            P = prod(max.(1, ω))
            for ωᵢ ∈ 0:n
                P * max(1, ωᵢ) > n && break
                push!(newS, [ω; ωᵢ])
            end
        end
        S = newS
    end
    reduce(hcat, collect.(S))
end