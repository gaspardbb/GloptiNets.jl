"""
We try to make the 
"""

using GloptiNets
using GloptiNets: GloptiNets as GN

d, rank, blocksize, nblocks = 5, 8, 16, 32
γ = ones(d) * 1.0
proba = ApproxBesselSampler(ℤ, 2 * γ; tol=1e-9)
g = PSDBlockBesselFourier(
    rand(d, blocksize, nblocks),
    begin
        coeffs = randn(blocksize, rank, nblocks)
        coeffs ./ √sum(abs2.(coeffs))
    end,
    γ
)

ωs, ps = samplesprobas(proba, 10)

# If on CUDA
# using CUDA
# CUDA.allowscalar(false)
# g, ωs, r = gpu(g), cu(ωs), cu(r)

r_base = GN.fourier_baseline(g, ωs)
r_fast = similar(r_base)
GN.fourier!(r_fast, g, ωs)
@show r_base ≈ r_fast

# r_masks = similar(r_base)
# GN.fourier!_wmasks(r_masks, g, ωs)
# @show r_base ≈ r_masks

using BenchmarkTools

ωs, ps = samplesprobas(proba, 32 * 16)
r = zeros(Complex{eltype(ps)}, size(ps))
@btime sum(GN.fourier_baseline($g, $ωs))
@btime GN.fourier!($r, $g, $ωs)
@btime GN.fourier!_wmasks($r, $g, $ωs)

# ωs = collect(ωs')
# GN.fourier!_row(r, g, ωs)

# using BenchmarkTools
# using Plots
# plotlyjs()

# plt_ns = 2 .^ (0:8)
# plt_col = similar(plt_ns, Float64)
# plt_row = similar(plt_ns, Float64)
# for (i, n) in enumerate(plt_ns)
#     @show n
#     ωs, ps = samplesprobas(proba, 32 * n)
#     r = zeros(Complex{eltype(ps)}, size(ps))
#     plt_col[i] = @belapsed GN.fourier!($r, $g, $ωs) seconds = 1
#     ωs = collect(ωs')
#     plt_row[i] = @belapsed GN.fourier!_row($r, $g, $ωs) seconds = 1
# end
# @btime GN.fourier!($r, $g, $ωs)
# ωs⊤ = collect(ωs')
# @btime GN.fourier!_row($r, $g, $ωs⊤)
# # @btime GN.fourier!($params..., $r, $g, $ωs)


# GN.fourier(g, ωs)

# x = randn(1000)
# r = randn(1000)

# function addinplace(x, r)
#     @simd for h ∈ eachindex(x)
#         x[h] = x[h] + r[h]
#     end
# end

# @allocated addinplace(x, r)

# @allocated @. x = x + r

# using BenchmarkTools
# @ballocated ($x .+= $r .* v) setup = (v = randn())