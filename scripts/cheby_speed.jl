"""
Making the Chebychev coefficient computation FAST.
"""

using GloptiNets
using GloptiNets: GloptiNets as GN

d, rank, blocksize, nblocks = 5, 8, 16, 32
γ = ones(d) * 1.0
proba = ApproxBesselSampler(ℕ, 2 * γ; tol=1e-9)
g = PSDBlockBesselCheby(
    rand(d, blocksize, nblocks),
    begin
        coeffs = randn(blocksize, rank, nblocks)
        coeffs ./ √sum(abs2.(coeffs))
    end,
    γ
)

ωs, ps = samplesprobas(proba, 10)

r_base = GN.cheby_baseline(g, ωs)
r_fast = similar(r_base)
GN.cheby!(r_fast, g, ωs)
@show r_base ≈ r_fast

# r_masks = similar(r_base)
# GN.fourier!_wmasks(r_masks, g, ωs)
# @show r_base ≈ r_masks

using BenchmarkTools

ωs, ps = samplesprobas(proba, 32 * 16)
@btime sum(GN.cheby_baseline($g, $ωs))
@btime sum(GN.cheby($g, $ωs))
r = zeros(eltype(ps), size(ps))
@btime GN.cheby!($r, $g, $ωs)
@btime GN.fourier!_wmasks($r, $g, $ωs)