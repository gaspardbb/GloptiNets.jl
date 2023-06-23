"""
This file generates random polynomial.

The polynomials are functions in `H₂`, i.e. with `s=2`, for which we keep only the coefficients of degrees smaller than `N`. 

In the first case, `N` is chosen so that the number of frequencies of the resulting polynomial is aroung `1000`, so that polynomial evaluation stays negligible in front of the model's optimization. 

Some polynomial may have a different seed, because the alternate projection scheme used to generate the said polynomial failed. 
"""

using GloptiNets
import Random

const path_poly = "data"

for (d, N) ∈ zip(3:5, (12, 7, 6))
    proba = ApproxBesselSampler(ℤ, 2ones(d); tol=1e-8)
    for S ∈ (1, 2, 4, 8)
        Random.seed!(0)
        f = GloptiNets.random_pos_polytrigo(proba, N, float(S))
        filename = "randompoly-s2-d$d-N$N-S$S"
        GloptiNets.save(f, joinpath(path_poly, filename))
        @info "Wrote $filename"
    end
end

"""
This creates lot of random polynomial in dimension `3`, with `N=12`, so that we can compare the value of the certificate function of the Hilbert norm, with a fixed number of parameters.
"""

const path_xp_hnorm2 = "data/hnorm2"

d, N = 3, 12
proba = ApproxBesselSampler(ℤ, 2ones(d); tol=1e-8)
for (i, S) ∈ enumerate((20^(1 / 10)) .^ (0:10))
    success = false
    for seed ∈ 0:9
        Random.seed!(seed)
        f = GloptiNets.random_pos_polytrigo(proba, N, S)
        if (GloptiNets.Hnorm2(f, proba) - S) < 0.01
            # found a good polynomial
            filename = "randompoly-s2-d3-N12-i$i"
            GloptiNets.save(f, joinpath(path_xp_hnorm2, filename))
            success = true
            @info "Wrote $filename"
            break
        end
    end
    !success && @warn "Did not find a polynomial for i=$i."
end

"""
This creates polynomials which can be solved -- for some -- by TSSOS, providing grounds for comparison.
"""

const path_vs_tssos = "data/vs_tssos/cheby"

S = 1.0
# for (d, Ns) ∈ zip((3, 4), ((5, 7, 9), (3, 5, 7)))
for (d, Ns) ∈ zip((4,), ((3, 4, 5),))
    # proba = ApproxBesselSampler(ℕ, 2ones(d); tol=1e-8)
    proba = ApproxBesselSampler(ℕ, 2ones(d); tol=1e-12)
    for N ∈ Ns
        Random.seed!(0)
        f = GloptiNets.random_pos_polycheby(proba, N, S)
        filename = "randompoly-s2-d$d-N$N"
        GloptiNets.save(f, joinpath(path_vs_tssos, filename))
        @info "Wrote $filename"
    end
end

# This writes the maximum value and the norm of the polynomials in `path_toconsider`

# path_toconsider = path_poly
# path_toconsider = path_xp_hnorm2
path_toconsider = path_vs_tssos

using TOML
out_info = Dict()
for file ∈ readdir(path_toconsider)
    !endswith(file, ".jld2") && continue
    file = first(split(file, ".jld2"))
    f = GloptiNets.load(PolyCheby, joinpath(joinpath(path_toconsider, file)))
    out_info[file] = Dict(
        "maxval" => GloptiNets.estimate_max(f, 20)[1],
        "hnorm2" => GloptiNets.Hnorm2(f, ApproxBesselSampler(ℕ, ones(dim(f)) * 2; tol=1e-8))  # Assume the variance to be 2
    )
end

open(joinpath(path_toconsider, "infos.toml"), "w") do f
    TOML.print(f, out_info)
end
