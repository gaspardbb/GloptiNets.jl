using DynamicPolynomials
using TSSOS

using GloptiNets
using GloptiNets: GloptiNets as GN


path_poly = "data/vs_tssos"

for (d, N) ∈ (
    (3, 5),
    (4, 3),
    (3, 7),
    (4, 5),
    (3, 9),
    (4, 7),
)
    f = GN.load(joinpath(path_poly, "randompoly-s2-d$d-N$N"))
    @polyvar x[1:2dim(f)]
    f_tssos = GN.evaluate_poly(f, x)
    @show ncoeffs = GN.ncoeffs(f)
    # t = @elapsed opt, _, _ = cs_tssos_first([f_tssos], x, dim(f), N; nb=2dim(f), TS="CS", QUIET=true)
    # @show d, ncoeffs, t, opt
end


# d, N = 3, 5
# proba = ApproxBesselSampler(ℤ, 2ones(d); tol=1e-8)
# f = GN.random_pos_polytrigo(proba, N, 1.0)
# @polyvar x[1:2dim(f)]
# f_tssos = GN.evaluate_poly(f, x)
# @elapsed @show opt, _, _ = cs_tssos_first([f_tssos], x, dim(f), N; nb=2dim(f), TS="CS")

@info @elapsed

opt, sol, data = cs_tssos_first([f_tssos], x, dim(f), N; nb=2dim(f), TS="CS")