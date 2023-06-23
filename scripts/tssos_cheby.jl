using DynamicPolynomials
using TSSOS

using GloptiNets
using GloptiNets: GloptiNets as GN

path_poly = "data/vs_tssos/cheby"

for (d, N) ∈ (
    (4, 3),
    (4, 4),
    (4, 5),
)
    f = GN.load(PolyCheby, joinpath(path_poly, "randompoly-s2-d$d-N$N"))
    @polyvar x[1:dim(f)]
    f_tssos = GN.evaluate_poly(f, x)
    inequalities = [1 - x[i]^2 for i ∈ 1:d]
    ncoeffs = GN.ncoeffs(f)
    ncoeffs_canon = nterms(f_tssos)
    t = @elapsed opt, _, _ = cs_tssos_first([f_tssos; inequalities], x, Int(ceil(d * N / 2)); nb=0, numeq=0, TS="CS", QUIET=true)
    @show d, N, ncoeffs, ncoeffs_canon, t, opt
end

# (d, N, ncoeffs, ncoeffs_canon, t, opt) = (4, 3, 255, 256, 6.17155475, 3.4286487043184336e-7)
# (d, N, ncoeffs, ncoeffs_canon, t, opt) = (4, 4, 624, 625, 152.917940792, 2.160262306139463e-9)