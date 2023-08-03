using GloptiNets
using JLD2

path_save = "data/samplers"
@assert isdir(path_save)

γ = 2
tol = 1e-12  # We need 1/(dim * nfreqs) ≫ tol
dim = 3
nfreqs = 32 * 100_000_000
proba = ApproxBesselSampler(ℤ, γ * ones(dim); tol=tol)
t_proba = @elapsed ωs, ns, ps = GloptiNets.samplesprobas_bycat(proba, nfreqs, Int)
@info "Sampled in $t_proba s! # of unique frequencies: $(size(ps, 1))"

jldsave(joinpath(path_save, "var$(γ)_dim$(dim).jld2");
    ωs=ωs, ns=ns, ps=ps,
    hparams=(; γ=γ, tol=tol, dim=dim, nfreqs=nfreqs)
)