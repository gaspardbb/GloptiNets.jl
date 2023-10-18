using GloptiNets
using JLD2

path_save = "data/samplers"
@assert isdir(path_save)

γ = 2
tol = 1e-12
dim = 4
nfreqs = 32 * 100_000_000
proba = ApproxBesselSampler(ℤ, γ * ones(dim); tol=tol)
t_proba = @elapsed ωs, ns, ps = GloptiNets.samplesprobas_bycat(proba, nfreqs, Int)
@info "Sampled in $t_proba s! # of unique frequencies: $(size(ps, 1))"

jldsave(joinpath(path_save, "var$(γ)_dim$(dim).jld2");
    ωs=ωs, ns=ns, ps=ps,
    hparams=(; γ=γ, tol=tol, dim=dim, nfreqs=nfreqs)
)

f = jldopen(joinpath(path_save, "var$(γ)_dim$(dim).jld2"))
close(f)

# Smaller set of frequencies for test purposes; 40k frequencies would take 4h to run, so keeping only 1k for identifying best model
ind = sortperm(ps)[end-1000+1:end]
ωs_small = ωs[:, ind]
ps_small = ps[ind]
ns_small = Int.(round.(ps_small / sum(ps_small) * nfreqs))
jldsave(joinpath(path_save, "var$(γ)_dim$(dim).jld2");
    ωs=ωs, ns=ns, ps=ps,
    hparams=(; γ=γ, tol=tol, dim=dim, nfreqs=nfreqs)
)
close(f)