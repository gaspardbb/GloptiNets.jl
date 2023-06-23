"""
Experiments for studying ``certificates vs. # of parameters''. 

    julia --project scripts/xp_fourier_params_variation.jl
"""

using GloptiNets
using Dates
using DataFrames
using TOML

const usegpu = true
const path_data = "data/hnorm2"
!isdir("xps") && mkdir("xps")

cuda_device = 0
using CUDA: device!
device!(cuda_device)

v = (
    lrdecay=:cos,
    opt=:momentum,
    lr=4e-1,
    nepochs=4000,
    batchsize=2048,
    variances=1.0,
    lossfunc_param=10.0,
    proba_tol=1e-8,
    gparams=(8, 16, 32),
    regtype=["HSNormP"],
    reg=0.0125,
    f_idx=1,
    f_var=2,
    f_d=3,
    f_N=12,
    T=Float32
)

f = GloptiNets.load(PolyTrigo, joinpath(path_data, "randompoly-s$(v.f_var)-d$(v.f_d)-N$(v.f_N)-i$(v.f_idx)"))
f = convert(v.T, f)
rank, blocksize, nblocks = v.gparams
γ = ones(v.T, dim(f)) * convert(v.T, v.variances)
g = PSDBlockBesselFourier(
    rand(v.T, dim(f), blocksize, nblocks),
    begin
        coeffs = randn(v.T, blocksize, rank, nblocks)
        coeffs ./ √sum(abs2.(coeffs))
    end,
    γ
)

usegpu && ((f, g) = (gpu(f), gpu(g)))

optimizer_params = (; optimizer_type=v.opt, optimizer_lrdecay=v.lrdecay, optimizer_lrinit=v.lr)
interpolate(f, g, RegHSNormU, (; val=v.reg);
    optimizer_params, v.nepochs, v.batchsize,
    lossfunc_symb=:lse,
    lossfunc_param=v.lossfunc_param,
    show_progress=true)
time_hs_norm2 = @elapsed hs_norm2 = GloptiNets.HSnorm2(g)
time_linfsamples = @elapsed linfsamples = maximum(GloptiNets.l∞norm_samples(f, g; nsamples=8192) for _ ∈ 1:10)

usegpu && ((f, g) = (cpu(f), cpu(g)))

proba = ApproxBesselSampler(ℤ, 2 * g.variances; tol=v.proba_tol)
mom_nbatches = 32

fnorm_hypercube = GloptiNets.norms_numapprox(f, g, proba, 10).diff_fnorm
dotprod_bound = GloptiNets.dotproduct_bound(f, g, proba)
stdbound = √(dotprod_bound + hs_norm2)

results = []
for mom_batchsize ∈ Int.(round.(exp.(LinRange(log(10), log(5000), 20))))
    @show mom_batchsize
    for seed ∈ 1:10
        ωs, ns, ps = GloptiNets.samplesprobas_bycat(proba, mom_nbatches * mom_batchsize, Int)
        (; vals_mom, vals_mean) = GloptiNets.mom_estimator(f, g, ωs, ns, ps;
            nbatch=mom_nbatches, batchsize=mom_batchsize
        )
        bound_cheby = vals_mean + stdbound / √(mom_nbatches * mom_batchsize * 0.018)
        bound_mom = vals_mom + 2stdbound / √mom_batchsize
        push!(results, Dict(
            "mom_batchsize" => mom_batchsize,
            "bound_cheby" => bound_cheby,
            "bound_mom" => bound_mom,
            "vals_mean" => vals_mean,
            "seed" => seed
        ))
    end
end
results = DataFrame(results)

using Statistics: mean
out = combine(
    groupby(results, :mom_batchsize), :bound_cheby => (x -> (bound_min=minimum(x), bound_max=maximum(x), bound_mean=mean(x))) => AsTable
)

using Plots
gr()
pgfplotsx()

xplot = out[!, :mom_batchsize] * mom_nbatches
p = plot(
    size=(400, 200),
    xlim=(minimum(xplot), maximum(xplot)),
    ylim=(0, maximum(out[!, :bound_max]) * 1.1),
    xscale=:log10,
    leg=:topright,
)
hline!(p, [linfsamples], label="L∞ norm")
hline!(p, [fnorm_hypercube], label="F norm")
plot!(p, xplot, out[!, :bound_mean], label="Our bound (Thm. 3)", ribbon=(out[!, :bound_mean] - out[!, :bound_min], out[!, :bound_max] - out[!, :bound_mean]))
p
savefig(p, "myfig.tikz")

savefig(p, "fourier_fnorm.pdf")