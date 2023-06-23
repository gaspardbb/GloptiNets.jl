"""
Experiments for studying ``Certificate vs. Hnorm2 of f''.
"""

using GloptiNets
using Dates
using TOML

const usegpu = true
const cuda_device = 3
const path_xps = "xps/fourier_hnorm2_variation"
const path_data = "data/hnorm2"
!isdir("xps") && mkdir("xps")
!isdir(path_xps) && mkdir(path_xps)
const mom_batchsize = 5_000
const mom_nbatches = 32

using CUDA: device!
device!(cuda_device)

config = (
    lrdecay=[
        :poly,
        :cos,],
    opt=[
        :momentum],
    lr=[1e-1],
    nepochs=[2000],
    batchsize=[2048],
    variances=[
        1.0,
    ],
    lossfunc_param=[
        10.0
    ],
    proba_tol=[
        1e-8
    ],
    gparams=[
        # rank, blocksize, nblocks
        (8, 16, 128),
    ],
    regtype=["HSNormU"],
    reg=[
        # 1e-7,
        # 1e-6,
        # 1e-5,
        1e-4,
    ],
    f_idx=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    f_var=[2],
    f_d=[3,],
    f_N=[12,],
)

ntotal = prod(length.(values(config)))
for (cur_run, v) in enumerate(Iterators.product(values(config)...))
    v = (; zip(keys(config), v)...)
    @info "Run $cur_run/$ntotal. Config:\n$(v)"

    path_save = mkdir(joinpath(path_xps, Dates.format(now(), "yymmdd-HHMMSS-s")))
    open(joinpath(path_save, "config.toml"), "w") do f
        TOML.print(f, Dict(pairs(v))) do x
            x isa Symbol && return string(x)
            x isa Tuple && return collect(x)
            x
        end
    end

    f = GloptiNets.load(joinpath(path_data, "randompoly-s$(v.f_var)-d$(v.f_d)-N$(v.f_N)-i$(v.f_idx)"))
    rank, blocksize, nblocks = v.gparams
    γ = ones(dim(f)) * v.variances
    g = PSDBlockBesselFourier(
        rand(dim(f), blocksize, nblocks),
        begin
            coeffs = randn(blocksize, rank, nblocks)
            coeffs ./ √sum(abs2.(coeffs))
        end,
        γ
    )

    usegpu && ((f, g) = (gpu(f), gpu(g)))

    optimizer_params = (; optimizer_type=v.opt, optimizer_lrdecay=v.lrdecay, optimizer_lrinit=v.lr)
    time_interpolate = @elapsed interpolate(f, g, RegHSNormU, (; val=v.reg);
        optimizer_params, v.nepochs, v.batchsize,
        lossfunc_symb=:lse,
        lossfunc_param=v.lossfunc_param,
        show_progress=true)
    time_hs_norm2 = @elapsed hs_norm2 = GloptiNets.HSnorm2(g)
    time_linfsamples = @elapsed linfsamples = maximum(GloptiNets.l∞norm_samples(f, g; nsamples=2048) for _ ∈ 1:10)

    usegpu && ((f, g) = (cpu(f), cpu(g)))

    proba = ApproxBesselSampler(ℤ, 2 * g.variances; tol=v.proba_tol)
    time_mom = @elapsed (; vals_mom, vals_mean) = GloptiNets.mom_estimator(f, g, proba; nsamples=mom_nbatches * mom_batchsize)
    dotprod_bound = GloptiNets.dotproduct_bound(f, g, proba)
    stdbound = √(dotprod_bound + hs_norm2)

    cur_result = Dict(
        "bound_cheby" => -vals_mean - stdbound / √(mom_nbatches * mom_batchsize * 0.018),
        "bound_mom" => -vals_mom - 2stdbound / √mom_batchsize,
        "linfsamples" => linfsamples,
        "hs_norm2" => hs_norm2,
        "vals_mom" => vals_mom,
        "vals_mean" => vals_mean,
        "mom_nsamples" => mom_nbatches * mom_batchsize,
        "time_interpolate" => time_interpolate,
        "time_hs_norm2" => time_hs_norm2,
        "time_linfsamples" => time_linfsamples,
        "time_mom" => time_mom,
    )

    # Main results in a TOML file 
    open(joinpath(path_save, "results.toml"), "w") do f
        TOML.print(f, cur_result)
    end

    @info "Run $cur_run/$ntotal done. Result:\n$(cur_result)"
end
