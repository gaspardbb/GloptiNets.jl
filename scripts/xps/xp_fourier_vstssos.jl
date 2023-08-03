"""
GloptiNets for Trigonometric polynomials, benchmarked on the same polynomials as TSSOS (Table 1)

julia --project scripts/xps/xp_fourier_vstssos.jl 
"""

using GloptiNets
using Dates
using TOML
using CUDA
CUDA.allowscalar(false)  # Just in case we start performing scalar indexing in new commits

const usegpu = true
cuda_device = 0
const path_xps = "xps/fourier-vstssos"
const path_data = "data/vs_tssos/fourier"
!isdir("xps") && mkdir("xps")
!isdir(path_xps) && mkdir(path_xps)
mom_batchsize = 100_000_000
const mom_nbatches = 32

using CUDA: device!
device!(cuda_device)

config = (
    lrdecay=[
        :cos,
    ],
    opt=[
        :momentum],
    lr=[
        4e-1,
        # 2e-1,
    ],
    nepochs=[2000],
    batchsize=[2048],
    lbfgs_nepochs=[30],
    lbfgs_batchsize=[8192],
    lbfgs_itermax=[100],
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
        # (4, 32, 8),
        (8, 128, 16),
    ],
    regtype=[NoReg],
    reg=[
        0.0,
        # 0.00078125,
        # 0.0015625,
        # 0.0015625,
        # 0.003125,
        # 0.00625,
        # 0.0125,
    ],
    f_var=[2],
    f_d=[3],
    f_N=[5, 7, 9,],
    # f_d=[4],
    # f_N=[3, 5, 7],
    T=[Float64,],
)

@assert length(config.f_d) == 1
@assert length(config.variances) == 1
@assert length(config.T) == 1
@assert length(config.proba_tol) == 1
d, var, T, proba_tol = config.f_d[1], config.variances[1], config.T[1], config.proba_tol[1]
γ = ones(T, d) * convert(T, var)  # We should have roughly 2γ ≈ f_var
proba = ApproxBesselSampler(ℤ, 2γ; tol=proba_tol)
@info "Started sampling the frequencies..."
t_proba = @elapsed ωs, ns, ps = GloptiNets.samplesprobas_bycat(proba, mom_nbatches * mom_batchsize, Int)
@info "Sampled $(mom_nbatches * mom_batchsize) elements in $t_proba time! # of unique frequencies: $(size(ps, 1))"


ntotal = prod(length.(values(config)))
for (cur_run, v) in enumerate(Iterators.product(values(config)...))
    v = (; zip(keys(config), v)...)
    @info "Run $cur_run/$ntotal. Config:\n$(v)"

    path_save = mkdir(joinpath(path_xps, Dates.format(now(), "yymmdd-HHMMSS-s")))
    open(joinpath(path_save, "config.toml"), "w") do f
        TOML.print(f, Dict(pairs(v))) do x
            x isa Union{Symbol,DataType} && return string(x)
            x isa Tuple && return collect(x)
            x
        end
    end

    f = GloptiNets.load(PolyTrigo, joinpath(path_data, "randompoly-s$(v.f_var)-d$(v.f_d)-N$(v.f_N)"))
    f = convert(v.T, f)
    rank, blocksize, nblocks = v.gparams
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
    time_gd = @elapsed interpolate(f, g, v.regtype, (; val=v.reg);
        optimizer_params, v.nepochs, v.batchsize,
        lossfunc_symb=:lse,
        lossfunc_param=v.lossfunc_param,
        show_progress=true)
    time_hs_norm2_gd = @elapsed hs_norm2_gd = GloptiNets.HSnorm2(g)
    time_linfsamples_gd = @elapsed linfsamples_gd = maximum(GloptiNets.l∞norm_samples(f, g; nsamples=4096) for _ ∈ 1:10)

    usegpu && ((f, g) = (cpu(f), cpu(g)))

    time_mom_gd = @elapsed (; vals_mom, vals_mean) = GloptiNets.mom_estimator(f, g, ωs, ns, ps; nbatch=mom_batchsize, batchsize=mom_nbatches)
    dotprod_bound = GloptiNets.dotproduct_bound(f, g, proba)
    stdbound_gd = √(dotprod_bound + hs_norm2_gd)

    certif_gd_mom, certif_gd_mean = vals_mom, vals_mean
    certif_gd_mean_var = stdbound_gd / √(mom_nbatches * mom_batchsize * 0.018)
    certif_gd_mom_var = 2stdbound_gd / √mom_batchsize

    usegpu && ((f, g) = (gpu(f), gpu(g)))

    time_bfgs = @elapsed GloptiNets.lbfgs(f, g, v.regtype, (; val=v.reg);
        nepochs=v.lbfgs_nepochs, batchsize=v.lbfgs_batchsize, iterperepochs=v.lbfgs_itermax,
        lossfunc_symb=:lse,
        lossfunc_param=10.0,
        show_progress=true)
    time_hs_norm2_bfgs = @elapsed hs_norm2_bfgs = GloptiNets.HSnorm2(g)
    time_linfsamples_bfgs = @elapsed linfsamples_bfgs = maximum(GloptiNets.l∞norm_samples(f, g; nsamples=4096) for _ ∈ 1:10)

    usegpu && ((f, g) = (cpu(f), cpu(g)))

    time_mom_bfgs = @elapsed (; vals_mom, vals_mean) = GloptiNets.mom_estimator(f, g, ωs, ns, ps; nbatch=mom_batchsize, batchsize=mom_nbatches)
    dotprod_bound_bfgs = GloptiNets.dotproduct_bound(f, g, proba)
    stdbound_bfgs = √(dotprod_bound_bfgs + hs_norm2_bfgs)

    certif_bfgs_mom, certif_bfgs_mean = vals_mom, vals_mean
    certif_bfgs_mean_var = stdbound_bfgs / √(mom_nbatches * mom_batchsize * 0.018)
    certif_bfgs_mom_var = 2stdbound_bfgs / √mom_batchsize


    cur_result = Dict(
        "certif_gd_mean" => -certif_gd_mean - certif_gd_mean_var,
        "certif_gd_mean_bias" => certif_gd_mean,
        "certif_gd_mean_var" => certif_gd_mean_var,
        "certif_gd_mom" => -certif_gd_mom - certif_gd_mom_var,
        "certif_gd_mom_bias" => certif_gd_mom,
        "certif_gd_mom_var" => certif_gd_mom_var,
        "linfsamples_gd" => linfsamples_gd,
        "hs_norm2_gd" => hs_norm2_gd,
        "mom_nsamples" => mom_nbatches * mom_batchsize,
        "time_gd" => time_gd,
        "time_hs_norm2_gd" => time_hs_norm2_gd,
        "time_mom_gd" => time_mom_gd,
        "time_linfsamples_gd" => time_linfsamples_gd,
        # BFGS
        "certif_bfgs_mean" => -certif_bfgs_mean - certif_bfgs_mean_var,
        "certif_bfgs_mean_bias" => certif_bfgs_mean,
        "certif_bfgs_mean_var" => certif_bfgs_mean_var,
        "certif_bfgs_mom" => -certif_bfgs_mom - certif_bfgs_mom_var,
        "certif_bfgs_mom_bias" => certif_bfgs_mom,
        "certif_bfgs_mom_var" => certif_bfgs_mom_var,
        "linfsamples_bfgs" => linfsamples_bfgs,
        "hs_norm2_bfgs" => hs_norm2_bfgs,
        "time_bfgs" => time_bfgs,
        "time_hs_norm2_bfgs" => time_hs_norm2_bfgs,
        "time_mom_bfgs" => time_mom_bfgs,
        "time_linfsamples_bfgs" => time_linfsamples_bfgs,
    )

    # Main results in a TOML file 
    open(joinpath(path_save, "results.toml"), "w") do f
        TOML.print(f, cur_result)
    end

    @info "Run $cur_run/$ntotal done. Result:\n$(cur_result)"
end
