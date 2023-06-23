"""
Experiments for studying ``certificates vs. # of parameters''. 
julia --project scripts/xp_fourier_params_variation.jl
"""

using GloptiNets
using Dates
using TOML

const usegpu = true
const cuda_device = 3
const path_xps = "xps/params_variation"
const path_data = "data/hnorm2"
!isdir("xps") && mkdir("xps")
!isdir(path_xps) && mkdir(path_xps)
mom_batchsize = 1_000_000
const mom_nbatches = 32

using CUDA: device!
device!(cuda_device)

config = (
    lrdecay=[
        :cos,
        # :poly,
    ],
    opt=[
        :momentum],
    lr=[
        # 8e-1,
        4e-1,
        # 2e-1,
        # 1e-1
    ],
    nepochs=[5000],
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
        (1, 2, 1),
        (2, 4, 1),
        (4, 8, 1),
        (4, 16, 1),
        (4, 32, 1),
        (8, 64, 1),
        (8, 128, 1),
        (8, 256, 1),
        # (8, 512, 1),
        # (1, 2, 2),
        # (1, 2, 4),
        # (1, 2, 8),
        # (1, 2, 16),
        # (1, 2, 32),
        # (1, 2, 64),
        # (1, 2, 128),
        # (2, 4, 4),
        # (2, 4, 8),
        # (2, 4, 16),
        # (2, 4, 32),
        # (2, 4, 64),
        # (2, 4, 128),
        # (16, 32, 32),
        # (16, 32, 128)
        # (8, 16, 8),
        # (8, 16, 16),
        # (8, 16, 32),
        # (8, 16, 64),
        # (8, 16, 128),
        # (16, 32, 64),
        # (4, 8, 8),
        # (4, 8, 16),
        # (4, 8, 32),
        # (4, 8, 64),
        # (4, 8, 128),
        # (4, 8, 256),
        # (4, 8, 512),
    ],
    regtype=["HSNormP"],
    reg=[
        0.0015625,
        # 0.003125,
        # 0.00625,
        # 0.0125,
        # 0.025,
        # 0.05,
        # 5e-9,
        # 1e-8,
        # 5e-8,
        # 1e-7,
    ],
    f_idx=[1],
    f_var=[2],
    f_d=[3,],
    f_N=[12,],
    T=[Float64,]
)

@assert length(config.f_d) == 1
@assert length(config.variances) == 1
@assert length(config.T) == 1
@assert length(config.proba_tol) == 1
d, var, T, proba_tol = config.f_d[1], config.variances[1], config.T[1], config.proba_tol[1]
γ = ones(T, d) * convert(T, var)
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

    f = GloptiNets.load(PolyTrigo, joinpath(path_data, "randompoly-s$(v.f_var)-d$(v.f_d)-N$(v.f_N)-i$(v.f_idx)"))
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

    time_interpolate = @elapsed for κ ∈ (1,)
        optimizer_params = (; optimizer_type=v.opt, optimizer_lrdecay=v.lrdecay, optimizer_lrinit=v.lr)
        interpolate(f, g, RegHSNormU, (; val=v.reg);
            optimizer_params, v.nepochs, v.batchsize,
            lossfunc_symb=:lse,
            lossfunc_param=v.lossfunc_param * κ,
            show_progress=true)
    end
    time_hs_norm2 = @elapsed hs_norm2 = GloptiNets.HSnorm2(g)
    time_linfsamples = @elapsed linfsamples = maximum(GloptiNets.l∞norm_samples(f, g; nsamples=2048) for _ ∈ 1:10)

    usegpu && ((f, g) = (cpu(f), cpu(g)))

    time_mom = @elapsed (; vals_mom, vals_mean) = GloptiNets.mom_estimator(f, g, ωs, ns, ps)
    dotprod_bound = GloptiNets.dotproduct_bound(f, g, proba)
    stdbound = √(dotprod_bound + hs_norm2)

    cur_result = Dict(
        "bound_cheby" => -vals_mean - stdbound / √(mom_nbatches * mom_batchsize * 0.018),
        "bound_mom" => -vals_mom - 2stdbound / √mom_batchsize,
        "hs_norm2" => hs_norm2,
        "vals_mom" => vals_mom,
        "vals_mean" => vals_mean,
        "mom_nsamples" => mom_nbatches * mom_batchsize,
        "time_interpolate" => time_interpolate,
        "time_hs_norm2" => time_hs_norm2,
        "time_mom" => time_mom,
        "linfsamples" => linfsamples,
        "time_linfsamples" => time_linfsamples,
    )

    # Main results in a TOML file 
    open(joinpath(path_save, "results.toml"), "w") do f
        TOML.print(f, cur_result)
    end

    @info "Run $cur_run/$ntotal done. Result:\n$(cur_result)"
end