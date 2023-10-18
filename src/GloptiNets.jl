module GloptiNets

using TestItems

using LinearAlgebra
import Random
using Flux.NNlib: batched_mul
using CUDA

using Optim
using ProgressMeter
using Bessels: besselix, besseli0x, besseli1x, besseli
using JLD2: jldsave, jldopen
using Flux: Flux
using ParameterSchedulers
using Statistics: mean, median
using Distributions: Categorical

export get_optimizer, interpolate, NoReg, RegHSNormU, RegHSNormP, RegOrthNorm

include("besselsampler.jl")
export ApproxBesselSampler, ℕ, ℤ, samplesprobas
include("psdmodels.jl")
export PSDBlockBesselCheby, PSDBlockBesselFourier
export gpu, cpu
include("polynomials.jl")
export PolyCheby, PolyTrigo
export evaluate, evaluate_cos, cheby, fourier
export blocksize, rank, nblocks, dim, ncoeffs, offset, coefficients_scaled  # accessors
include("besselmixture.jl")
export BesselMixtureTrigo
include("experimental.jl")

#= 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Train functions and other utilities                                      │
  └──────────────────────────────────────────────────────────────────────────┘
 =#

_realfunc(f::PolyCheby, x) = x
_realfunc(f::PolyTrigo, x) = real(x)
_coeff(f::PolyCheby{T}) where {T} = one(T)
_coeff(f::PolyTrigo{T}) where {T} = 2one(T)

"""
Given `f` a Chebychev or trigonometric polynomial, `g` a Bessel model and `proba` a probability distribution, computes the dot product
```
    ⟨f, f - 2g⟩
```
in the RKHS induced by `proba`. This is useful to compute ``‖f - g‖²`` as 
```
‖f - g‖² = ⟨f, f - 2g⟩ + ‖g‖².
```
"""
function dotproduct_bound(f, g, proba) end
function dotproduct_bound(f::AbstractPoly{T,U,D}, g::AbstractBlockPSDModel{T,D}, proba::ApproxBesselSampler{T,D}) where {T,U,D}
    # Without offset c
    ĝ = _basis_proj(g, hcat(zeros(U, dim(f)), frequencies(f)))
    ps = pdf(proba, hcat(zeros(U, dim(f)), frequencies(f)))

    dc_part = _realfunc(f, offset(f) * (offset(f) - 2ĝ[1]) / ps[1])
    ac_part = @views (
        _coeff(f) * _realfunc(f, sum(
            (1 / _coeff(f)) * conj(coefficients(f)) .* ((1 / _coeff(f)) * coefficients(f) .- 2ĝ[2:end]) ./ ps[2:end]
        ))
    )

    dc_part + ac_part
end
function dotproduct_bound(f::BesselMixtureTrigo{T}, g::AbstractBlockPSDModel{T,ℤ}, proba::ApproxBesselSampler{T,ℤ}) where {T}
    @assert scale(f) .== proba.scale[1] "Implemented for Bessel mixture with uniform scale"
    hnorm2(f)
end

"""
A bound on the variance of the estimator defined with
```
X = |û(ω)| / p(ω), u = f - g.
```
The variance is upper bounded by ``‖u‖²`` in the norm induced by the probability ``p``. 
"""
Hnorm2_bound(f, g, proba) = dotproduct_bound(f, g, proba) + HSnorm2(g)

"""
Hilbert norm in the RKHS associated to `proba` for a polynomial `f`.
"""
function Hnorm2(f::AbstractPoly{T,U,D}, proba::ApproxBesselSampler{T,D}) where {T,U,D}
    ps = pdf(proba, hcat(zeros(U, dim(f)), frequencies(f)))

    abs2(offset(f)) / ps[1] + @views (
        (1 / _coeff(f)) * sum(
            abs2.(coefficients(f)) ./ ps[2:end]
        )
    )
end


_grid(::Type{ℕ}, n) = 0:n
_grid(::Type{ℤ}, n) = -n:n
"""
Approximation of the F-norm and the H̄² norm for `f`, `g`, and `f - g`, by computing their coefficients on a grid of size `n`.
"""
function norms_numapprox(f::AbstractPoly{T,U,D}, g::AbstractBlockPSDModel{T,D}, proba::ApproxBesselSampler{T,D}, n::Int) where {T,U,D}
    ωs = reduce(hcat, map(collect,
        Iterators.product(
            Iterators.repeated(_grid(D, n), dim(f))...
        )
    ))
    ps = pdf(proba, ωs)
    f̂ = _basis_proj(f, ωs)
    ĝ = _basis_proj(g, ωs)
    diff_fnorm = sum(abs.(f̂ .- ĝ))
    diff_hnorm = √sum(abs2.(f̂ .- ĝ) ./ ps)
    f_fnorm = sum(abs.(f̂))
    f_hnorm = √sum(abs2.(f̂) ./ ps)
    g_fnorm = sum(abs.(ĝ))
    g_hnorm = √sum(abs2.(ĝ) ./ ps)
    dotproduct = sum(conj(f̂) .* ĝ ./ ps)
    @show dotproduct
    (;
        diff_fnorm=diff_fnorm,
        diff_hnorm=diff_hnorm,
        f_fnorm=f_fnorm,
        f_hnorm=f_hnorm,
        g_fnorm=g_fnorm,
        g_hnorm=g_hnorm
    )
end

_basis_proj(f::Union{AbstractPoly{T,U,ℕ},AbstractBlockPSDModel{T,ℕ}}, ωs) where {T,U} = cheby(f, ωs)
_basis_proj(f::Union{ObjFunc{T,ℤ},AbstractBlockPSDModel{T,ℤ}}, ωs) where {T} = fourier(f, ωs)
"""
Approximate the F-norm with a median of mean. Combined with the variance, this allows a bound with confidence `1 - δ` on the mean of the random variable being estimated.

# Parameters
- `nsamples`: number of frequencies sampled to compute the MoM
- `nbatch`: number of batch we take the median on. With `nbatch=32`, the confidence is `delta < 0.03`. Having `rem(nsamples, nbatch) = 0` enables taking exactly `nsamples`. 
"""
function fnorm_approx(f::AbstractPoly{T,U,D}, g::AbstractBlockPSDModel, proba::ApproxBesselSampler{T,D}; nsamples=1024, nbatch=32) where {T,U,D<:Domain}
    (; vals_mom, batchsize, vals_mean) = mom_estimator(f, g, proba; nsamples=nsamples, nbatch=nbatch)
    σ = √Hnorm2_bound(f, g, proba)
    bound = vals_mom + 2σ / √batchsize
    confidence = 1 - exp(-nbatch / 8)
    bound_cheby = vals_mean + σ / √(nbatch * batchsize * (1 - confidence))
    (; bound_mom=bound, bound_cheby=bound_cheby,
        σ=σ, confidence=confidence,
        vals_mom=vals_mom, vals_mean=vals_mean)
end

function mom_estimator(f::AbstractPoly{T,U,D}, g::AbstractBlockPSDModel, proba::ApproxBesselSampler{T,D}; nsamples=1024, nbatch=32) where {T,U,D<:Domain}
    batchsize = div(nsamples, nbatch)
    nsamples_eff = nbatch * batchsize

    ωs, ns, ps = samplesprobas_bycat(proba, nsamples_eff)

    mom_estimator(f, g, ωs, ns, ps; nbatch=nbatch, batchsize=batchsize)
end

function mom_estimator(f::ObjFunc, g::AbstractBlockPSDModel, ωs, ns, ps; nbatch, batchsize)
    r̂ = abs.(
        _basis_proj(f, ωs) .- _basis_proj(g, ωs)
    ) ./ ps
    vals_mom = mom(r̂, ns, nbatch, batchsize)
    (; vals_mom=vals_mom, batchsize=batchsize, vals_mean=sum(r̂ .* ns) / sum(ns))
end

"Gives an optimizer which can be passed to Flux's optimise. Allows for writing the optimizer to a config file."
function get_optimizer(type::Symbol, lrdecay::Symbol, lrinit::AbstractFloat, nepochs::Integer)
    opt = if type == :momentum
        Flux.Optimise.Momentum()
    elseif type == :descent
        Flux.Optimise.Descent()
    elseif type == :adam
        Flux.Optimise.Adam()
    else
        error("Unknown optimizer: $(type)")
    end
    if lrdecay == :poly
        ParameterSchedulers.Scheduler(Poly(lrinit, 1, nepochs), opt)
    elseif lrdecay == :cos
        ParameterSchedulers.Scheduler(CosAnneal(; λ0=lrinit, λ1=0.0, period=nepochs), opt)
    elseif lrdecay == :constant
        opt
    else
        error("Unknown scheduler: $(lrdecay)")
    end
end

"Checks that no items in a collection (usually Flux' Params objects) have NaN in them."
hasnan(grads) = any(any(isnan.(v)) for v in values(grads) if typeof(v) <: Array)

_samples!(X, g::PSDBlockBesselCheby) = begin
    Random.rand!(X)
    @. X = acos(X * 2.0 - 1.0) / 2π
end
_samples!(X, g::PSDBlockBesselFourier) = Random.rand!(X)
"Create samples of a grid in dimension `d` with `nᵈ` values regularly spaced."
grid(::Type{T}, d, n) where {T} = begin
    out = zeros(T, d, n^d)
    for k in 1:d
        r, R = n^(d - k), n^(k - 1)
        out[k, :] = reduce(hcat,
            reduce(hcat, T(xᵢ) * ones(T, r)' for xᵢ ∈ LinRange(0.0, 1.0, n + 1)[1:end-1])::LinearAlgebra.Adjoint{T,Vector{T}}
            for _ ∈ 1:R)::LinearAlgebra.Adjoint{T,Vector{T}}
    end
    out
end
function _samples_deterministic!(X, g::PSDBlockBesselFourier{T}) where {T}
    dim = size(X, 1)
    nperdim = Int(floor(Float64(size(X, 2))^(1 / dim)))
    ntot = nperdim^dim * dim
    grid_cpu = grid(T, dim, nperdim)
    copyto!(X, grid_cpu[1:ntot])
    Random.rand!(X[ntot+1:end])
end

_evaluate!(r, h::PolyCheby, X) = evaluate_cos!(r, h, X)
_evaluate!(r, h::Union{PolyTrigo,BesselMixtureTrigo}, X) = evaluate!(r, h, X)
_evaluate(h::Union{PolyCheby,PSDBlockBesselCheby}, X) = evaluate_cos(h, X)
_evaluate(h::Union{PolyTrigo,BesselMixtureTrigo,PSDBlockBesselFourier}, X) = h(X)

abstract type AbstractReg end
function init end
function update! end
function loss end

struct NoReg <: AbstractReg end
init(::Type{NoReg}, g, params) = NoReg()
update!(reg::NoReg, g) = nothing
loss(reg::NoReg, g) = zero(eltype(g))

struct RegHSNormU{T} <: AbstractReg
    val::T
end
init(::Type{RegHSNormU}, g, params) = RegHSNormU(params.val)
update!(reg::RegHSNormU, g) = nothing
loss(reg::RegHSNormU, g) = reg.val * √HSnorm2_upper(g)

struct RegHSNormP{T} <: AbstractReg
    val::T
end
init(::Type{RegHSNormP}, g, params) = RegHSNormP(params.val)
update!(reg::RegHSNormP, g) = nothing
loss(reg::RegHSNormP, g) = reg.val * √HSnorm2_proxy(g)

struct RegOrthNorm{T,M<:AbstractArray{T,2},F<:Factorization{T}} <: AbstractReg
    Z::M
    C::F
    val::T
end

_kernel_func(x, y, g::PSDBlockBesselCheby) = besselkernel_cheby_cos(x, y; γ=variances(g))
_kernel_func(x, y, g::PSDBlockBesselFourier) = besselkernel_fourier(x, y; γ=variances(g))
function init(::Type{RegOrthNorm}, g::AbstractBlockPSDModel{T}, params) where {T}
    (; nsamples, val) = params
    Z = Array{T}(undef, dim(g), nsamples)
    isgpu(g) && (Z = CuArray(Z))
    C = cholesky(_kernel_func(Z, Z, g) .^ 2 + regularization(g) * I)
    RegOrthNorm(Z, C, val)
end

function update!(reg::RegOrthNorm, g)
    _samples!(reg.Z, g)
    reg.C.U .= cholesky(_kernel_func(reg.Z, reg.Z, g) .^ 2 + regularization(g) * I).U
    nothing
end

function loss(reg::RegOrthNorm, g)
    reg.val * √(HSnorm2_upper(g) - sum(abs2.(reg.C.L \ _evaluate(g, reg.Z))))
end

"Interpolate a *positive* polynomial."
function interpolate(f::ObjFunc{T,D}, g::AbstractBlockPSDModel{T,D}, reg_type, reg_params;
    optimizer_params, nepochs, batchsize,
    lossfunc_symb=:mse,
    lossfunc_param=1.0,
    show_progress=true
) where {T<:AbstractFloat,D<:Domain}
    (; optimizer_type, optimizer_lrdecay, optimizer_lrinit) = optimizer_params
    optimizer = get_optimizer(optimizer_type, optimizer_lrdecay, optimizer_lrinit, nepochs)
    @assert !xor(isgpu(f), isgpu(g))

    params = Flux.params(trainable_params(g))
    X = Array{T}(undef, dim(g), batchsize)
    yf = Vector{T}(undef, batchsize)
    if isgpu(f)
        X = CuArray(X)
        yf = CuArray(yf)
    end
    # Not sure if it is advised to have a conditional loss function?
    lossfunc = lossfunc_symb == :mse ? (
        (ŷ, y) -> Flux.mse(ŷ, y)
    ) : (
        (ŷ, y) -> 1 / lossfunc_param * Flux.logsumexp(lossfunc_param * abs.(ŷ - y)) - 1 / lossfunc_param * log(batchsize)  # So that the LSE is closer to the max of the batch of points
    )

    reg = init(reg_type, g, reg_params)

    pbar = Progress(nepochs; enabled=show_progress)
    # r = zero(T)
    for _ ∈ 1:nepochs
        _samples!(X, g)
        _evaluate!(yf, f, X)
        update!(reg, g)

        # r += lossfunc(_evaluate(g, X), yf)
        loss_epoch, grads_epoch = Flux.withgradient(params) do
            (
                lossfunc(_evaluate(g, X), yf) + loss(reg, g)
                # Previous setup: 
                # regfunc_ps * √HSnorm2_upper(g)
            )
        end
        hasnan(grads_epoch) && break
        Flux.Optimise.update!(optimizer, params, grads_epoch)
        next!(pbar; showvalues=[(:loss, loss_epoch)])
        # next!(pbar; showvalues=[(:r, r)])
    end
end

using Optim

"From a `Params` object, make a vector. Requires all the params to have the same type. Works for params on CUDA."
function params2vec(params)
    reduce(vcat, vec(p) for p ∈ params)
end

function params2vec!(v, params)
    i = 1
    for p ∈ params
        s = length(reinterpret(eltype(v), p))
        v[i:i+s-1] = reinterpret(eltype(v), p)
        i += s
    end
end

"Copy the content of a vector into a `Params` object. Performs no check."
function vec2params!(params, v)
    i = 1
    for p ∈ params
        s = length(p)
        p[:] .= v[i:i+s-1]
        i += s
    end
end

function lbfgs(f::ObjFunc{T,D}, g::AbstractBlockPSDModel{T,D}, reg_type, reg_params;
    nepochs, batchsize, iterperepochs=1000,
    lossfunc_symb=:mse,
    lossfunc_param=1.0,
    show_progress=true,
    deterministic=false
) where {T<:AbstractFloat,D<:Domain}
    @assert !xor(isgpu(f), isgpu(g))
    @assert !(deterministic && nepochs > 1) "Deterministic sampling is used, so do only one epoch."

    params = Flux.params(trainable_params(g))
    X = Array{T}(undef, dim(g), batchsize)
    yf = Vector{T}(undef, batchsize)
    if isgpu(f)
        X = CuArray(X)
        yf = CuArray(yf)
    end
    lossfunc = lossfunc_symb == :mse ? (
        (ŷ, y) -> Flux.mse(ŷ, y)
    ) : (
        (ŷ, y) -> 1 / lossfunc_param * Flux.logsumexp(lossfunc_param * abs.(ŷ - y)) - 1 / lossfunc_param * log(batchsize)  # So that the LSE is closer to the max of the batch of points
    )

    reg = init(reg_type, g, reg_params)

    pbar = Progress(nepochs; enabled=show_progress)
    for _ ∈ 1:nepochs
        deterministic ? _samples_deterministic!(X, g) : _samples!(X, g)
        _evaluate!(yf, f, X)
        update!(reg, g)

        function _get_grads!(G, x)
            vec2params!(params, x)
            grads = Flux.gradient(params) do
                lossfunc(_evaluate(g, X), yf) + loss(reg, g)
            end
            params2vec!(G, grads)
        end
        function _get_val(x)
            vec2params!(params, x)
            lossfunc(_evaluate(g, X), yf) + loss(reg, g)
        end
        optimize(_get_val, _get_grads!, params2vec(params), LBFGS(), Optim.Options(
            iterations=iterperepochs,
            store_trace=false,
            show_trace=false))

        # next!(pbar; showvalues=[(:loss, loss_epoch)])
        next!(pbar)
    end
end

function l∞norm_samples(f::ObjFunc{T}, g::AbstractBlockPSDModel{T}; nsamples=4096) where {T}
    X = Array{T}(undef, dim(g), nsamples)
    isgpu(f) && (X = CuArray(X))
    _samples!(X, g)
    maximum(abs.(_evaluate(f, X) - _evaluate(g, X)))
end

_basis_proj_diff(f::Union{AbstractPoly{T,U,ℕ},AbstractBlockPSDModel{T,ℕ}}, ωs) where {T,U} = cheby_diff(f, ωs)
_basis_proj_diff(f::Union{AbstractPoly{T,U,ℤ},AbstractBlockPSDModel{T,ℤ}}, ωs) where {T,U} = error()  #fourier_diff(f, ωs)

"""
Optimize the certificate directly. Does not work as is, as it requires a [custom rrule](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/writing_good_rules.html) for `fourier(g, ωs)` and `cheby(g, ωs)`. 
"""
function certif_opt!(g::AbstractBlockPSDModel{T,D}, f::AbstractPoly{T,U,D}, ωs, ns, ps;
    optimizer_params, nepochs,
    show_progress=true
) where {T<:AbstractFloat,U,D<:Domain}
    (; optimizer_type, optimizer_lrdecay, optimizer_lrinit) = optimizer_params
    optimizer = get_optimizer(optimizer_type, optimizer_lrdecay, optimizer_lrinit, nepochs)
    @assert !xor(isgpu(f), isgpu(g))


    params = Flux.params(trainable_params(g))
    pbar = Progress(nepochs; enabled=show_progress)
    # r = zero(T)
    for _ ∈ 1:nepochs
        yf = _basis_proj(f, ωs)
        # r += lossfunc(_evaluate(g, X), yf)
        loss_epoch, grads_epoch = Flux.withgradient(params) do
            (
                sum(abs.(
                    yf .- _basis_proj_diff(g, ωs)
                ) ./ ps .* ns) / sum(ns) + HSnorm2(g)
            )
        end
        hasnan(grads_epoch) && break
        Flux.Optimise.update!(optimizer, params, grads_epoch)
        next!(pbar; showvalues=[(:loss, loss_epoch)])
        # next!(pbar; showvalues=[(:r, r)])
    end
end

end
