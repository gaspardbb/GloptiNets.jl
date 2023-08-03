abstract type ObjFunc{T<:AbstractFloat,D<:Domain} end
abstract type AbstractPoly{T,U,D<:Domain} <: ObjFunc{T,D} end

"""
A polynomial given in the Chebychev basis.

# Notations
- `m` is the number of coefficients
- `d` is the dimension
"""
struct PolyCheby{T<:AbstractFloat,U<:Integer,V<:AbstractVector{T},M<:AbstractMatrix{U}} <: AbstractPoly{T,U,ℕ}
    offset::T
    coefficients::V  # m
    frequencies::M   # d, m

    function PolyCheby(offset, coefficients, frequencies; nochecks=false)
        if !nochecks  # To avoid indexing on CuArrays
            @assert size(coefficients, 1) == size(frequencies, 2)
            @assert allunique(eachcol(frequencies)) "The frequencies must be unique."
            @assert all(any(ω .≠ 0) for ω ∈ eachcol(frequencies)) "The zero frequency should not be included."
        end

        new{eltype(coefficients),eltype(frequencies),typeof(coefficients),typeof(frequencies)}(offset, coefficients, frequencies)
    end
end
offset(f::PolyCheby) = f.offset
coefficients(f::PolyCheby) = f.coefficients
frequencies(f::PolyCheby) = f.frequencies
ncoeffs(f::PolyCheby) = size(coefficients(f), 1)
dim(f::PolyCheby) = size(frequencies(f), 1)

Base.convert(::Type{Any}, f::PolyCheby) = f
Base.convert(::Type{T}, f::PolyCheby) where {T} = PolyCheby(
    convert(T, offset(f)),
    convert(Vector{T}, coefficients(f)),
    frequencies(f);
    nochecks=true
)
gpu(f::PolyCheby) = PolyCheby(
    offset(f),
    CuArray(coefficients(f)),
    CuArray(frequencies(f));
    nochecks=true
)
cpu(f::PolyCheby) = PolyCheby(
    offset(f),
    collect(coefficients(f)),
    collect(frequencies(f));
    nochecks=true
)

# TODO: add fourier or cheby in the filename
save(f::PolyCheby, filename) = jldsave("$filename.jld2"; offset=offset(f), coefficients=coefficients(f), frequencies=frequencies(f))
load(::Type{PolyCheby}, filename) = begin
    file = jldopen("$filename.jld2", "r+")
    f = PolyCheby(file["offset"], file["coefficients"], file["frequencies"])
    close(file)
    f
end

function evaluate_cos!(r::AbstractVector{T}, f::PolyCheby{T}, U::AbstractMatrix{T}) where {T}
    d, m, n = dim(f), ncoeffs(f), size(U, 2)

    sum!(reshape(r, 1, :), (
        dropdims(prod(cospi.(2(
                reshape(frequencies(f), d, m, 1) .* reshape(U, d, 1, n)  # d, m, n
            )), dims=1), dims=1)  # m, n
    ) .* coefficients(f))  # n
    r .+= offset(f)
    nothing
end

"""
    evaluate_cos(f, U)

Evaluate the polynomial when composed with cosine, i.e. we require ``U ∈ (0, 1/2)``, and we compute 
```
    f.(cos(2π U))
```

# Remarks
Instead of evaluating with Clenshaw's algorithm, we use the fact that 
```
    Hₙ(cos x) = cos(n x)
```
which should be better suited to GPU computation when we move the data to the GPU. 
"""
function evaluate_cos(f::PolyCheby{T}, U::AbstractMatrix{T}) where {T}
    @assert size(U, 1) == dim(f)
    # @assert all(0 .<= U .<= 1 / 2)
    r = similar(coefficients(f), size(U, 2))
    evaluate_cos!(r, f, U)
    r
end

"""
    evaluate(f, X)

Evaluate the polynomial for input ``X ∈ (-1, 1)``.
"""
function evaluate(f::PolyCheby{T}, X::AbstractMatrix{T}) where {T}
    @assert size(X, 1) == dim(f)
    @assert all(-1 .<= X .<= 1)

    evaluate_cos(f, acos.(X) / (2π))
end

"Returns the Chebychev coefficient of `f` in `ω ∈ ωs`."
function cheby(f::PolyCheby{T,U}, ωs::AbstractMatrix{U}) where {T,U}
    @assert size(ωs, 1) == dim(f)

    rs = zeros(T, size(ωs, 2))
    for (i, ω) ∈ enumerate(eachcol(ωs))
        all(ω .== 0) && (rs[i] = offset(f); continue)
        ind = findfirst(ν -> all(ν .== ω), eachcol(frequencies(f)))
        ind !== nothing && (rs[i] = coefficients(f)[ind]; continue)
    end

    rs
end

"A lower bound on `f` corresponding to the lower bound hierarchy when `g=0`."
function lowerbound(f::PolyCheby)
    offset(f) - sum(abs.(coefficients(f)))
end

"Computes the value and gradient of `f ∘ cos` in `x`. Follows Zygote's `withgradient` API."
function withgradient_cos(f::PolyCheby{T,U}, x::AbstractVector{T}) where {T,U}
    ∏ = dropdims(prod(cospi.(2 * frequencies(f) .* x), dims=1), dims=1)  # m
    val = offset(f) + sum(coefficients(f) .* ∏)
    grads = dropdims(sum(coefficients(f)' .* (-2π * frequencies(f)) .* tan.(2π * frequencies(f) .* x) .* ∏', dims=2), dims=2)  # d
    (val=val, grad=(grads,))
end

@testitem "Gradient of f" begin
    using GloptiNets: withgradient_cos

    f = PolyCheby(
        randn(),
        randn(3),
        [
            1 0 2
            0 1 1
        ]
    )

    x, y = randn(2)
    Δ = 1e-4
    h = [
        x x+Δ x-Δ x x
        y y y y+Δ y-Δ
    ]
    (val_g, grad_g) = withgradient_cos(f, [x, y])
    val, x₊, x₋, y₊, y₋ = evaluate_cos(f, h)
    @test val ≈ val_g
    @test [
        (x₊ - x₋) / (2Δ)
        (y₊ - y₋) / (2Δ)
    ] ≈ grad_g[1] rtol = 1e-4
end

"Returns `fg!` which can then be used with `Optim.optimize`."
function get_fg!(f::PolyCheby)
    function fg!(F, G, x)
        (; val, grad) = withgradient_cos(f, x)
        G !== nothing && copy!(G, grad[1])
        F !== nothing && return val
        nothing
    end
    fg!
end

function candidate_min(f::PolyCheby, x₀)
    res = Optim.optimize(
        Optim.only_fg!(get_fg!(f)),
        x₀, Optim.BFGS())
    x★ = Optim.minimizer(res)
    f★ = evaluate_cos(f, [x★;;])[1]
    x★ = cospi.(2Optim.minimizer(res))
    f★, x★
end


_cheby_recursion(x, n) = n == 0 ? one(x) : (n == 1 ? x : 2x * _cheby_recursion(x, n - 1) - _cheby_recursion(x, n - 2))

"Evaluate the polynomial on an abstract vector, with operation compatible with DynamicPolynomials."
function evaluate_poly(f::PolyCheby, x::AbstractVector)
    @assert length(x) == dim(f)

    r = offset(f)
    for (c, ω) ∈ zip(coefficients(f), eachcol(frequencies(f)))
        r += c * prod(
            _cheby_recursion(x[i], ω[i]) for i ∈ 1:dim(f)
        )
    end
    r
end

@testitem "evaluate poly with DynamicPolynomials" begin
    using GloptiNets: evaluate_poly
    using DynamicPolynomials
    f = PolyCheby(
        0.5,
        [1.0, 1.5, 2.0, 2.5, 3.0],
        [
            0 1 0 3 4
            1 0 2 1 5
        ]
    )

    @polyvar x[1:dim(f)]
    f_poly = evaluate_poly(f, x)
    z = rand(dim(f), 10) * 2 .- 1
    @test evaluate(f, z) ≈ f_poly.(eachcol(z))
end

"""
Returns a polynomial which approximates a function in `H`, the RKHS associated to `proba`. It has frequencies for all the `ω` for which `‖ω‖∞ ≤ N` (Koroborov space). Each frequency has norm `p̂(ω)`, i.e. `|f̂(ω)|²/p̂(ω) = p̂(ω)`, so that taking all the frequencies converge to 1.
"""
function random_polycheby(proba::ApproxBesselSampler{T,ℕ}, N; offset0=false)::PolyCheby{T,Int,Vector{T},Matrix{Int}} where {T}  # Add type annotation to make random_pos_polytrigo type stable
    # TODO: merge with random_polytrigo
    N > minimum(support(proba)) && error("N must be smaller than the minimum support of the distribution.")
    ωs = reduce(hcat,
        collect(ω) for ω ∈ Iterators.product((0:N for _ ∈ 1:dim(proba))...)
    )[:, 2:end]  # discard 0
    coeffs = rand((-one(T), one(T)), size(ωs, 2)) .* pdf(proba, ωs)

    z0 = offset0 ? zero(T) : prod(proba.weights[i][1] for i ∈ 1:dim(proba))
    PolyCheby(z0, coeffs, ωs; nochecks=true)
end

"#TODO: merge with `random_pos_polytrigo`."
function random_pos_polycheby(proba::ApproxBesselSampler{T,ℕ}, N, target_hnorm2) where {T}
    f = random_polycheby(proba, N; offset0=true)
    R² = Hnorm2(f, proba)
    q₀ = prod(proba.weights[i][1] for i ∈ 1:dim(proba))
    f★, _ = estimate_min(f, 50; show_progress=false)
    # Condition for the Hnorm to be target_hnorm2: α² R² + f★²/q₀ = target_hnorm2
    α = √(target_hnorm2 / (R² + abs2(f★) / q₀))
    # Condition for the polynomial to be positive: c + α * f = 0
    c = -α * f★
    PolyCheby(c, α * f.coefficients, f.frequencies; nochecks=true)
end


"
    PolyTrigo(offset, coefficients, frequencies)

A trigonometric polynomial of the form
``
    f(x) = c₀ + ∑ⱼ ½ cⱼ e^(2 im π ωⱼᵀx) + ½ c̄ⱼ e^(-2 im π ωⱼᵀx)
``
"
struct PolyTrigo{T<:AbstractFloat,U<:Integer,V<:AbstractVector{Complex{T}},M<:AbstractMatrix{U}} <: AbstractPoly{T,U,ℤ}
    offset::T
    coefficients::V
    frequencies::M

    function PolyTrigo(offset, coefficients, frequencies; nochecks=false)
        if !nochecks
            d, m = size(frequencies)
            @assert m == size(coefficients, 1)
            @assert zeros(eltype(frequencies), d) ∉ eachcol(frequencies) "DC component should not be in the frequencies"
            @assert !any(-ω ∈ eachcol(frequencies) for ω ∈ eachcol(frequencies)) "There is one frequency ω which is also present with -ω"
        end
        new{real(eltype(coefficients)),eltype(frequencies),typeof(coefficients),typeof(frequencies)}(offset, coefficients, frequencies)
    end
end
offset(f::PolyTrigo) = f.offset
frequencies(f::PolyTrigo) = f.frequencies
coefficients(f::PolyTrigo) = f.coefficients
ncoeffs(f::PolyTrigo) = size(coefficients(f), 1)
dim(f::PolyTrigo) = size(frequencies(f), 1)
degrees(f::PolyTrigo) = dropdims(sum(abs.(frequencies(f)), dims=1), dims=1)

Base.convert(::Type{Any}, f::PolyTrigo) = f
Base.convert(::Type{T}, f::PolyTrigo) where {T} = PolyTrigo(
    convert(T, offset(f)),
    convert(Vector{Complex{T}}, coefficients(f)),
    frequencies(f);
    nochecks=true
)
gpu(f::PolyTrigo) = PolyTrigo(
    offset(f),
    CuArray(coefficients(f)),
    CuArray(frequencies(f));
    nochecks=true
)
cpu(f::PolyTrigo) = PolyTrigo(
    offset(f),
    collect(coefficients(f)),
    collect(frequencies(f));
    nochecks=true
)

function (f::PolyTrigo{T,U})(x::AbstractVector{T}) where {T,U}
    # PERF: calling f.(eachcol(X)) is type unstable!
    @assert size(x, 1) == dim(f)
    offset(f) + sum(real(
        coefficients(f) .* cispi.(2 * frequencies(f)' * x)
    ))
end

function evaluate!(r::AbstractVector{T}, f::PolyTrigo{T,U}, X::AbstractMatrix{T}) where {T,U}
    sum!(reshape(r, :, 1), real(
        cispi.(2 * X' * frequencies(f)) .* transpose(coefficients(f))
    ))
    r .+= offset(f)
    nothing
end

function (f::PolyTrigo{T,U})(X::AbstractMatrix{T}) where {T,U}
    @assert size(X, 1) == dim(f)
    r = similar(X, size(X, 2))
    evaluate!(r, f, X)
    r
end

"Computes the value and gradient of `f` in `x`. Follows Zygote's `withgradient` API."
function withgradient(f::PolyTrigo{T,U}, x::AbstractVector{T}) where {T,U}
    r = coefficients(f) .* cispi.(2 * frequencies(f)' * x)
    d = 2im * π * frequencies(f)
    val = offset(f) + sum(real(r))
    grads = dropdims(sum(real(transpose(r) .* d), dims=2), dims=2)
    (val=val, grad=(grads,))
end

function fnorm(f::PolyTrigo)
    abs(offset(f)) + sum(abs.(coefficients(f)))
end

function fourier(f::PolyTrigo{T,U}, ωs::AbstractMatrix{U}) where {T,U}
    @assert dim(f) == size(ωs, 1)
    rs = zeros(Complex{T}, size(ωs, 2))

    for (i, ω) ∈ enumerate(eachcol(ωs))
        all(ω .== 0) && (rs[i] = offset(f); continue)
        for (j, ν) ∈ enumerate(eachcol(frequencies(f)))
            ν == ω && (rs[i] = coefficients(f)[j] / 2; break)
            ν == -ω && (rs[i] = conj(coefficients(f)[j]) / 2; break)
        end
    end

    rs
end

"Evaluate the polynomial on an abstract vector, with operation compatible with DynamicPolynomials."
function evaluate_poly(f::PolyTrigo, x::AbstractVector)
    @assert length(x) == 2dim(f)

    z = x[1:dim(f)]
    z̄ = x[dim(f)+1:end]

    r = offset(f)
    for (c, ω) ∈ zip(
        vcat(
            coefficients(f), conj(coefficients(f))
        ), Iterators.flatten((
            eachcol(frequencies(f)),
            eachcol(-frequencies(f)),
        ))
    )
        r += c / 2 * prod(
            (ω[i] >= 0 ? z[i]^ω[i] : z̄[i]^(-ω[i])) for i ∈ 1:dim(f)
        )
    end
    r
end

@testitem "evaluate poly with DynamicPolynomials" begin
    # TODO write a random poly and that test will be completed  
    using GloptiNets: evaluate_poly
    using DynamicPolynomials
    f = PolyTrigo()

    @polyvar x[1:2dim(f)]
    f_poly = evaluate_poly(f)
    z = randn(2)
    @test f(z) ≈ f_poly([
        cispi.(2z)
        cispi.(-2z)
    ]) broken = true
end

"""
Returns a polynomial which approximates a function in `H`, the RKHS associated to `proba`. It has frequencies for almost all the `ω` of degree smaller than `N`. Each frequency has norm `p̂(ω)`, i.e. `|f̂(ω)|²/p̂(ω) = p̂(ω)`, so that taking all the frequencies converge to 1.

# Remarks
If `S` is the support of `proba`, we consider the frequencies in `(-S, S)ᵈ⁻¹ × (1, S)` rather than `(-S, S)ᵈ` as the latter would require checking if `-ω` is in the set for each `ω` in the set. This is not necessary for the former.
"""
function random_polytrigo(proba::ApproxBesselSampler{T,ℤ}, N)::PolyTrigo{T,Int,Vector{Complex{T}},Matrix{Int}} where {T}  # Add type annotation to make random_pos_polytrigo type stable
    ωs = reduce(hcat,
        collect(ω) for ω ∈ Iterators.product((-Sᵢ:Sᵢ for Sᵢ ∈ support(proba)[1:end-1])..., 1:support(proba)[end]) if sum(abs.(ω)) ≤ N
    )
    coeffs = randn(Complex{T}, size(ωs, 2))
    coeffs .= coeffs ./ abs.(coeffs) .* pdf(proba, ωs)

    z0 = prod(proba.weights[i][1] for i ∈ 1:dim(proba)) * rand((-one(T), one(T)))
    PolyTrigo(z0, coeffs, ωs)
end

"""
# TODO This is way too complex. Do as with PolyCheby. 
"""
function random_pos_polytrigo(proba::ApproxBesselSampler{T,ℤ}, N, target_hnorm2; nrounds_max=100) where {T}
    f = random_polytrigo(proba, N)
    q₀ = prod(proba.weights[i][1] for i ∈ 1:dim(proba))
    f̂₀ = offset(f)
    R = Hnorm2(f, proba) - abs2(f̂₀) / q₀

    α = one(T)
    c = f̂₀
    for _ ∈ 1:nrounds_max
        residual = target_hnorm2 - abs2(f̂₀ - c) / q₀
        residual < 0 && (α = zero(T); break)
        α = √(residual / R)
        f.coefficients .*= α # √α
        R = α^2 * R
        c, _ = estimate_min(f, 50; show_progress=false)
        abs(α - one(T)) ≤ 1e-6 && break
    end
    abs(α - one(T)) > 1e-6 && @warn "Did not converge to target norm. α = $α"
    PolyTrigo(f̂₀ - c, f.coefficients, f.frequencies; nochecks=true)
end

save(f::PolyTrigo, filename) = jldsave("$filename.jld2"; offset=offset(f), coefficients=coefficients(f), frequencies=frequencies(f))
load(::Type{PolyTrigo}, filename) = begin
    file = jldopen("$filename.jld2", "r+")
    f = PolyTrigo(file["offset"], file["coefficients"], file["frequencies"])
    close(file)
    f
end

"A lower bound on `f` corresponding to the lower bound hierarchy when `g=0`"
function lowerbound(f::PolyTrigo)
    offset(f) - sum(abs.(coefficients(f)))
end

function get_fg!(f::ObjFunc)
    function fg!(F, G, x)
        (; val, grad) = withgradient(f, x)
        G !== nothing && copy!(G, grad[1])
        F !== nothing && return val
        nothing
    end
    fg!
end

function candidate_min(f::ObjFunc, x₀)
    res = Optim.optimize(
        Optim.only_fg!(get_fg!(f)),
        x₀, Optim.BFGS())
    x★ = Optim.minimizer(res) .% 1.0  # Just in case there are numerical issues
    f(x★), x★
end


_candidate_position(f::PolyCheby{T}) where {T} = rand(T, dim(f)) .- 0.5
_candidate_position(f::PolyTrigo{T}) where {T} = rand(T, dim(f))
function estimate_min(f, ntries; show_progress=true)
    f★, x★ = +Inf, zeros(dim(f))
    pbar = Progress(ntries; enabled=show_progress)
    for _ ∈ 1:ntries
        fcand, xcand = candidate_min(f, _candidate_position(f))
        (fcand < f★) && ((f★, x★) = (fcand, xcand))
        next!(pbar; showvalues=[(:f★, f★)])
    end
    finish!(pbar)
    f★, x★
end

_constructor(f::PolyTrigo, ps...) = PolyTrigo(ps...; nochecks=true)
_constructor(f::PolyCheby, ps...) = PolyCheby(ps...; nochecks=true)
function estimate_max(f, ntries; showprogress=true)
    fm = _constructor(f, -offset(f), -coefficients(f), frequencies(f))
    f★, x★ = estimate_min(fm, ntries; show_progress=showprogress)
    -f★, x★
end

function isgpu(f::AbstractPoly) end
isgpu(f::PolyCheby) = typeof(coefficients(f)) <: CuArray
isgpu(f::PolyTrigo) = typeof(coefficients(f)) <: CuArray

Base.show(io::IO, ::MIME"text/plain", f::AbstractPoly) = print(io,
    """
    $(nameof(typeof(f))):
    • ncoeffs: $(ncoeffs(f))
    • dim: $(dim(f))\
    """
)
Base.show(io::IO, f::AbstractPoly) = print(io,
    """
    $(nameof(typeof(f)))(\
    nc=$(ncoeffs(f)),\
    d=$(dim(f)))\
    """
)