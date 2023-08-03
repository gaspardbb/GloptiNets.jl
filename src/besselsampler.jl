abstract type Domain end
struct ℕ <: Domain end
struct ℤ <: Domain end

"""
Approximation of a sampler using the Bessel distribution as pdf. Instead of using
```
p(ω) = I(ω, s), ω ∈ (ℕ or ℤ),
```
where ``I`` is the modified Bessel function of the first kind, it approximates `p` on a _finite_ support. It is defined on `ℤ` by symmetry around `0`. 
"""
struct ApproxBesselSampler{T<:AbstractFloat,Domain}
    scale::Vector{T}
    weights::Vector{Vector{T}}
end

support(proba::ApproxBesselSampler) = length.(proba.weights) .- 1
dim(proba::ApproxBesselSampler) = length(proba.weights)
domain(proba::ApproxBesselSampler{T,D}) where {T,D} = D

"""
Builds an `ApproxBesselSampler` given a list of scale. 

# Arguments
- `maxsupport::Integer`: the maximum support of the probability. For ``|ω| > maxsupport``, will return ``p(ω) = 0``.
- `tol::AbstractFloat`: the proportion of the support to leave out. The probability will satisfy ``∑ p(ω) = 1 - tol``.
"""
function ApproxBesselSampler(D::Type{<:Domain}, scale::AbstractVector{T};
    maxsupport=1000,
    tol=1e-4) where {T}

    weights = Vector{T}[]
    for i ∈ axes(scale, 1)
        sᵢ = scale[i]
        # Proba of obtaining 0 and |1| (1 or -1 if d=ℤ)
        wᵢ = [besseli0x(sᵢ), 2besseli1x(sᵢ)]
        # Support considered so far: p₀, p₁ if domain is ℕ,
        # otherwise p₀, p₁, p₋₁ if domain is ℤ.
        ∑ = wᵢ[1] + wᵢ[2]
        for ω ∈ 2:maxsupport
            p = besselix(ω, sᵢ)
            push!(wᵢ, 2p)
            ((∑ += 2p) > 1 - tol) && break
        end
        # TODO check that the renomarlization is motivated theoretically 
        wᵢ ./= sum(wᵢ)  # Normalize to 1 

        push!(weights, wᵢ)
    end

    ApproxBesselSampler{T,D}(scale, weights)
end

_ι(D::Type{ℕ}, ω) = 1
_ι(D::Type{ℤ}, ω) = ω ≠ 0 ? 2 : 1
_c(D::Type{ℕ}) = 1
_c(D::Type{ℤ}) = 2
_flipsign(D::Type{ℕ}, T, n) = one(T)
_flipsign(D::Type{ℤ}, T, n) = (rand(Bool, n) .* 2one(T) .- one(T))
_flipsign!(ω, D::Type{ℕ}) = ω
_flipsign!(ω, D::Type{ℤ}) = begin
    @inbounds for i ∈ axes(ω, 1)
        ω[i] *= rand(Bool) ? 1 : -1
    end
end

function samplesprobas(proba::ApproxBesselSampler{T,D}, nfreqs, ::Type{U}) where {T,D,U}
    ωs = zeros(U, dim(proba), nfreqs)
    ps = ones(T, nfreqs)

    samples = zeros(U, nfreqs)
    for i ∈ axes(ωs, 1)
        # Sample from the categorical distribution
        samples .= rand(Categorical(proba.weights[i]), nfreqs)
        # Probabilities are the weights; if d=ℤ we have sampled |ω|; need to choose the sign and divide proba by 2
        ps .*= proba.weights[i][samples] ./ _ι.(D, samples .- 1)
        # The samples are given with -1 (because Categorical gives values starting from 1) and with random sign if D=ℤ
        ωs[i, :] .= (samples .- 1) .* _flipsign(D, T, nfreqs)
    end

    ωs, ps
end
samplesprobas(proba::ApproxBesselSampler, nfreqs) = samplesprobas(proba::ApproxBesselSampler, nfreqs, Int)


_sampleone!(ω, proba::ApproxBesselSampler{T,D}) where {T,D} = begin
    for i ∈ axes(ω, 1)
        ω[i] = rand(Categorical(proba.weights[i])) - 1
    end
    _flipsign!(ω, D)
end

"
Samples `nfreqs` from `proba`. Returns a tuple `(ωs, ns, ps)`. `ωs` are the frequencies sampled. For each `ω ∈ ωs`, `ns` are the number of occurences of each (so that `∑ ns = nfreqs`), and `ps` the probabilities of each.

# Implementation

Uses a hash table to store the frequencies which have already been seen. This works better for `D = ℕ`: if `D = ℤ`, we have to consider all the possible different signs, resulting in a search space `2ᵈ` bigger, with `d` the dimension of the proba.
"
function samplesprobas_bycat(proba::ApproxBesselSampler{T,D}, nfreqs, ::Type{U}) where {T,D,U}
    ω = zeros(U, dim(proba))
    d_ns = Dict{Vector{U},U}()
    d_ps = Dict{Vector{U},T}()
    for _ ∈ 1:nfreqs
        _sampleone!(ω, proba)
        ω ∈ keys(d_ns) && (d_ns[copy(ω)] += 1; continue)

        d_ns[copy(ω)] = 1
        d_ps[copy(ω)] = prod(proba.weights[i][abs(ω[i])+1] / _ι.(D, abs(ω[i])) for i ∈ axes(ω, 1))
    end
    reduce(hcat, keys(d_ns)), collect(values(d_ns)), collect(values(d_ps))
end
samplesprobas_bycat(proba::ApproxBesselSampler, nfreqs) = samplesprobas_bycat(proba::ApproxBesselSampler, nfreqs, Int)

"
Returns a list of frequencies and their probabilities. The frequencies are all the _positive_ ones we can get from `proba` whose probability is greater than machine precision; the other ones are discarded. Typically, the resulting grid has a size `≪ Nᵈ`, if `N` is the support of `proba` and `d` the dimension. 

This is useful for sampling from the grid afterwards, instead of using a hashtable as in [`samplesprobas_bycat`](@ref).

# Performances

This function is not type-stable, because of `Iterators.product` of variable size. One option is to make the dimension of `proba` a static parameter, or a [value-type](https://docs.julialang.org/en/v1/manual/types/#%22Value-types%22). 
"
function _get_grid(::Type{U}, proba::ApproxBesselSampler{T,D}) where {T,D,U}  # TODO: not type stable; make dimension of proba a value type 
    ωs = zeros(U, dim(proba), prod(support(proba)))
    ps = ones(T, prod(support(proba)))

    nunique = 0
    p = one(T)
    for ω ∈ Iterators.product((1:support(proba)[i] for i ∈ 1:dim(proba))...)
        p = prod(proba.weights[i][ω[i]] for i ∈ 1:dim(proba))
        p < eps(T) && continue
        nunique += 1
        @views ωs[:, nunique] .= ω
        ps[nunique] = p
    end
    # Allocates to free the rest of the memory
    ωs = ωs[:, 1:nunique]
    ps = ps[1:nunique]
    ps ./= sum(ps)

    ωs, ps
end

"
Same as [`samplesprobas_bycat`](@ref), but uses categorical distribution over the possible frequencies instead of using a hash table. Should be much more efficient for very high `nfreqs`.

# Implementation

Implementation is only provided for `D = ℕ`. For `D = ℤ`, this would require storing the probability for each dimension and choosing a random sign (hence taking the conjugate at random) when we need to compute the estimator of the F-norm. 
"
function samplesprobas_bycat_wgrid(proba::ApproxBesselSampler{T,ℕ}, nfreqs, ::Type{U}) where {T,U}
    ωs, ps = _get_grid(U, proba)
    rs = rand(Categorical(ps), nfreqs)

    counts = Dict{eltype(rs),eltype(rs)}()
    for ind ∈ axes(rs, 1)
        rs[ind] ∈ keys(counts) ? (counts[rs[ind]] += 1) : (counts[rs[ind]] = 1)
    end
    ind = collect(keys(counts))
    ns = collect(values(counts))

    ωs[:, ind], ns, ps[ind]
end

"""
For each `ω ∈ ωs`, computes the probability of obtaining `ω` with `proba`.
"""
function pdf(proba::ApproxBesselSampler{T,D}, ωs::AbstractMatrix{U}) where {T,D,U<:Integer}
    @assert dim(proba) == size(ωs, 1)
    ps = ones(T, size(ωs, 2))

    for i ∈ axes(ωs, 1)
        for j ∈ axes(ωs, 2)
            ps[j] *= begin
                ω = abs(ωs[i, j])
                if ω < length(proba.weights[i])
                    proba.weights[i][ω+1] / _ι(D, ω)
                else
                    zero(T)
                end
            end
        end
    end

    ps
end

"Mean from weighted samples. `y` are the unique values, and `N` are the number of samples to draw without replacement. `cum` contains the cumulative count of `ns` and will be modified."
function mean!(cum, y, N)
    @assert N ≤ cum[end]
    r = zero(eltype(y))
    x, ind = zero(eltype(cum)), zero(eltype(cum))
    for _ ∈ 1:N
        x = rand(1:cum[end])
        ind = searchsortedfirst(cum, x)
        @inbounds @views cum[ind:end] .-= 1
        r += y[ind]
    end
    r / N
end

"
Median-of-Mean estimator for weighted samples. `y` are the unique values, and `ns` their respective counts.
"
function mom(y, ns, nbatch, batchsize)
    @assert nbatch * batchsize ≤ sum(ns) "You asked too many samples: `nbatch * batchsize > N`"

    cum = cumsum(ns)
    batches_mean = zeros(eltype(y), nbatch)

    for i ∈ 1:nbatch
        batches_mean[i] = mean!(cum, y, batchsize)
    end

    median(batches_mean)
end

Base.show(io::IO, ::MIME"text/plain", proba::ApproxBesselSampler) = print(io,
    """
    ApproxBesselSampler($(domain(proba))):
    • support: $(support(proba))
    • dim: $(dim(proba))
    • domain: $(domain(proba))\
    """
)
Base.show(io::IO, proba::ApproxBesselSampler) = print(io,
    """
    ApproxBesselSampler($(domain(proba)), d=$(dim(proba)))\
    """
)
