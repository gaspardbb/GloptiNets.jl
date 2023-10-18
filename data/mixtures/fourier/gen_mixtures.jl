import Random
Random.seed!(0)
using JLD2
using GloptiNets
const GN = GloptiNets


ncoeffs, dim, scale, hnorm = 100, 3, 2.0, 2.0
f = GN.random_pos_besselmixture(ncoeffs, dim, scale, hnorm; ntries_min=100)


GN.save(f, joinpath("data/mixtures/fourier", "N$(ncoeffs)-d$(dim)-hnorm$(Int(hnorm))"))