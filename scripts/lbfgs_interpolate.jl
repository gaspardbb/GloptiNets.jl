using GloptiNets
using GloptiNets: GloptiNets as GN

T = Float64
usegpu = false
f = GN.load(PolyTrigo, "/Users/gbeugnot/Documents/these/code/gloptinets/GloptiNets.jl/data/vs_tssos/fourier/randompoly-s2-d3-N5")
f = convert(T, f)
rank, blocksize, nblocks = (2, 3, 4)
γ = ones(dim(f))
g = PSDBlockBesselFourier(
    rand(T, dim(f), blocksize, nblocks),
    begin
        coeffs = randn(T, blocksize, rank, nblocks)
        coeffs ./ √sum(abs2.(coeffs))
    end,
    γ
)

usegpu && ((f, g) = (gpu(f), gpu(g)))

GN.lbfgs(f, g, RegHSNormP, (; val=1e-1);
    nepochs=3, batchsize=128,
    lossfunc_symb=:lse,
    lossfunc_param=10.0,
    show_progress=true)