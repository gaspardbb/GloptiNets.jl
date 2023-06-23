using GloptiNets
using Test
# To run the packages in the code
# using TestItemRunner

# @run_package_tests

@testset "GloptiNets.jl" begin
    @testset "GPU" begin
        using CUDA
        if CUDA.functional()
            CUDA.allowscalar(false)

            dim, rank, blocksize, nblocks = 2, 3, 4, 5
            γ = 1.0 .+ rand(dim) / 10

            g = PSDBlockBesselCheby(
                rand(dim, blocksize, nblocks),
                randn(blocksize, rank, nblocks),
                γ
            )

            h = gpu(g)
            X = cu(rand(2, 10))
            @test evaluate_cos(g, Matrix{Float64}(X)) ≈ collect(evaluate_cos(h, X))

            g = PSDBlockBesselFourier(
                rand(dim, blocksize, nblocks),
                randn(blocksize, rank, nblocks),
                γ
            )
            h = gpu(g)
            @test g(Matrix{Float64}(X)) ≈ collect(h(X))

            g = PolyCheby(
                randn(),
                randn(3),
                [0 1 2; 1 0 1]
            )
            h = gpu(g)
            @test evaluate_cos(g, Matrix{Float64}(X)) ≈ collect(evaluate_cos(h, X))

            g = PolyTrigo(
                randn(),
                randn(ComplexF64, 3),
                [0 -1 2; 1 0 1]
            )
            h = gpu(g)
            @test g(Matrix{Float64}(X)) ≈ collect(h(X))
        end
    end
end
