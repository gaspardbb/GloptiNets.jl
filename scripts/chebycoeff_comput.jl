"""
Use numerical integration to test the computation of the Chebychev coefficient of a product of kernel function. 
"""


s = 1.1

h(x, s) = exp(s * (cos(2π * x) - 1))
u(x) = acos(x) / 2π
K(x, y, s) = (h(u(x) + u(y), s) + h(u(x) - u(y), s)) / 2

y, z = 2rand(2) .- 1
m₊ = (u(y) + u(z)) / 2
m₋ = (u(y) - u(z)) / 2
σ₊ = cos(2π * m₊)
σ₋ = cos(2π * m₋)

using QuadGK
using Bessels: besseli

ω = 2
htrue = quadgk(
    x -> K(x, y, s) * K(x, z, s) * cos(ω * acos(x)) / √(1 - x^2),
    -(1 - 1e-8), (1 - 1e-8))[1] / π * (ω != 0 ? 2 : 1)
htruecos = quadgk(
    xx -> K(cos(2π * xx), y, s) * K(cos(2π * xx), z, s) * cos(2π * ω * xx),
    0, 1)[1] * (ω != 0 ? 2 : 1)
hcomput = exp(-2s) / 2 * (ω != 0 ? 2 : 1) * (
              cos(2π * ω * m₊) * besseli(ω, 2s * σ₋) +
              cos(2π * ω * m₋) * besseli(ω, 2s * σ₊)
          )

@assert ≈(htrue, htruecos, rtol=1e-3)
@assert ≈(hcomput, htruecos, rtol=1e-6)