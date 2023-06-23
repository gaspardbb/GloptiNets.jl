using GloptiNets
using GloptiNets: GloptiNets as GN
using DataFrames
using TOML
using Plots
plotlyjs()

# xptype = PolyTrigo
xptype = PolyCheby
xpname = xptype === PolyTrigo ? "fourier" : "cheby"

path_xps = "xps/$xpname-vstssos"
@assert isdir(path_xps)

nfailedxp = 0

poly_config = TOML.parsefile("data/vs_tssos/$xpname/infos.toml")
results = []
for name_curxp in readdir(path_xps)
    global nfailedxp, results
    path_curxp = joinpath(path_xps, name_curxp)
    @show name_curxp
    !startswith(name_curxp, "230518") && continue
    path_curconfig = joinpath(path_curxp, "config.toml")
    path_curresult = joinpath(path_curxp, "results.toml")
    !(isfile(path_curconfig) && isfile(path_curresult)) && (nfailedxp += 1; continue)

    curconfig = TOML.parsefile(path_curconfig)
    curresult = TOML.parsefile(path_curresult)

    var, d, N = curconfig["f_var"], curconfig["f_d"], curconfig["f_N"]
    poly_name = "randompoly-s$var-d$d-N$N"
    f = GN.load(xptype, "data/vs_tssos/$xpname/$poly_name")
    f_ncoeffs = size(f.coefficients, 1)
    f_hnorm2 = poly_config[poly_name]["hnorm2"]
    f_maxval = poly_config[poly_name]["maxval"]

    push!(results, merge(curconfig, curresult, Dict(
        "name" => name_curxp,
        "f_hnorm2" => f_hnorm2,  # Or f_hnorm2
        "f_maxval" => f_maxval,
        "HS_sup" => curresult["hs_norm2"] - f_hnorm2,
        "f_ncoeffs" => f_ncoeffs,
    )))
end
results = DataFrame(results)
select!(results, Not([:opt, :proba_tol, :lossfunc_param]))
@info "$(size(results, 1)) xps collected. $nfailedxp failed xps."
transform!(results, [:bound_mom, :bound_cheby] => ByRow(max) => :bound_best)
transform!(results, :gparams => ByRow(t -> (t[1] + 3) * t[2] * t[3]) => :nparams)

out = combine(groupby(results, [:nparams, :f_d, :f_N]), [:bound_best, :lrdecay, :reg, :f_hnorm2, :lr, :gparams, :time_hs_norm2, :time_interpolate, :time_mom, :f_ncoeffs] => (
    (bound_best, lrdecay, reg, f_hnorm2, lr, gparams, time_hs_norm2, time_interpolate, time_mom, f_ncoeffs) -> begin
        idx = argmax(bound_best)
        (bound_best_max=bound_best[idx], lrdecay_max=lrdecay[idx], reg_max=reg[idx], f_hnorm2=f_hnorm2[idx], lr=lr[idx], g_rank=gparams[idx][1], g_bs=gparams[idx][2], g_nb=gparams[idx][3],
            time_total=time_hs_norm2[idx] + time_interpolate[idx] + time_mom[idx], f_ncoeffs=f_ncoeffs[idx])
    end
) => AsTable
)
sort!(out, [:f_d, :f_N])

using Plots
gr()

p = plot(size=(400, 200),
    xlabel="# params",
    xscale=:log10,
    ylabel="Certificate",
    yscale=:log10,
)
for (k, df) ∈ pairs(groupby(out, :g_bs))
    sort!(df, :nparams)
    plot!(p, df[!, :nparams], -df[!, :bound_best_max],
        label=k[:g_bs],
        marker=:circle,
    )
end
p
savefig(p, "params_variation.pdf")



#Row │ nparams  f_d    f_N    bound_best_max  lrdecay_max  reg_max  f_hnorm2  lr       g_rank  g_bs   g_nb  
#    │ Int64    Int64  Int64  Float64         String       Float64  Float64   Float64  Int64   Int64  Int64 
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
#  1 │     896      3      5      -0.0240751  cos           1.0e-7  1.0           0.5       4      8     16
#  3 │   11264      3      5      -0.0121785  cos           5.0e-7  1.0           0.2       8     16     64
#  4 │     896      3      7      -0.0317357  cos           1.0e-5  1.0           0.5       4      8     16
#  6 │   11264      3      7      -0.0148168  cos           1.0e-6  1.0           0.5       8     16     64
#  7 │     896      3      9      -0.0305482  cos           1.0e-7  1.0           0.5       4      8     16
#  9 │   11264      3      9      -0.0154154  cos           1.0e-8  1.0           0.2       8     16     64
# 10 │     896      4      3      -0.128591   cos           1.0e-5  0.999997      0.5       4      8     16
# 12 │   11264      4      3      -0.0310481  cos           1.0e-8  0.999997      0.5       8     16     64
# 13 │     896      4      5      -0.139152   cos           1.0e-5  0.999999      0.5       4      8     16
# 15 │   11264      4      5      -0.0403387  cos           1.0e-6  0.999999      0.5       8     16     64
# 16 │     896      4      7      -0.153206   cos           1.0e-5  0.999999      0.5       4      8     16
# 18 │   11264      4      7      -0.0528856  cos           1.0e-8  0.999999      0.5       8     16     64


# Row │ nparams  f_d    f_N    bound_best_max  lrdecay_max  reg_max  f_hnorm2  lr       g_rank  g_bs   g_nb   time_total  f_ncoeffs 
# │ Int64    Int64  Int64  Float64         String       Float64  Float64   Float64  Int64   Int64  Int64  Float64     Int64     
# ─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 1 │   11264      4      3      -0.078207   cos           1.0e-8       1.0      0.5       8     16     64     542.579        255
# 2 │     896      4      3      -0.104807   cos           1.0e-8       1.0      0.2       4      8     16     114.232        255
# 3 │   11264      4      4      -0.0695963  cos           5.0e-7       1.0      0.5       8     16     64     551.966        624
# 4 │     896      4      4      -0.0942897  cos           1.0e-6       1.0      0.2       4      8     16     118.506        624
# 5 │   11264      4      5      -0.0774635  cos           1.0e-7       1.0      0.5       8     16     64     618.532       1295
# 6 │     896      4      5      -0.107742   cos           1.0e-7       1.0      0.2       4      8     16     129.675       1295