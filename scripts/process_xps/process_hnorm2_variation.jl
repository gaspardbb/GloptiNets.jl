using GloptiNets
using DataFrames
using TOML
using Plots
plotlyjs()

path_xps = "xps/fourier_hnorm2_variation"
@assert isdir(path_xps)

nfailedxp = 0

poly_config = TOML.parsefile("data/hnorm2/infos.toml")
results = []
for name_curxp in readdir(path_xps)
    global nfailedxp, results
    path_curxp = joinpath(path_xps, name_curxp)
    !startswith(name_curxp, "23") && continue
    path_curconfig = joinpath(path_curxp, "config.toml")
    path_curresult = joinpath(path_curxp, "results.toml")
    !(isfile(path_curconfig) && isfile(path_curresult)) && (nfailedxp += 1; continue)

    curconfig = TOML.parsefile(path_curconfig)
    curresult = TOML.parsefile(path_curresult)

    var, d, N, idx = curconfig["f_var"], curconfig["f_d"], curconfig["f_N"], curconfig["f_idx"]
    poly_name = "randompoly-s$var-d$d-N$N-i$idx"
    f = GloptiNets.load(PolyTrigo, "data/hnorm2/$poly_name")
    f_hnorm2 = poly_config[poly_name]["hnorm2"]
    f_maxval = poly_config[poly_name]["maxval"]

    push!(results, merge(curconfig, curresult, Dict(
        "name" => name_curxp,
        "f_hnorm2" => f_hnorm2,  # Or f_hnorm2
        "f_maxval" => f_maxval,
        "HS_sup" => curresult["hs_norm2"] - f_hnorm2,
    )))
end
results = DataFrame(results)
select!(results, Not([:opt, :proba_tol, :lossfunc_param]))
@info "$(size(results, 1)) xps collected. $nfailedxp failed xps."
transform!(results, [:bound_mom, :bound_cheby] => ByRow(max) => :bound_best)

results_selection = sort(results, :bound_cheby; rev=true)
select!(results_selection, Not([:time_hs_norm2, :time_linfsamples, :time_interpolate]))

out = combine(groupby(results, :f_idx), [:bound_best, :lrdecay, :reg, :f_hnorm2] => (
    (bound_best, lrdecay, reg, f_hnorm2) -> begin
        idx = argmax(bound_best)
        (bound_best_max=bound_best[idx], lrdecay_max=lrdecay[idx], reg_max=reg[idx], f_hnorm2=f_hnorm2[idx])
    end
) => AsTable
)

using Plots
gr()
pgfplotsx()

plot(out[!, :f_hnorm2], -out[!, :bound_best_max],
    xlabel="‖f‖²",
    xticks=[collect(1:10); 20], #[1, 2, 5, 20],
    xformatter=x -> "$(Int(round(x)))",
    xscale=:log10,
    xflip=true,
    ylabel="Certificate",
    # yticks=[collect(1:10); 20], #[1, 2, 5, 20],
    # yformatter=x -> "$(Int(round(x)))",
    yscale=:log10,
    # yflip=true,
    marker=:circle,
    legend=:none,
    size=(400, 200)
)
savefig("hnorm2_variation.tex")
savefig("hnorm2_variation.pdf")