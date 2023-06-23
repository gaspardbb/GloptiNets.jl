using GloptiNets
using DataFrames
using TOML
using Plots
plotlyjs()

path_xps = "xps/param-norm"
@assert isdir(path_xps)

nfailedxp = 0

poly_config = TOML.parsefile("data/infos.toml")
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
    # curconfig["f_S"] = 10

    s, d, N, S = curconfig["f_var"], curconfig["f_d"], curconfig["f_N"], curconfig["f_hnorm2"]
    poly_name = "randompoly-s$s-d$d-N$N-S$S"
    f = GloptiNets.load("data/$poly_name")
    f_hnorm2 = poly_config[poly_name]["hnorm2"]
    f_maxval = poly_config[poly_name]["maxval"]

    push!(results, merge(curconfig, curresult, Dict(
        "name" => name_curxp,
        "f_hnorm2" => S,  # Or f_hnorm2
        "f_maxval" => f_maxval,
        "HS_sup" => curresult["hs_norm2"] - f_hnorm2,
    )))
end
results = DataFrame(results)
select!(results, Not([:opt, :proba_tol, :lossfunc_param]))
@info "$(size(results, 1)) xps collected. $nfailedxp failed xps."
transform!(results, [:bound_mom, :bound_cheby] => ByRow(max) => :bound_best)



# results_selection = sort(filter(:f => x -> x == 1, results), :"c*-linf"; rev=true)[1:30, :]
# results_selection = sort(results, :bound_cheby; rev=true)

results_selection = sort(results, :bound_cheby; rev=true)
select!(results_selection, Not([:time_hs_norm2, :time_linfsamples, :time_interpolate]))

# combine(groupby(results, :f_hnorm2), [:bound_best, :lrdecay, :reg] => (
#        (x, y, z) -> begin
#        idx = argmax(x)
#        (bound_best_max=x[idx], lrdecay_max=y[idx], reg_max=z[idx])
#        end
#        ) => AsTable
#        )