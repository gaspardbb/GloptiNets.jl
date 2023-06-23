using GloptiNets
using DataFrames
using TOML
using Plots
using Dates
plotlyjs()

path_xps = "xps/params_variation"
@assert isdir(path_xps)

nfailedxp = 0

poly_config = TOML.parsefile("data/hnorm2/infos.toml")
results = []
for name_curxp in readdir(path_xps)
    global nfailedxp, results
    path_curxp = joinpath(path_xps, name_curxp)
    !startswith(name_curxp, "23") && continue
    dt_curxp = DateTime(name_curxp, dateformat"yymmdd-HHMMSS-s")
    dt_curxp < DateTime("0023-05-17T15:30:00.000") && continue
    path_curconfig = joinpath(path_curxp, "config.toml")
    path_curresult = joinpath(path_curxp, "results.toml")
    !(isfile(path_curconfig) && isfile(path_curresult)) && (nfailedxp += 1; continue)

    curconfig = TOML.parsefile(path_curconfig)
    curresult = TOML.parsefile(path_curresult)

    var, d, N, idx = curconfig["f_var"], curconfig["f_d"], curconfig["f_N"], curconfig["f_idx"]
    poly_name = "randompoly-s$var-d$d-N$N-i$idx"
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
transform!(results, :gparams => ByRow(t -> (t[1] + 3) * t[2] * t[3]) => :nparams)

out = combine(groupby(results, :nparams), [:bound_best, :lrdecay, :reg, :f_hnorm2, :lr, :gparams, :vals_mean, :mom_nsamples] => (
    (bound_best, lrdecay, reg, f_hnorm2, lr, gparams, vals_mean, mom_nsamples) -> begin
        idx = argmax(bound_best)
        (bound_best_max=bound_best[idx], lrdecay_max=lrdecay[idx], reg_max=reg[idx], f_hnorm2=f_hnorm2[idx], lr=lr[idx], g_rank=gparams[idx][1], g_bs=gparams[idx][2], g_nb=gparams[idx][3], vals_mean=vals_mean[idx], mom_nsamples=mom_nsamples[idx])
    end
) => AsTable
)
sort!(out, :nparams)



using Plots
gr()

pgfplotsx()
plot(
    out[!, :nparams], -out[!, :bound_best_max],
    marker=:circle,
    size=(400, 200),
    xlabel="# params",
    xscale=:log10,
    ylabel="Certificate",
    yscale=:log10,
    legend=false,
)
savefig("params_variation.tex")
# savefig("params_variation.pdf")

plot!(out[!, :nparams], out[!, :vals_mean])


p = plot(size=(400, 200),
    xlabel="# params",
    xscale=:log10,
    ylabel="Certificate",
    yscale=:log10,
)
for (k, df) ∈ pairs(groupby(out, :g_bs))
    k[:g_bs] == 32 && continue
    sort!(df, :nparams)
    plot!(p, df[!, :nparams], -df[!, :bound_best_max],
        label=k[:g_bs],
        marker=:circle,
    )
end
p
savefig(p, "params_variation_all.pdf")

filter!(out -> out.mom_nsamples == 160000, out)

x = [first(eachrow(out)).nparams]
y = [-first(eachrow(out)).bound_best_max]
allx = copy(x)
ally = copy(y)
for row ∈ eachrow(out)[2:end]
    curx, cury = row.nparams, -row.bound_best_max
    push!(allx, curx)
    push!(ally, cury)
    cury < y[end] && (push!(x, curx), push!(y, cury))
end

p = plot(size=(400, 200),
    xlabel="# params",
    xscale=:log10,
    ylabel="Certificate",
    yscale=:log10,
    ylims=(1e-2, 1e0),
    legend=false
)
# scatter!(p, allx, ally, label="All", marker=:circle, alpha=0.5)
plot!(p, x, y, marker=:circle)
p
savefig(p, "params_variation_bestonly.pdf")