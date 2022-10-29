
using DataFrames
using CSV

function collect_results(rootpath, nfiles)

    alldfs = []
    for i=1:nfiles
        filepath = rootpath * "step1/results_df_run_" * string(i) * ".csv"
        mydf = CSV.read(filepath, DataFrame)
        push!(alldfs, mydf)
    end

    alldfs = vcat(alldfs...)
    return alldfs
end



myrootpath = "G:/My Drive/optnets/analysis/gmm-server/gmm-v7/cmd-results-v6-nonbrt-only-opt/"
alldfs = collect_results(myrootpath, 25)
sort!(alldfs, :obj_vals)

myrootpath = "G:/My Drive/optnets/analysis/gmm-server/gmm-v7/cmd-results-v5-joint-opt/"
alldfs = collect_results(myrootpath, 25)
sort!(alldfs, :obj_vals)