

function get_estimates(gmm_main)
    # (currently, for 2-stage GMM only)
    gmm_main2 = gmm_main["results_stage2"]

    estimates_row = gmm_main2[gmm_main2.is_best_vec .== 1, :]
    @assert size(estimates_row)[1] == 1

    return estimates_row[1, r"param_"] |> Vector
end

"""
    ci_level: percentile (0-50), by default 2.5 meaning we compute 95% confidence intervals
"""
function get_model_results(gmm_results; ci_level=2.5, keep_only_converged=true)


    model_results = []

    # get parameter estimates 
        estimates_vec = get_estimates(gmm_results["gmm_main_results"])
        n_params = length(estimates_vec)

        model_results = Vector{Any}(undef, n_params)
        for i=1:n_params
            model_results[i] = Dict(
                "param name" => gmm_results["gmm_options"]["param_names"][i],
                "estimate" => estimates_vec[i]
            )
        end

    # confidence intervals asymptotic
        z_low  = quantile(Normal(), ci_level / 100.0)
        z_high = quantile(Normal(), 1.0 - ci_level / 100.0)

        asy_cilow = estimates_vec .+ z_low  .* gmm_results["asy_stderr"]
        asy_cihigh = estimates_vec .+ z_high .* gmm_results["asy_stderr"]

        for i=1:n_params
            model_results[i]["asy_stderr"] = gmm_results["asy_stderr"][i]
            model_results[i]["asy_cilow"] = asy_cilow[i]
            model_results[i]["asy_cihigh"] = asy_cihigh[i]
            model_results[i]["ci_level"] = ci_level
        end

    # confidence intervals based on bootstrap
        gmm_boots = gmm_results["gmm_boot_results"]
        # boot_estimates = zeros(length(gmm_boots), n_params)

        # one large DF
        boot_df = vcat([myel["results_stage2"] for myel in gmm_boots]...)

        # for i=1:length()
        #     gmm_boot = gmm_boots[i]
        #     boot_estimates[i,:] = get_estimates(gmm_boot)
        # end

        # compute confidence intervals
        ci_levels = [ci_level, 100 - ci_level]
        for i=1:n_params
        
            if keep_only_converged
                boot_vec = boot_df[boot_df.opt_converged .== 1, string("param_", i)]
            else
                boot_vec = boot_df[:, string("param_", i)]
            end

            boot_vec = skipmissing(boot_vec) |> collect
            # boot_vec = sort(boot_vec)

            boot_cilow, boot_cihigh = percentile(boot_vec, ci_levels) 

            model_results[i]["boot_cilow"] = boot_cilow
            model_results[i]["boot_cihigh"] = boot_cihigh
            model_results[i]["ci_level"] = ci_level
            
        end

    print_model_results(model_results)
    
    # return [estimates_vec, boot_df]
end

## Build the table
function pn1(mynumber; d=1)
    (d==0) && return @sprintf("%2.0f", mynumber)
    (d==1) && return @sprintf("%2.1f", mynumber)
    (d==2) && return @sprintf("%2.2f", mynumber)
    (d==3) && return @sprintf("%2.3f", mynumber)
end

function print_model_results(model_results;)
    println("\nGMM estimates [bootstrap CI], (asymptotic CI), se. Using CI level ", model_results[1]["ci_level"], "]")
    for param_results in model_results

        est = pn1(param_results["estimate"], d=2)
        blo = pn1(param_results["boot_cilow"], d=2)
        bhi = pn1(param_results["boot_cihigh"], d=2)

        alo = pn1(param_results["asy_cilow"], d=2)
        ahi = pn1(param_results["asy_cihigh"], d=2)

        se = pn1(param_results["asy_stderr"], d=2)

        println(est, " [", blo, ", ", bhi, "]", " (", alo, ", ", ahi, ") ", se)
    end
end