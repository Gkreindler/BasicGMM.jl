
# Todo: how to handle (a) errors in certain runs, (b) lack of convergence
" Access the "
function get_estimates(gmm_results; onestep=true)

    if onestep
        gmm_results_df = gmm_results["results_stage1"]
    else # two-step optimal
        gmm_results_df = gmm_results["results_stage2"]
    end

    estimates_row = gmm_results_df[gmm_results_df.is_optimum, :]
    @assert size(estimates_row)[1] == 1

    return estimates_row[1, r"param_"] |> Vector
end




# TODO: break up into smaller functions -> one per CIs with given level, etc.
"""
    ci_level: percentile (0-50), by default 2.5 meaning we compute 95% confidence intervals
"""
function get_model_results(gmm_results; ci_level=2.5, keep_only_converged=true)


    model_results = []

    # get parameter estimates 
        estimates_vec = get_estimates(gmm_results["gmm_main_results"], onestep=gmm_results["gmm_options"]["one_step_gmm"])
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

        # TODO: use get_estimates(gmm_results; twostage=true)

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
                boot_vec = boot_df[boot_df.opt_converged, string("param_", i)]
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


function print_results(gmm_results; ci_level=2.5)

    # prep
    gmm_options = gmm_results["gmm_options"]

    printstyled("======================================================================\n", color=:yellow)
    if gmm_options["one_step_gmm"]
    println(" One-step GMM \n")
    else
    println(" Two-step GMM (optimal weighting matrix) \n")
    end
    println(" # moments: ", gmm_results["n_moms"], ". # parameters: ", gmm_results["n_params"], 
           ". # observations: ", gmm_results["n_observations"], ".\n")

    # parameter estimates
    main_results = gmm_results["gmm_main_results"]
    if main_results["outcome"] != "failed"
        # header
        println(" ---------------------------------------------------------------------")
        println(" Estimate             Asymptotic CI                 Bootstrap CI      ")
        println(" ---------------------------------------------------------------------")

        for i=1:gmm_results["n_params"]

            # estimate
            theta = main_results["theta_hat"][i]
            output_line = @sprintf("%g", theta)
            output_line = " " * rpad(output_line, 20, " ")

            # asymptotic confidence interval
            if gmm_options["asy_var"]
                z_low  = quantile(Normal(), ci_level / 100.0)
                z_high = quantile(Normal(), 1.0 - ci_level / 100.0)

                se = gmm_results["asy_stderr"][i]
                asy_cilow = theta  .+ z_low  .* se
                asy_cihigh = theta .+ z_high .* se

                temp = "[" * @sprintf("%g", asy_cilow) * ", " * @sprintf("%g", asy_cihigh) * "]"
                output_line *= rpad(temp, 30, " ") 
            end

            # bootstrap confidence interval
            if gmm_options["run_boot"]
                ci_levels = [ci_level, 100 - ci_level]

                boot_vec = [boot_result["theta_hat"][i] for boot_result=gmm_results["gmm_boot_results"]]

                # ! skipping missing!
                boot_vec = skipmissing(boot_vec) |> collect

                boot_cilow, boot_cihigh = percentile(boot_vec, ci_levels)

                temp = "[" * @sprintf("%g", boot_cilow) * ", " * @sprintf("%g", boot_cihigh) * "]"
                output_line *= rpad(temp, 30, " ")
            end

            println(output_line)
        end

        println(" ---------------------------------------------------------------------")
        print(" ", (100-2*ci_level), "% level confidence intervals. ")
        if gmm_options["run_boot"]
            println(" # bootstrap iterations: ", gmm_options["boot_n_runs"], ".\n")
        else
            println("\n")
        end

    end

    ### Optimization info:
    print(" Number of initial conditions: ", gmm_results["main_n_initial_cond"], " (main estimation)")
    if gmm_results["boot_n_initial_cond"] > 0
        println(", ", gmm_results["boot_n_initial_cond"], " (bootstrap)")
    else
        println()
    end

    ### Did all estimations converge?
    if gmm_options["run_boot"]
        print(" Main      estimation optimization: ")
    end

    if gmm_results["gmm_main_results"]["outcome"] == "success"
        printstyled("All iterations converged.\n", color=:green)
    else
        printstyled(" Some iterations did not converge:\n", color=:orange)
        for detail_outcome=gmm_results["outcome_stage1_detail"]
            printstyled(" Stage 1: " * detail_outcome, color=:orange)
        end
        if haskey(gmm_results, "outcome_stage2_detail")
            for detail_outcome=gmm_results["outcome_stage2_detail"]
                printstyled(" Stage 2: " * detail_outcome, color=:orange)
            end
        end
    end

    # TODO: boot iterations that converged.
    if gmm_options["run_boot"]
        print(" Bootstrap estimation optimization: ")

        # TODO: handle failed runs
        boot_df = [boot_result["results_stage1"] for boot_result=gmm_results["gmm_boot_results"]]
        boot_df = vcat(boot_df...)

        if ~gmm_options["one_step_gmm"]
            boot_df2 = [boot_result["results_stage1"] for boot_result=gmm_results["gmm_boot_results"]]
            boot_df2 = vcat(boot_df2...)
            boot_df = vcat(boot_df, boot_df2)
        end

        boot_df_results = combine(groupby(boot_df, "boot_run_idx"), :opt_converged => mean => :mean_opt_converged)

        n_boot_idx_notall_converged = sum(boot_df_results.mean_opt_converged .< 1.0)
        n_boot_idx_none_converged = sum(boot_df_results.mean_opt_converged .== 0.0)

        if n_boot_idx_notall_converged == 0
            printstyled("All iterations converged.\n", color=:green)
        else
            printstyled(" Some iterations did not converge:\n", color=:orange)
            printstyled(" ", n_boot_idx_notall_converged, " bootstrap runs had at least one iteration not converge:\n", color=:orange)
            printstyled(" ", n_boot_idx_none_converged, " bootstrap runs had no iteration converge:\n", color=:orange)
        end
    end


    printstyled("======================================================================\n", color=:yellow)
end