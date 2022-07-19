
# Todo: how to handle (a) errors in certain runs, (b) lack of convergence
# function get_estimates(est_results, est_results_df; onestep=true)

#     if est_results["outcome"] == "skipped"
#         return est_results["theta_hat"]
#     end

#     if onestep
#         est_results_df = est_results["results_stage1"]
#     else # two-step optimal
#         est_results_df = est_results["results_stage2"]
#     end

#     estimates_row = est_results_df[est_results_df.is_optimum, :]
#     @assert size(estimates_row)[1] == 1

#     return estimates_row[1, r"param_"] |> Vector
# end

function print_results(;
            est_options,
            est_results, 
            boot_results=nothing, 
            boot_df=nothing,
            ci_level=2.5)

    # prep
    gmm_options = est_options["gmm_options"]

    printstyled("======================================================================\n", color=:yellow)

    if gmm_options["estimator"] == "gmm1step"
        println(" One-step GMM \n")
    elseif gmm_options["estimator"] == "gmm2step"    
        println(" Two-step GMM (optimal weighting matrix) \n")
    elseif gmm_options["estimator"] == "cmd"    
        println(" Classical Minimum Distance \n")
    elseif gmm_options["estimator"] == "cmd_optimal"    
        println(" Classical Minimum Distance with Optimal Weighting Matrix\n")
    end
    
    n_moms_estim = est_options["n_moms"]
    if ~isnothing(est_options["moms_subset"])    
        n_moms_dropd = est_options["n_moms_full"] - est_options["n_moms"]
        
        print(" # moments: ", n_moms_estim, " (plus ", n_moms_dropd, " not used). ")
    else
        print(" # moments: ", n_moms_estim, ". ")
    end
    
    n_params = est_options["n_params"]
    if ~isnothing(est_options["theta_fix"])
        
        n_params_estimated = length(findall(isnothing, est_options["theta_fix"]))

        print("# parameters: ", n_params_estimated, " (plus ", n_params - n_params_estimated, " fixed). ")
    else
        print("# parameters: ", n_params, ". ")
    end
    
    println("# observations: ", est_options["n_observations"], ".\n")

    if ~isnothing(est_options["moms_subset"])
        println(" Estimation uses moments: ", est_options["moms_subset"], "\n")
    end

    n_params = est_options["n_params"]
    if isnothing(est_options["theta_fix"])
        idxs_fixed = []

        smaller_idxs = 1:n_params
    else
        theta_fix = est_options["theta_fix"]
        idxs_fixed = findall(x -> ~isnothing(x), theta_fix)
        idxs_estim = findall(isnothing, theta_fix)

        # mapping from 1:n_params to the smaller 1:n_params_to_estimate
        smaller_idxs = zeros(Int64, n_params)
        smaller_idxs[idxs_estim] = 1:length(idxs_estim)
    end

    # parameter estimates
    results = est_results
    if results["outcome"] != "failed"

        # header
        if isnothing(gmm_options["param_names"])
            println(" -------------------------------------------------------------------------")
            println(" #   Estimate             Asymptotic CI                 Bootstrap CI      ")
            println(" -------------------------------------------------------------------------")
            l1 = " "
        else
            wmax = max(20, min(30, maximum(length.(gmm_options["param_names"]))) + 4)
            l1 = rpad(" ", wmax, "-")
            l2 = rpad(" #  Param Name ", wmax + 2, " ")
            println(l1 * "------------------------------------------------------------------------")
            println(l2 * " Estimate             Asymptotic CI                 Bootstrap CI      ")
            println(l1 * "------------------------------------------------------------------------")
        end

        for i=1:est_options["n_params"]

            output_line = " " * string(i) * ") "
            
            if ~isnothing(gmm_options["param_names"])
                output_line *= rpad(gmm_options["param_names"][i], wmax - 2, " ")
            end

            if i in idxs_fixed
                theta = est_options["theta_fix"][i]
                output_line *= " " * rpad(@sprintf("%g", theta), 20, " ") * "(fixed parameter)"
            else

                j = smaller_idxs[i]

                # estimate
                theta = results["theta_hat"][j]
                output_line *= " " * rpad(@sprintf("%g", theta), 20, " ") 

                # asymptotic confidence interval
                if gmm_options["var_asy"]
                    z_low  = quantile(Normal(), ci_level / 100.0)
                    z_high = quantile(Normal(), 1.0 - ci_level / 100.0)

                    se = est_results["asy_stderr"][j]
                    asy_cilow = theta  .+ z_low  .* se
                    asy_cihigh = theta .+ z_high .* se

                    temp = "[" * @sprintf("%g", asy_cilow) * ", " * @sprintf("%g", asy_cihigh) * "]"
                    output_line *= rpad(temp, 30, " ") 
                end

                # bootstrap confidence interval
                if ~(isnothing(gmm_options["var_boot"]) || (gmm_options["var_boot"] == "")) 
                    ci_levels = [ci_level, 100 - ci_level]

                    boot_vec = [boot_result["theta_hat"][j] for boot_result=boot_results if boot_result["outcome"] != "fail"]

                    # ! skipping missing!
                    boot_vec = skipmissing(boot_vec) |> collect

                    boot_cilow, boot_cihigh = percentile(boot_vec, ci_levels)

                    temp = "[" * @sprintf("%g", boot_cilow) * ", " * @sprintf("%g", boot_cihigh) * "]"
                    output_line *= rpad(temp, 30, " ")
                end
            end

            println(output_line)
        end

        println(l1 * "-------------------------------------------------------------------------")
        print(" ", Int64(floor(100-2*ci_level)), "% level confidence intervals. ")
        if gmm_options["var_boot"] == "quick"
            println("Quick bootstrap.\n")
        elseif gmm_options["var_boot"] == "slow"
            println("Slow bootstrap, # iterations: ", gmm_options["boot_n_runs"], ".\n")
        else
            println("\n")
        end

    end

    ### Optimization info:
    print(" Number of initial conditions: ", est_options["main_n_initial_cond"], " (main estimation)")
    if gmm_options["var_boot"] == "slow"
        println(", ", boot_results[1]["boot_n_initial_cond"], " (bootstrap)")
    else
        println()
    end

    ### Did all estimations converge?
    if ~isnothing(gmm_options["var_boot"])
        print(" Main estimation optimization: ")
    end

    if est_results["outcome"] == "success"
        printstyled("All iterations converged.\n", color=:green)
    else
        printstyled(" Some iterations did not converge:\n", color=:orange)
        for detail_outcome=results["outcome_stage1_detail"]
            printstyled(" Stage 1: " * detail_outcome, color=:orange)
        end
        if haskey(results, "outcome_stage2_detail")
            for detail_outcome=results["outcome_stage2_detail"]
                printstyled(" Stage 2: " * detail_outcome, color=:orange)
            end
        end
    end

    # TODO: boot iterations that converged.
    if gmm_options["var_boot"] == "slow"
        print(" Bootstrap estimation optimization: ")

        # TODO: handle failed runs

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
