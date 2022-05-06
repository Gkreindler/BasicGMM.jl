
# Todo: how to handle (a) errors in certain runs, (b) lack of convergence
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

function print_results(gmm_results; ci_level=2.5)

    # prep
    gmm_options = gmm_results["gmm_options"]

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
    
    print(" # moments: ", gmm_results["n_moms_full"])
    if ~isnothing(gmm_options["subset_moms"])
        print(" (", gmm_results["n_moms_full"] - gmm_results["n_moms"], " fixed). ")
    else
        print(". ")
    end
    
    print("# parameters: ", gmm_results["n_params"])
    if ~isnothing(gmm_options["fix_params"])
        print(" (", length(gmm_options["fix_params"]), " fixed). ")
    else
        print(". ")
    end
    
    println("# observations: ", gmm_results["n_observations"], ".\n")

    if ~isnothing(gmm_options["subset_moms"])
        println(" Estimation uses moments: ", gmm_options["subset_moms"], "\n")
    end

    n_params = gmm_results["n_params"]
    if isnothing(gmm_options["fix_params"])
        fixed_param_idxs = Set()

        smaller_idxs = 1:n_params
    else
        fixed_param_idxs = Set(collect(keys(gmm_options["fix_params"])))

        idxs = keys(gmm_options["fix_params"]) |> collect |> sort
        other_idxs = sort(collect(setdiff(Set(1:n_params), Set(idxs))))

        smaller_idxs = zeros(Int64, n_params)
        for i=1:n_params
            j = findfirst(x -> x==i, other_idxs)
            isnothing(j) || (smaller_idxs[i] = j)            
        end

    end

    # parameter estimates
    main_results = gmm_results["gmm_main_results"]
    if main_results["outcome"] != "failed"
        # header
        println(" ---------------------------------------------------------------------")
        println(" Estimate             Asymptotic CI                 Bootstrap CI      ")
        println(" ---------------------------------------------------------------------")

        for i=1:gmm_results["n_params"]

            if i in fixed_param_idxs
                theta = gmm_options["fix_params"][i]
                output_line = @sprintf("%g", theta)
                output_line = " " * rpad(output_line, 20, " ") * "(fixed parameter)"
            else

                j = smaller_idxs[i]

                # estimate
                theta = main_results["theta_hat"][j]
                output_line = @sprintf("%g", theta)
                output_line = " " * rpad(output_line, 20, " ")

                # asymptotic confidence interval
                if gmm_options["var_asy"]
                    z_low  = quantile(Normal(), ci_level / 100.0)
                    z_high = quantile(Normal(), 1.0 - ci_level / 100.0)

                    se = gmm_results["asy_stderr"][j]
                    asy_cilow = theta  .+ z_low  .* se
                    asy_cihigh = theta .+ z_high .* se

                    temp = "[" * @sprintf("%g", asy_cilow) * ", " * @sprintf("%g", asy_cihigh) * "]"
                    output_line *= rpad(temp, 30, " ") 
                end

                # bootstrap confidence interval
                if ~(isnothing(gmm_options["var_boot"]) || (gmm_options["var_boot"] == "")) 
                    ci_levels = [ci_level, 100 - ci_level]

                    boot_vec = [boot_result["theta_hat"][j] for boot_result=gmm_results["gmm_boot_results"]]

                    # ! skipping missing!
                    boot_vec = skipmissing(boot_vec) |> collect

                    boot_cilow, boot_cihigh = percentile(boot_vec, ci_levels)

                    temp = "[" * @sprintf("%g", boot_cilow) * ", " * @sprintf("%g", boot_cihigh) * "]"
                    output_line *= rpad(temp, 30, " ")
                end
            end

            println(output_line)
        end

        println(" ---------------------------------------------------------------------")
        print(" ", (100-2*ci_level), "% level confidence intervals. ")
        if gmm_options["var_boot"] == "quick"
            println("Quick bootstrap.\n")
        elseif gmm_options["var_boot"] == "slow"
            println("Slow bootstrap, # iterations: ", gmm_options["boot_n_runs"], ".\n")
        else
            println("\n")
        end

    end

    ### Optimization info:
    print(" Number of initial conditions: ", gmm_results["main_n_initial_cond"], " (main estimation)")
    if gmm_options["var_boot"] == "slow"
        println(", ", gmm_results["boot_n_initial_cond"], " (bootstrap)")
    else
        println()
    end

    ### Did all estimations converge?
    if ~isnothing(gmm_options["var_boot"])
        print(" Main estimation optimization: ")
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
    if gmm_options["var_boot"] == "slow"
        print(" Bootstrap estimation optimization: ")

        # TODO: handle failed runs
        boot_df = [boot_result["results_stage1"] for boot_result=gmm_results["gmm_boot_results"]]
        boot_df = vcat(boot_df...)

        if gmm_options["2step"]
            boot_df2 = [boot_result["results_stage2"] for boot_result=gmm_results["gmm_boot_results"]]
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
