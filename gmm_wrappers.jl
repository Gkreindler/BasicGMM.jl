

using LinearAlgebra
using LsqFit # modified version that accepts time limit. Use https://github.com/Gkreindler/LsqFit.jl


"""
Creates a random matrix of initial conditions, taking boudns into account
"""
function random_initial_conditions(theta0, theta_lower, theta_upper, n_init)

    n_params = length(theta0)

    theta0_mat = zeros(n_init, n_params)

    for i=1:n_init
        for j=1:n_params

            θL = theta_lower[j]
            θH = theta_upper[j]

            if (θL == -Inf) & (θH == Inf)
                theta0_mat[i,j] = (-1.5 + 3.0 * rand()) * theta0[j]
            elseif (θL > -Inf) & (θH == Inf)
                theta0_mat[i,j] = θL + (theta0[j] - θL) * (0.5 + rand())
            elseif (θL == -Inf) & (θH < Inf)
                theta0_mat[i,j] = θH - (θH - theta0[j]) * (0.5 + rand())
            else
                @assert (θL > -Inf) & (θH < Inf)
                theta0_mat[i,j] = θL + (θH - θL) * rand()
            end
        end
    end

    return theta0_mat
end

"""
    theta = typically the first stage estimate
    momfn = moment function loaded with data and other parameters
"""
function vcov_gmm_iid(theta, momfn)
    # compute matrix of moments
    mom_matrix = momfn(theta)

    # compute variance covariance matrix under iid assumption
    # ensure it's Hermitian
    n_observations = size(mom_matrix, 1)
    vcov_matrix = Hermitian(transpose(mom_matrix) * mom_matrix / n_observations)

    return vcov_matrix
end

# TODO: describe what this wrapper does
"""
"""
function gmm_obj(;theta, Whalf, momfn, show_theta=false)

	# print parameter vector at current step
	show_theta && println(">>> theta ", theta, " ")

	# compute moments
	mymoms = momfn(theta)

	# multiply by (Cholesky) half matrice and take means
	mean_mymoms = vec(mean(mymoms, dims=1) * Whalf)

	# write the value of the objective function at the end of the line
	show_theta && println(">>> obj value:", transpose(mean_mymoms) * mean_mymoms)

	return mean_mymoms
end


function curve_fit_wrapper(
				idx,
				myobjfunction,
				Whalf,
				n_moms,
				theta_initial_vec,
				theta_lower,
				theta_upper;
				write_results_to_file=0,
				individual_run_results_path="",
				maxIter=1000,
				time_limit=-1,
                show_trace=true
			)

    show_trace && println("starting iteration ", idx)

	# 	moments are already differenced out so we target zero:
	ymoms = zeros(n_moms)

	# call curve_fit from LsqFit.jl
	timeittook = @elapsed result = curve_fit(
				myobjfunction, # objective function, takes theta and Whalf as arguments
				Whalf, # pass cholesky half as "data" to myobjfunction
				ymoms, # zeros
				theta_initial_vec,
				lower=theta_lower,
				upper=theta_upper,
				maxIter=maxIter,
				time_limit=time_limit, # this is added relative to official LsqFit.jl . Use https://github.com/Gkreindler/LsqFit.jl
                show_trace=show_trace
			)

    # TODO: remove this?
	if show_trace
		println(">>> iteration ", idx, " took: ", timeittook, " seconds. Converged: ", result.converged)
		println(">>> optimum ", result.param)
		println(">>> obj val ", norm(result.resid))
	end

    # results dictionary
    results_df = Dict{String, Any}(
        "obj_vals" => norm(result.resid),
        "opt_converged" => result.converged,
		"opt_runtime" => timeittook
    )
    for i=1:length(result.param)
        results_df[string("param_", i)] = result.param[i]
    end

    # save to file
	if write_results_to_file == 2
	    outputfile = string(individual_run_results_path,"results_df_run_",idx,".csv")
		CSV.write(outputfile, DataFrame(results_df))
	end

	return results_df
end

"""

    gmm_2step()

Generalized method of moments (GMM) or classical minimum distance (CMD), with (optional) two-step optimal weighting matrix.

momfn = the moment function
    - should take a single vector argument
    - data etc is already "loaded" in this function
theta0 = initial condition (vector)
theta_lower, theta_upper = bounds (vectors, can include +/-Inf)
vcov_fn = function to compute variance covariance matrix. By default, vcov_gmm_iid which assumes data is iid.
n_theta0 = number of initial conditions
n_moms = size of moment function (number of moments) TODO: get this auto
results_dir_path = where to write results
Wstep1 = weighting matrix (default = identity matrix).
        Should be Hermitian (this will be enforced).
        Will be Cholesky decomposed
normalize_weight_matrix = boolean, if true, aim for the initial objective function to be <= O(1)
jacobian = provide jacobian function
write_results_to_file = 
    0 = nothing written to file
    1 = write objects to file, including one csv with all runs (once for first stage, once for second stage)
    2 = write objects to file, including one csv per each run
run_parallel  = individual runs in parallel vs in serial (default=parallel)
show_trace = show trace of curve_fit from LsqFit?
maxIter    = maximum iterations for curve_fit from LsqFit
show_progress = pass on to objective function
"""
function gmm_2step(;
			momfn_loaded,
			theta0,
			theta_lower,
			theta_upper,
            vcov_fn=vcov_gmm_iid,
			run_parallel=true,
            two_step=false,
			n_moms=nothing,
			Wstep1=nothing,
            normalize_weight_matrix=false,
			results_dir_path="",
            write_results_to_file=0,
			maxIter=1000,
			time_limit=-1,
            show_trace=false,
			show_theta=false,
            show_progress=false)

## Basic checks
    if write_results_to_file ∉ [0, 1, 2]
        error("write_results_to_file should be 0, 1, or 2")
    end

## Store estimation results here
    gmm_results = Dict{String, Any}()

## if theta0 is a vector (one set of initial conditions), turn to 1 x n_params matrix
    # ? ever needed?
	if isa(theta0, Vector)
		theta0 = Matrix(transpose(theta0))
	end
    n_params = size(theta0, 2)
    n_theta0 = size(theta0, 1)

## if not provided, compute number of moments by running moment function once
	if isnothing(n_moms)
		n_moms = size(momfn_loaded(theta0[1,:]), 2)
	end

## save initial conditions to file
	theta0_df = DataFrame("iteration" => 1:n_theta0)
	for i=1:n_params
		theta0_df[!, string("param_", i, "_initial")] = vec(theta0[:, i])
	end

    # gmm_results["theta0_df"] = theta0_df
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"theta_initial_df.csv")
        CSV.write(outputfile, theta0_df)
    end

## Initial weighting matrix W
	if isnothing(Wstep1)
        show_progress && println("GMM => using identity W1")

        # if not provided, use identity weighting matrix
        @assert isnothing(Wstep1)
		Wstep1 = diagm(ones(n_moms))
	end

	## save
    gmm_results["Wstep1"] = Wstep1
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"Wstep1.csv")
        CSV.write(outputfile, Tables.table(Wstep1), header=false)
    end

	# cholesky half. satisfies Whalf * transpose(Whalf) = W
	initialWhalf = Matrix(cholesky(Hermitian(Wstep1)).L)
	@assert norm(initialWhalf * transpose(initialWhalf) - Wstep1) < 1e-10

## normalize weight matrix such that the objective function is <= O(1) (very roughly speaking)
    if normalize_weight_matrix
        # initial guess = median along all initial conditions
		theta_initial = median(theta0, dims=1) |> vec

		# evaluable moments
		mom_matrix = momfn_loaded(theta_initial)

        # norm
        mom_norm = 1.0 + sqrt(norm(mean(mom_matrix, dims=1)))

        initialWhalf = initialWhalf .* det(initialWhalf)^(-1/n_moms) ./ mom_norm

        show_progress && println("GMM => Normalizing weight matrix.")
    end

## GMM first stage
    show_progress && println("GMM => Launching stage 1, number of initial conditions: ", n_theta0)

	# curve_fit in LsqFit requires to give data (x) to the objective function 
	# we hack this to give the GMM weighting matrix (Cholesky half)

	# optional: save results for each initial condition in a subdirectory
    results_subdir_path = string(results_dir_path, "step1/")
	if write_results_to_file == 2
		show_progress && print("GMM => Creating subdirectory to save one file per initial condition vector...")
		isdir(results_subdir_path) || mkdir(results_subdir_path)
	end

	# define objective function with moment function "loaded"
	gmm_obj_fn = (anyWhalf, any_theta) ->
					gmm_obj(theta=any_theta,
							Whalf=anyWhalf,
							momfn=momfn_loaded, # <- loaded moment function
							show_theta=show_theta)

    # run in parallel
	if run_parallel && n_theta0 > 1

	    all_results_df = pmap(
	        idx -> curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							initialWhalf,
							n_moms,
	                        theta0[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=results_subdir_path,
							maxIter=maxIter,
							time_limit=time_limit,
                            show_trace=show_trace
						), 1:n_theta0)
	else

        # not in parallel
		all_results_df = Vector{Any}(undef, n_theta0)
		for idx=1:n_theta0
	        all_results_df[idx] = curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							initialWhalf,
							n_moms,
	                        theta0[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=results_subdir_path,
							maxIter=maxIter,
							time_limit=time_limit,
							show_trace=show_trace)
		end
	end

    # DataFrame with all results
    all_results_df = vcat(DataFrame.(all_results_df)...)
    
	# pick smallest objective value (among those that have converged) 
    all_results_df.obj_vals_converged = all_results_df.obj_vals

    # TODO: allow iterations that have not converged (+ warning in docs)
    # if not converged, replace objective value with +Infinity
    all_results_df[.~all_results_df.opt_converged, :obj_vals_converged] .= Inf
	
    if minimum(all_results_df.obj_vals_converged) == Inf
        gmm_results["outcome_stage1"] = gmm_results["outcome"] = "fail"
        gmm_results["outcome_stage1_detail"] = ["none of the iterations converged"]

        return gmm_results

    elseif any(.~all_results_df.opt_converged)

        n_converged = sum(all_results_df.opt_converged)

        gmm_results["outcome_stage1"] = gmm_results["outcome"] = "some_errors"
        gmm_results["outcome_stage1_detail"] = [string(n_converged) * "/" * string(n_theta0) * " iterations converged"]

        if minimum(all_results_df.obj_vals_converged) > minimum(all_results_df.obj_vals)
            push!(gmm_results["outcome_detail"], "minimum objective value occurs in iteration that did not converge")
        end

    else
        gmm_results["outcome_stage1"] = gmm_results["outcome"] = "success"
        gmm_results["outcome_stage1_detail"] = ["all iterations converged"]
    end
    
    # pick best
    idx_optimum = argmin(all_results_df.obj_vals_converged)
	all_results_df.is_optimum = ((1:n_theta0) .== idx_optimum)
	
    # select just the estimated parameters
    theta_hat_stage1 = gmm_results["theta_hat_stage1"] = all_results_df[idx_optimum, r"param_"] |> collect
	obj_val_stage1 = all_results_df[idx_optimum, :obj_vals]

	show_progress && println("GMM => Stage 1 optimal theta   ", theta_hat_stage1)
	show_progress && println("GMM => Stage 1 optimal obj val ", obj_val_stage1)

    # save
    # gmm_results["results_stage1"] = copy(all_results_df)
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"results_step1_df.csv")
	    CSV.write(outputfile, all_results_df)
    end 

	## if one-step -> stop here
    if ~two_step
        full_df = hcat(theta0_df, all_results_df)

        gmm_results["theta_hat"] = gmm_results["theta_hat_stage1"]
        return gmm_results, full_df
    end

## Optimal Weighting Matrix
    show_progress && println("GMM => Computing optimal weighting matrix")

    # by default, call vcov_gmm_iid() which computes the variance covariance matrix assuming data is iid
    # can also use user-provided function, useful when using classical minimum distance and non-iid data
	Wstep2 = vcov_fn(theta_hat_stage1, momfn_loaded)

    # ! sketchy
	# if super small determinant:
	# if det(Wstep2) < 1e-100
	# 	show_progress && println(" Matrix determinant very low: 1e-100. Adding 0.001 * I.")
	# 	Wstep2 = Wstep2 + 0.0001 * I
	# end

	# invert (still Hermitian)
	Wstep2 = inv(Wstep2)

    # save
    gmm_results["Wstep2"] = Wstep2
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"Wstep2.csv")
        CSV.write(outputfile, Tables.table(Wstep2), header=false)
    end

	# cholesky half. satisfies Whalf * transpose(Whalf) = W
	optimalWhalf = Matrix(cholesky(Wstep2).L)
	@assert norm(optimalWhalf * transpose(optimalWhalf) - Wstep2) < 1e-10

    ## normalize weight matrix such that the objective function is <= O(1) (very roughly speaking)
    if normalize_weight_matrix
        optimalWhalf = optimalWhalf .* det(optimalWhalf)^(-1/n_moms)

        show_progress && println("GMM => Normalizing weight matrix.")
    end

## GMM second stage
    show_progress && println("GMM => Launching stage 2, number of initial conditions: ", n_theta0)

	# optional: save results for each initial condition in a subdirectory
    results_subdir_path = string(results_dir_path, "step2/")
	if write_results_to_file == 2
		show_progress && print("GMM => Creating subdirectory to save one file per initial condition vector...")
		isdir(results_subdir_path) || mkdir(results_subdir_path)
	end

	# run in parallel
	if run_parallel && n_theta0 > 1

	    all_results2_df = pmap(
	        idx -> curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							optimalWhalf,
							n_moms,
	                        theta0[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=results_subdir_path,
							maxIter=maxIter,
							time_limit=time_limit,
							show_trace=show_trace), 1:n_theta0)
	else

        # not in parallel
		all_results2_df = Vector{Any}(undef, n_theta0)
		for idx=1:n_theta0
			all_results2_df[idx]=curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							optimalWhalf,
							n_moms,
	                        theta0[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=results_subdir_path,
							maxIter=maxIter,
							time_limit=time_limit,
							show_trace=show_trace)
		end
	end

    # one df with all results
    all_results2_df = vcat(DataFrame.(all_results2_df)...)

    rename!(all_results2_df,
        :obj_vals => :obj_vals_step2,
        :opt_converged => :opt_converged_step2,
        :opt_runtime => :opt_runtime_step2)
    
    for i=1:n_params
        rename!(all_results2_df, "param_" * string(i) => "param_" * string(i) * "_step2")
    end

    # pick smallest objective value (among those that have converged) 
    all_results2_df.obj_vals_converged_step2 = all_results2_df.obj_vals_step2

    # TODO: allow iterations that have not converged (+ warning in docs)
    # if not converged, replace objective value with +Infinity
    all_results2_df[.~all_results2_df.opt_converged_step2, :obj_vals_converged_step2] .= Inf

    if minimum(all_results2_df.obj_vals_converged_step2) == Inf
        gmm_results["outcome_stage2"] = gmm_results["outcome"] = "fail"
        gmm_results["outcome_stage2_detail"] = ["none of the iterations converged"]

        return gmm_results

    elseif any(.~all_results2_df.opt_converged_step2)

        n_converged = sum(all_results2_df.opt_converged_step2)

        gmm_results["outcome_stage2"] = "some_errors"
        gmm_results["outcome_stage2_detail"] =[string(n_converged) * "/" * string(n_theta0) * " iterations converged"]

        if minimum(all_results2_df.obj_vals_converged_step2) > minimum(all_results2_df.obj_vals_step2)
            push!(gmm_results["outcome_stage2_detail"], "minimum objective value occurs in iteration that did not converge")
        end

    else
        gmm_results["outcome_stage2"] = "success"
        gmm_results["outcome_stage2_detail"] = ["all iterations converged"]
    end

	# pick best
	idx_optimum = argmin(all_results2_df.obj_vals_converged_step2)
	all_results2_df.is_optimum_step2 = ((1:n_theta0) .== idx_optimum)
	
    # select just the estimated parameters
    theta_hat_stage2 = gmm_results["theta_hat_stage2"] = all_results2_df[idx_optimum, r"param_[0-9]*_step2"] |> collect
	obj_val_stage2 = all_results2_df[idx_optimum, :obj_vals_step2]

    gmm_results["theta_hat"] = gmm_results["theta_hat_stage2"]

	show_progress && println("GMM => stage 2 optimal theta   ", theta_hat_stage2)
	show_progress && println("GMM => stage 2 optimal obj val ", obj_val_stage2)

    # save
    # gmm_results["results_stage2"] = copy(all_results2_df)
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"results_step2_df.csv")
	    CSV.write(outputfile, all_results2_df)
    end 

    # overall outcome of the GMM
    if (gmm_results["outcome_stage1"] == "some_errors") || (gmm_results["outcome_stage2"] == "some_errors")
        gmm_results["outcome"] = "some_errors"
    end

    full_df = hcat(theta0_df, all_results_df, all_results2_df)

	show_progress && println("GMM => complete")

    return gmm_results, full_df
end



## Serial (not parallel) bootstrap with multiple initial conditions

function bootstrap_2step(;
                    boot_run_idx,
					momfn,
					data,
					theta0_boot,
                    theta_lower,
                    theta_upper,
                    theta_hat, # so that we can evaluate the new moment at old parameters
					rootpath_boot_output,
                    sample_data_fn=nothing,
					boot_rng=nothing,
					# run_parallel=false,          # currently, always should be false
					write_results_to_file=false,
					maxIter=100,
					time_limit=-1,
                    show_trace=false,
					throw_exceptions=true,
                    show_theta=false,
                    show_progress=false
					)

try
	show_progress && print(".")

    ## load data and prepare
    # TODO: do and document this better -- default = assume data is a dictionary of vectors/matrices
    # TODO: should also be able to use a boostrapping function
    # TODO: sample_data_fn(DATA::Any, boot_rng::RandomNumberSeed)

    if isnothing(sample_data_fn)
        data_dict_boot = copy(data)
        firstdatakey = first(sort(collect(keys(data_dict_boot))))
        n_observations = size(data[firstdatakey], 1)

        boot_sample = StatsBase.sample(boot_rng, 1:n_observations, n_observations)

        for mykey in keys(data_dict_boot)
            if length(size(data[mykey])) == 1
                data_dict_boot[mykey] = data[mykey][boot_sample]
            elseif length(size(data[mykey])) == 2
                data_dict_boot[mykey] = data[mykey][boot_sample, :]
            end
        end	
    else
        
        # apply the provided function
        data_dict_boot = sample_data_fn(data, boot_rng)
    end

## define the moment function with Boostrap Data
	momfn_loaded = theta -> momfn(theta, data_dict_boot)

## run 2-step GMM and save results
	boot_result, boot_result_df = gmm_2step(
			momfn_loaded=momfn_loaded,
			theta0=theta0_boot,
			theta_lower=theta_lower,
			theta_upper=theta_upper,
			run_parallel=false,
			results_dir_path=rootpath_boot_output,
			write_results_to_file=write_results_to_file, ## TODO: do what here?
			show_trace=show_trace,
			maxIter=maxIter,
			time_limit=time_limit,
            show_theta=show_theta,
			show_progress=false)

    
    # evaluate new boot moment function at the old parameter values
    boot_result["mom_at_theta_hat"] = momfn_loaded(theta_hat)

    boot_result["boot_run_idx"] = boot_run_idx
    boot_result_df[!, "boot_run_idx"] .= boot_run_idx
    # if haskey(boot_result, "results_stage1")
    #     boot_result["results_stage1"][!, "boot_run_idx"] .= boot_run_idx
    # end
    # if haskey(boot_result, "results_stage2")
    #     boot_result["results_stage2"][!, "boot_run_idx"] .= boot_run_idx
    # end

    return boot_result, boot_result_df
catch e
	println("BOOTSTRAP_EXCEPTION for ", rootpath_boot_output)

	bt = catch_backtrace()
	msg = sprint(showerror, e, bt)
	println(msg)

	if throw_exceptions
		throw(e)
    else
        boot_result = Dict{String, Any}(
            "boot_run_idx" => boot_run_idx,
            "outcome" => fail,
            "outcome_detail" => ["bootstrap error " * string(boot_run_idx)]
        )

        # TODO: add DF with all "failed"
        
	end
end

end


function boot_cleanup(;
			rootpath_input, boot_folder,
			idx_batch, idx_run, idx_boot_file,
			mymomfunction,
			show_trace=false,
			maxIter=100,
			time_limit=-1
			)
	#=
		1. read optW and check not empty
		2. load data and big mats
		3. collect initial condtions to run
		4. define function
		5. run wrapper and write to file
		6. write all results to 2nd stage file
	=#
	println("\n\n\n... Processing batch ", idx_batch, " run ", idx_run, "\n\n\n")

	## Step 1. Read optW and check it's ok
	mypath = string(boot_folder, "gmm-stage1-optW.csv")
	optW = readdlm(mypath, ',', Float64)

	# define everywhere
	optimalWhalf = Matrix(cholesky(Hermitian(optW)).L)
	@assert size(optW) == (60, 60)

	## Read initial conditions
	mypath = string(boot_folder, "gmm-initial-conditions.csv")
	theta0_boot_df = CSV.read(mypath, DataFrame)
	select!(theta0_boot_df, r"param_")
	theta0_boot = Matrix(theta0_boot_df)

	# display(theta0_boot[1,:])

	## Step 2. Load data
	begin
		## initial params
		n_ha = 79
		n_hd = 79 # one per 5 minutes

		hdgrid = vec(Array(range(-2.5,stop=4.0,length=n_hd)))
		hagrid = vec(Array(range(-2.0,stop=4.5,length=n_ha)))

		bin_l = -2.5
		bin_h = 2.5
		n_hd_bins = 61

		n_hd_mom_l = 7 # corresponds to -2h
		n_hd_mom_h = 55 # corresponds to 2h

		n_moms = (n_hd_mom_h - n_hd_mom_l + 1) + 2 + 12

		## load boot samples:
		bootstrap_samples = Dict{String, Vector{Int64}}()

		## Load boot samples to file
		path = string(boot_folder,"boot_sample_a.csv")
		bootstrap_samples["a"] = readdlm(path, ',', Float64) |> vec
		# display(bootstrap_samples["a"])

		path = string(boot_folder,"boot_sample_na.csv")
		bootstrap_samples["na"] = readdlm(path, ',', Float64) |> vec

		## load data and prepare
		BIGMAT_BOOT, CHARGEMAT_BOOT, COMPMAT_BOOT, DATAMOMS_BOOT =
			load_and_prep(rootpath_input, hdgrid=hdgrid, hagrid=hagrid, bootstrap_samples=bootstrap_samples)
		DATAMOMS_BOOT["n_hd_bins"] = n_hd_bins
		DATAMOMS_BOOT["hd_bins_hl"] = [bin_h, bin_l, n_hd_mom_l, n_hd_mom_h]
		DATAMOMS_BOOT["hdgrid"] = hdgrid

		n_resp_all, n_resp_a, n_resp_a_ct23, n_resp_a_tr = DATAMOMS_BOOT["n_resp_vec"]
	end


	## Step 4. Define function (with boot data loaded)
		my_momfn = theta -> mymomfunction(theta,
							BIGMAT=BIGMAT_BOOT,
							CHARGEMAT=CHARGEMAT_BOOT,
							COMPMAT=COMPMAT_BOOT,
							DATAMOMS=DATAMOMS_BOOT)

	## Step 5. Run wrapper and write to file

		# objective function with moment function "loaded"
		gmm_obj_fn = (anyWhalf, any_theta) ->
			gmm_obj(theta=any_theta, Whalf=anyWhalf, momfn=my_momfn, debug=false)

		# subdirectory for individual run results
		print("GMM => 3. creating subdirectory to save individual run results...")
		results_subdir_path = string(boot_folder, "stage2")
		isdir(results_subdir_path) || mkdir(results_subdir_path)

		# run GMM
		# n_runs = length(step2_runs)
		n_moms = size(optW)[1]

		curve_fit_wrapper(idx_boot_file,
					gmm_obj_fn, optimalWhalf,
					n_moms,
					theta0_boot[idx_boot_file,:],
					theta_lower,
					theta_upper,
					write_results_to_file=true,
					individual_run_results_path=string(boot_folder, "stage2/"),
					maxIter=maxIter,
					time_limit=time_limit,
                    show_trace=show_trace)

end


function vector_theta_fix(mytheta, theta_fix)
    
    # indices of parameters to estimate
    idxs_estim = findall(isnothing, theta_fix)
    
    # different for vectors and matrices
    if isa(mytheta, Vector)
        return mytheta[idxs_estim]
    else
        return mytheta[:, idxs_estim]
    end

end


"""
Calls the moment function `momfn` combining `mytheta` and the fixed parameters in `theta_fix`
"""
function momfn_theta_fix(momfn, mytheta, mydata_dict, theta_fix)

    # indices of parameters that are fixed and those that need to be estimated
    idxs_fixed = findall(x -> ~isnothing(x), theta_fix)
    idxs_estim = findall(isnothing, theta_fix)

    vals_fixed = [theta_fix[idx] for idx=idxs_fixed]

    # make full parameter vector
    mytheta_new = zeros(length(theta_fix))
    mytheta_new[idxs_fixed] .= vals_fixed
    mytheta_new[idxs_estim] .= mytheta

    return momfn(mytheta_new, mydata_dict)
end

function omega_subset(myomega, moms_subset)
    if isa(myomega, Matrix)
        return myomega[moms_subset, moms_subset]
    elseif isa(myomega, Function)
        return (theta, momfn) -> myomega(theta, momfn)[moms_subset, moms_subset]
    end
end


"""
myfactor: higher factor = larger changes in parameters
max range -- in order to avoid sampling outside boundaries
"""
function compute_jacobian(;
        momfn,
        data,
        theta_hat,
        theta_upper=nothing, 
        theta_lower=nothing,
        theta_fix=nothing,
        moms_subset=nothing,
        myfactor=1.0)
    
    # subset parameter and moments, if applicable
    if ~isnothing(theta_fix)

        # define moment function taking as input only the variable entries in theta
        momfn2 = (mytheta, mydata_dict) -> momfn_theta_fix(momfn, mytheta, mydata_dict, theta_fix)
        
        # subset the other entries
        theta_hat   = vector_theta_fix(theta_hat, theta_fix)
        theta_upper = vector_theta_fix(theta_upper, theta_fix)
        theta_lower = vector_theta_fix(theta_lower, theta_fix)
            
    else
        momfn2 = momfn
    end

    if ~isnothing(moms_subset)
        # moms_subset is a sorted vector of distinct indices between 1 and n_moms 
        momfn3 = (mytheta, mydata_dict) -> momfn2(mytheta, mydata_dict)[:, moms_subset]
    else
        momfn3 = momfn2
    end
    
    # function that computes averaged moments
    mymomfunction_main_avg = mytheta -> mean(momfn3(mytheta, data), dims=1)

    # numerical jacobian
    my_max_range = 0.9 * min(minimum(abs.(theta_hat .- theta_lower)), minimum(abs.(theta_hat .- theta_upper)))

    # compute jacobian
    myjac = jacobian(central_fdm(5, 1, factor=myfactor, max_range=my_max_range), mymomfunction_main_avg, theta_hat)

    return myjac[1]
end


"""
    run_gmm(; momfn, data, theta0, theta0_boot=nothing, theta_upper=nothing, theta_lower=nothing, gmm_options=nothing)

Wrapper for GMM/CMD with multiple initial conditions and optional bootstrap inference.

# Arguments
- momfn: the moment function momfn(theta, data)
- data: any object
- theta0: matrix of size main_n_start_pts x n_params, main_n_start_pts = gmm_options["main_n_start_pts] and n_params is length of theta
- theta0_boot: matrix of size (boot_n_start_pts * boot_n_runs) x n_params
- theta_lower: vector of lower bounds (default is -Inf)
- theta_upper: vector of upper bounds (default is +Inf)

Note: all arguments must be named (indicated by the ";" at the start), meaning calling [1] will work but [2] will not:
[1] run_gmm(momfn=my_moment_function, data=mydata, theta0=my_theta0)
[2] run_gmm(my_moment_function, mydata, my_theta0)
"""

function run_gmm(;
		momfn,
		data,
		theta0, 
        theta0_boot=nothing,
        theta_upper=nothing, 
        theta_lower=nothing,
        W=nothing,
        omega=nothing, # nothing, function or matrix
        theta_fix=nothing,
        moms_subset=nothing,
        sample_data_fn=nothing, # function for slow bootstrapping
		gmm_options=nothing,
	)

## make local copy and run checks
    gmm_options = copy(gmm_options)
    if ~isnothing(omega)
        omega = copy(omega)
    end

    if isnothing(omega) && (get(gmm_options, "estimator", "") == "cmd")
        error("If using CMD must provide omega the var-covar of the moments")
    end

    # runchecks(theta0, theta0_boot, theta_upper, theta_lower, gmm_options)

## Number of parameters
    # if only one initial condition as Vector, convert to 1 x n_params matrix
    if isa(theta0, Vector)
        theta0 = Matrix(transpose(theta0))
    end
    n_params = size(theta0)[2]

    # Default parameter bounds
    if isnothing(theta_lower) 
        theta_lower = fill(-Inf, n_params)
    end
    if isnothing(theta_upper) 
        theta_upper = fill(Inf, n_params)
    end

## Default options
	gmm_options_default = Dict(

        "param_names" => nothing, # vector of parameter names (strings)
        "n_observations" => nothing, # number of observations (data size)
        "n_moms" => nothing, # number of moments

        # one-step or two-step GMM
        "estimator" => "gmm2step", #"gmm1step" or "gmm2step" or "cmd" or "cmd_optimal"

        # main gmm estimation
		"run_main" 			=> true, # run main estimation? False if only want to run bootstrap
		        # "main_n_theta0" => 1, # number of independent optimization runs with different initial conditions
        # "main_Wstep1" => nothing,
        "normalize_weight_matrix" => false,

		"main_write_results_to_file" => 0, # 0, 1 or 2, see definition in gmm_2step()
        "main_run_parallel" => false, # different starting conditions run in parallel
		"main_maxIter" 		=> 1000, # maximum number of iterations for curve_fit() from LsqFit.jl
		"main_time_limit" 	=> -1, # maximum time for curve_fit() from LsqFit.jl (Use https://github.com/Gkreindler/LsqFit.jl)
        "main_show_trace" 	=> false, # display optimization trace
		"main_show_theta" 	=> false, # during optimization, print current value of theta for each evaluation of momfn + value of objective function

        # inference:
        "var_asy" => true,  # Compute asymptotic variance covariance matrix
        "var_boot" => nothing,  # bootstrap. nothing or "quick" or "slow"

        # bootstrap:
        "boot_run_parallel" => false, # each bootstrap run in parallel.
		"boot_n_runs" 		=> 100, # number of bootstrap runs 
		        # "boot_n_theta0"=> 1, # for each bootstrap run: number of independent optimization runs with different starting conditions
		"boot_write_results_to_file" => 0, # similar to "main_write_results_to_file"
		"boot_maxIter" 		=> 1000, # 
		"boot_time_limit"	=> -1, # 
		"boot_throw_exceptions" => true, # if "false" will not crash when one bootstrap run has error. Error will be recorded in gmm_results["gmm_boot_results"]
        # ! add errors from boot in final results gmm_results["gmm_boot_results"]
        "boot_show_trace" 	=> false, # 
		"boot_show_theta"	=> false, # 

        # paths:
		"rootpath_input" => "",
		"rootpath_output" => "",

        # misc
        "show_progress" => true, # print overall progress/steps

        # use estimated parameters when the optimizer did not converge (due to iteration limit or time limit). 
        # Attention! Results may be wrong/misleading! Only use for testing or if this is OK for use case.
        "use_unconverged_results" => false
	)

    # for options that are not provided, use the default
    if isnothing(gmm_options) 
        gmm_options = Dict{String, Any}()
    end

	for mykey in keys(gmm_options_default)
		if ~haskey(gmm_options, mykey)
			gmm_options[mykey] = gmm_options_default[mykey]
		end
	end

## get number of observations in the data and number of moments
    if isnothing(gmm_options["n_observations"]) || isnothing(gmm_options["n_moms"])
        theta_test = theta0[1, :]
        mymoms = momfn(theta_test, data)
        gmm_options["n_observations"] = size(mymoms)[1]
        gmm_options["n_moms_full"] = size(mymoms)[2]
    end

## Fix some of the parameters (estimate only the others)
    if ~isnothing(theta_fix)

        # define moment function taking as input only the variable entries in theta
        momfn2 = (mytheta, mydata_dict) -> momfn_theta_fix(momfn, mytheta, mydata_dict, theta_fix)
        
        # subset the other entries
        theta0      = vector_theta_fix(theta0, theta_fix)
        theta_upper = vector_theta_fix(theta_upper, theta_fix)
        theta_lower = vector_theta_fix(theta_lower, theta_fix)
        isnothing(theta0_boot) || (theta0_boot = vector_theta_fix(theta0_boot, theta_fix))
            
    else
        momfn2 = momfn
    end

## Subset of moments?
    if ~isnothing(moms_subset)

        # basic checks: moms_subset is a sorted vector of distinct indices between 1 and n_moms 
        sort!(moms_subset)
        # TODO: add checks for moms_subset

        momfn3 = (mytheta, mydata_dict) -> momfn2(mytheta, mydata_dict)[:, moms_subset]

        gmm_options["n_moms"] = length(moms_subset)

        # update the variance-covariance matrix to the subset of moment we're using
        omega1 = omega_subset(omega, moms_subset)

    else
        momfn3 = momfn2
        gmm_options["n_moms"] = gmm_options["n_moms_full"]
        omega1 = omega
    end

## two-step estimator?
    gmm_options["2step"] = gmm_options["estimator"]  == "gmm2step"

## CMD
    if gmm_options["estimator"] == "cmd_optimal"
        # optimal W = Ω⁻¹
        W = inv(Symmetric(omega1))
        @assert issymmetric(W)

    elseif (gmm_options["estimator"] in ["cmd", "gmm1step"]) && isnothing(W)
        # optimal W = Ω⁻¹
        W = diagm(ones(gmm_options["n_moms"]))
        @assert issymmetric(W)

    elseif ~issymmetric(W)
        @warn "The provided weighting matrix W is not symmetric."
        W = Symmetric(W)
    end

## Number of initial conditions
    main_n_initial_cond = size(theta0, 1)
    if isnothing(theta0_boot)
        boot_n_initial_cond = 0
    else
        boot_n_initial_cond = size(theta0_boot, 1)
    end



## Store estimation results here
    # store parameters and options
    full_results = Dict{String, Any}(
        "gmm_options" => gmm_options,
        "theta0" => theta0,
        "theta0_boot" => theta0_boot,
        "theta_upper" => theta_upper,
        "theta_lower" => theta_lower,
        "W" => W,
        "omega" => omega,
        "theta_fix" => theta_fix,
        "moms_subset" => moms_subset,
        "n_observations" => gmm_options["n_observations"],
        "n_moms" => gmm_options["n_moms"],
        "n_moms_full" => gmm_options["n_moms_full"],
        "n_params" => n_params,
        "main_n_initial_cond" => main_n_initial_cond,
        "boot_n_initial_cond" => boot_n_initial_cond
    )

    

    
## Misc
    show_progress = gmm_options["show_progress"]
    
    boot_result_json = nothing
    boot_result_dfs  = nothing

## Load data into moments function
	momfn_loaded = theta -> momfn3(theta, data)

## Run two-step GMM / CMD with optimal weighting matrix
    # if gmm_options["run_main"]

    show_progress && println("Starting main estimation")

    # TODO: add "use_unconverged_results" option
    if gmm_options["run_main"] 
        full_results["results"], main_results_df  = gmm_2step(
            momfn_loaded    =momfn_loaded,

            theta0  =theta0,
            theta_lower     =theta_lower,
            theta_upper     =theta_upper,

            two_step    = gmm_options["2step"],
            Wstep1=W,
            normalize_weight_matrix=gmm_options["normalize_weight_matrix"],
            vcov_fn=omega,
            
            results_dir_path=gmm_options["rootpath_output"],
            write_results_to_file=gmm_options["main_write_results_to_file"],
            
            run_parallel    =gmm_options["main_run_parallel"],
            maxIter      =gmm_options["main_maxIter"],
            time_limit   =gmm_options["main_time_limit"],

            show_trace   =gmm_options["main_show_trace"],
            show_theta      =gmm_options["main_show_theta"],
            show_progress   =show_progress)
    
    else
        full_results["results"] = Dict(
            "outcome" => "skipped",
            "theta_hat" => vec(theta0)
        )
    end

    ## Asymptotic Variance
    if (gmm_options["var_asy"] || gmm_options["var_boot"] == "quick") && full_results["results"]["outcome"] != "fail"

        show_progress && println("Computing asymptotic variance")

        # Get estimated parameter vector
        theta_hat = full_results["results"]["theta_hat"]
        
        # function that computes averaged moments
        mymomfunction_main_avg = theta -> mean(momfn_loaded(theta), dims=1)

        # TODO: add autodiff option
        ## numerical jacobian
        # higher factor = larger changes in parameters
        myfactor = 1.0

        # max range -- in order to avoid sampling outside boundaries
        my_max_range = 0.9 * min(minimum(abs.(theta_hat .- theta_lower)), minimum(abs.(theta_hat .- theta_upper)))

        # compute jacobian
        myjac = jacobian(central_fdm(5, 1, factor=myfactor, max_range=my_max_range), mymomfunction_main_avg, theta_hat)

        G = myjac[1]

        # different formulas if optimal (2step GMM or optimal CMD) or any other weight matrix
        # same for 2-step optimal GMM and CMD
        # https://ocw.mit.edu/courses/14-386-new-econometric-methods-spring-2007/b8a285cadaa8203272ad3cbce3ef445f_ngmm07.pdf
        # https://ocw.mit.edu/courses/14-384-time-series-analysis-fall-2013/7ddedae5317fdd5424ff924688df7c7c_MIT14_384F13_rec12.pdf

        if gmm_options["estimator"] == "cmd"
            # (G'WG)⁻¹G' W Ω W G(G'WG)⁻¹
            # Ω = Symmetric(gmm_options["cmd_omega"])

            Ω = Symmetric(omega1)
            W = Symmetric(W)

            bread = inv(transpose(G) * W * G) 
            V = bread * transpose(G) * W * Ω * W * G * bread
        end

        if gmm_options["estimator"] == "cmd_optimal"
            # W = Ω⁻¹ so the above simplifies to (G'WG)⁻¹
            # Ω = Symetric(gmm_options["cmd_omega"])
            W = Symmetric(W)

            V = inv(transpose(G) * W * G) 
        end

        if gmm_options["estimator"] == "gmm1step"
            # (G'WG)⁻¹G' W Ω W G(G'WG)⁻¹

            if isnothing(omega1)
                vcov_fn = vcov_gmm_iid
            else
                vcov_fn = omega1
            end
            theta_hat_stage1 = full_results["results"]["theta_hat_stage1"]
            Ω = vcov_fn(theta_hat_stage1, momfn_loaded)

            W = Symmetric(W)

            bread = inv(transpose(G) * W * G) 
            V = bread * transpose(G) * W * Ω * W * G * bread
        end

        if gmm_options["estimator"] == "gmm2step"
            # W = Ω⁻¹ so the above simplifies to (G'WG)⁻¹
            W = Symmetric(full_results["results"]["Wstep2"])
            V = inv(transpose(G) * W * G) 
        end

        # treat GMM and CMD differently
        n_observations = gmm_options["n_observations"]

        full_results["G"] = G
        full_results["asy_vcov"] = V / n_observations
        full_results["asy_stderr"] = sqrt.(diag(V / n_observations))

        
        ### Quick bootstrap
        # https://schrimpf.github.io/GMMInference.jl/bootstrap/
        # https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/resources/mit14_382s17_lec3/
        # https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/resources/mit14_382s17_lec5/
        if gmm_options["var_boot"] == "quick"
            
            # √n(θ̂ -θ₀) ∼ (G'AG)⁻¹G'Aϵ where G is Jacobian and A is the weighting matrix
            M = (transpose(G) * W * G) \ (transpose(G) * W)

            # Z=g(X,θ̂ ), demeaned
            Z = momfn_loaded(theta_hat) .- mean(momfn_loaded(theta_hat), dims=1)

            boot_n_runs = gmm_options["boot_n_runs"]
            rng = MersenneTwister(123);
            boot_result_json = Vector(undef, boot_n_runs)
            for i=1:boot_n_runs
                boot_sample = StatsBase.sample(rng, 1:n_observations, n_observations)

                Z_boot = mean(Z[boot_sample, :], dims=1)

                theta_hat_boot = theta_hat + vec(M * transpose(Z_boot))

                boot_result_json[i] = Dict(
                    "theta_hat" => theta_hat_boot,
                    "boot_sample" => boot_sample
                )
            end
        end

        # write to file?
        if gmm_options["main_write_results_to_file"] > 0
            outputfile = string(gmm_options["rootpath_output"], "asy_vcov.csv")
            CSV.write(outputfile, Tables.table(full_results["asy_vcov"]), header=false)

            outputfile = string(gmm_options["rootpath_output"], "jacobian.csv")
            CSV.write(outputfile, Tables.table(full_results["G"]), header=false)
        end

    end
    # end

## Run "slow" bootstrap where we re-run the minimization each time    
    if gmm_options["var_boot"] == "slow"
        show_progress && println("Starting boostrap")

        if gmm_options["boot_write_results_to_file"] > 0
            rootpath_boot_output = gmm_options["rootpath_output"] * "boot/"
            isdir(rootpath_boot_output) || mkdir(rootpath_boot_output)
        end

        # Get estimated parameter vector
        theta_hat = theta_hat = full_results["results"]["theta_hat"]

        # Define moment function for bootstrap -- target moment value at estimated theta
        mom_at_theta_hat = mean(momfn_loaded(theta_hat), dims=1)

        momfn_boot = (mytheta, mydata_dict) -> (momfn3(mytheta, mydata_dict) .- mom_at_theta_hat)

        # one random number generator per bootstrap run
        current_rng = MersenneTwister(123);
        boot_rngs = Vector{Any}(undef, gmm_options["boot_n_runs"])

        show_progress && println("Creating random number generator for boot run:")
        for i=1:gmm_options["boot_n_runs"]
            show_progress && print(".")
            
            # increment and update "current" random number generator
            boot_rngs[i] = Future.randjump(current_rng, big(10)^20)
            current_rng = boot_rngs[i]

            # can check that all these are different
            # print(boot_rngs[i])  
        end
        show_progress && println(".")

        # Todo: what are folder options for boot? (similar to main?)
        # Create folders where we save estimation results
        boot_folders = Vector{String}(undef, gmm_options["boot_n_runs"])
        for i=1:gmm_options["boot_n_runs"]
            boot_folders[i] = rootpath_boot_output * "boot_run_" * string(i) * "/"
            isdir(boot_folders[i]) || mkdir(boot_folders[i])
        end

        # Run bootstrap
        boot_n_runs = gmm_options["boot_n_runs"]

        show_progress && println("Bootstrap runs:")
        if gmm_options["boot_run_parallel"]
            boot_results = pmap(
            idx -> bootstrap_2step(
                        boot_run_idx=idx,
                        momfn=momfn_boot,
                        data=data,
                        theta0_boot=theta0_boot,
                        theta_lower=theta_lower,
                        theta_upper=theta_upper,
                        theta_hat=theta_hat,
                        rootpath_boot_output=boot_folders[idx],
                        sample_data_fn=sample_data_fn,
                        boot_rng=boot_rngs[idx],
                        write_results_to_file=gmm_options["boot_write_results_to_file"],
                        maxIter=gmm_options["boot_maxIter"],
                        time_limit=gmm_options["boot_time_limit"],
                        throw_exceptions=gmm_options["boot_throw_exceptions"],
                        show_trace=false,
                        show_theta=gmm_options["boot_show_theta"],
                        show_progress=show_progress
                    ), 1:boot_n_runs)
        else

            boot_results = Vector{Any}(undef, boot_n_runs)
            for boot_run_idx=1:boot_n_runs
                boot_results[boot_run_idx] = bootstrap_2step(
                        boot_run_idx=boot_run_idx,
                        momfn=momfn_boot,
                        data=data,
                        theta0_boot=theta0_boot,
                        theta_lower=theta_lower,
                        theta_upper=theta_upper,
                        theta_hat=theta_hat,
                        rootpath_boot_output=boot_folders[boot_run_idx],
                        sample_data_fn=sample_data_fn,
                        boot_rng=boot_rngs[boot_run_idx],
                        write_results_to_file=gmm_options["boot_write_results_to_file"],
                        maxIter=gmm_options["boot_maxIter"],
                        time_limit=gmm_options["boot_time_limit"],
                        throw_exceptions=gmm_options["boot_throw_exceptions"],
                        show_trace=false,
                        show_progress=show_progress,
                        show_theta=gmm_options["boot_show_theta"]
                    )
            end
        end
        show_progress && println()

        # JSON and CSV with boot results
        boot_result_json = []
        boot_result_dfs = []
        for boot_result=boot_results
            mydict, mydf = boot_result

            push!(boot_result_json, mydict)
            push!(boot_result_dfs, mydf)
        end

        boot_result_dfs = vcat(boot_result_dfs...)

        if gmm_options["boot_write_results_to_file"] > 0
            open(rootpath_boot_output * "results_boot.json" ,"w") do f
                JSON.print(f, boot_result_json, 4)
            end
    
            CSV.write(rootpath_boot_output * "results_boot_df.csv", boot_result_dfs)
        end

    end # end

    if gmm_options["main_write_results_to_file"] > 0
        open(gmm_options["rootpath_output"] * "results.json" ,"w") do f
            JSON.print(f, full_results, 4)
        end

        CSV.write(gmm_options["rootpath_output"] * "results_df.csv", main_results_df)
    end

    return full_results, main_results_df, boot_result_json, boot_result_dfs
end
