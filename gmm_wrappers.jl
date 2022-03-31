

using LinearAlgebra
using LsqFit 		# has to be modified version that accepts time limit


# TODO: describe what this wrapper does
"""
"""
function gmm_obj(;theta, Whalf, momfn, debug=false)

	# print parameter vector at current step
	debug && print("theta ", theta, " ")

	# compute moments
	mymoms = momfn(theta)

	# multiply by (Cholesky) half matrice and take means
	mean_mymoms = vec(mean(mymoms, dims=1) * Whalf)

	# write the value of the objective function at the end of the line
	debug && println(transpose(mean_mymoms) * mean_mymoms)

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
				my_show_trace=true,
				my_maxIter=1000,
				my_time_limit=-1,
				debug=false
			)

	debug && println("starting iteration ", idx)

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
				show_trace=my_show_trace,
				maxIter=my_maxIter,
				time_limit=my_time_limit # this is added relative to official LsqFit.jl . Use https://github.com/Gkreindler/LsqFit.jl
			)

    # TODO: remove this?
	if debug
		println(">>> iteration ", idx, " took: ", timeittook, " seconds. Converged: ", result.converged)
		println(">>> optimum ", result.param)
		println(">>> obj val ", norm(result.resid))
	end

    # results dictionary
    results_df = Dict{String, Any}(
        "obj_vals" => norm(result.resid),
        "opt_converged" => result.converged .+ 0,
		"opt_runtime" => timeittook
    )
    for i=1:length(result.param)
        results_df[string("param_", i)] = result.param[i]
    end

    # save to file
	if write_results_to_file == 2
	    outputfile = string(individual_run_results_path,"gmm-stage1-run",idx,".csv")
		CSV.write(outputfile, results_df)
	end

	return results_df
end

"""
	momfn = the moment function
		- should take a single vector argument
		- data etc is already "loaded" in this function
	theta_initials = initial condition (vector)
	theta_lower, theta_upper = bounds (vectors, can include +/-Inf)
	n_runs = number of initial conditions
	n_moms = size of moment function (number of moments) TODO: get this auto
	results_dir_path = where to write results
	Wstep1 = weighting matrix (default = identity matrix).
			Should be Hermitian (this will be enforced).
			Will be Cholesky decomposed
	jacobian = provide jacobian function
	write_results_to_file = 
        0 = nothing written to file
        1 = write objects to file, including one csv with all runs (once for first stage, once for second stage)
        2 = write objects to file, including one csv per each run
	run_parallel  = individual runs in parallel vs in serial (default=parallel)
	my_show_trace = show trace of curve_fit from LsqFit?
	my_maxIter    = maximum iterations for curve_fit from LsqFit
	debug 		  = pass on to objective function
"""
function gmm_2step(;
			momfn_loaded,
			theta_initials,
			theta_lower,
			theta_upper,
			n_runs,
			run_parallel=true,
            run_one_step_only=false,
			n_moms=nothing,
			Wstep1=nothing,
			Wstep1_from_moms=false,
			results_dir_path="",
            write_results_to_file=0,
			my_show_trace=false,
			my_maxIter=1000,
			my_time_limit=-1,
			debug=false)

## Basic checks
    if write_results_to_file ∉ [0, 1, 2]
        error("write_results_to_file should be 0, 1, or 2")
    end

    n_params = size(theta_initials, 2)

## Store GMM results here
    gmm_main_object = Dict{String, Any}()

## if theta_initials is a vector, turn to 1 x K matrix
	if isa(theta_initials, Vector)
		theta_initials = Matrix(transpose(theta_initials))
	end

## if not provided, compute number of moments by running moment function once
	if isnothing(n_moms)
		n_moms = size(momfn_loaded(theta_initials[1,:]), 2)
	end

## save starting conditions
## print to csv file
	initial_conditions_df = DataFrame(
		"iteration" => 1:n_runs
	)
	for i=1:size(theta_initials)[2]
		initial_conditions_df[!, string("param_", i)] = vec(theta_initials[:, i])
	end

	# save
    gmm_main_object["initial_conditions_df"] = initial_conditions_df
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"gmm_initial_conditions.csv")
        CSV.write(outputfile, initial_conditions_df)
    end

## Initial weighting matrix Wstep1
	if ~isnothing(Wstep1) && Wstep1_from_moms
		error("cannot have both Wstep1 matrix AND Wstep1_from_moms=true")
	end

    # Optional: compute initial weighting matrix based on moments covariances at initial "best guess" theta
	if Wstep1_from_moms

		# initial guess = median along all runs
		theta_initial = median(theta_initials, dims=1) |> vec

		# evaluable moments
		mom_matrix = momfn_loaded(theta_initial)

		# compute optimal W matrix -> ensure it's Hermitian
		nmomsize = size(mom_matrix, 1)
		Wstep1 = Hermitian(transpose(mom_matrix) * mom_matrix / nmomsize)

		# if super small determinant:
		if det(Wstep1) < 1e-100
			debug && println(" Matrix determinant very low: 1e-100. Adding 0.001 * I.")
			Wstep1 = Wstep1 + 0.001 * I
		end

		# invert (still Hermitian)
		Wstep1 = inv(Wstep1)
		Wstep1 = Wstep1 * det(Wstep1)^(-1/size(Wstep1,1))

		debug && println("Step 1 weighting matrix: determinant=", det(Wstep1))

	# if not provided, use identity weighting matrix
	elseif isnothing(Wstep1)
		Wstep1 = diagm(ones(n_moms))
	end

	## save
    gmm_main_object["Wstep1"] = Wstep1
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"gmm_Wstep1.csv")
        CSV.write(outputfile, Tables.table(Wstep1), header=false)
    end

	# cholesky half. 
    # satisfies Whalf * transpose(Whalf) = W
	initialWhalf = Matrix(cholesky(Hermitian(Wstep1)).L)
	@assert norm(initialWhalf * transpose(initialWhalf) - Wstep1) < 1e-10

## FIRST stage
    println("GMM => Launching FIRST stage, n parallel runs: ", n_runs)

	# curve_fit in LsqFit requires to give data (x) to the objective function 
	# we hack this to give the GMM weighting matrix (Cholesky half)

	# subdirectory for individual run results
	if write_results_to_file == 2
		debug && print("GMM => 1. creating subdirectory to save individual run results...")
		results_subdir_path = string(results_dir_path, "stage1")
		isdir(results_subdir_path) || mkdir(results_subdir_path)
	end

	# define objective function with moment function "loaded"
	gmm_obj_fn = (anyWhalf, any_theta) ->
					gmm_obj(theta=any_theta,
							Whalf=anyWhalf,
							momfn=momfn_loaded, # <- loaded moment function
							debug=debug)

	if run_parallel && n_runs > 1

	    all_results_df = pmap(
	        idx -> curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							initialWhalf,
							n_moms,
	                        theta_initials[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=string(results_dir_path, "stage1/"),
	                        my_show_trace=my_show_trace,
							my_maxIter=my_maxIter,
							my_time_limit=my_time_limit,
							debug=debug
						), 1:n_runs)
	else

        # not in parallel
		all_results_df = Vector{Any}(undef, n_runs)
		for idx=1:n_runs
	        all_results_df[idx] = curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							initialWhalf,
							n_moms,
	                        theta_initials[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=string(results_dir_path, "stage1/"),
	                        my_show_trace=my_show_trace,
							my_maxIter=my_maxIter,
							my_time_limit=my_time_limit,
							debug=debug)
		end
	end

    # one df with all results
    all_results_df = vcat(DataFrame.(all_results_df)...)
    
    # obj_vals = [norm(all_runs_results[idx]["result"].resid) for idx=1:n_runs]
    # opt_thetas = [all_runs_results[idx]["result"].param for idx=1:n_runs]
	# opt_converged = [all_runs_results[idx]["result"].converged for idx=1:n_runs]
    # opt_runtime = [all_runs_results[idx]["timeittook"] for idx=1:n_runs]

	# pick best
	idx_best = argmin(all_results_df.obj_vals)
	all_results_df.is_best_vec = ((1:n_runs) .== idx_best) .+ 0
	
    # select just the estimated parameters
    theta_stage1 = all_results_df[idx_best, r"param_"] |> collect
	obj_val_stage1 = all_results_df[idx_best, :obj_vals]

	debug && println("GMM => 1. FIRST STAGE optimal theta   ", theta_stage1)
	debug && println("GMM => 1. FIRST STAGE optimal obj val ", obj_val_stage1)

    # save
    gmm_main_object["results_stage1"] = all_results_df
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"gmm_results_stage1_all.csv")
	    CSV.write(outputfile, all_results_df)
    end 
    ## print to csv file
	# stage1_df = DataFrame(
	# 	"run" => 1:n_runs,
	# 	"obj_vals" => obj_vals,
	# 	"opt_converged" => opt_converged .+ 0,
	# 	"opt_runtime" => opt_runtime,
	# 	"is_best_vec" => is_best_vec
	# )
	# for i=1:length(all_runs_results[1]["result"].param)
	# 	stage1_df[!, string("param_", i)] = [all_runs_results[idx]["result"].param[i] for idx=1:n_runs]
	# end

	

	# TODO: if one-step -> stop here
    if run_one_step_only
        return gmm_main_object
    end

## Optimal Weighting Matrix
	println("GMM => Computing optimal weighting matrix")

    # compute matrix of moments
	mom_matrix = momfn_loaded(theta_stage1)

 	# compute optimal W matrix -> ensure it's Hermitian
	nmomsize = size(mom_matrix, 1)
	Wstep2 = Hermitian(transpose(mom_matrix) * mom_matrix / nmomsize)

    # ! TODO: isn't this sketchy?
	# if super small determinant:
	if det(Wstep2) < 1e-100
		debug && println(" Matrix determinant very low: 1e-100. Adding 0.001 * I.")
		Wstep2 = Wstep2 + 0.0001 * I
	end

	# invert (still Hermitian)
	Wstep2 = inv(Wstep2)

    # ! cut
	# normalize?
	if debug
		println("original Determinant and matrix")
		display(det(Wstep2))
		display(Wstep2)
	end

    # normalize
	Wstep2 = Wstep2 * det(Wstep2)^(-1/size(Wstep2,1))

	if debug
		println("Determinant and matrix:")
		display(det(Wstep2))  # close to 1
		display(Wstep2)
	end

    ## save
    gmm_main_object["Wstep2"] = Wstep2
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"gmm_Wstep2.csv")
        CSV.write(outputfile, Tables.table(Wstep2), header=false)
    end

	# cholesky half. satisfies Whalf * transpose(Whalf) = W
	optimalWhalf = Matrix(cholesky(Wstep2).L)
	@assert norm(optimalWhalf * transpose(optimalWhalf) - Wstep2) < 1e-10


## SECOND stage
    println("GMM => Launching SECOND stage, n parallel runs: ", n_runs)

	debug && display(optimalWhalf)

	# subdirectory for individual run results
	if write_results_to_file == 2
		print("GMM => 3. creating subdirectory to save individual run results...")
		results_subdir_path = string(results_dir_path, "stage2")
		isdir(results_subdir_path) || mkdir(results_subdir_path)
	end

	# run GMM
	if run_parallel && n_runs > 1

	    all_results_df = pmap(
	        idx -> curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							optimalWhalf,
							n_moms,
	                        theta_initials[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=string(results_dir_path, "stage2/"),
	                        my_show_trace=my_show_trace,
							my_maxIter=my_maxIter,
							my_time_limit=my_time_limit,
							debug=debug), 1:n_runs)
	else

		all_results_df = Vector{Any}(undef, n_runs)
		for idx=1:n_runs
			all_results_df[idx]=curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							optimalWhalf,
							n_moms,
	                        theta_initials[idx,:], theta_lower, theta_upper,
							write_results_to_file=write_results_to_file,
							individual_run_results_path=string(results_dir_path, "stage2/"),
	                        my_show_trace=my_show_trace,
							my_maxIter=my_maxIter,
							my_time_limit=my_time_limit,
							debug=debug)
		end
	end

    # one df with all results
    all_results_df = vcat(DataFrame.(all_results_df)...)

	# pick best
	idx_best = argmin(all_results_df.obj_vals)
	all_results_df.is_best_vec = ((1:n_runs) .== idx_best) .+ 0
	
    # select just the estimated parameters
    theta_stage2 = all_results_df[idx_best, r"param_"] |> collect
	obj_val_stage2 = all_results_df[idx_best, :obj_vals]

	debug && println("GMM => 3. SECOND STAGE optimal theta   ", theta_stage2)
	debug && println("GMM => 3. SECOND STAGE optimal obj val ", obj_val_stage2)

    # save
    gmm_main_object["results_stage2"] = all_results_df
    if write_results_to_file > 0
        outputfile = string(results_dir_path,"gmm_results_stage2_all.csv")
	    CSV.write(outputfile, all_results_df)
    end 

    ## print to csv file
	# stage1_df = DataFrame(
	# 	"run" => 1:n_runs,
	# 	"obj_vals" => obj_vals,
	# 	"opt_converged" => opt_converged .+ 0,
	# 	"opt_runtime" => opt_runtime,
	# 	"is_best_vec" => is_best_vec
	# )
	# for i=1:length(all_runs_results[1]["result"].param)
	# 	stage1_df[!, string("param_", i)] = [all_runs_results[idx]["result"].param[i] for idx=1:n_runs]
	# end

	# #
    # if write_results_to_file > 0
    #     outputfile = string(results_dir_path,"gmm_results_stage2_all.csv")
    #     CSV.write(outputfile, stage1_df)
    # end

	println("GMM => complete")

    return gmm_main_object
end



## Serial (not parallel) bootstrap with multiple initial conditions

function bootstrap_2step(;
					n_runs,
                    boot_run_idx,
					momfn,
					data,
					ESTIMATION_PARAMS,
					rootpath_boot_output,
					boot_rng=nothing,
					# bootstrap_samples=nothing,
					Wstep1_from_moms=true,
					# run_parallel=false,          # currently, always should be false
					write_results_to_file=false,
					my_show_trace=false,
					my_maxIter=100,
					my_time_limit=-1,
					debug=false,
					throw_exceptions=true
					)

try
## Announce
	println("\nBOOT => Starting Bootstrap ", rootpath_boot_output)

## load data and prepare
# TODO: do and document this better -- default = like this, but should also be able to use a function
# TODO: sample_data_fn(DATA::Any, boot_rng::RandomNumberSeed)

	data_dict_boot = copy(data_dict)
	N = size(data_dict["data"], 1)
	data_dict_boot["data"] = data_dict["data"][StatsBase.sample(boot_rng, 1:N, N), :]

## define the moment function with Boostrap Data
	momfn_loaded = theta -> momfn(theta, data_dict_boot)

## run 2-step GMM and save results
	boot_result = gmm_2step(
			momfn_loaded=momfn_loaded,
			theta_initials=ESTIMATION_PARAMS["theta_initials_boot"],
			theta_lower=ESTIMATION_PARAMS["theta_lower"],
			theta_upper=ESTIMATION_PARAMS["theta_upper"],
			Wstep1_from_moms=Wstep1_from_moms,
			n_runs=n_runs,
			run_parallel=false,
			results_dir_path="",
			write_results_to_file=write_results_to_file, ## TODO: do what here?
			my_show_trace=my_show_trace,
			my_maxIter=my_maxIter,
			my_time_limit=my_time_limit,
			debug=debug)

    print(boot_result)

    boot_result["results_stage1"][:, "boot_run_idx"] .= boot_run_idx
    boot_result["results_stage2"][:, "boot_run_idx"] .= boot_run_idx

    return boot_result
catch e
	println("BOOTSTRAP_EXCEPTION for ", rootpath_boot_output)

	bt = catch_backtrace()
	msg = sprint(showerror, e, bt)
	println(msg)

	if throw_exceptions
		throw(e)
	end
end

end


function boot_cleanup(;
			rootpath_input, boot_folder,
			idx_batch, idx_run, idx_boot_file,
			mymomfunction,
			my_show_trace=false,
			my_maxIter=100,
			my_time_limit=-1
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
	theta_initials_boot_df = CSV.read(mypath, DataFrame)
	select!(theta_initials_boot_df, r"param_")
	theta_initials_boot = Matrix(theta_initials_boot_df)

	# display(theta_initials_boot[1,:])

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
					theta_initials_boot[idx_boot_file,:],
					theta_lower,
					theta_upper,
					write_results_to_file=true,
					individual_run_results_path=string(boot_folder, "stage2/"),
					my_show_trace=my_show_trace,
					my_maxIter=my_maxIter,
					my_time_limit=my_time_limit)

end


# TODO: really need this?
# ! NO
"""
This little "currying" function allows us to make the moment function
(with data already "loaded") available to all workers.

Note that f must accept these three parameters in order:
1. parameters vector to estimate
2. data object
3. other parameters
"""
# gmm_curry(f, MYDATA, MYPARAMS) = theta -> f(theta, MYDATA, MYPARAMS)
# gmm_curry(f, MYDATA) = theta -> f(theta, MYDATA)


function run_gmm(;
		momfn,
		data,
		ESTIMATION_PARAMS,
		gmm_options=nothing,
	)

## Default options

    # TODO: drop the xxx?
	isnothing(gmm_options) && (gmm_options = Dict{String, Any}("xxx" => 1.0))

	gmm_options_default = Dict(
		"run_main" 			=> true,
		"main_n_start_pts" 	=> 1,
		"main_run_parallel" => false,
		"main_Wstep1_from_moms" => false,
		"main_write_results_to_file" => 0,
		"main_show_trace" 	=> true,
		"main_maxIter" 		=> 1000,
		"main_time_limit" 	=> -1,
		"main_debug" 		=> true,

        "param_names" => nothing,

		"run_boot" 			=> false,
        "boot_run_parallel" => false,
		"boot_n_runs" 		=> 200,
		"boot_n_start_pts" 	=> 1,
		"boot_write_results_to_file" => 0,
		"boot_show_trace" 	=> true,
		"boot_maxIter" 		=> 1000,
		"boot_time_limit"	=> -1,
		"boot_throw_exceptions" => false,
		"boot_debug" 		=> false,

		"rootpath_input" => "",
		"rootpath_output" => "",
		"rootpath_boot_output" => "",
	)

	for mykey in keys(gmm_options_default)
		if ~haskey(gmm_options, mykey)
			gmm_options[mykey] = gmm_options_default[mykey]
		end
	end

    # parameter names
    if isnothing(gmm_options["param_names"])
        n_params = size(ESTIMATION_PARAMS["theta_initials"])[2]
        gmm_options["param_names"] = [string("param_", i) for i=1:n_params]
    end

## Store results here
    gmm_results = Dict{String, Any}(
        "gmm_options" => gmm_options
    )

## Load data into function that is available to all workers
	momfn_loaded = theta -> momfn(theta, data)

## Run main GMM

    if gmm_options["run_main"]
        println("Starting GMM 2 step")

        # save params to file
        # params_df = DataFrame(ESTIMATION_PARAMS["param_factors"])
        # rename!(params_df, [:param_name, :factor, :fixed_value])
        # params_df.fixed_value = replace(params_df.fixed_value, nothing => missing)
        # CSV.write(string(rootpath_output, "gmm-param-names.csv"), params_df)

        gmm_results["gmm_main_results"] = gmm_2step(
                momfn_loaded    =momfn_loaded,
                theta_initials  =ESTIMATION_PARAMS["theta_initials"],
                theta_lower     =ESTIMATION_PARAMS["theta_lower"],
                theta_upper     =ESTIMATION_PARAMS["theta_upper"],
                Wstep1_from_moms=gmm_options["main_Wstep1_from_moms"],
                n_runs          =gmm_options["main_n_start_pts"],
                run_parallel    =gmm_options["main_run_parallel"],
                results_dir_path=gmm_options["rootpath_output"],
                write_results_to_file=gmm_options["main_write_results_to_file"],
                my_show_trace   =gmm_options["main_show_trace"],
                my_maxIter      =gmm_options["main_maxIter"],
                my_time_limit   =gmm_options["main_time_limit"],
                debug           =gmm_options["main_debug"])
    end


## Bootstrap
    if gmm_options["run_boot"]
        println("Starting Bootstrap")

        # Random number generators. One per bootstrap run. This is being extra careful.
        master_rng = MersenneTwister(123);
        boot_rngs = Vector{Any}(undef, gmm_options["boot_n_runs"])

        for i=1:gmm_options["boot_n_runs"]
            println("creating random number generator for boot run ", i)

            # each bootstrap run gets a different random seed
            boot_rngs[i] = Future.randjump(master_rng, big(10)^20 * i)
        end

        # Todo: what are folder options for boot? (similar to main?)
        # Folders
        boot_folders = Vector{String}(undef, gmm_options["boot_n_runs"])
        for i=1:gmm_options["boot_n_runs"]
            boot_folders[i] = string(gmm_options["rootpath_boot_output"], "boot_run_", i, "/")
        end

        # Run bootstrap
        boot_n_runs = gmm_options["boot_n_runs"]

        if gmm_options["boot_run_parallel"]
            boot_results = pmap(
            idx -> bootstrap_2step(
                        n_runs=gmm_options["boot_n_start_pts"],
                        boot_run_idx=idx,
                        momfn=momfn,
                        data=data,
                        ESTIMATION_PARAMS=ESTIMATION_PARAMS,
                        rootpath_boot_output="",
                        boot_rng=boot_rngs[idx],
                        Wstep1_from_moms=true,
                        write_results_to_file=gmm_options["boot_write_results_to_file"],
                        my_maxIter=gmm_options["boot_maxIter"],
                        my_time_limit=gmm_options["boot_time_limit"],
                        throw_exceptions=gmm_options["boot_throw_exceptions"],
                        my_show_trace=false,
                        debug=gmm_options["boot_debug"]
                    ), 1:boot_n_runs)
        else

            boot_results = Vector{Any}(undef, boot_n_runs)
            for boot_run_idx=1:boot_n_runs
                boot_results[boot_run_idx] = bootstrap_2step(
                        n_runs=gmm_options["boot_n_start_pts"],
                        boot_run_idx=boot_run_idx,
                        momfn=momfn,
                        data=data,
                        ESTIMATION_PARAMS=ESTIMATION_PARAMS,
                        rootpath_boot_output="",
                        boot_rng=boot_rngs[boot_run_idx],
                        Wstep1_from_moms=true,
                        write_results_to_file=gmm_options["boot_write_results_to_file"],
                        my_maxIter=gmm_options["boot_maxIter"],
                        my_time_limit=gmm_options["boot_time_limit"],
                        throw_exceptions=gmm_options["boot_throw_exceptions"],
                        my_show_trace=false,
                        debug=gmm_options["boot_debug"]
                    )
            end
        end

        gmm_results["gmm_boot_results"] = boot_results

    end # end

    return gmm_results
end
