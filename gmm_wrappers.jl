

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
				write_result_to_file=false,
				results_dir_path="",
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
				time_limit=my_time_limit # this is added relative to official LsqFit.jl
			)

	if debug
		println(">>> iteration ", idx, " took: ", timeittook, " seconds. Converged: ", result.converged)
		println(">>> optimum ", result.param)
		println(">>> obj val ", norm(result.resid))
	end

	# save each round of results to csv file
	# this option is useful if some initial conditions are taking too long
	# the corresponding files will not be created -> you can debug for the
	# corresponding initial conditions
	if write_result_to_file

		results_df = DataFrame(
			"obj_vals" => norm(result.resid),
			"opt_results" => result.converged .+ 0,
			"opt_runtime" => timeittook
		)
		for i=1:length(result.param)
			results_df[:, string("param_", i)] = [result.param[i]]
		end

	    outputfile = string(results_dir_path,"gmm-stage1-run",idx,".csv")
		CSV.write(outputfile, results_df)
	end

	return Dict(
		"result" => result,
		"timeittook" => timeittook
	)
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
	write_result_to_file = write individual run results to files (default=no)
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

			n_moms=nothing,
			Wstep1=nothing,
			Wstep1_from_moms=false,

			results_dir_path="",
			write_result_to_file=false,
			my_show_trace=false,
			my_maxIter=1000,
			my_time_limit=-1,
			debug=false)

## if theta_initials is a vector, turn to 1 x K matrix
	if isa(theta_initials, Vector)
		theta_initials = Matrix(transpose(theta_initials))
	end

## if not provided, compute size of moments by running moment function once
	if isnothing(n_moms)
		n_moms = size(momfn_loaded(theta_initials[1,:]), 2)
	end

## save starting conditions
## print to csv file
	start_conditions_df = DataFrame(
		"iteration" => 1:n_runs
	)
	for i=1:size(theta_initials)[2]
		start_conditions_df[!, string("param_", i)] = vec(theta_initials[:, i])
	end

	# TODO: when do we save to file?
	outputfile = string(results_dir_path,"gmm-initial-conditions.csv")
	CSV.write(outputfile, start_conditions_df)

## Compute initial weighting matrix based on moments covariances at initial "best guess" theta
	if ~isnothing(Wstep1) && Wstep1_from_moms
		error("cannot have both Wstep1 matrix AND Wstep1_from_moms=true")
	end

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

	# if not provided, identity weighting matrix
	elseif isnothing(Wstep1)
		Wstep1 = diagm(ones(n_moms))
	end

	# TODO: when do we save to file?
	## Save matrix to file
	outputfile = string(results_dir_path,"gmm-stage0-optW.csv")
	CSV.write(outputfile, Tables.table(Wstep1), header=false)

	# cholesky half. satisfies
	# Whalf * transpose(Whalf) = W
	initialWhalf = Matrix(cholesky(Hermitian(Wstep1)).L)
	@assert norm(initialWhalf * transpose(initialWhalf) - Wstep1) < 1e-10

## FIRST stage
    println("GMM => Launching FIRST stage, n parallel runs: ", n_runs)

	# curve_fit in LsqFit requires to give the objective function data (x)
	# we hack this to give the GMM weighting matrix (Cholesky half)

	# subdirectory for individual run results
	# TODO: when do we save to file?
	if write_result_to_file
		debug && print("GMM => 1. creating subdirectory to save individual run results...")
		results_subdir_path = string(results_dir_path, "stage1")
		isdir(results_subdir_path) || mkdir(results_subdir_path)
	end

	# objective function with moment function "loaded"
	gmm_obj_fn = (anyWhalf, any_theta) ->
					gmm_obj(theta=any_theta,
							Whalf=anyWhalf,
							momfn=momfn_loaded,
							debug=debug)


	if run_parallel && n_runs > 1

	    all_runs_results = pmap(
	        idx -> curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							initialWhalf,
							n_moms,
	                        theta_initials[idx,:], theta_lower, theta_upper,
							write_result_to_file=write_result_to_file,
							results_dir_path=string(results_dir_path, "stage1/"),
	                        my_show_trace=my_show_trace,
							my_maxIter=my_maxIter,
							my_time_limit=my_time_limit,
							debug=debug
						), 1:n_runs)
	else

		all_runs_results = Vector{Any}(undef, n_runs)
		for idx=1:n_runs
	        all_runs_results[idx] = curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							initialWhalf,
							n_moms,
	                        theta_initials[idx,:], theta_lower, theta_upper,
							write_result_to_file=write_result_to_file,
							results_dir_path=string(results_dir_path, "stage1/"),
	                        my_show_trace=my_show_trace,
							my_maxIter=my_maxIter,
							my_time_limit=my_time_limit,
							debug=debug)
		end
	end

    # collect results
    obj_vals = [norm(all_runs_results[idx]["result"].resid) for idx=1:n_runs]
    opt_thetas = [all_runs_results[idx]["result"].param for idx=1:n_runs]
	opt_results = [all_runs_results[idx]["result"].converged for idx=1:n_runs]
    opt_runtime = [all_runs_results[idx]["timeittook"] for idx=1:n_runs]


	# pick best
	idx_best = argmin(obj_vals)
	is_best_vec = ((1:n_runs) .== idx_best) .+ 0
	theta_stage1 = opt_thetas[idx_best]
	obj_val_stage1 = obj_vals[idx_best]

	debug && println("GMM => 1. FIRST STAGE optimal theta   ", theta_stage1)
	debug && println("GMM => 1. FIRST STAGE optimal obj val ", obj_val_stage1)

    ## print to csv file
	stage1_df = DataFrame(
		"run" => 1:n_runs,
		"obj_vals" => obj_vals,
		"opt_converged" => opt_results .+ 0,
		"opt_runtime" => opt_runtime,
		"is_best_vec" => is_best_vec
	)
	for i=1:length(all_runs_results[1]["result"].param)
		stage1_df[!, string("param_", i)] = [all_runs_results[idx]["result"].param[i] for idx=1:n_runs]
	end

	# TODO: optional
    outputfile = string(results_dir_path,"gmm-stage1-all.csv")
	CSV.write(outputfile, stage1_df)

	# TODO: if one-step -> stop here

## Optimal Weighting Matrix
	println("GMM => Computing optimal weighting matrix")

	mom_matrix = momfn_loaded(theta_stage1)

 	# compute optimal W matrix -> ensure it's Hermitian
	nmomsize = size(mom_matrix, 1)
	Wstep2 = Hermitian(transpose(mom_matrix) * mom_matrix / nmomsize)

	# if super small determinant:
	if det(Wstep2) < 1e-100
		debug && println(" Matrix determinant very low: 1e-100. Adding 0.001 * I.")
		Wstep2 = Wstep2 + 0.001 * I
	end

	# invert (still Hermitian)
	Wstep2 = inv(Wstep2)

	# normalize?
	if debug
		println("original Determinant and matrix")
		display(det(Wstep2))
		display(Wstep2)
	end

	Wstep2 = Wstep2 * det(Wstep2)^(-1/size(Wstep2,1))

	if debug
		println("Determinant and matrix:")
		display(det(Wstep2))  # close to 1
		display(Wstep2)
	end

	## Save matrix to file
	outputfile = string(results_dir_path,"gmm-stage1-optW.csv")
	CSV.write(outputfile, Tables.table(Wstep2), header=false)

	# cholesky half. satisfies
	# Whalf * transpose(Whalf) = W
	optimalWhalf = Matrix(cholesky(Wstep2).L)
	# debug && println("checking if Cholesky decomposition worked: (1) difference matrix, (2) norm")
	# debug && display(optimalWhalf * transpose(optimalWhalf) - Wstep2)
	# debug && display(norm(optimalWhalf * transpose(optimalWhalf) - Wstep2))
	@assert norm(optimalWhalf * transpose(optimalWhalf) - Wstep2) < 1e-10


## SECOND stage
    println("GMM => Launching SECOND stage, n parallel runs: ", n_runs)

	debug && display(optimalWhalf)

	# subdirectory for individual run results
	if write_result_to_file
		print("GMM => 3. creating subdirectory to save individual run results...")
		results_subdir_path = string(results_dir_path, "stage2")
		isdir(results_subdir_path) || mkdir(results_subdir_path)
	end

	# run GMM
	if run_parallel && n_runs > 1

	    all_runs_results = pmap(
	        idx -> curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							optimalWhalf,
							n_moms,
	                        theta_initials[idx,:], theta_lower, theta_upper,
							write_result_to_file=write_result_to_file,
							results_dir_path=string(results_dir_path, "stage2/"),
	                        my_show_trace=my_show_trace,
							my_maxIter=my_maxIter,
							my_time_limit=my_time_limit,
							debug=debug), 1:n_runs)
	else

		all_runs_results = Vector{Any}(undef, n_runs)
		for idx=1:n_runs
			all_runs_results[idx]=curve_fit_wrapper(
							idx,
							gmm_obj_fn,
							optimalWhalf,
							n_moms,
	                        theta_initials[idx,:], theta_lower, theta_upper,
							write_result_to_file=write_result_to_file,
							results_dir_path=string(results_dir_path, "stage2/"),
	                        my_show_trace=my_show_trace,
							my_maxIter=my_maxIter,
							my_time_limit=my_time_limit,
							debug=debug)
		end
	end

    # collect results
    obj_vals = [norm(all_runs_results[idx]["result"].resid) for idx=1:n_runs]
    opt_thetas = [all_runs_results[idx]["result"].param for idx=1:n_runs]
	opt_results = [all_runs_results[idx]["result"].converged for idx=1:n_runs]
    opt_runtime = [all_runs_results[idx]["timeittook"] for idx=1:n_runs]

	# pick best
	idx_best = argmin(obj_vals)
	is_best_vec = ((1:n_runs) .== idx_best) .+ 0
	theta_stage2 = opt_thetas[idx_best]

	obj_val_stage2 = obj_vals[idx_best]

	debug && println("GMM => 3. SECOND STAGE optimal theta   ", theta_stage2)
	debug && println("GMM => 3. SECOND STAGE optimal obj val ", obj_val_stage2)


    ## print to csv file
	stage1_df = DataFrame(
		"run" => 1:n_runs,
		"obj_vals" => obj_vals,
		"opt_converged" => opt_results .+ 0,
		"opt_runtime" => opt_runtime,
		"is_best_vec" => is_best_vec
	)
	for i=1:length(all_runs_results[1]["result"].param)
		stage1_df[!, string("param_", i)] = [all_runs_results[idx]["result"].param[i] for idx=1:n_runs]
	end

	# TODO: optional
    outputfile = string(results_dir_path,"gmm-stage2-all.csv")
	CSV.write(outputfile, stage1_df)

	println("GMM => complete")

	# todo: return object
end



## Serial (not parallel) bootstrap with multiple initial conditions

function bootstrap_2step(;
					n_runs,
					momfn,
					data,
					ESTIMATION_PARAMS,
					rootpath_boot_output,
					boot_rng=nothing,
					# bootstrap_samples=nothing,
					Wstep1_from_moms=true,
					# run_parallel=false,          # currently, always should be false
					write_result_to_file=false,
					my_show_trace=false,
					my_maxIter=100,
					my_time_limit=-1,
					debug=false,
					throw_exceptions=true
					)

try
## Announce
	println("\nBOOT => Starting Bootstrap ", rootpath_boot_output)

## folder
	# isdir(rootpath_boot_output) || mkdir(rootpath_boot_output)

## load data and prepare

	# TODO: this better
	# if isnothing(boot_rng)
	# 	if isnothing(bootstrap_samples)
	# 		error("must provide boot_rng or boot_sample")
	# 	end
	# else
	# 	if ~isnothing(bootstrap_samples)
	# 		error("can only provide boot_rng or boot_sample, not both")
	# 	end
	# 	bootstrap_samples = Dict(
	# 		"a" => StatsBase.sample(boot_rng, 1:205, 205),
	# 		"na" => StatsBase.sample(boot_rng, 1:99, 99)
	# 	)
	# end

	# bootstrap_samples = nothing

	## Save boot samples to file
	# outputfile = string(rootpath_boot_output, "boot_sample_a.csv")
	# CSV.write(outputfile, Tables.table(bootstrap_samples["a"]), header=false)
	# outputfile = string(rootpath_boot_output, "boot_sample_na.csv")
	# CSV.write(outputfile, Tables.table(bootstrap_samples["na"]), header=false)
	#
	# # save params to file
	# params_df = DataFrame(ESTIMATION_PARAMS["param_factors"])
	# rename!(params_df, [:param_name, :factor, :fixed_value])
	# params_df.fixed_value = replace(params_df.fixed_value, nothing => missing)
	# CSV.write(string(rootpath_boot_output, "gmm-param-names.csv"), params_df)


## Sample existing loaded matrices:
	# BIGMAT_BOOT, CHARGEMAT_BOOT, COMPMAT_BOOT, DATAMOMS_BOOT =
	# 	sample_bigmats(
	# 	DATA["BIGMAT"], DATA["CHARGEMAT"], DATA["COMPMAT"], DATA["DATAMOMS"],
	# 	bootstrap_samples["a"], bootstrap_samples["na"])
	#
	# data_boot=Dict(
	# 	"BIGMAT" => BIGMAT_BOOT,
	# 	"CHARGEMAT" => CHARGEMAT_BOOT,
	# 	"COMPMAT" => COMPMAT_BOOT,
	# 	"DATAMOMS" => DATAMOMS_BOOT
	# )

	data_dict_boot = copy(data_dict)
	N = size(data_dict["data"], 1)
	data_dict_boot["data"] = data_dict["data"][StatsBase.sample(boot_rng, 1:N, N), :]

## define the moment function with Boostrap Data
	# mymomfunction_loaded = curry(momfn, DATA_BOOT, ESTIMATION_PARAMS)
	momfn_loaded = theta -> momfn(theta, data_dict_boot)

## run 2-step GMM and save results
	gmm_2step(
			momfn_loaded=momfn_loaded,
			theta_initials=ESTIMATION_PARAMS["theta_initials"],
			theta_lower=ESTIMATION_PARAMS["theta_lower"],
			theta_upper=ESTIMATION_PARAMS["theta_upper"],
			Wstep1_from_moms=Wstep1_from_moms,
			n_runs=n_runs,
			run_parallel=false,
			results_dir_path="", 		 # TODO: better
			write_result_to_file=false,	 # TODO: better
			my_show_trace=my_show_trace,
			my_maxIter=my_maxIter,
			my_time_limit=my_time_limit,
			debug=debug)

	# gmm_parallel_2step(
	# 		momfn=mymomfunction_loaded,
	# 		theta_initials=ESTIMATION_PARAMS["theta_initials_boot"],
	# 		theta_lower=ESTIMATION_PARAMS["theta_lower"],
	# 		theta_upper=ESTIMATION_PARAMS["theta_upper"],
	# 		Wstep1_from_moms=Wstep1_from_moms,
	# 		n_runs=n_runs,
	# 		results_dir_path=rootpath_boot_output,
	# 		write_result_to_file=write_result_to_file,  # Don't write each step
	# 		run_parallel=run_parallel,          # Will run the entire boot function in parallel so this is serial
	# 		my_show_trace=my_show_trace,
	# 		my_maxIter=my_maxIter, my_time_limit=my_time_limit,
	# 		debug=false)

catch e
	println("BOOTSTRAP_EXCEPTION for ", rootpath_boot_output)
	# println("BOOTSTRAP_EXCEPTION for ", rootpath_boot_output)
	# println("BOOTSTRAP_EXCEPTION for ", rootpath_boot_output)

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
					write_result_to_file=true,
					results_dir_path=string(boot_folder, "stage2/"),
					my_show_trace=my_show_trace,
					my_maxIter=my_maxIter,
					my_time_limit=my_time_limit)

end


# TODO: really need this?
"""
This little "currying" function allows us to make the moment function
(with data already "loaded") available to all workers.

Note that f must accept these three parameters in order:
1. parameters vector to estimate
2. data object
3. other parameters
"""
# gmm_curry(f, MYDATA, MYPARAMS) = theta -> f(theta, MYDATA, MYPARAMS)
gmm_curry(f, MYDATA) = theta -> f(theta, MYDATA)


function run_gmm(;
		momfn,
		data,
		ESTIMATION_PARAMS,
		gmm_options=nothing,
	)

## Default options
	isnothing(gmm_options) && (gmm_options = Dict{String, Any}("x" => 1.0))

	gmm_options_default = Dict(
		"run_main" 			=> true,
		"main_n_start_pts" 	=> 1,
		"main_run_parallel" => false,
		"main_Wstep1_from_moms" => false,
		"main_write_result_to_file" => false,   ### DROP THIS?
		"main_show_trace" 	=> true,
		"main_maxIter" 		=> 1000,
		"main_time_limit" 	=> -1,
		"main_debug" 		=> true,

		"run_boot" 			=> false,
		"boot_n_runs" 		=> 200,
		"boot_n_start_pts" 	=> 1,
		"boot_write_result_to_file" => false,
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

## Load data into function that is available to all workers
	# momfn_loaded = gmm_curry(momfn, data)
	momfn_loaded = theta -> momfn(theta, data)

## Run main GMM

if gmm_options["run_main"]
	println("Starting GMM 2 step")

	# save params to file
	# params_df = DataFrame(ESTIMATION_PARAMS["param_factors"])
	# rename!(params_df, [:param_name, :factor, :fixed_value])
	# params_df.fixed_value = replace(params_df.fixed_value, nothing => missing)
	# CSV.write(string(rootpath_output, "gmm-param-names.csv"), params_df)

	gmm_2step(
			momfn_loaded=momfn_loaded,
			theta_initials=ESTIMATION_PARAMS["theta_initials"],
			theta_lower=ESTIMATION_PARAMS["theta_lower"],
			theta_upper=ESTIMATION_PARAMS["theta_upper"],
			Wstep1_from_moms=gmm_options["main_Wstep1_from_moms"],
			n_runs=gmm_options["main_n_start_pts"],
			run_parallel=gmm_options["main_run_parallel"],
			results_dir_path=gmm_options["rootpath_output"],
			write_result_to_file=gmm_options["main_write_result_to_file"],
			my_show_trace=gmm_options["main_show_trace"],
			my_maxIter=gmm_options["main_maxIter"],
			my_time_limit=gmm_options["main_time_limit"],
			debug=gmm_options["main_debug"])
end




## Bootstrap
if gmm_options["run_boot"]
	println("Starting Bootstrap")

	# Random number generators (being extra careful) one per bootstrap run
	master_rng = MersenneTwister(123);
	boot_rngs = Vector{Any}(undef, gmm_options["boot_n_runs"])

	for i=1:gmm_options["boot_n_runs"]
		println("creating random number generator for boot run ", i)

		# each bootstrap run gets a different random seed
		# as we run the bootrap in separate rounds, large initial skip
		# boostrap_skip = (boot_round-1)*boot_n_runs + i
		boostrap_skip = i

		boot_rngs[i] = Future.randjump(master_rng, big(10)^20 * boostrap_skip)
	end

	# Folders
	boot_folders = Vector{String}(undef, gmm_options["boot_n_runs"])
	for i=1:gmm_options["boot_n_runs"]
		boot_folders[i] = string(gmm_options["rootpath_boot_output"], "boot_run_", i, "/")
	end

	# Run bootstrap
	pmap(
	idx -> bootstrap_2step(
				n_runs=gmm_options["boot_n_start_pts"],
				momfn=momfn,
				data=data,
				ESTIMATION_PARAMS=ESTIMATION_PARAMS,
				rootpath_boot_output="",
				boot_rng=boot_rngs[idx],
				Wstep1_from_moms=true,
				write_result_to_file=gmm_options["boot_write_result_to_file"],
				my_maxIter=gmm_options["boot_maxIter"],
				my_time_limit=gmm_options["boot_time_limit"],
				throw_exceptions=gmm_options["boot_throw_exceptions"],
				my_show_trace=false,
				debug=gmm_options["boot_debug"]
				# rootpath_boot_output=boot_folders[idx],
			), 1:gmm_options["boot_n_runs"])
end


end
