using Distributed
n_procs = 4
if (n_procs > 1) && (n_procs > length(workers()))
	rmprocs(workers())
	display(workers())
	addprocs(n_procs)
	display(workers())
end

@everywhere begin
    using Future

    using Statistics
    using StatsBase
    using Distributions

    using DataFrames
    using CSV
    using JSON
    using Random

    using PrettyPrint
    using Printf

    using FiniteDifferences

    using FixedEffectModels
    using GLM
end 


@everywhere include("gmm_wrappers.jl")
@everywhere include("gmm_display.jl")

## Generate data for testing. 
    # The model is a logit choice model over two driving routes (short and long), where utility is a function of the time difference and any potential congestion charge on the "short" route
    # Utility is denominated in the currency (e.g. dollars)
    # Approx half of the agents are "treated" in an experiment where they face a fixed charge for using the short route.
    # The model parameters are alpha = value of travel time (in minutes) and sigma = logit variance parameter

    @everywhere include("example_model_logit.jl")

    # true parameters (alpha, sigma)
    true_theta = [1.5, 10.0]

    # do this (once) on the local worker
    rng = MersenneTwister(123);
    data_dict, model_params = generate_data_logit(N=100, rng=rng)

    # copy to all workers
    @everywhere data_dict = $data_dict
    @everywhere model_params = $model_params

## Define moments function with certain parameters already "loaded"
    
    # get data moments
    @everywhere M, V = moms_data_cmd(data_dict)

    # model moments minus data moments
    @everywhere moments_gmm_loaded = (mytheta, mydata_dict) -> (moms_model_cmd(
        mytheta=mytheta, 
        mydata_dict=mydata_dict, 
        model_params=model_params) .- M)

    # test
    @everywhere moments_gmm_loaded([1.0, 5.0], data_dict)


## GMM options
    gmm_options = Dict{String, Any}(
        "main_run_parallel" => true,
        "estimator" => "cmd",
        "var_boot" => "slow",
        "boot_n_runs" => 10,
        "boot_run_parallel" => true,
        "boot_throw_exceptions" => true,
        "main_write_results_to_file" => 2,
        "boot_write_results_to_file" => 2,
        "rootpath_output" => "G:/My Drive/optnets/analysis/temp/"
    )

## Initial conditions (matrix for multiple initial runs) and parameter box constraints
    main_n_initial_cond = 100
    boot_n_initial_cond = 100

    theta_lower = [0.0, 0.0]
    theta_upper = [Inf, Inf]

    theta0      = random_initial_conditions([1.0 5.0], theta_lower, theta_upper, main_n_initial_cond)
    theta0_boot = random_initial_conditions([1.0 5.0], theta_lower, theta_upper, boot_n_initial_cond)

## Run estimation
est_results, est_results_df = run_estimation(
        momfn=moments_gmm_loaded,
		data=data_dict,
		theta0=theta0,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        omega=V,
		gmm_options=gmm_options)

### Run inference
boot_results, boot_df = run_inference(
        momfn=moments_gmm_loaded,
        data=data_dict,
        theta0_boot=theta0_boot,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        omega=V,
        sample_data_fn=nothing,
        gmm_options=gmm_options,
        est_results=est_results)

## print model_results
print_results(est_results=est_results, boot_results=boot_results, boot_df=boot_df)

