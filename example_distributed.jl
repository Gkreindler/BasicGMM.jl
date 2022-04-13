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
    using Random

    using PrettyPrint
    using Printf

    using FiniteDifferences
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
    data_dict, model_params = generate_data_logit(N=200, rng=rng)

    # copy to all workers
    @everywhere data_dict = $data_dict
    @everywhere model_params = $model_params

## Define moments function with certain parameters already "loaded"
    
    # on each worker
    @everywhere moments_gmm_loaded = (mytheta, mydata_dict) -> moments_gmm(
            theta=mytheta, 
            mydata_dict=mydata_dict, 
            model_params=model_params)

    # test
    @everywhere moments_gmm_loaded([1.0, 5.0], data_dict)


## GMM options
    gmm_options = Dict{String, Any}(
        "main_run_parallel" => true,
        "var_boot" => "slow",
        "boot_n_runs" => 100,
        "boot_run_parallel" => true,
        "boot_throw_exceptions" => true
    )

## Initial conditions (matrix for multiple initial runs) and parameter box constraints
    main_n_initial_cond = 100
    boot_n_initial_cond = 100

    theta0 = repeat([1.0 5.0], main_n_initial_cond, 1)
    theta0_boot = repeat([1.0 5.0], boot_n_initial_cond, 1)
    theta_lower = [0.0, 0.0]
    theta_upper = [Inf, Inf]

## Run GMM
    gmm_results = run_gmm(momfn=moments_gmm_loaded,
		data=data_dict,
		theta0=theta0,
        theta0_boot=theta0_boot,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
		gmm_options=gmm_options
	)

## print model_results
    print_results(gmm_results)

