using Distributed
n_procs = 2
if n_procs > 1
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

### Model definitions
@everywhere include("model_logit.jl")

## Generate data for testing

    # true parameters
    true_theta = [1.5, 10.0]

    # do this (once) on the local worker
    data_dict, model_params = generate_data_logit(N=200)

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
        # "main_n_initial_cond" => 100,
        # "boot_n_initial_cond" => 100,
        "main_run_parallel" => true,
        "run_boot" => true,
        "boot_n_runs" => 100,
        "boot_run_parallel" => true,
        "boot_throw_exceptions" => true
    )

    main_n_initial_cond = 100
    boot_n_initial_cond = 100

theta0 = repeat([1.0 5.0], main_n_initial_cond, 1)
theta0_boot = repeat([1.0 5.0], boot_n_initial_cond, 1)
theta_lower = [0.0, 0.0]
theta_upper = [Inf, Inf]

gmm_results = run_gmm(momfn=moments_gmm_loaded,
		data=data_dict,
		theta0=theta0,
        theta0_boot=theta0_boot,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
		gmm_options=gmm_options
	)

# print model_results
get_model_results(gmm_results)

