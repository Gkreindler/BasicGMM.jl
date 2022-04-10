
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

include("gmm_wrappers.jl")
include("gmm_display.jl")

### Model definitions
include("model_logit.jl")

## true parameters
true_theta = [1.5, 10.0]

## Generate data for testing

    # true parameters
    true_theta = [1.5, 10.0]

    data_dict, model_params = generate_data_logit(N=200)


## Define moments function with certain parameters already "loaded"
    moments_gmm_loaded = (mytheta, mydata_dict) -> moments_gmm(
        theta=mytheta, 
        mydata_dict=mydata_dict, 
        model_params=model_params)


moments_gmm_loaded([1.0, 5.0], data_dict)

gmm_options = Dict{String, Any}(
	"main_run_parallel" => false,
	"run_boot" => true,
	"boot_n_runs" => 10,
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

# model_results
# get_model_results(gmm_results)

printstyled("ha!\n", color=:red)