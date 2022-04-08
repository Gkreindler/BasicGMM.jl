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

fdsf

## true parameters
true_theta = [1.5, 10.0]


## Define Data
begin
    N = 1000
    # travel time difference route 1 - route 0 (minutes)
    travel_time_diff = rand(N) * 20

    # charges (only in treatment group) for route 0
    price = 10.0
    treated = (rand(N) .< 0.5).* 1.0

    data = hcat(travel_time_diff, treated)

    model_params = Dict(
        "price" => price
    )

    takeup_data = data_synthetic(true_theta, data=data, model_params=model_params, asymptotic=false)
end


# test
#   a = logit_takeup_route1([1.0, 20.0], data=data, model_params=model_params)

data_dict = Dict(
	"data" => data,
	"takeup_data" => takeup_data
)

# Define data moments for "true" parameters
    moms_data = moments_gmm(
                    theta=true_theta, 
                    mydata_dict=data_dict,
                    model_params=model_params)


mean(moms_data, dims=1) |> display

# run GMM
# run_gmm()

moments_gmm_loaded = (mytheta, mydata_dict) -> moments_gmm(
        theta=mytheta, 
        mydata_dict=mydata_dict, 
        model_params=model_params)


moments_gmm_loaded([1.0, 5.0], data_dict)

gmm_options = Dict{String, Any}(
	"main_debug" => false,
    "main_show_trace" => false,
	"main_n_start_pts" => 100,
    "boot_n_start_pts" => 100,
	"main_run_parallel" => false,
	"run_boot" => true,
	"boot_n_runs" => 100,
	"boot_throw_exceptions" => true,
)

main_n_start_pts = gmm_options["main_n_start_pts"]
boot_n_start_pts = gmm_options["boot_n_start_pts"]

ESTIMATION_PARAMS = Dict(
	"theta_initials" => repeat([1.0 5.0], main_n_start_pts, 1),
    "theta_initials_boot" => repeat([1.0 5.0], boot_n_start_pts, 1),
	"theta_lower" => [0.0, 0.0],
	"theta_upper" => [Inf, Inf]
)


gmm_results = run_gmm(momfn=moments_gmm_loaded,
		data=data_dict,
		ESTIMATION_PARAMS=ESTIMATION_PARAMS,
		gmm_options=gmm_options
	)

# model_results
get_model_results(gmm_results)