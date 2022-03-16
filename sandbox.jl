
using Distributed

rmprocs(workers())
display(workers())
addprocs(6)
display(workers())

@everywhere begin
	using Future

	using Statistics
	using StatsBase
	using DataFrames
	using CSV
	using Random
end

@everywhere include("gmm_wrappers.jl")

# """
# route 1 takeup
# """
@everywhere function logit_takeup_route1(
        theta::Vector{Float64};
        data::Matrix{Float64},
        model_params::Dict{String, Float64},
        MAXDIFF=200.0)

    alpha = theta[1]
    sigma = theta[2]
    price = model_params["price"]

    temp = ( - alpha .* data[:, 1] .+ price .* data[:, 2]) ./ sigma
    temp = max.(min.(temp, MAXDIFF), -MAXDIFF)
    temp = exp.(temp)

    return temp ./ ( 1.0 .+ temp)
end

@everywhere function moments_model(
        theta::Vector{Float64};
        data::Matrix{Float64},
        model_params::Dict{String, Float64}
    )

    takeupr1 = logit_takeup_route1(theta, data=data, model_params=model_params)

    # moment 1 = takeup in control group
    # moment 2 = takeup in treatment group

    return hcat(takeupr1 .* (1.0 .- data[:, 2]),
                takeupr1 .* data[:, 2])
end

@everywhere function moments_gmm(
        theta::Vector{Float64};
        data_dict,
        model_params::Dict{String, Float64}
    )

    moms_model = moments_model(theta, data=data_dict["data"], model_params=model_params)

    return moms_model .- data_dict["moms_data"]
end

## Define Data
@everywhere begin
    N = 100
    # travel time difference route 1 - route 0 (minutes)
    travel_time_diff = rand(N) * 20

    # charges (only in treatment group) for route 0
    price = 10.0
    treated = (rand(N) .< 0.5).* 1.0

    data = hcat(travel_time_diff, treated)

    model_params = Dict(
        "price" => price
    )
end

# test
# logit_takeup_route1([1.0, 20.0], data=data, model_params=model_params)

# true parameters
true_theta = [1.5, 10.0]

# Define data moments for "true" parameters
moms_data = moments_model(true_theta, data=data, model_params=model_params)

mean(moms_data, dims=1) |> display

# run GMM
# run_gmm()

# @everywhere
@everywhere moments_gmm_loaded = (mytheta, mydata_dict) -> moments_gmm(
        mytheta, data_dict=mydata_dict, model_params=model_params)

data_dict = Dict(
	"data" => data,
	"moms_data" => moms_data
)
@everywhere data_dict = $data_dict

moments_gmm_loaded([1.0, 5.0], data_dict)

ESTIMATION_PARAMS = Dict(
	"theta_initials" => repeat([1.0 5.0], 1, 1),
	"theta_lower" => [0.0, 0.0],
	"theta_upper" => [Inf, Inf]
)

gmm_options = Dict{String, Any}(
	"main_debug" => false,
	"main_n_start_pts" => 1,
	"main_run_parallel" => true,
	"run_boot" => true,
	"boot_n_runs" => 20,
	"boot_throw_exceptions" => true,
)

run_gmm(momfn=moments_gmm_loaded,
		data=data_dict,
		ESTIMATION_PARAMS=ESTIMATION_PARAMS,
		gmm_options=gmm_options
	)
