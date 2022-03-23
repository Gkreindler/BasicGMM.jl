using Future
using Statistics
using StatsBase
using DataFrames
using CSV
using Random

include("gmm_wrappers.jl")




# """
# model function -- returns probability to choose route 1
# """
function logit_takeup_route1(
    theta::Vector{Float64};
    data::Matrix{Float64},
    model_params::Dict{String, Float64},
    MAXDIFF=200.0)

    alpha = theta[1]
    sigma = theta[2]
    price = model_params["price"] # this will be zero for control group

    temp = ( - alpha .* data[:, 1] .+ price .* data[:, 2]) ./ sigma
    temp = max.(min.(temp, MAXDIFF), -MAXDIFF)
    temp = exp.(temp)

    return temp ./ ( 1.0 .+ temp)
end

function data_synthetic(theta::Vector{Float64};
    data::Matrix{Float64},
    model_params::Dict{String, Float64})

    data_probs = logit_takeup_route1(theta; data=data, model_params=model_params)

    data_choices = rand(size(data,1)) .<= data_probs
    data_choices = convert.(Float64, data_choices)

    return data_choices
end

# """
# Moments from model or data
# """
function moments(;
        takeup::Vector{Float64},
        data::Matrix{Float64})

    return hcat(takeup .* (1.0 .- data[:, 2]),
                takeup .* data[:, 2])    
end

# """
# moments from the model
# """
# function moments_model(
#     theta::Vector{Float64};
#     data::Matrix{Float64},
#     model_params::Dict{String, Float64})

#     takeupr1 = logit_takeup_route1(theta, data=data, model_params=model_params)

#     # moment 1 = takeup in control group
#     # moment 2 = takeup in treatment group

#     return hcat(takeupr1 .* (1.0 .- data[:, 2]),
#                 takeupr1 .* data[:, 2])
# end

# """
# moments (model minus data)
# """
function moments_gmm(;
    theta::Vector{Float64},
    data::Matrix{Float64},
    takeup_data::Vector{Float64},
    model_params::Dict{String, Float64})

    takeup_model = logit_takeup_route1(theta, data=data, model_params=model_params)

    moms_model = moments(takeup=takeup_model, data=data)
    moms_data  = moments(takeup=takeup_data, data=data)

    return moms_model .- moms_data
end


## true parameters
true_theta = [1.5, 10.0]


## Define Data
begin
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

    takeup_data = data_synthetic(true_theta, data=data, model_params=model_params)
end


# test
#   a = logit_takeup_route1([1.0, 20.0], data=data, model_params=model_params)


# Define data moments for "true" parameters
    moms_data = moments_gmm(
                    theta=true_theta, 
                    data=data,
                    takeup_data=takeup_data,
                    model_params=model_params)


mean(moms_data, dims=1) |> display

# run GMM
# run_gmm()

moments_gmm_loaded = (mytheta, mydata_dict) -> moments_gmm(
        theta=mytheta, 
        data=mydata_dict["data"], 
        takeup_data=mydata_dict["takeup_data"], 
        model_params=model_params)

data_dict = Dict(
	"data" => data,
	"takeup_data" => takeup_data
)
# data_dict = $data_dict

moments_gmm_loaded([1.0, 5.0], data_dict)

ESTIMATION_PARAMS = Dict(
	"theta_initials" => repeat([1.0 5.0], 1, 1),
	"theta_lower" => [0.0, 0.0],
	"theta_upper" => [Inf, Inf]
)

gmm_options = Dict{String, Any}(
	"main_debug" => true,
	"main_n_start_pts" => 1,
	"main_run_parallel" => false,
	"run_boot" => false,
	"boot_n_runs" => 20,
	"boot_throw_exceptions" => true,
)

run_gmm(momfn=moments_gmm_loaded,
		data=data_dict,
		ESTIMATION_PARAMS=ESTIMATION_PARAMS,
		gmm_options=gmm_options
	)
