


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
    price = model_params["price"] # this will be switched on only for treatment group

    travel_time_diff = data[:, 1]
    treated = data[:, 2]

    temp = ( - alpha .* travel_time_diff .+ price .* treated) ./ sigma
    temp = max.(min.(temp, MAXDIFF), -MAXDIFF)
    temp = exp.(temp)

    return temp ./ ( 1.0 .+ temp)
end

function data_synthetic(theta::Vector{Float64};
    data::Matrix{Float64},
    model_params::Dict{String, Float64}, asymptotic=false)

    data_probs = logit_takeup_route1(theta; data=data, model_params=model_params)

    if asymptotic
        return data_probs
    end

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

    treated = data[:, 2]

    return hcat(takeup .* (1.0 .- treated),
                takeup .* treated)
end

# """
# moments (model minus data)
# """
function moments_gmm(;
    theta::Vector{Float64},
    mydata_dict::Dict{String, Array{Float64}},
    model_params::Dict{String, Float64})

    data = mydata_dict["data"]
    takeup_data = mydata_dict["takeup_data"]

    takeup_model = logit_takeup_route1(theta, data=data, model_params=model_params)

    moms_model = moments(takeup=takeup_model, data=data)
    moms_data  = moments(takeup=takeup_data, data=data)

    return moms_model .- moms_data
end