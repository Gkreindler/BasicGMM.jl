


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