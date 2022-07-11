
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

using FixedEffectModels
using GLM

# Random.seed!(1234)

include("gmm_wrappers.jl")
include("gmm_display.jl")

## Generate data for testing. 
    # The model is a logit choice model over two driving routes (short and long), where utility is a function of the time difference and any potential congestion charge on the "short" route
    # Utility is denominated in the currency (e.g. dollars)
    # Approx half of the agents are "treated" in an experiment where they face a fixed charge for using the short route.
    # The model parameters are alpha = value of travel time (in minutes) and sigma = logit variance parameter

    include("example_model_logit.jl")

    # true parameters (alpha, sigma)
    true_theta = [1.5, 10.0]

    rng = MersenneTwister(123);
    data_dict, model_params = generate_data_logit(N=50000, rng=rng)

## Define moment functions
    function moms_data_cmd(data_dict)
        df = DataFrame(
        "takeup_data" => data_dict["takeup_data"],
        "treat" => data_dict["data"][:, 2],
        "control" => 1.0 .- data_dict["data"][:, 2]
        )
        # reg(df, term(:takeup_data) ~ term(:treat) + term(:control))

        mymodel = lm(@formula(takeup_data ~ 0 +  treat + control), df)
        
        M = coef(mymodel) |> transpose |> Matrix
        V = vcov(mymodel)

        return M, V
    end

    function moms_model_cmd(;
        mytheta::Vector{Float64},
        mydata_dict::Dict{String, Array{Float64}},
        model_params::Dict{String, Float64})

        data = mydata_dict["data"]
        # takeup_data = mydata_dict["takeup_data"]

        takeup_model = logit_takeup_route1(mytheta, data=data, model_params=model_params)

        # moms_model = moments(takeup=takeup_model, data=data)
        treat_vec = data[:, 2] .== 1.0
        contr_vec = data[:, 2] .== 0.0

        @assert all(treat_vec + contr_vec .== 1) 

        M = zeros(1, 2)
        M[1,1] = mean(takeup_model[treat_vec])
        M[1,2] = mean(takeup_model[contr_vec])

        return M
    end

## Define moments function with certain parameters already "loaded"

    # get data moments
    M, V = moms_data_cmd(data_dict)

    # model moments minus data moments
    moments_gmm_loaded = (mytheta, mydata_dict) -> (moms_model_cmd(
        mytheta=mytheta, 
        mydata_dict=mydata_dict, 
        model_params=model_params) .- M)

    # Test
    theta0 = [1.5, 10.0]
    moments_gmm_loaded(theta0, data_dict)


## GMM options
    gmm_options = Dict{String, Any}(
        "main_run_parallel" => false,
        "estimator" => "cmd",
        "cmd_omega" => V,  # variance-coveriance matrix
        "main_write_results_to_file" => 2,
        "rootpath_output" => "G:/My Drive/optnets/analysis/temp/"
    )

## Initial conditions (matrix for multiple initial runs) and parameter box constraints
    main_n_initial_cond = 20
    boot_n_initial_cond = 20

    theta_lower = [0.0, 0.0]
    theta_upper = [Inf, Inf]

    theta0      = random_initial_conditions([1.0 5.0], theta_lower, theta_upper, main_n_initial_cond)
    theta0_boot = random_initial_conditions([1.0 5.0], theta_lower, theta_upper, main_n_initial_cond)

## Run GMM
    main_results, main_df, boot_results, boot_df = run_gmm(momfn=moments_gmm_loaded,
		data=data_dict,
		theta0=theta0,
        theta0_boot=theta0_boot,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        omega=V,
		gmm_options=gmm_options
	)

## print model_results
    print_results(main_results=main_results, boot_results=boot_results, boot_df=boot_df)