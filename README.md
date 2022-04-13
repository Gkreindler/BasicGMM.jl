# BasicGMM.jl

Functions to run GMM/CMD. (Preliminary/In progress.)

# Install
Note: need to install a slightly custom version of `LsqFit.jl` from https://github.com/Gkreindler/LsqFit.jl
Run:
1. ]remove LsqFit
1. ]add https://github.com/Gkreindler/LsqFit.jl

# Usage
The user provides two objects:
1. A function `moments(theta,data)` that returns an NxM matrix, where `theta` is the parameter vector, `N` is the number of observations, and `M` is the number of moments.
1. An object `data`. (Can be anything. By default Dict{String, Any} with values tha are vectors or matrices with 1st dimension of size `N`. This ensures automatic "slow" bootstrap.)

# Examples
See examples in `example_serial.jl` and `example_distributed.jl`.

# Features
1. (DONE) run GMM (or classical minimum distance) one-step or two-step with optimal weight matrix
1. (DONE) parameter box constraints
1. (DONE) optimize using multiple initial conditions (serial or embarrassingly parallel with Distributed.jl)
asymptotic variance-covariance matrix
1. (DONE) “quick” bootstrap
1. (DONE) “slow” bootstrap (serial or embarrassingly parallel)
1. (DONE) output estimation results text
1. (Pending) output estimation results latex
1. (Pending) save results to files, re-start estimation based on incomplete results (e.g. when bootstrap run #63 fails after many hours of running!)
1. (Pending) compute sensitivity measure (Andrews et al 2017)
1. (Pending) (using user-provided function to generate data from model) Monte Carlo simulation to compute size and power.
1. (Pending) (using user-provided function to generate data from model) Monte Carlo simulation of estimation finite sample properties (simulate data for random parameter values ⇒ run GMM ⇒ compare estimated parameters with underlying true parameters)

# Notes
- the optimizer is a slightly modified version of LsqFit.jl, because the GMM/CMD objective is a sum of squares (using the Cholesky decomposition of the weighting matrix). In principle, other optimizers can be used.
- in optimization, the gradient is currently computed using finite differences (surely automatic differentiation can be added)

