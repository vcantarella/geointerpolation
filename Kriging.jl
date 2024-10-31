include("Variogram.jl")
include("VariogramModels.jl")
using LinearAlgebra
using Statistics
using Optim



# Function to perform cross-validation to find the optimal regularization parameter
function cross_validate_regularization(grid::Matrix{T}, points::Matrix{T}, values::Vector{T}, model::Function, p::Vector{T}, μ::T, σ²::T, search_radius_h::T, search_radius_v::T, min_points::Int, max_points::Int, λ::T) where T
    errors = []
    for reg in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        μₖ, _ = ordinary_kriging_omnihorizontal(grid, points, values, model, p, μ, σ², search_radius_h, search_radius_v, min_points, max_points, λ, reg)
        error = mean((values .- μₖ).^2, dims=1)
        push!(errors, (reg, error))
    end
    return minimum(errors, by=x->x[2])[1]
end

# TODO: verify the correctness of the code
function simple_kriging(grid::Matrix{T}, points::Matrix{T},
                values::Vector{T}, model::Function,
                p::Vector{T}, μ::T,
                σ²::T,search_radius::T) where T
    dist_matrix = distance_matrix(points)
    covariance_matrix = model(dist_matrix, p)
    # Assuming p[2] is the sill
    sill = p[2]
    covariance_matrix = sill.-covariance_matrix
    index = []
    k = zeros(T, size(grid, 1))
    n = size(grid, 1)
    m = size(points, 1)
    μₖ = zeros(T, n)
    σ²ₖ = zeros(T, n)
    for i in 1:n
        index = Int[]
        cov_v = Vector{T}()
        for j in 1:m
            dist = distance(grid[i, :], points[j, :])
            if dist < search_radius
                push!(index, j)
                push!(cov_v, sill - model(dist, p))
            end
        end
        if length(index) == 0
            μₖ[i] = μ
            σ²ₖ[i] = σ²
            continue
        end
        weights = Symmetric(covariance_matrix[index, index]) \ cov_v
        # estimating the mean and variance of the estimator
        μₖ[i] = μ + sum(weights.*(values[index] .- μ))
        σ²ₖ[i] = σ² - sum(weights.*cov_v)
    end
    return μₖ, σ²ₖ
end

function simple_kriging_omnihorizontal(grid::Matrix{T}, points::Matrix{T},
                values::Vector{T}, model::Function,
                p::Vector{T}, μ::T,
                σ²::T,search_radius::T,
                min_points=10, max_points = 20) where T
    dist_matrix_h = distance_matrix(points[:, 1:2])
    dist_matrix_v = distance_matrix(hcat(points[:, 3]))
    #unravel parameters for the function
    covariance_matrix = model(dist_matrix_h,dist_matrix_v, p...)
    # Assuming p[3] is the sill
    sill = p[3]
    covariance_matrix = sill.-covariance_matrix
    # make sure the covariance is the variance error at the diagonal
    # for i in axes(covariance_matrix, 1)
    #     covariance_matrix[i, i] = σ²
    # end
    index = []
    k = zeros(T, size(grid, 1))
    n = size(grid, 1)
    m = size(points, 1)
    μₖ = zeros(T, n)
    σ²ₖ = zeros(T, n)
    for i in 1:n
        index = Int[]
        cov_v = Vector{T}()
        for j in 1:m
            dist_h = distance(grid[i, 1:2], points[j, 1:2])
            dist_v = abs(grid[i, 3] - points[j, 3])
            if sqrt(dist_h^2 + dist_v^2) < search_radius
                push!(index, j)
                push!(cov_v, sill - model(dist_h,dist_v, p...))
            end
        end
        if length(index) < min_points
            μₖ[i] = μ
            σ²ₖ[i] = σ²
            continue
        end
        # select the 20 most relevant points
        arg_cov = sortperm(cov_v, rev = true)
        cov_v = cov_v[arg_cov]
        index = index[arg_cov]
        if length(index) > max_points
            index = index[1:max_points]
            cov_v = cov_v[1:max_points]
        end
        weights =  Symmetric(covariance_matrix[index, index]) \ cov_v
        # estimating the mean and variance of the estimator
        μₖ[i] = (1-sum(weights))*μ .+ sum(weights.*(values[index]))
        σ²ₖ[i] = σ² - sum(weights.*cov_v)
    end
    return μₖ, σ²ₖ
end


function ordinary_kriging(grid::Matrix{T}, points::Matrix{T},
                values::Vector{T}, model::Function,
                p::Vector{T}, σ²::T,
                search_radius::T) where T
    dist_matrix = distance_matrix(points)
    cov_matrix = model.(dist_matrix, p)
    # Assuming p[2] is the sill
    sill = p[2]
    cov_matrix = sill.-cov_matrix
    index = []
    k = zeros(T, size(grid, 1))
    n = size(grid, 1)
    m = size(points, 1)
    μₖ = zeros(T, n)
    σ²ₖ = zeros(T, n)
    for i in 1:n
        index = []
        cov_v = []
        for j in 1:m
            dist = distance(grid[i, :], points[j, :])
            if dist < search_radius
                push!(index, j)
                push!(cov_v, sill - model(dist, p))
            end
        end
        # Append ones to the last row and last column of cov_matrix
        
        cov_matrix_i = [cov_matrix[index, index] ones(length(index)); ones(length(index))' 0]
        cov_v = [cov_v; 1]
        weights = Symmetric(cov_matrix_i) \ cov_v
        lagrange_mult = weights[end]
        μₖ[i] = sum(weights[1:end-1].*values[index])
        σ²ₖ[i] = σ² - lagrange_mult- sum(weights[1:end-1].*cov_v[1:end-1])
    end
    return μₖ, σ²ₖ
end

function ordinary_kriging_omnihorizontal(grid::Matrix{T}, points::Matrix{T},
    values::Vector{T}, model::Function,
    p::Vector{T}, μ::T,
    σ²::T,search_radius_h::T,search_radius_v::T,
    min_points = 10, max_points = 20, regularization::T = 1e-10) where T


    dist_matrix_h = distance_matrix(points[:, 1:2])
    dist_matrix_v = distance_matrix(hcat(points[:, 3]))
    #unravel parameters for the function
    covariance_matrix = model(dist_matrix_h,dist_matrix_v, p...)
    # Assuming p[3] is the sill
    sill = p[3]
    covariance_matrix = sill.-covariance_matrix
    index = []
    k = zeros(T, size(grid, 1))
    n = size(grid, 1)
    m = size(points, 1)
    μₖ = zeros(T, n)
    σ²ₖ = zeros(T, n)
    for i in 1:n
        index = Int[]
        cov_v = Vector{T}()
        for j in 1:m
            dist_h = distance(grid[i, 1:2], points[j, 1:2])
            dist_v = abs(grid[i, 3] - points[j, 3])
            if (dist_h < search_radius_h) &  (dist_v < search_radius_v)
                push!(index, j)
                push!(cov_v, sill - model(dist_h,dist_v, p...))
            end
        end
        if length(index) < min_points
            μₖ[i] = NaN
            σ²ₖ[i] = NaN
            continue
        end
        # select the 20 most relevant points
        arg_cov = sortperm(cov_v, rev = true)
        cov_v = cov_v[arg_cov]
        index = index[arg_cov]
        if length(index) > max_points
            index = index[1:max_points]
            cov_v = cov_v[1:max_points]
        end
        cov_matrix_i = [covariance_matrix[index, index]  ones(length(index)); ones(length(index))' 0]
        cov_matrix_i += regularization * I # Add regularization
        cov_v = [cov_v; 1]
        weights = Symmetric(cov_matrix_i) \ cov_v
        @assert sum(weights[1:end-1]) ≈ 1
        # estimating the mean and variance of the estimator
        lagrange_mult = weights[end]
        μₖ[i] = sum(weights[1:end-1].*values[index])
        σ²ₖ[i] = σ² - lagrange_mult- sum(weights[1:end-1].*cov_v[1:end-1])
    end

    return μₖ, σ²ₖ
end

# # Example usage
# values = [your data values here]
# optimal_lambda = find_optimal_lambda(values)
# optimal_regularization = cross_validate_regularization(grid, points, values, model, p, μ, σ², search_radius_h, search_radius_v, min_points, max_points, optimal_lambda)

# println("Optimal λ: ", optimal_lambda)
# println("Optimal regularization: ", optimal_regularization)
