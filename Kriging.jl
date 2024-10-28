include("Variogram.jl")
include("VariogramModels.jl")
using Base.Threads

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
        weights = covariance_matrix[index, index] \ cov_v
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
                max_points = 20) where T
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
    @threads for i in 1:n
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
        if length(index) == 0
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
        weights = covariance_matrix[index, index] \ cov_v
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
        cov_v = [cov_v; 0]
        weights = cov_matrix_i \ cov_v
        lagrange_mult = weights[end]
        μₖ[i] = sum(weights[1:end-1].*values[index])
        σ²ₖ[i] = σ² - lagrange_mult- sum(weights[1:end-1].*cov_v[1:end-1])
    end
    return μₖ, σ²ₖ
end

function ordinary_kriging_omnihorizontal(grid::Matrix{T}, points::Matrix{T},
    values::Vector{T}, model::Function,
    p::Vector{T}, μ::T,
    σ²::T,search_radius::T,
    max_points = 20) where T
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
    @threads for i in 1:n
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
        if length(index) == 0
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
        cov_matrix_i = [covariance_matrix[index, index] ones(length(index)); ones(length(index))' 0]
        cov_v = [cov_v; 0]
        weights = cov_matrix_i \ cov_v
        # estimating the mean and variance of the estimator
        lagrange_mult = weights[end]
        μₖ[i] = sum(weights[1:end-1].*values[index])
        σ²ₖ[i] = σ² - lagrange_mult- sum(weights[1:end-1].*cov_v[1:end-1])
    end
    return μₖ, σ²ₖ
end
