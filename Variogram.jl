include("VariogramModels.jl")

# function to compute distance between two points
function distance(p1::Vector{T}, p2::Vector{T}) where T
    @assert 2 <= length(p1) <= 3 "p1 must be a vector of length 2 or 3"
    @assert length(p2) == length(p1) "p2 have the same lenght as p1"
    return sqrt(reduce(+, (p1-p2).^2))
end

function distance_matrix(points::Matrix{T}) where T
    # calculates the distance matrix between points (lazily)
    @assert 2 <= size(points, 2) <= 3 "points must be 2 or 3 dimensional"
    @assert size(points, 1) > 2 "points must have more than 2 points"
    dist_matrix = zeros(T, size(points, 1), size(points, 1))
    for i in 1:size(points, 1)
        for j in i:size(points, 1)
            if i != j
                if dist_matrix[i, j] == 0
                    dist = distance(points[i, :], points[j, :])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                end
            end
        end
    end
    return dist_matrix
end

"""
        universal_variogram(data::Array{Float64, 2}, h::Float64, model::Symbol)

    Calculate the empirical semi-variogram for a given dataset using a specified model

    # Arguments
    - `data::Array{T, 2}`: A 2D array where each row represents a spatial location and the corresponding value.
    - `values::Vector{T}`: A vector of values corresponding to the spatial locations in `data`.
    - `step::T`: The lag regular interval distance at which the variogram is calculated.
    - `tol::T`: The tolerance for the lag distance.
    
    # Returns
    - `Float64`: The calculated variogram value for the given lag distance and model.

    # Description
    The `universal_variogram` function computes the variogram, which is a fundamental tool in geostatistics used to describe the spatial continuity or roughness of a dataset. The function supports multiple models to fit the variogram to the data, allowing for flexibility in capturing different spatial patterns.

    # Examples
"""
function universal_variogram(points::Matrix{T}, values::Vector{T},
     step::T, tol::T,
     max_h = nothing,
     ) where T
    # calculates the distance matrix between points (lazily) and then computes the 
    # universal variogram.
    @assert size(values, 1) == size(points, 1) "values must have the same length as points"

    dist_matrix = distance_matrix(points)
    if max_h === nothing
        max_step = maximum(dist_matrix)
    else
        max_step = max_h
    end
    step_range = 0:step:max_step
    γₛ = zeros(T, length(step_range))
    k = 1
    step_valid = ones(Bool, length(step_range))
    for st in step_range
        γ = 0.0
        n = 0
        for i in 1:size(points, 1)
            for j in i:size(points, 1)
                if dist_matrix[i, j] > st - tol && dist_matrix[i, j] < st + tol
                    γ += 0.5 * (values[i] - values[j])^2
                    n += 1
                end
            end
        end
        if n > 0
            γ /= n
            println("Step: $st, Variogram: $γ")
        else
            step_valid[k] = false
        end
        γₛ[k] = γ
        k += 1
    end
    γₛ = γₛ[step_valid]
    step_range = step_range[step_valid]
    return γₛ, step_range
end

# Calculate the model variogram at given steps (for universal variograms)

