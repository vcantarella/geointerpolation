include("VariogramModels.jl")
using LinearAlgebra

# function to compute distance between two points
function distance(p1::Vector{T}, p2::Vector{T}) where T
    #@assert 2 <= length(p1) <= 3 "p1 must be a vector of length 2 or 3"
    @assert length(p2) == length(p1) "p2 have the same lenght as p1"
    return sqrt(reduce(+, (p1-p2).^2))
end

function distance_matrix(points::Matrix{T}) where T
    # calculates the distance matrix between points (lazily)
    #@assert 2 <= size(points, 2) <= 3 "points must be 2 or 3 dimensional"
    #@assert size(points, 1) > 2 "points must have more than 2 points"
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
        omni_variogram(data::Array{Float64, 2}, h::Float64, model::Symbol)

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
function omni_variogram(points::Matrix{T}, values::Vector{T},
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
        for i in 1:(size(points, 1)-1)
            for j in (i+1):size(points, 1)
                if  (st - tol) < dist_matrix[i, j] < (st + tol)
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
function unit_vector_crossing_point(v::Vector{T}, p::Vector{T}) where T
    # Normalize the vector v to ensure it is a unit vector
    v_unit = v / norm(v)
    
    # Scale the unit vector by the coordinates of point p
    direction_vector = v_unit .* p
    
    return direction_vector
end
@doc raw"""
    project_point_on_line(a::Vector{T}, v::Vector{T}, p::Vector{T})
    Let `p` be a point on a line and v the vector of that line direction.
    Then the projection of a point `a` on the line is the orthogonal projection of `a` on the line.
    The general solution of the projection is:

    `` prj = \frac{(a - p)^T v}{v^T v} v ``

    Then the distance between the projection point and the point `a` is:

    `` d = ||prj - a|| ``

    And the directional distance between the point p and the projection point is:

    `` h = ||prj - p|| ``

"""
function project_point_on_line(a::Vector{T}, v::Vector{T}, p::Vector{T}) where T
    
    # Calculate the vector from point a to point p
    ap = p - a
    
    # Project the vector ap onto the unit vector v_unit
    projection_point = p + (a-p)' * v / (v' * v) * v
    
    # Calculate the projection point
    projection_distance = norm(projection_point - a)
    directional_distance = norm(projection_point - p)
    
    return projection_point, projection_distance, directional_distance
end


function directional_variogram(points::Matrix{T}, values::Vector{T},
    step::T, tol::T,
    direction::Vector{T}, halfbandwitch_tol::T,
    max_h = nothing,
    ) where T
    # calculates the distance matrix between points (lazily) and then computes the 
    # universal variogram.
    @assert size(values, 1) == size(points, 1) "values must have the same length as points"
    if max_h === nothing
        max_step = maximum(points)-minimum(points)
    else
        max_step = max_h
    end
    step_range = 0:step:max_step
    γₛ = zeros(T, length(step_range))
    step_valid = ones(Bool, length(step_range))
    ns = zeros(Int, length(step_range))
    for i in 1:(size(points, 1)-1)
        for j in (i+1):size(points, 1)
            projection_point, prj_distance, dir_distance = project_point_on_line(points[j,:], direction, points[i,:])
            if dir_distance < halfbandwitch_tol
                k = 1
                for st in step_range
                    if prj_distance > st - tol && prj_distance < st + tol
                        γ = 0.5 * (values[i] - values[j])^2
                        γₛ[k] += γ
                        ns[k] += 1
                    end
                    k += 1
                end
            end
        end
    end
    for k in eachindex(γₛ)
        if ns[k] > 0
            γₛ[k] /= ns[k]
        else
            step_valid[k] = false
        end
   end
   γₛ = γₛ[step_valid]
   step_range = step_range[step_valid]
   return γₛ, step_range
end

"""
Sibut an anisotropic variogram in the vertical direction.

Simpler implementation than a full anisotropic variogram where we can calculate a uniform variogram in the horizontal direction
but anisotropic vertical direction.
"""
function omni_horizontal_variogram(points::Matrix{T}, values::Vector{T},
    step_h::T, tol_h::T,
    step_v::T, tol_v::T,
    bandwitch_v::T, max_h = nothing
    ) where T
    @assert size(values, 1) == size(points, 1) "values must have the same length as points"
    if max_h === nothing
        max_step = maximum(points)-minimum(points)
    else
        max_step = max_h
    end
    # horizontal_range
    step_range = 0:step_h:max_step
    γₛₕ = zeros(T, length(step_range))
    step_validₕ = ones(Bool, length(step_range))
    nsₕ = zeros(Int, length(step_range))
    # vertical_range
    max_v = maximum(points[:,3])-minimum(points[:,3])
    step_vert = 0:step_v:max_v
    γₛᵥ = zeros(T, length(step_vert))
    step_validᵥ = ones(Bool, length(step_vert))
    nsᵥ = zeros(Int, length(step_vert))
    for i in 1:(size(points, 1)-1)
        for j in (i+1):size(points, 1)
            vector_distance = points[j,:]-points[i,:]
            horizontal_distance = norm(vector_distance[1:2])
            vertical_distance = abs(vector_distance[3])
            for k in eachindex(step_range)
                st = step_range[k]
                if (st - tol_h) < horizontal_distance < (st + tol_h)
                    γ = (values[i] - values[j])^2
                    γₛₕ[k] += γ
                    nsₕ[k] += 1
                end
            end
            for j in eachindex(step_vert)
                st = step_vert[j]
                if  (st - tol_v) < vertical_distance < (st + tol_v)
                    if horizontal_distance < bandwitch_v
                        γ = (values[i] - values[j])^2
                        nsᵥ[j] += 1
                        γₛᵥ[j] += γ
                    end
                end
            end
        end
    end
    for k in eachindex(γₛₕ)
        if nsₕ[k] > 0
            γₛₕ[k] = γₛₕ[k]/(nsₕ[k] * 2)
        else
            step_validₕ[k] = false
        end
   end
   γₛₕ = γₛₕ[step_validₕ]
   step_range = step_range[step_validₕ]

   for k in eachindex(γₛᵥ)
        if nsᵥ[k] > 0
            γₛᵥ[k] = γₛᵥ[k]/(nsᵥ[k] * 2)
        else
            step_validᵥ[k] = false
        end
   end
    γₛᵥ = γₛᵥ[step_validᵥ]
    step_vert = step_vert[step_validᵥ]
   
   return γₛₕ, step_range, γₛᵥ, step_vert
end


