using LinearAlgebra
using Random
using Statistics
# Function to calculate the inverse distance weighting
function idw_interpolation(grid::Matrix{T}, points::Matrix{T},
     values::Vector{T}, power::T) where T
    n = size(grid, 1)
    m = size(points, 1)
    interpolated_values = zeros(T, n)

    for i in 1:n
        weights = zeros(T, m)
        for j in 1:m
            dist = norm(grid[i, :] .- points[j, :])
            if dist == 0
                weights[j] = Inf
            else
                weights[j] = 1 / dist^power
            end
        end
        if any(isinf.(weights))
            interpolated_values[i] = values[findfirst(isinf, weights)]
        else
            weights /= sum(weights)
            interpolated_values[i] = sum(weights .* values)
        end
    end

    return interpolated_values
end


# Function to perform k-fold cross-validation
function cross_validate_idw(points::Matrix{T}, values::Vector{T},
    k::Int, power_range::Vector{T}) where T
    n = size(points, 1)
    fold_size = div(n, k)
    errors = zeros(T, length(power_range))

    for (p_idx, power) in enumerate(power_range)
        fold_errors = zeros(T, k)
        shuffled_indices = shuffle(1:n)

        for fold in 1:k
            test_indices = shuffled_indices[(fold-1)*fold_size+1:fold*fold_size]
            train_indices = setdiff(shuffled_indices, test_indices)

            train_points = points[train_indices, :]
            train_values = values[train_indices]
            test_points = points[test_indices, :]
            test_values = values[test_indices]

            interpolated_values = idw_interpolation(test_points, train_points, train_values, power)
            fold_errors[fold] = mean(abs.(interpolated_values .- test_values))
        end

        errors[p_idx] = mean(fold_errors)
    end

    best_power_idx = argmin(errors)
    best_power = power_range[best_power_idx]

    return best_power, errors
end
