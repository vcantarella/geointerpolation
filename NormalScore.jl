using Distributions
using Statistics
using StatsBase
using Interpolations

# Function to calculate the normal score transformation
function normal_score_transformation(values::Vector{T}) where T
    f = ecdf(values)
    normal_scores = quantile.(Normal(0, 1), f.(values))

    return normal_scores, f
end

# Function to reverse the normal score transformation
function reverse_normal_score_transformation(normal_scores::Vector{T}, original_values::Vector{T}) where T
    sorted_values = sort(original_values)
    cumulative_probabilities = ecdf(sorted_values).(sorted_values)
    interpolation = linear_interpolation(cumulative_probabilities, sorted_values, extrapolation_bc = Line())
    cdf_values = cdf.(Normal(0, 1), normal_scores)
    reverse_values = interpolation.(cdf_values)
    return reverse_values
end