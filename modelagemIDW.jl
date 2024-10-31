include("IDW3D.jl")
using CSV, DataFrames
using CairoMakie
using Statistics
using Base.Threads
using Optim
using Distributions
BLAS.set_num_threads(1)
println("Num of BLAS Threads: $(BLAS.get_num_threads())")
println("Num of Threads: $(Threads.nthreads())")

# Function for Box-Cox transformation
function boxcox_transform(values::Vector{T}, λ::T) where T
    if λ == 0
        return log.(values)
    else
        return (values .^ λ .- 1) ./ λ
    end
end

# Function for inverse Box-Cox transformation
function inverse_boxcox_transform(transformed_values::Vector{T}, λ::T) where T
    if λ == 0
        return exp.(transformed_values)
    else
        return (λ .* transformed_values .+ 1) .^ (1 / λ)
    end
end

# Function to find the optimal lambda using maximum likelihood estimation
function find_optimal_lambda(values::Vector{T}) where T
    log_likelihood(λ) = -sum(logpdf(Normal(mean(boxcox_transform(values, λ)), std(boxcox_transform(values, λ))), boxcox_transform(values, λ)))
    result = optimize(log_likelihood, -5.0, 5.0)
    return Optim.minimizer(result)
end


# Read the data
data = CSV.read("../tabela_quimica_inicial.csv", DataFrame)

# select columns x,y and prof [m]
x = data[:, :x]
y = data[:, :y]
z = data[:, "prof [m]"]

points = [x y z]

# select values
col_names = names(data)
Al = data[:, "Alumínio Dissolvido"]./27e3
Fe = data[:, "Ferro Dissolvido"]./55.8e3
NO3 = data[:, "Nitrato"]./62e3
F = data[:, "Fluoreto"]./19e3
svoc = data[:, "Di(2-Etilhexil)ftalato (DEHP)"]./ #formula: C24H38O4
    ((24*12.01 + 38*1.01 + 4*16.00)*1e3)
Se = data[:, "Selênio Dissolvido"]./78.96e3
Co = data[:, "Cobalto Dissolvido"]./58.93e3

# make a values dataframe
values = DataFrame(Al = Al, Fe = Fe, NO3 = NO3, F = F, svoc = svoc, Se = Se, Co = Co)

# make a points dataframe
points_df = DataFrame(x = x, y = y, z = z)

missing_coords = findall(ismissing, points)
points_ls = []
comple_vals = []
normal_scores = []
geods = []
lambdas = []
namess = ["NO3", "Al", "Fe", "F", "svoc", "Se", "Co"]
# fix missing values and points
for (j,data) in enumerate([NO3, Al, Fe, F, svoc, Se, Co])
    missing_conc = findall(ismissing, data)
    indexes_misscoords = [miss[1] for miss in missing_coords]
    indexes_missdata = union(indexes_misscoords, missing_conc)
    # remove missing values
    points_data = points[setdiff(1:end, indexes_missdata), :]
    data = data[setdiff(1:end, indexes_missdata)]
    # correct type
    points_data = convert(Matrix{Float64}, points_data)
    data = convert(Vector{Float64}, data)
    lambda = find_optimal_lambda(data)
    if j >= 5
        lambda = 0.
    end
    print(lambda)
    print("$(namess[j])\n")
    print("minimum: $(minimum(data))\n")
    print("maximum: $(maximum(data))\n")
    normal_ = boxcox_transform(data, lambda)
    exp_score = inverse_boxcox_transform(normal_, lambda)
    @assert data ≈ exp_score
    push!(normal_scores, normal_)
    push!(lambdas, lambda)
    push!(points_ls, points_data)
    push!(comple_vals, data)
end

# Run cross-validation for concentrations
power_range = 0.1:0.1:3
best_powers = []
errors_plot = []
for (j, names) in enumerate(namess)
    best_power, errors = cross_validate_idw(points_ls[j], normal_scores[j], 5, collect(power_range))
    push!(best_powers, best_power)
    push!(errors_plot, errors)
end

# # Plot the cross-validation errors
# fig = Figure(resolution = (800, 600))
# ax = Axis(fig[1, 1], xlabel = "Power", ylabel = "Mean Absolute Error")
# lines!(ax, power_range, errors_no3, label = "NO3", color = :blue)
# lines!(ax, power_range, errors_Al, label = "Al", color = :red)
# lines!(ax, power_range, errors_Fe, label = "Fe", color = :green)
# lines!(ax, power_range, errors_F, label = "F", color = :yellow)
# lines!(ax, power_range, errors_Se, label = "Se", color = :purple)
# lines!(ax, power_range, errors_Co, label = "Co", color = :orange)
# lines!(ax, power_range, errors_svoc, label = "svoc", color = :black)
# fig

# Run IDW interpolation with the best power
##load the grid
xyz = CSV.read("test_datasets/tab_xyz_active.csv", DataFrame)
xyz = xyz[!, [:x, :y, :z]]
xyz = Matrix(xyz)
tab_conc = DataFrame()
@threads for i in eachindex(namess)
    interpolated_values = idw_interpolation(xyz, points_ls[i], normal_scores[i], best_powers[i])
    tab_conc[!,namess[i]] = inverse_boxcox_transform(interpolated_values, lambdas[i])
    CSV.write("test_datasets/tab_concIDW.csv", tab_conc)
    println("$(namess[i]) done")
end


