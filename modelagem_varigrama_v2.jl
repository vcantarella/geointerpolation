using CSV, DataFrames, CairoMakie
using Statistics
using LinearAlgebra
using Base.Threads
using GeoStats
using Optim


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

include("Variogram.jl")
include("VariogramModels.jl")
include("Kriging.jl")
include("NormalScore.jl")

# Read the data
data = CSV.read("../tabela_quimica_inicial.csv", DataFrame)


# Example usage
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

# Dataframe for the QGIS
pqgis = hcat(points_df, values)
CSV.write("test_datasets/tab_qgis.csv", pqgis)
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
    print(lambda)
    print("$(namess[j])\n")
    print("minimum: $(minimum(data))\n")
    print("maximum: $(maximum(data))\n")
    normal_ = boxcox_transform(data, lambda)
    push!(normal_scores, normal_)
    push!(lambdas, lambda)
    push!(points_ls, points_data)
    push!(comple_vals, data)
    # export complete datasets
    tab = hcat(points_data, data)
    tab = DataFrame(tab, [:x, :y, :z, :conc])
    CSV.write("test_datasets/tab_xyz_$(namess[j]).csv", tab)
    geo = georef(tab, ("x", "y", "z"))
    push!(geods, geo)
end
xyz = CSV.read("test_datasets/tab_xyz_active.csv", DataFrame)
xyz = xyz[!, [:x, :y, :z]]
xyz = Matrix(xyz)
emp_var = EmpiricalVariogram(geods[1], :conc; maxlag = 300, nlags = 20)
varioplot(emp_var,title="Variograma Empírico NO3")
model = Kriging(GaussianVariogram(range=300.))
geotab = georef(xyz, ("x", "y", "z"))
interp = geods[1] |> Interpolate(geotab.geometry, model)

# Checking the normal transformation
for (i, elem) in enumerate(["NO3", "Al", "Fe", "F", "svoc", "Se", "Co"])
    # normal_scores[i] = reverse_normal_score_transformation(normal_scores[i], comple_vals[i])
    exp_score = inverse_boxcox_transform(normal_scores[i], lambdas[i])
    @assert comple_vals[i] ≈ exp_score
end
params = [
    [420 4 5];
    [150 3 1.2];
    [300 3 0.8];
    [80 3 0.4];
]
#     [100 8 1e-10];
#     [200 8 4e-6];
#     [490 8 1e-6]
# ]
#--------------- Variogram modeling-------------------
fig = Figure(size = (550, 1200), title = "Variograma Transformado")
for (i, elem) in enumerate(["NO3", "Al", "Fe", "F"])#, "svoc", "Se", "Co"])
    # Calculate omni_horizontal variogram
    gamma_h,step_h,gamma_v,step_v = omni_horizontal_variogram(points_ls[i], normal_scores[i], 10.0, 5., 0.2, 0.1, 5.)
    # plot
    p = params[i,:]
    p = vec(p')
    ax = Axis(fig[i, 1], subtitle = "Horizontal")
    scatter!(ax, step_h, gamma_h, color = :blue, markersize = 10, label = "Horizontal Variogram")
    lines!(ax, 0:5:600, gaussian3D.(0:5:600,0., p...), color = :blue, linewidth = 2, label = "Gaussian Model")
    ax2 = Axis(fig[i, 2], subtitle = "Vertical")
    scatter!(ax2, step_v, gamma_v, color = :red, markersize = 10, label = "Vertical Variogram")
    lines!(ax2, 0:0.01:5, gaussian3D.(0.,0:0.01:5, p...), color = :red, linewidth = 2, label = "Gaussian Vertical Model")
end
fig
save("test_datasets/variogram_models1.png", fig)

#saving parameters
pars_df = DataFrame(params, [:a_h, :a_v, :c])
pars_df
CSV.write("test_datasets/parameters1.csv", pars_df)
# ------ Cross Validation ------




#--------------- Kriging-------------------
# simple kriging


tab_conc = DataFrame()
elem = ["NO3", "Al", "Fe", "F"]#, "svoc", "Se", "Co"]
@threads for i in eachindex(elem)
    p = params[i,:]
    p = vec(p')
    background = minimum(comple_vals[i])
    μ = background
    σ² = var(comple_vals[i])
    reg = σ²*1e-10
    μₖ, σ²ₖ = ordinary_kriging_omnihorizontal(xyz, points_ls[i], normal_scores[i], gaussian3D,
     p, μ, σ², p[1],p[2], 3, 8, reg)
    μₖ = inverse_boxcox_transform(μₖ, lambdas[i])
    μₖ[μₖ .< 0] .= minimum(comple_vals[i])
    μₖ[isnan.(μₖ)] .= minimum(comple_vals[i])
    σ²ₖ[σ²ₖ .< 0] .= var(comple_vals[i])
    σ²ₖ[isnan.(σ²ₖ)] .= var(comple_vals[i])
    tab_conc[!, elem[i]] = μₖ
    println("$(elem[i]) done")
end

CSV.write("test_datasets/tab_conc.csv", tab_conc)
#---- fixing negative values:
