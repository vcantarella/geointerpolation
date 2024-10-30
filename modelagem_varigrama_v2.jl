using CSV, DataFrames, CairoMakie
using Statistics
using LinearAlgebra
using Base.Threads

BLAS.set_num_threads(1)

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
missing_coords = findall(ismissing, points)
points_ls = []
comple_vals = []
normal_scores = []
# fix missing values and points
for data in [NO3, Al, Fe, F, svoc, Se, Co]
    missing_conc = findall(ismissing, data)
    indexes_misscoords = [miss[1] for miss in missing_coords]
    indexes_missdata = union(indexes_misscoords, missing_conc)
    # remove missing values
    points_data = points[setdiff(1:end, indexes_missdata), :]
    data = data[setdiff(1:end, indexes_missdata)]
    # correct type
    points_data = convert(Matrix{Float64}, points_data)
    data = convert(Vector{Float64}, data)
    normal_, f = normal_score_transformation(data)
    push!(normal_scores, normal_)
    push!(points_ls, points_data)
    push!(comple_vals, data)
end

# Checking the normal transformation
for (i, elem) in enumerate(["NO3", "Al", "Fe", "F", "svoc", "Se", "Co"])
    normal_scores[i] = reverse_normal_score_transformation(normal_scores[i], comple_vals[i])
    @assert comple_vals[i] ≈ normal_scores[i]
end
params = [
    [420 6 40];
    [450 6 1.2];
    [450 6 3.];
    [100 6 0.4];
    [150 6 1e-10];
    [450 6 4e-6];
    [500 6 1e-6]
]
#--------------- Variogram modeling-------------------
fig = Figure(size = (550, 1200), title = "Variograma Transformado")
for (i, elem) in enumerate(["NO3", "Al", "Fe", "F", "svoc", "Se", "Co"])
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
    # save("test_datasets/variogram_$elem.svg", fig)
end
fig
save("test_datasets/variogram_models.png", fig)

#saving parameters
pars_df = DataFrame(params, [:a_h, :a_v, :c])
pars_df
CSV.write("test_datasets/parameters.csv", pars_df)

#--------------- Kriging-------------------
# simple kriging

xyz = CSV.read("test_datasets/tab_xyz.csv", DataFrame)
xyz = xyz[!, [:x, :y, :z]]
xyz = Matrix(xyz)
tab_conc = DataFrame()
elem = ["NO3", "Al", "Fe", "F", "svoc", "Se", "Co"]
@threads for i in eachindex(elem)
    p = params[i,:]
    p = vec(p')
    μₖ, σ²ₖ = simple_kriging_omnihorizontal(xyz, points_ls[i], normal_scores[i], gaussian3D, p, mean(normal_scores[i]), var(normal_scores[i]), 2000., 10, 90)
    μₖ[μₖ .< 0] .= minimum(normal_scores[i])
    μₖ[isnan.(μₖ)] .= minimum(normal_scores[i])
    σ²ₖ[σ²ₖ .< 0] .= var(normal_scores[i])
    σ²ₖ[isnan.(σ²ₖ)] .= var(normal_scores[i])
    tab_conc[!, elem[i]] = reverse_normal_score_transformation(μₖ, comple_vals[i])
end
CSV.write("test_datasets/tab_conc.csv", tab_conc)
#---- fixing negative values:
