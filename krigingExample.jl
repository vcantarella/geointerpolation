include("Variogram.jl")
include("VariogramModels.jl")
include("Kriging.jl")
using CSV, DataFrames
using Statistics
using Plots
using Optim

# Load the data
data = CSV.read("test_datasets/2D_Porosity_Small.csv", DataFrame)
points = Matrix(data[!, [:X, :Y]])
values = data[:, :Porosity]

# Variogram calculation
a = 60.
C = 30.
p = [a, C]
# Calculate the variogram
stp = 1.0
tol = 0.5
γₛ, step_range = universal_variogram(points, values, stp, tol)
γˢ = spherical.(step_range, a, C)
γᵍ = gaussian.(step_range, a, C)

# Fitting the variogram model
model = (step_range, p)->gaussian.(step_range, p[1], p[2])
plt = scatter(step_range, γₛ, label="Variogram", xlabel="Lag Distance", ylabel="Variogram Value",
 title="Empirical Variogram",
 markershape=:circle, markercolor=:blue, legend=:topleft)

plot!(plt, step_range, model(p, step_range), label="Initial Model", linecolor=:black, linewidth=2)

loss = (p, step_range, γₛ)->sum((model(p, step_range) .- γₛ).^2)
loss_s = (p)->loss(p, step_range, γₛ)
res = optimize(loss_s, p, LBFGS(); autodiff = :forward)
p0 = Optim.minimizer(res)
plot!(plt, step_range, model(p0, step_range), label="Fitted Model", linecolor=:purple, linewidth=2)

# Create a grid:
dx = 1.0
x = range(0+dx/2, stop=100, step=dx)
y = range(0+dx/2, stop=100, step=dx)
# grid as a meshgrid like in numpy
grid_y = hcat([[i for i in x] for j in y]...)
# flip the grid
grid_y = reverse(grid_y, dims=1)
grid_x = hcat([[j for i in y] for j in x]...)

cell_centers = hcat(grid_x[:], grid_y[:])

# Simple Kriging
μ = mean(values)
σ² = var(values)
search_radius = p0[1]
μₖ, σ²ₖ = simple_kriging(cell_centers, points, values, model, p0, μ, σ², search_radius)
μₖ = reshape(μₖ, size(grid_x))
σ²ₖ = reshape(σ²ₖ, size(grid_x))

# Plotting the results
plt = heatmap(x, y, μₖ, title="Simple Kriging Mean", xlabel="X", ylabel="Y", color=:viridis)
plt = heatmap(x, y, σ²ₖ, title="Simple Kriging Variance", xlabel="X", ylabel="Y", color=:viridis)