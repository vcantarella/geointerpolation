using CSV
using DataFrames
using Optim
using Plots
include("Variogram.jl")
include("VariogramModels.jl")

# Load the data
data = CSV.read("test_datasets/2D_Porosity_Small.csv", DataFrame)
points = Matrix(data[!, [:X, :Y]])
values = data[:, :Porosity]

# Calculate the variogram
stp = 1.0
tol = 0.5
γₛ, step_range = universal_variogram(points, values, stp, tol)

#Plot the variogram
plt = scatter(step_range, γₛ, label="Variogram", xlabel="Lag Distance", ylabel="Variogram Value",
 title="Empirical Variogram",
 markershape=:circle, markercolor=:blue, legend=:topleft)

# Restrict to maximum lag distance (around 70 m)

max_h = 70.0
γₛ, step_range = universal_variogram(points, values, stp, tol, max_h)

plt = scatter(step_range, γₛ, label="Variogram", xlabel="Lag Distance", ylabel="Variogram Value",
 title="Empirical Variogram",
 markershape=:circle, markercolor=:blue, legend=:topleft)

a = 60.
C = 30.
p = [a, C]

γˢ = spherical.(step_range, a, C)
γᵍ = gaussian.(step_range, a, C)
plot!(plt, step_range, γˢ, label="Spherical Model", linecolor=:red, linewidth=2)
plot!(plt, step_range, γᵍ, label="Gaussian Model", linecolor=:green, linewidth=2)

# Fitting the variogram model
model = (p, step_range)->gaussian.(step_range, p[1], p[2])
plot!(plt, step_range, model(p, step_range), label="Initial Model", linecolor=:black, linewidth=2)

loss = (p, step_range, γₛ)->sum((model(p, step_range) .- γₛ).^2)
loss_s = (p)->loss(p, step_range, γₛ)
res = optimize(loss_s, p, LBFGS(); autodiff = :forward)
p0 = Optim.minimizer(res)
plot!(plt, step_range, model(p0, step_range), label="Fitted Model", linecolor=:purple, linewidth=2)
