using CairoMakie
using LinearAlgebra
include("Variogram.jl")
using CSV, DataFrames
using DataInterpolations

data = CSV.read("test_datasets/2D_Porosity_Small.csv", DataFrame)
points = Matrix(data[!, [:X, :Y]])
values = data[:, :Porosity]
f = Figure(size = (800, 500))
# Calculate the variogram
stp = 1.0
tol = 0.5
max_h = 70.0
ax = PolarAxis(f[1, 1], title = "VariogramMap")
rs = 1:max_h
phis = range(0, 2pi, 12)[1:12]
unit_v = [[cos(ϕ), sin(ϕ)] for ϕ in phis]
gamma = zeros(length(phis), length(rs))
for i in eachindex(unit_v)
    gammas, steps = directional_variogram(points, values, stp, tol, unit_v[i], 40., max_h)
    interp = LinearInterpolation(gammas, steps, extrapolate=true)
    gammas = [interp(r) for r in rs]
    gamma[i, :] = gammas
end
cs = [r+cos(4phi) for phi in phis, r in rs]
p = voronoiplot!(ax, phis, rs, gamma, show_generators = false, strokewidth = 0)
rlims!(ax, 0.0, 50)
Colorbar(f[2, 2], p, vertical = false, flipaxis = false)