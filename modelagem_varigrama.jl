using CSV, DataFrames, CairoMakie
using Statistics
include("Variogram.jl")
include("VariogramModels.jl")
include("Kriging.jl")

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

# save to csv
CSV.write("test_datasets/values.csv", values)
CSV.write("test_datasets/points.csv", points_df)

#--------------- Variogram modeling-------------------
# getting missing values indexes:
missing_coords = findall(ismissing, points)
missing_conc = findall(ismissing, NO3)
indexes_no3 = [miss[1] for miss in missing_coords]
indexes_no3 = union(indexes_no3, missing_conc)
# remove missing values
points_no3 = points[setdiff(1:end, indexes_no3), :]
NO3 = NO3[setdiff(1:end, indexes_no3)]
# correct type
points_no3 = convert(Matrix{Float64}, points_no3)
NO3 = convert(Vector{Float64}, NO3)
# Calculate omni_horizontal variogram
gamma_h,step_h,gamma_v,step_v = omni_horizontal_variogram(points_no3[NO3 .> minimum(NO3),:], NO3[NO3 .> minimum(NO3)], 10.0, 3., 0.2, 0.1, 5.)
gamma_h2, step_h2 = omni_variogram(points_no3[NO3 .> minimum(NO3),:], NO3[NO3 .> minimum(NO3)], 10.0, 3.,)
# plot
fig = Figure(size = (550, 1200), title = "Variograma Nitrato")
ax = Axis(fig[1, 1], subtitle = "Horizontal")
scatter!(ax, step_h, gamma_h, color = :blue, markersize = 10, label = "Horizontal Variogram")
scatter!(ax, step_h2, gamma_h2, color = :green, markersize = 10, label = "Horizontal Variogram 2")
lines!(ax, 0:5:600, gaussian3D.(0:5:600,0., 300,3, 30), color = :blue, linewidth = 2, label = "Gaussian Model")
ax2 = Axis(fig[1, 2], subtitle = "Vertical")
scatter!(ax2, step_v, gamma_v, color = :red, markersize = 10, label = "Vertical Variogram")
lines!(ax2, 0:0.01:5, gaussian3D.(0.,0:0.01:5, 300,3, 30), color = :red, linewidth = 2, label = "Gaussian Vertical Model")
fig
# save parameters
p_no3 = [300 3 30]

# getting missing values indexes:
missing_coords = findall(ismissing, points)
missing_conc = findall(ismissing, Al)
indexes_al = [miss[1] for miss in missing_coords]
indexes_al = union(indexes_al, missing_conc)
# remove missing values
points_al = points[setdiff(1:end, indexes_al), :]
Al = Al[setdiff(1:end, indexes_al)]
# correct type
points_al = convert(Matrix{Float64}, points_al)
Al = convert(Vector{Float64}, Al)
# Calculate omni_horizontal variogram
gamma_h,step_h,gamma_v,step_v = omni_horizontal_variogram(points_al[Al .> minimum(Al),:], Al[Al .> minimum(Al)], 10.0, 5., 0.2, 0.1, 5.)
# plot
#fig = Figure(size = (800, 600))
ax = Axis(fig[2, 1], subtitle = "Horizontal")
scatter!(ax, step_h, gamma_h, color = :blue, markersize = 10, label = "Horizontal Variogram")
lines!(ax, 0:5:600, gaussian3D.(0:5:600,0., 100,2, 1.), color = :blue, linewidth = 2, label = "Gaussian Model")
ax2 = Axis(fig[2, 2], subtitle = "Vertical")
scatter!(ax2, step_v, gamma_v, color = :red, markersize = 10, label = "Vertical Variogram")
lines!(ax2, 0:0.01:5, gaussian3D.(0.,0:0.01:5, 100,2, 1.), color = :red, linewidth = 2, label = "Gaussian Vertical Model")
fig
p_al = [100 2 1]

# getting missing vfeues indexes:
missing_coords = findall(ismissing, points)
missing_conc = findall(ismissing, Fe)
indexes_fe = [miss[1] for miss in missing_coords]
indexes_fe = union(indexes_fe, missing_conc)
# remove missing vfeues
points_fe = points[setdiff(1:end, indexes_fe), :]
Fe = Fe[setdiff(1:end, indexes_fe)]
# correct type
points_fe = convert(Matrix{Float64}, points_fe)
Fe = convert(Vector{Float64}, Fe)
# Cfeculate omni_horizontfe variogram
gamma_h,step_h,gamma_v,step_v = omni_horizontal_variogram(points_fe, Fe, 10.0, 5., 0.2, 0.1, 5.)
# plot
ax = Axis(fig[3, 1], subtitle = "Horizontal")
scatter!(ax, step_h, gamma_h, color = :blue, markersize = 10, label = "Horizontfe Variogram")
lines!(ax, 0:5:600, gaussian3D.(0:5:600,0., 150,1.5, 1.1), color = :blue, linewidth = 2, label = "Gaussian Model")
ax2 = Axis(fig[3, 2], subtitle = "Vertical")
scatter!(ax2, step_v, gamma_v, color = :red, markersize = 10, label = "Verticfe Variogram")
lines!(ax2, 0:0.01:5, gaussian3D.(0.,0:0.01:5, 150,1.5, 1.1), color = :red, linewidth = 2, label = "Gaussian Verticfe Model")
fig
p_fe = [150 1.5 1.1]

# getting missing vfues indexes:
missing_coords = findall(ismissing, points)
missing_conc = findall(ismissing, F)
indexes_f = [miss[1] for miss in missing_coords]
indexes_f = union(indexes_f, missing_conc)
# remove missing vfues
points_f = points[setdiff(1:end, indexes_f), :]
F = F[setdiff(1:end, indexes_f)]
# correct type
points_f = convert(Matrix{Float64}, points_f)
F = convert(Vector{Float64}, F)
# Cfculate omni_horizontf variogram
gamma_h,step_h,gamma_v,step_v = omni_horizontal_variogram(points_f, F, 10.0, 5., 0.2, 0.1, 5.)
# plot
ax = Axis(fig[4, 1], subtitle = "Horizontal")
scatter!(ax, step_h, gamma_h, color = :blue, markersize = 10, label = "Horizontf Variogram")
lines!(ax, 0:5:600, gaussian3D.(0:5:600,0., 60,2, 0.3), color = :blue, linewidth = 2, label = "Gaussian Model")
ax2 = Axis(fig[4, 2],subtitle = "Vertical")
scatter!(ax2, step_v, gamma_v, color = :red, markersize = 10, label = "Verticf Variogram")
lines!(ax2, 0:0.01:5, gaussian3D.(0.,0:0.01:5, 60,2, 0.3), color = :red, linewidth = 2, label = "Gaussian Verticf Model")
fig
p_f = [60 2 0.3]

# getting missing vfues indexes:
missing_coords = findall(ismissing, points)
missing_conc = findall(ismissing, Se)
indexes_se = [miss[1] for miss in missing_coords]
indexes_se = union(indexes_se, missing_conc)
# remove missing vfues
points_se = points[setdiff(1:end, indexes_se), :]
Se = Se[setdiff(1:end, indexes_se)]
# correct type
points_se = convert(Matrix{Float64}, points_se)
Se = convert(Vector{Float64}, Se)
# Cfculate omni_horizontf variogram
gamma_h,step_h,gamma_v,step_v = omni_horizontal_variogram(points_se, Se, 10.0, 5., 0.2, 0.1, 5.)
# plot
ax = Axis(fig[5, 1],subtitle = "Horizontal")
scatter!(ax, step_h, gamma_h, color = :blue, markersize = 10, label = "Horizontf Variogram")
lines!(ax, 0:5:600, gaussian3D.(0:5:600,0., 120,1, 3e-6), color = :blue, linewidth = 2, label = "Gaussian Model")
ax2 = Axis(fig[5, 2], subtitle = "Vertical")
scatter!(ax2, step_v, gamma_v, color = :red, markersize = 10, label = "Verticf Variogram")
lines!(ax2, 0:0.01:5, gaussian3D.(0.,0:0.01:5, 120,1, 3e-6), color = :red, linewidth = 2, label = "Gaussian Verticf Model")
fig
p_se = [120 1 3e-6]

# getting missing vfues indexes:
missing_coords = findall(ismissing, points)
missing_conc = findall(ismissing, Co)
indexes_co = [miss[1] for miss in missing_coords]
indexes_co = union(indexes_co, missing_conc)
# remove missing vfues
points_co = points[setdiff(1:end, indexes_co), :]
Co = Co[setdiff(1:end, indexes_co)]
# correct type
points_co = convert(Matrix{Float64}, points_co)
Co = convert(Vector{Float64}, Co)
# Cfculate omni_horizontf variogram
gamma_h,step_h,gamma_v,step_v = omni_horizontal_variogram(points_co, Co, 10.0, 5., 0.2, 0.1, 5.)
# plot
ax = Axis(fig[6, 1])
scatter!(ax, step_h, gamma_h, color = :blue, markersize = 10, label = "Horizontf Variogram")
lines!(ax, 0:5:600, gaussian3D.(0:5:600,0., 40,2, 5e-7), color = :blue, linewidth = 2, label = "Gaussian Model")
ax2 = Axis(fig[6, 2])
scatter!(ax2, step_v, gamma_v, color = :red, markersize = 10, label = "Verticf Variogram")
lines!(ax2, 0:0.01:5, gaussian3D.(0.,0:0.01:5, 40,2, 5e-7), color = :red, linewidth = 2, label = "Gaussian Verticf Model")
fig
p_co = [40 2 5e-7]

# getting missing vfues indexes:
missing_svords = findall(ismissing, points)
missing_svnc = findall(ismissing, svoc)
indexes_sv = [miss[1] for miss in missing_svords]
indexes_sv = union(indexes_sv, missing_svnc)
# remove missing vfues
points_sv = points[setdiff(1:end, indexes_sv), :]
svoc = svoc[setdiff(1:end, indexes_sv)]
# correct type
points_sv = convert(Matrix{Float64}, points_sv)
svoc = convert(Vector{Float64}, svoc)
# Cfculate omni_horizontf variogram
gamma_h,step_h,gamma_v,step_v = omni_horizontal_variogram(points_sv, svoc, 10.0, 5., 0.2, 0.1, 5.)
# plot
ax = Axis(fig[7, 1])
scatter!(ax, step_h, gamma_h, color = :blue, markersize = 10, label = "Horizontf Variogram")
lines!(ax, 0:5:600, gaussian3D.(0:5:600,0., 80,1, 1e-10), color = :blue, linewidth = 2, label = "Gaussian Model")
ax2 = Axis(fig[7, 2])
scatter!(ax2, step_v, gamma_v, color = :red, markersize = 10, label = "Verticf Variogram")
lines!(ax2, 0:0.01:5, gaussian3D.(0.,0:0.01:5, 80,1, 1e-10), color = :red, linewidth = 2, label = "Gaussian Verticf Model")
fig
p_sv = [80 1 1e-10]
save("test_datasets/variogram.svg", fig)

#saving parameters
pars = [p_no3; p_al; p_fe; p_f; p_se; p_co; p_sv]
pars_df = DataFrame(pars, [:a_h, :a_v, :c])
pars_df
CSV.write("test_datasets/parameters.csv", pars_df)

#--------------- Kriging-------------------
# simple kriging

xyz = CSV.read("test_datasets/tab_xyz.csv", DataFrame)
xyz = xyz[!, [:x, :y, :z]]
xyz = Matrix(xyz)
p_no3 = convert(Vector{Float64}, vec(p_no3'))
dist_matrix_h = distance_matrix(points_no3[:, 1:2])
dist_matrix_v = distance_matrix(hcat(points_no3[:, 3]))
cov_m = p_no3[3] .- gaussian3D(dist_matrix_h, dist_matrix_v, p_no3...)
grid = xyz[100000:100100,:]
dist_h = [distance(grid[i,1:2], points_no3[j,1:2]) for i in 1:size(grid,1), j in 1:size(points_no3,1)]
dist_v = [abs(grid[i,3]- points_no3[j,3]) for i in 1:size(grid,1), j in 1:size(points_no3,1)]
cov_v = p_no3[3] .- gaussian3D.(dist_h[end,:], dist_v[end,:], p_no3...)

μₖ, σ²ₖ = _kriging_omnihorizontal(xyz, points_no3, NO3, gaussian3D, p_no3, minimum(NO3), (minimum(NO3)*0.1)^2, p_no3[1], 10, 80)
μₖ[μₖ .< 0] .= minimum(NO3)
μₖ[isnan.(μₖ)] .= minimum(NO3)
σ²ₖ[σ²ₖ .< 0] .= (minimum(NO3)*0.1)^2
σ²ₖ[isnan.(σ²ₖ)] .= (minimum(NO3)*0.1)^2
tab_conc = DataFrame(NO3 = μₖ)
tab_var = DataFrame(NO3 = σ²ₖ)
CSV.write("test_datasets/tab_conc.csv", tab_conc)
CSV.write("test_datasets/tab_var.csv", tab_var)

p_al = convert(Vector{Float64}, vec(p_al'))
μₖ, σ²ₖ = ordinary_kriging_omnihorizontal(xyz, points_al, Al, gaussian3D, p_al, minimum(Al), var(Al), p_al[1], 10, 80)
μₖ[μₖ .< 0] .= minimum(Al)
μₖ[isnan.(μₖ)] .= minimum(Al)
σ²ₖ[σ²ₖ .< 0] .= var(Al)
σ²ₖ[isnan.(σ²ₖ)] .= var(Al)
tab_conc[!, :Al] = μₖ
tab_var[!, :Al] = σ²ₖ
CSV.write("test_datasets/tab_conc.csv", tab_conc)
CSV.write("test_datasets/tab_var.csv", tab_var)

# Loop at the remaining substances
p_fe = convert(Vector{Float64}, vec(p_fe'))
μₖ, σ²ₖ = ordinary_kriging_omnihorizontal(xyz, points_fe, Fe, gaussian3D, p_fe, minimum(Fe), var(Fe), p_fe[1], 10, 80)
μₖ[μₖ .< 0] .= minimum(Fe)
μₖ[isnan.(μₖ)] .= minimum(Fe)
σ²ₖ[σ²ₖ .< 0] .= var(Fe)
σ²ₖ[isnan.(σ²ₖ)] .= var(Fe)
tab_conc[!, :Fe] = μₖ
tab_var[!, :Fe] = σ²ₖ
CSV.write("test_datasets/tab_conc.csv", tab_conc)
CSV.write("test_datasets/tab_var.csv", tab_var)

p_f = convert(Vector{Float64}, vec(p_f'))
μₖ, σ²ₖ = ordinary_kriging_omnihorizontal(xyz, points_f, F, gaussian3D, p_f, minimum(F), var(F), p_f[1], 10, 80)
μₖ[μₖ .< 0] .= minimum(F)
μₖ[isnan.(μₖ)] .= minimum(F)
σ²ₖ[σ²ₖ .< 0] .= var(F)
σ²ₖ[isnan.(σ²ₖ)] .= var(F)
tab_conc[!, :F] = μₖ
tab_var[!, :F] = σ²ₖ
CSV.write("test_datasets/tab_conc.csv", tab_conc)
CSV.write("test_datasets/tab_var.csv", tab_var)

p_se = convert(Vector{Float64}, vec(p_se'))
μₖ, σ²ₖ = ordinary_kriging_omnihorizontal(xyz, points_se, Se, gaussian3D, p_se, minimum(Se), var(Se), p_se[1], 10, 80)
μₖ[μₖ .< 0] .= minimum(Se)
μₖ[isnan.(μₖ)] .= minimum(Se)
σ²ₖ[σ²ₖ .< 0] .= var(Se)
σ²ₖ[isnan.(σ²ₖ)] .= var(Se)
tab_conc[!, :Se] = μₖ
tab_var[!, :Se] = σ²ₖ
CSV.write("test_datasets/tab_conc.csv", tab_conc)
CSV.write("test_datasets/tab_var.csv", tab_var)

p_co = convert(Vector{Float64}, vec(p_co'))
μₖ, σ²ₖ = ordinary_kriging_omnihorizontal(xyz, points_co, Co, gaussian3D, p_co, minimum(Co), var(Co), p_co[1], 10, 80)
μₖ[μₖ .< 0] .= minimum(Co)
μₖ[isnan.(μₖ)] .= minimum(Co)
σ²ₖ[σ²ₖ .< 0] .= var(Co)
σ²ₖ[isnan.(σ²ₖ)] .= var(Co)
tab_conc[!, :Co] = μₖ
tab_var[!, :Co] = σ²ₖ
CSV.write("test_datasets/tab_conc.csv", tab_conc)
CSV.write("test_datasets/tab_var.csv", tab_var)

p_sv = convert(Vector{Float64}, vec(p_sv'))
μₖ, σ²ₖ = ordinary_kriging_omnihorizontal(xyz, points_sv, svoc, gaussian3D, p_sv, minimum(svoc), var(svoc), p_sv[1], 10, 80)
μₖ[μₖ .< 0] .= minimum(svoc)
μₖ[isnan.(μₖ)] .= minimum(svoc)
σ²ₖ[σ²ₖ .< 0] .= var(svoc)
σ²ₖ[isnan.(σ²ₖ)] .= var(svoc)
tab_conc[!, :svoc] = μₖ
tab_var[!, :svoc] = σ²ₖ
CSV.write("test_datasets/tab_conc.csv", tab_conc)
CSV.write("test_datasets/tab_var.csv", tab_var)

#---- fixing negative values:
