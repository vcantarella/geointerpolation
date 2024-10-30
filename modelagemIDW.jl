include("IDW3D.jl")
using CSV, DataFrames
using CairoMakie
using Statistics

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

missing_conc = findall(ismissing, Al)
indexes_al = [miss[1] for miss in missing_coords]
indexes_al = union(indexes_al, missing_conc)
# remove missing values
points_al = points[setdiff(1:end, indexes_al), :]
Al = Al[setdiff(1:end, indexes_al)]
# correct type
points_al = convert(Matrix{Float64}, points_al)
Al = convert(Vector{Float64}, Al)

missing_conc = findall(ismissing, Fe)
indexes_fe = [miss[1] for miss in missing_coords]
indexes_fe = union(indexes_fe, missing_conc)
# remove missing vfeues
points_fe = points[setdiff(1:end, indexes_fe), :]
Fe = Fe[setdiff(1:end, indexes_fe)]
# correct type
points_fe = convert(Matrix{Float64}, points_fe)
Fe = convert(Vector{Float64}, Fe)

missing_conc = findall(ismissing, F)
indexes_f = [miss[1] for miss in missing_coords]
indexes_f = union(indexes_f, missing_conc)
# remove missing vfues
points_f = points[setdiff(1:end, indexes_f), :]
F = F[setdiff(1:end, indexes_f)]
# correct type
points_f = convert(Matrix{Float64}, points_f)
F = convert(Vector{Float64}, F)

missing_conc = findall(ismissing, Se)
indexes_se = [miss[1] for miss in missing_coords]
indexes_se = union(indexes_se, missing_conc)
# remove missing vfues
points_se = points[setdiff(1:end, indexes_se), :]
Se = Se[setdiff(1:end, indexes_se)]
# correct type
points_se = convert(Matrix{Float64}, points_se)
Se = convert(Vector{Float64}, Se)

missing_conc = findall(ismissing, Co)
indexes_co = [miss[1] for miss in missing_coords]
indexes_co = union(indexes_co, missing_conc)
# remove missing vfues
points_co = points[setdiff(1:end, indexes_co), :]
Co = Co[setdiff(1:end, indexes_co)]
# correct type
points_co = convert(Matrix{Float64}, points_co)
Co = convert(Vector{Float64}, Co)

missing_svnc = findall(ismissing, svoc)
indexes_sv = [miss[1] for miss in missing_svords]
indexes_sv = union(indexes_sv, missing_svnc)
# remove missing vfues
points_sv = points[setdiff(1:end, indexes_sv), :]
svoc = svoc[setdiff(1:end, indexes_sv)]
# correct type
points_sv = convert(Matrix{Float64}, points_sv)
svoc = convert(Vector{Float64}, svoc)

# Run cross-validation for NO3
power_range = 1:0.1:3
best_power_no3, errors_no3 = cross_validate_idw(points_no3, NO3, 5, collect(power_range))
best_power_Al, errors_Al = cross_validate_idw(points_al, Al, 5, collect(power_range))
best_power_Fe, errors_Fe = cross_validate_idw(points_fe, Fe, 5, collect(power_range))
best_power_F, errors_F = cross_validate_idw(points_f, F, 5, collect(power_range))
best_power_Se, errors_Se = cross_validate_idw(points_se, Se, 5, collect(power_range))
best_power_Co, errors_Co = cross_validate_idw(points_co, Co, 5, collect(power_range))
best_power_svoc, errors_svoc = cross_validate_idw(points_sv, svoc, 5, collect(power_range))

# Plot the cross-validation errors
fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1], xlabel = "Power", ylabel = "Mean Absolute Error")
lines!(ax, power_range, errors_no3, label = "NO3", color = :blue)
lines!(ax, power_range, errors_Al, label = "Al", color = :red)
lines!(ax, power_range, errors_Fe, label = "Fe", color = :green)
lines!(ax, power_range, errors_F, label = "F", color = :yellow)
lines!(ax, power_range, errors_Se, label = "Se", color = :purple)
lines!(ax, power_range, errors_Co, label = "Co", color = :orange)
lines!(ax, power_range, errors_svoc, label = "svoc", color = :black)
fig

# Run IDW interpolation with the best power
##load the grid
xyz = CSV.read("test_datasets/tab_xyz.csv", DataFrame)
xyz = xyz[!, [:x, :y, :z]]
xyz = Matrix(xyz)
interpolated_values_no3 = idw_interpolation(xyz, points_no3, NO3, best_power_no3)
tab_conc = DataFrame(NO3 = interpolated_values_no3)
CSV.write("test_datasets/tab_concIDW.csv", tab_conc)
interpolated_values_Al = idw_interpolation(xyz, points_al, Al, best_power_Al)
tab_conc[!,"Al"] = interpolated_values_Al
CSV.write("test_datasets/tab_concIDW.csv", tab_conc)
interpolated_values_Fe = idw_interpolation(xyz, points_fe, Fe, best_power_Fe)
tab_conc[!,"Fe"] = interpolated_values_Fe
CSV.write("test_datasets/tab_concIDW.csv", tab_conc)
interpolated_values_F = idw_interpolation(xyz, points_f, F, best_power_F)
tab_conc[!,"F"] = interpolated_values_F
CSV.write("test_datasets/tab_concIDW.csv", tab_conc)
interpolated_values_Se = idw_interpolation(xyz, points_se, Se, best_power_Se)
tab_conc[!,"Se"] = interpolated_values_Se
CSV.write("test_datasets/tab_concIDW.csv", tab_conc)
interpolated_values_Co = idw_interpolation(xyz, points_co, Co, best_power_Co)
tab_conc[!,"Co"] = interpolated_values_Co
CSV.write("test_datasets/tab_concIDW.csv", tab_conc)
interpolated_values_svoc = idw_interpolation(xyz, points_sv, svoc, best_power_svoc)
tab_conc[!,"svoc"] = interpolated_values_svoc
CSV.write("test_datasets/tab_concIDW.csv", tab_conc)

