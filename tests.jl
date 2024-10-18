include("Variogram.jl")

# Test the distance function
p1 = [1.0, 2.0, 3.0]
p2 = [4.0, 5.0, 6.0]
@assert distance(p1, p2) ≈ 5.196152422706632

p3 = [6., 7., 8.]
p4 = [9., 10., 11.]

points = [p1'; p2'; p3'; p4']
values = [1.0, 2.0, 3.0, 4.0]
st = 1.0
tol = 0.5
dist_matrix = distance_matrix(points)
@show eachindex(dist_matrix)
for i in axes(dist_matrix, 1)
    for j in axes(dist_matrix, 2)
        if i != j
            @assert dist_matrix[i, j] ≈ distance(points[i, :], points[j, :])
        end
    end
end
γₛ, step_range = omni_variogram(points, values, st, tol)

# Example usage
v = [1.0, 2.0, 3.0]
p = [4.0, 5.0, 6.0]
direction_vector = unit_vector_crossing_point(v, p)
println(direction_vector)

# Example usage
a = [1.0, 2.0]
v = [1.0, 0.0]
p = [4.0, 5.0]
projection_point, prj_distance, dir_distance = project_point_on_line(a, v, p)
println(projection_point)
println(prj_distance)
println(dir_distance)
using Plots
all_points = vcat(p', a', projection_point')

plt = scatter(all_points[:,1], all_points[:,2], label=["P" "A" "Projection Point"], xlabel="X", ylabel="Y", zlabel="Z", title="Projection Point on Line")