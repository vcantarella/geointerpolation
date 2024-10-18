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
γₛ, step_range = universal_variogram(points, values, st, tol)