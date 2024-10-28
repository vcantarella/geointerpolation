@. spherical(h, a, c) = ifelse(h<a, c*(1.5*h./a - 0.5*(h/a)^3), c)
@. exponential(h, a, c) = c*(1 - exp(-3*h/a))
@. gaussian(h, a, c) = c*(1 - exp(-3*(h/a)^2))

function spherical3D(hₕ, hᵥ, aₕ, aᵥ, c)
    h = @. sqrt((hₕ/aₕ)^2 + (hᵥ/aᵥ)^2)
    return @. ifelse(h<1,c*(1.5*h - 0.5*h^3),c)
end

function exponential3D(hₕ, hᵥ, aₕ, aᵥ, c)
    h = @. sqrt((hₕ/aₕ)^2 + (hᵥ/aᵥ)^2)
    return @. c*(1 - exp(-3*h))
end

function gaussian3D(hₕ, hᵥ, aₕ, aᵥ, c)
    h = @. sqrt((hₕ/aₕ)^2 + (hᵥ/aᵥ)^2)
    return @. c*(1 - exp(-3* h ^2))
end
