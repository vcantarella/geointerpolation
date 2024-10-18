@. spherical(h, a, c) = ifelse(h<a, c*(1.5*h./a - 0.5*(h/a)^3), c)
@. exponential(h, a, c) = c*(1 - exp(-3*h/a))
@. gaussian(h, a, c) = c*(1 - exp(-3*(h/a)^2))