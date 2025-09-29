using Interpolations, LinearAlgebra, Statistics, Distributions
using CairoMakie, FastGaussQuadrature, Parameters, NLsolve

function set_parameters(; ρ = 0.01, r = 0.04, αb = 1.0, κ = 5.0, y = 0.1, γ = 3.0,
                        a_grid = range(0, 2, 1000), 
                        θ1 = 1e-4, θ2 = 0.1, θ3 = 1e-6, 
                        h = (t -> θ1 * exp(θ2 * t) + θ3), 
                        b = (a -> αb * (κ + a)^(1 - γ) / (1 - γ)))
    # find h(t) = 10
    h_inv(h_val) = log((h_val - θ3)/θ1)/θ2 
    max_t = h_inv(10)
    # uniform grids
    # t_grid = range(0, max_t, 10000)
    # adjusting for h 
    t_grid = h_inv.(range(h(0), h(max_t), 10000))
    max_a = a_grid[end]
    da = a_grid[2] - a_grid[1]
    dt = diff(t_grid)
    u(c) = c^(1-γ) / (1-γ)
    return (; ρ, r, αb, κ, y, γ, a_grid, t_grid, h, b, max_a, da, dt, u, max_t)
end

function T(V, t, dt; p)
    @unpack ρ, r, αb, κ, y, γ, a_grid, t_grid, h, b, max_a, da, u, max_t = p 

    fV_a = vcat((V[2:end] - V[1:end-1])/da, (r * max_a + y)^(-γ))
    bV_a = vcat(y^(-γ), (V[2:end] - V[1:end-1])/da) 
    fμ_a = r * a_grid .+ y .- fV_a.^(-1/γ)
    bμ_a = r * a_grid .+ y .- bV_a.^(-1/γ) 
    If = fμ_a .> 0
    Ib = bμ_a .< 0 
    upwind_V_a = If .* fV_a .+ Ib .* bV_a .+ (1 .- If .- Ib) .* (r * a_grid .+ y).^(-γ)
    v = u.(upwind_V_a.^(-1/γ)) .+ h(t) * b.(a_grid) 
    A = Tridiagonal(min.(0, bμ_a[2:end]), max.(0, fμ_a) - min.(0, bμ_a), -max.(0,fμ_a[1:end-1]))
    S = (ρ + h(t) + 1 / dt) * Diagonal(ones(length(V))) + 1 / da * A 
    new_V = S \ v 
    return new_V
end

function solve(p)
    @unpack ρ, r, αb, κ, y, γ, a_grid, t_grid, h, b, max_a, da, dt, u, max_t = p 

    # solve for time-homogeneous solution at h(t) = h(max_t) for t ≥ max_t. 
    sol = fixedpoint(v -> T(v, max_t, 0.001; p = p), b.(a_grid))

    V = zeros(length(t_grid), length(a_grid))
    V[end, :] = sol.zero
    for (i,t) in enumerate(reverse(t_grid[1:end-1]))
        V[end-i, :] = T(V[end-i+1, :], t, dt[end-i+1]; p = p)
    end
    fV_a = hcat(diff(V, dims = 2)/da, fill((r * max_a + y)^(-γ), size(V, 1))) 
    bV_a = hcat(fill(y^(-γ), size(V, 1)), diff(V, dims = 2)/da) 
    fμ_a = r * permutedims(repeat(a_grid, 1, size(V, 1))) .+ y .- fV_a.^(-1/γ) 
    bμ_a = r * permutedims(repeat(a_grid, 1, size(V, 1))) .+ y .- bV_a.^(-1/γ) 
    If = fμ_a .> 0
    Ib = bμ_a .< 0 
    upwind_V_a = If .* fV_a .+ Ib .* bV_a .+ (1 .- If .- Ib) .* (r * permutedims(repeat(a_grid, 1, size(V, 1))) .+ y).^(-γ)
    c = (upwind_V_a).^(-1/γ)
    return V, c
end

p = set_parameters()
V, c = solve(p)

begin
    f = Figure()
    ax = Axis(f[1,1])
    for (i, t) in enumerate(p.t_grid)
        if i % 100 == 0
            lines!(ax, p.a_grid, V[i, :], color = RGBf(t / p.max_t, 0.0, 1 - t / p.max_t))
        end
    end
    f
end

begin
    f = Figure()
    ax = Axis(f[1,1])
    for (i, t) in enumerate(p.t_grid)
        if i % 100 == 0
            lines!(ax, p.a_grid, c[i, :], color = RGBf(t / p.max_t, 0.0, 1 - t / p.max_t))
        end
    end
    f
end

