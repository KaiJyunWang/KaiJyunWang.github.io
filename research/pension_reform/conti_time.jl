using Interpolations, LinearAlgebra, Statistics, Distributions
using CairoMakie, FastGaussQuadrature, Parameters

function set_parameters(; r = 0.3, σ = 0.1, ρ = 0.1, αe = 0.94,
                        αb = 10.0, κ = 5.0, γ = 3.0, λy = 0.01,
                        λe = 0.02, λu = 0.01, yf = -8,
                        max_t = 140, max_a = 20, min_y = -8,
                        max_y = 1, μ = (t -> -2e-5 * t^2 + 1e-3 * t),
                        h = (t -> 1e-4 * exp(0.1 * t) + 1e-6),
                        b = (a -> αb / (1 - γ) * (κ + a)^(1 - γ))
                        )
        t_grid = range(0, max_t, length = 2000) 
        a_grid = range(0, max_a, length = 500)
        y_grid = range(min_y, max_y, length = 50)
        dt = t_grid[2] - t_grid[1]
        da = a_grid[2] - a_grid[1]
        e, w = gausshermite(30, normalize = true)
        exp_a_grid = repeat(a_grid, 1, length(e))
        exp_e_grid = permutedims(repeat(e, 1, length(a_grid)))
        exp_w = permutedims(repeat(w, 1, length(a_grid)))
    return (; r, σ, ρ, αe, αb, κ, γ, λy, λe, λu, yf, max_t, max_a, min_y, max_y, μ, h, b, t_grid, a_grid, y_grid, dt, da, e, w, exp_a_grid, exp_e_grid, exp_w)
end

p = set_parameters()

# quasi implicit scheme
function T(Ve, Vu, t; p = p)
    @unpack r, σ, ρ, αe, αb, κ, γ, λy, λe, λu, yf, max_t, max_a, min_y, max_y, μ, h, b, t_grid, a_grid, y_grid, dt, da, e, w, exp_a_grid, exp_e_grid, exp_w = p

    new_Ve = similar(Ve)
    new_Vu = similar(Vu)
    itp_Ve = linear_interpolation((a_grid, y_grid), Ve)
    
    for (j, y) in enumerate(y_grid)
        fVe_a = max.(vcat((Ve[2:end, j] - Ve[1:end-1, j])/da, (Ve[end, j] - Ve[end-1, j])/da), 1e-50)
        bVe_a = max.(vcat(αe * (κ + exp(y))^(-γ), (Ve[2:end, j] - Ve[1:end-1, j])/da), 1e-50)
        Phi_integral = dropdims(sum(itp_Ve.(exp_a_grid, clamp.(y .+ μ(t) .+ σ * exp_e_grid, min_y, max_y)) .* exp_w, dims = 2), dims = 2) 
        v = (h(t) * b.(a_grid) + λy * Phi_integral + λe * max.(Vu, Ve[:, j])) * dt + Ve[:, j]
        fμa = r * a_grid .+ exp(y) .+ αe^(1/γ) * γ / (1 - γ) * (fVe_a).^(-1/γ)
        bμa = r * a_grid .+ exp(y) .+ αe^(1/γ) * γ / (1 - γ) * (bVe_a).^(-1/γ)
        A = Tridiagonal(- min.(bμa[2:end], 0), - max.(fμa, 0) + min.(bμa, 0), max.(fμa[1:end-1], 0))
        S = Diagonal(fill((ρ + h(t) + λy + λe) * dt + 1, length(a_grid))) - dt / da * A
        new_Ve[:, j] = S \ v
    end
    fVu_a = max.(vcat((Vu[2:end] - Vu[1:end-1]) / da, (Vu[end] - Vu[end-1]) / da), 1e-50)
    bVu_a = max.(vcat((κ + exp(yf))^(-γ), (Vu[2:end] - Vu[1:end-1]) / da), 1e-50)
    F_integral = dropdims(sum(max.(itp_Ve.(exp_a_grid, clamp.(-8 .+ σ * exp_e_grid, min_y, max_y)), repeat(Vu, 1, length(e))) .* exp_w, dims = 2), dims = 2) 
    v = (h(t) * b.(a_grid) + λu * F_integral) * dt + Vu
    fμa = r * a_grid .+ exp(yf) .+ γ / (1 - γ) * (fVu_a).^(-1/γ)
    bμa = r * a_grid .+ exp(yf) .+ γ / (1 - γ) * (bVu_a).^(-1/γ)
    A = Tridiagonal(- min.(bμa[2:end], 0), - max.(fμa, 0) + min.(bμa, 0), max.(fμa[1:end-1], 0))
    S = Diagonal(fill((ρ + h(t) + λu) * dt + 1, length(a_grid))) - dt / da * A
    new_Vu = S \ v
    return new_Ve, new_Vu
end

function solve(; p = p)
    @unpack r, σ, ρ, αe, αb, κ, γ, λy, λe, λu, yf, max_t, max_a, max_y, μ, h, b, t_grid, a_grid, y_grid, dt, da, e, w, exp_a_grid, exp_e_grid, exp_w = p

    Ve = zeros(length(t_grid), length(a_grid), length(y_grid))
    Vu = zeros(length(t_grid), length(a_grid))
    Ve[end, :, :] = repeat(b.(a_grid), 1, length(y_grid))
    Vu[end, :] = b.(a_grid)

    for (i, t) in enumerate(reverse(t_grid[1:end-1]))
        Ve[end-i, :, :], Vu[end-i, :] = T(Ve[end-i+1, :, :], Vu[end-i+1, :], t; p = p)
    end
    return Ve, Vu
end

Ve, Vu = solve()

begin
    f = Figure(size = (800, 400))
    ax1 = Axis(f[1,1], title = "Ve")
    ax2 = Axis(f[1,2], title = "Vu")
    
    for (i, t) in enumerate(p.t_grid)
        if i % 50 == 0
            lines!(ax1, p.a_grid, Ve[i, :, 40], color = RGBf(0.1 + 0.9* i / length(p.t_grid), 0.0, 0.9 - 0.9* i / length(p.t_grid)))
            lines!(ax2, p.a_grid, Vu[i, :], color = RGBf(0.1 + 0.9* i / length(p.t_grid), 0.0, 0.9 - 0.9* i / length(p.t_grid)))
        end
    end
    
    #lines!(ax1, p.a_grid, Ve[600, :, 10], color = :teal, linewidth = 2)
    #lines!(ax2, p.a_grid, Vu[600, :], color = :brown, linewidth = 2)
    f
end

begin
    f = Figure()
    ax = Axis(f[1,1], title = "Ve - Vu") 
    for (i, t) in enumerate(p.t_grid)
        if i % 100 == 0
            lines!(ax, exp.(p.y_grid), Ve[i, 10, :] .- Vu[i, 10], color = RGBf(0.1 + 0.9* i / length(p.t_grid), 0.0, 0.9 - 0.9* i / length(p.t_grid)))
        end
    end
    f
end



