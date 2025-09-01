using Interpolations, LinearAlgebra, Statistics, Distributions
using CairoMakie, FastGaussQuadrature, Parameters



function set_parameters(; r = 0.02, σ = 0.1, ρ = 0.02, αe = 0.94,
                        αb = 1.0, κ = 1.0, γ = 2.0, λy = 0.1,
                        λe = 0.2, λu = 0.1, yf = 0.001,
                        max_t = 100f0, max_a = 30f0,
                        max_y = 5f0, μ = (t -> -2e-5 * t^2 + 1e-3 * t),
                        h = (t -> 1e-4 * exp(0.1 * t) + 1e-6),
                        b = (a -> αb / (1 - γ) * (κ + a)^(1 - γ))
                        )
        t_grid = range(0, max_t, length = 10000)
        a_grid = range(0, max_a, length = 100)
        y_grid = range(-2, 2, length = 100)
        dt = t_grid[2] - t_grid[1]
        da = a_grid[2] - a_grid[1]
        e, w = gausshermite(30, normalize = true)
        exp_a_grid = repeat(a_grid, 1, length(e))
        exp_e_grid = permutedims(repeat(e, 1, length(a_grid)))
        exp_w = permutedims(repeat(w, 1, length(a_grid)))
    return (; r, σ, ρ, αe, αb, κ, γ, λy, λe, λu, yf, max_t, max_a, max_y, μ, h, b, t_grid, a_grid, y_grid, dt, da, e, w, exp_a_grid, exp_e_grid, exp_w)
end

p = set_parameters()

function T(Ve, Vu, t; p = p)
    @unpack r, σ, ρ, αe, αb, κ, γ, λy, λe, λu, yf, max_t, max_a, max_y, μ, h, b, t_grid, a_grid, y_grid, dt, da, e, w, exp_a_grid, exp_e_grid, exp_w = p

    new_Ve = similar(Ve)
    new_Vu = similar(Vu)
    itp_Ve = linear_interpolation((a_grid, y_grid), Ve, extrapolation_bc = Line())

    for (j, y) in enumerate(y_grid)
        bVe_a = vcat(max((Ve[2,j] - Ve[1,j])/da, αe * exp(-γ * y)), (Ve[2:end, j] - Ve[1:end-1, j]) / da)
        fVe_a = vcat(max((Ve[2,j] - Ve[1,j])/da, αe * exp(-γ * y)), (Ve[3:end, j] - Ve[2:end-1, j]) / da, (Ve[end, j] - Ve[end-1, j]) / da)
        ve = (h(t) * b.(a_grid) + λy * dropdims(sum(itp_Ve.(exp_a_grid, y .+ μ(t) .+ σ * exp_e_grid) .* exp_w, dims = 2), dims = 2) + λe * max.(Ve[:,j], Vu)) * dt + Ve[:,j]
        fμea = r * a_grid .+ exp(y) .+ γ / (1 - γ) * (αe ./ fVe_a).^(1/γ)
        bμea = r * a_grid .+ exp(y) .+ γ / (1 - γ) * (αe ./ bVe_a).^(1/γ)
        Se = (1 + (ρ + h(t) + λy + λe) * dt) * Diagonal(ones(length(a_grid))) + dt / da * Tridiagonal(min.(0, bμea[2:end]), max.(fμea, 0) - min.(bμea, 0), -max.(fμea[1:end-1], 0))
        new_Ve[:, j] = Se \ ve
    end
    bVu_a = vcat(max((Vu[2] - Vu[1]) / da, yf^(-γ)), (Vu[2:end] - Vu[1:end-1]) / da)
    fVu_a = vcat(max((Vu[2] - Vu[1]) / da, yf^(-γ)), (Vu[3:end] - Vu[2:end-1]) / da, (Vu[end] - Vu[end-1]) / da)
    vu = (h(t) * b.(a_grid) + λu * dropdims(sum(itp_Ve.(exp_a_grid, log(0.1) .+ σ * exp_e_grid) .* exp_w, dims = 2), dims = 2)) * dt + Vu
    bμua = r * a_grid .+ yf .+ γ / (1 - γ) * (1 ./ bVu_a).^(1/γ)
    fμua = r * a_grid .+ yf .+ γ / (1 - γ) * (1 ./ fVu_a).^(1/γ)
    Su = (1 + (ρ + h(t) + λu) * dt) * Diagonal(ones(length(a_grid))) + dt / da * Tridiagonal(min.(0, bμua[2:end]), max.(fμua, 0) - min.(bμua, 0), -max.(fμua[1:end-1], 0))
    new_Vu = Su \ vu
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
    f = Figure()
    ax1 = Axis(f[1,1], title = "Ve")
    ax2 = Axis(f[1,2], title = "Vu")
    
    
    for (i, t) in enumerate(p.t_grid)
        if i % 10000 == 1
            lines!(ax1, p.a_grid, Ve[i, :, 1], color = RGBf(0.1 + i / length(p.t_grid), 0.0, 0.9 - i / length(p.t_grid)))
            lines!(ax2, p.a_grid, Vu[i, :], color = RGBf(0.1 + i / length(p.t_grid), 0.0, 0.9 - i / length(p.t_grid)))
        end
    end
    
    #lines!(ax1, p.a_grid, Ve[1, :, 1], color = :teal, linewidth = 2)
    #lines!(ax2, p.a_grid, Vu[1, :], color = :brown, linewidth = 2)
    f
end

