# To Plot a Prior
using DrWatson
using JLD2
using PyPlot, SlimPlotting
using Images

include("CNF/utils.jl")
include("config.jl")

params = Config.get_parameters()
down_rate = params["down_rate"]

function plot_vel(x, filename; smooth=false, gaussian=20)
    if smooth
        x = x_smooth = 1f0./Float32.(imfilter(1f0./x[:, :, 1, 1], Kernel.gaussian(gaussian)))
    end

    imshow(x, vmin=1.5, vmax=4.8,  interpolation="none", cmap="cet_rainbow4")
    savefig(filename, bbox_inches="tight", dpi=300)

    close()
end

@load "data/posteriors_iteration_j=1.jld2" x0
x = x0

for i in 1:10
    plot_vel(x0[:, :, 1, i]', "plots/posterior_1_$(i)_sample.png")
end

# @load "data/synthoseis_no_salt.jld2" data
# x = data[1:down_rate:end, 1:down_rate:end, :, :]

# # @load "data/initial.jld2" x

# plot_vel(x[:, :, 1, 1]', "plots/no_salt.png")
# plot_vel(x0[:, :, 1, 1]', "plots/no_salt_smoothed.png")
# plot_vel(x[:, :, 1, 1]', "plots/no_salt_smoothed_true.png", smooth=true)
