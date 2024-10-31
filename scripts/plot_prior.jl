# To Plot a Prior
using DrWatson
using JLD2
using PyPlot, SlimPlotting

include("CNF/utils.jl")
include("config.jl")

params = Config.get_parameters()
@load "data/posteriors_iteration_j=0.jld2" x0

# @load "data/initial.jld2" x
# x0 = x

gaussian = 20

imshow(x0[:, :, 1, 1]', vmin=minimum(x0), vmax=maximum(x0),  interpolation="none", cmap="cet_rainbow4")
savefig("plots/velocity_1.png", bbox_inches="tight", dpi=300)

close()

x_smooth = 1f0./Float32.(imfilter(1f0./x0[:, :, 1, 1], Kernel.gaussian(gaussian)))

imshow(x_smooth', vmin=minimum(x0), vmax=maximum(x0),  interpolation="none", cmap="cet_rainbow4")
savefig("plots/velocity_smooth_1.png", bbox_inches="tight", dpi=300)

close()
