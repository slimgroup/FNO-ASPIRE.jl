# To Plot a Prior
include("../CNF/utils.jl")
include("../config.jl")

params = Config.get_parameters()
@load "data/posteriors_iteration_j=0.jld2" x0

imshow(x0[:, :, 1, 1]', vmin=minimum(x0), vmax=maximum(x0),  interpolation="none", cmap="cet_rainbow4")
savefig("plots/velocity.png", bbox_inches="tight", dpi=300)

close()
