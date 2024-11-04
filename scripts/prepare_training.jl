using JLD2
using Images

include("../config.jl")

params = Config.get_parameters()
down_rate = params["down_rate"]

# First set of posteriors/fiducials/background-models
@load "data/synthoseis_no_salt.jld2" data;
x0 = data[1:down_rate:end, 1:down_rate:end, :, :]
println(size(x0))

for sample in 1:size(x0, 4)
    x0[:, :, 1, sample] = 1f0./Float32.(imfilter(1f0./x0[:, :, 1, sample], Kernel.gaussian(20)))
end

JLD2.@save "data/posteriors_iteration_j=0.jld2" x0;

# Ground truth
@load "data/synthoseis_salt.jld2" data;
x = data[1:down_rate:end, 1:down_rate:end, :, :]
println(size(x))
JLD2.@save "data/initial.jld2" x;
