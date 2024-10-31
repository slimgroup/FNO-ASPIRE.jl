using JLD2

include("../config.jl")

params = Config.get_parameters()

j, nsamples = parse.(Int, ARGS[1:2])
n_offsets = params["n_offsets"]
down_rate = params["down_rate"]
nx = params["nx"] ÷ down_rate
nz = params["nz"] ÷ down_rate

CIGs = zeros(Float32, n_offsets, nx, nz, 1, nsamples)

for i in 1:nsamples
    @load "data/$j/sample_$i.jld2" rtm
    CIGs[:, :, :, 1, i] = rtm
end

JLD2.@save "data/cigs_iteration_j=$j.jld2" CIGs

CIG0 = CIGs[:, :, :, 1:1, 1:1]

@load "data/posteriors_iteration_j=$(j-1).jld2" x0
JLD2.@save "data/posteriors_iteration_j=$(j-1).jld2" x0 CIG0
