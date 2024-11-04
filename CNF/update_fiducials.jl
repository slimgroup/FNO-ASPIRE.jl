import Pkg; Pkg.instantiate()
using LinearAlgebra, Random, Statistics
using ImageQualityIndexes
using Images, BSON, JLD2
using FFTW
using Augmentor
using Random
using DrWatson
using InvertibleNetworks, Flux, UNet
using PyPlot, SlimPlotting

Random.seed!(2023)

include("utils.jl")
include("../config.jl")

using .Config

T = Float32
device = gpu
num_post_samples = 64
batch_size = 8

params = Config.get_parameters()

offsets = params["n_offsets"]
down_rate = params["down_rate"]
nx = params["nx"] ÷ 2
nz = params["nz"] ÷ 2

j, nsamples, ntrain, epochs = parse.(Int, ARGS[1:4])

net_path = "weights/$j/CNF/K=9_L=3_e=$(epochs)_n_hidden=64_n_train=$(ntrain)_unet_lev=4.bson"
G = load_trained_network(net_path, offsets);

x0 = zeros(Float32, nx, nz, 1, nsamples);
@load "data/cigs_iteration_j=$j.jld2" CIGs;

Y = CIGs[:, :, :, 1, :];
Y = permutedims(Y, [2, 3, 1, 4]);

for idx in 1:nsamples
    y_hat = Y[:, :, :, idx:idx];
    x_temp = zeros(nx, nz, 1, 1);
    X_post = posterior_sampler(G, y_hat, x_temp; device=device, num_samples=num_post_samples, batch_size=batch_size) |> cpu
    
    x0[:, :, 1:1, idx:idx] = mean(X_post, dims=4)
end

# plot_velocity_model(x0[:, :, 1, 1]', "velocity.png", params);

# TODO: Streamline plotting
# imshow(x0[:, :, 1, 1]', vmin=minimum(x0), vmax=maximum(x0),  interpolation="none", cmap="cet_rainbow4")
# savefig("plots/velocity.png", bbox_inches="tight", dpi=300)
# close()

JLD2.@save "data/posteriors_iteration_j=$j.jld2" x0
