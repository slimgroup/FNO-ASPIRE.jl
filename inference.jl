# Inference ASPIRE

using JLD2
using ParametricDFNOs.DFNO_3D
using DrWatson
using MPI
using CUDA

include("config.jl")

using .Config

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

CUDA.device!(rank % 4)
partition = [1, pe_count]

nc_lift = 32
nblocks = 4
mx, mz, mh = 36, 36, 1

params = Config.get_parameters()

nh = 1
nx = params["nx"]
nz = params["nz"]
offsets = params["n_offsets"]

nc_in = offsets + 1 + 1 + 4 # offsets + 2 velocity models + indices
nc_out = offsets

@assert MPI.Comm_size(comm) == prod(partition)
modelConfig = DFNO_3D.ModelConfig(nc_in=nc_in, nc_lift=nc_lift, nc_out=nc_out, nx=nx, ny=nz, nz=nh, nt=1, mx=mx, my=mz, mz=mh, mt=1, nblocks=nblocks, partition=partition, dtype=Float32, relu01=false)

y_obs = 
x0 = 

for j = 1:J
    x_path = "data/posteriors_iteration_j=$(j-1).jld2"

    # TODO: Calculate summary statistic yhat for yobs around fiducials x0

    # TODO: Update fiducials x0 by averaging posterior samples conditioned on yhat
    
end
