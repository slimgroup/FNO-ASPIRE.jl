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

offset, use_fno = parse.(Int, ARGS[1:2])

# TODO: Make test Files, so its easy for FNO to read
@load "data/test/initial.jld2" x y
@load "data/test/cigs_iteration_j=0.jld2" CIGs
@load "data/test/posteriors_iteration_j=0.jld2" x0 CIG0

y_obs = y[offset + 1]
x0 = x0[offset + 1]
x = x[:, :, 1:1, ]

function migrate(sim::TrueSimulator, x_path, y_path)

end

function migrate(sim::FNO_Simulator, x_path, y_path)
    # TODO: MOve to FNO_Simulator
    @assert MPI.Comm_size(comm) == prod(partition)
    modelConfig = DFNO_3D.ModelConfig(nc_in=nc_in, nc_lift=nc_lift, nc_out=nc_out, nx=nx, ny=nz, nz=nh, nt=1, mx=mx, my=mz, mz=mh, mt=1, nblocks=nblocks, partition=partition, dtype=Float32, relu01=false)

    model = DFNO_3D.Model(modelConfig)
    θ = DFNO_3D.initModel(model)

end

for j = 1:J
    x_path = "data/test/posteriors_iteration_j=$(j-1).jld2"
    y_path = "data/test/cigs_iteration_j=$j.jld2"
    _, _, x_valid, _ = read_velocity_cigs_offsets_as_nc(x_path, y_path, modelConfig, ntrain=offset, nvalid=nvalid)

    filename = "mt=25_mx=10_my=10_mz=10_nblocks=20_nc_in=5_nc_lift=20_nc_mid=128_nc_out=1_nd=20_nt=51_nx=20_ny=20_nz=20_p=8.jld2"
    DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)

    x_valid = reshape(x_valid, nc_in, nx, nz, :)
    yhat = migrate(simulator, x_path, y_path)

    post = CNF-1(z; yhat)


    # TODO: Calculate summary statistic yhat for yobs around fiducials x0

    # TODO: Update fiducials x0 by averaging posterior samples conditioned on yhat
    
end
