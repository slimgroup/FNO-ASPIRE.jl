import Pkg; Pkg.instantiate()
using ParametricDFNOs.DFNO_3D
using DrWatson
using MPI
using CUDA

include("utils.jl")
include("../config.jl")

using .Utils
using .Config

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

CUDA.device!(rank % 4)
partition = [1,pe_count]

nc_lift = 32
nblocks = 4
mx, mz, mh = 36, 36, 1

nbatch, epochs, j, ntrain, nvalid = 8, 2, 1, 16, 8
nbatch, epochs, j, ntrain, nvalid = parse.(Int, ARGS[1:5])

params = Config.get_parameters()

offsets = params["n_offsets"]
down_rate = params["down_rate"]
nx = params["nx"] ÷ 2
nz = params["nz"] ÷ 2

use_nz = false
labels = @strdict use_nz offsets nbatch epochs ntrain nvalid j

nc_in = use_nz ? 7 : offsets + 1 + 1 + 4 # offsets + 2 velocity models + indices
nc_out = use_nz ? 1 : offsets

nh = use_nz ? offsets : 1
mh = use_nz ? mh : 1

@info "Initializing model..."

@assert MPI.Comm_size(comm) == prod(partition)
modelConfig = DFNO_3D.ModelConfig(nc_in=nc_in, nc_lift=nc_lift, nc_out=nc_out, nx=nx, ny=nz, nz=nh, nt=1, mx=mx, my=mz, mz=mh, mt=1, nblocks=nblocks, partition=partition, dtype=Float32, relu01=false)

x_path = "data/posteriors_iteration_j=$(j-1).jld2"
y_path = "data/cigs_iteration_j=$j.jld2"

x_train, y_train, x_valid, y_valid = read_velocity_cigs_offsets_as_nc(x_path, y_path, modelConfig, ntrain=ntrain, nvalid=nvalid)

# x_train = reshape(x_train, nc_in, nx, nz, :)
# y_train = reshape(y_train, nc_out, nx, nz, :)
# x_valid = reshape(x_valid, nc_in, nx, nz, :)
# y_valid = reshape(y_valid, nc_out, nx, nz, :)

@info "Loaded data..."

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

# # To train from a checkpoint
# filename = "mt=25_mx=10_my=10_mz=10_nblocks=20_nc_in=5_nc_lift=20_nc_mid=128_nc_out=1_nd=20_nt=51_nx=20_ny=20_nz=20_p=8.jld2"
# DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)

trainConfig = DFNO_3D.TrainConfig(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train,
    x_valid=nvalid == 0 ? x_train : x_valid,
    y_valid=nvalid == 0 ? y_train : y_valid,
    plot_every=5,
    nbatch=nbatch
)

DFNO_3D.train!(trainConfig, model, θ, plotEval=Utils.plot_cig_eval_wrapper(use_nz, labels=labels))

MPI.Finalize()
