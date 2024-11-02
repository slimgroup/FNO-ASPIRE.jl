using ParametricDFNOs.DFNO_3D
using DrWatson
using MPI
using CUDA
using JLD2
using HDF5

include("FNO/utils.jl")
include("config.jl")

using .Utils
using .Config

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

CUDA.device!(rank % 4)
nvalid, offset_is_nz = parse.(Int, ARGS[1:2])

filename = "/pscratch/sd/r/richardr/FNO_CIG/FNO-CIG/weights/DFNO_3D/mt=1_mx=36_my=36_mz=1_nblocks=4_nc_in=27_nc_lift=32_nc_mid=128_nc_out=21_nd=256_nt=1_ntrain=500_nvalid=10_nx=256_ny=128_nz=1_p=1.jld2"
dataset_path = "/pscratch/sd/r/richardr/FNO_CIG/FNO-CIG/results/concatenated_data_99_quantile_scaled.jld2"

file = filename # projectdir("weights", "DFNO_3D", filename)
file = load(file)

nblocks = file["nblocks"]
mx = file["mx"] 
mz = file["my"]
mh = file["mz"] 
nd = file["nd"] 
nc_lift = file["nc_lift"]
ntrain = file["ntrain"]
# nvalid = file["nvalid"]

partition = [1,pe_count]
params = Config.get_parameters()

down_rate = params["down_rate"]
nx = params["nx"] ÷ down_rate
nz = params["nz"] ÷ down_rate
offsets = params["n_offsets"]

use_nz = Bool(offset_is_nz)
labels = @strdict use_nz offsets

nc_in = use_nz ? 7 : offsets + 1 + 1 + 4 # offsets + 2 velocity models + indices
nc_out = use_nz ? 1 : offsets

nh = use_nz ? offsets : 1
mh = use_nz ? mh : 1 

@info "Initializing model..."
@assert MPI.Comm_size(comm) == prod(partition)

function read_velocity_cigs_offsets_as_nc_old(path::String, modelConfig::DFNO_3D.ModelConfig; ntrain::Int, nvalid::Int)

    params = Config.get_parameters()
    offset_start = params["read_offset_start"]
    offsets = params["n_offsets"]
    total = params["n_total"]

    # Assumption that x is (nx, nz, 1, n). x0 is (nx, nz). CIG0 is (nh, nx, nz). CIG is (nh, nx, nz, 1, n)
    function read_x_tensor_helper(file_name, key, indices; flip=false, init_index=1)
        data = nothing
        h5open(file_name, "r") do file
            x_data = file[key[1]]
            cigs_data = file[key[2]]

            # Read proper indices of x and x0. NOTE: Disclude 3 because no z = 1 for background
            x = x_data[indices[1], indices[2], 1, indices[4]]
            x0 = x_data[indices[1], indices[2], 1, init_index]
            cig0 = cigs_data[offset_start:offset_start+offsets-1, indices[1], indices[2], 1, init_index]

            # Reshape to prepare for augmentation
            x = reshape(x, :, map(range -> length(range), indices[1:4])...)
            x0 = reshape(x0, :, map(range -> length(range), indices[1:3])..., 1)
            cig0 = reshape(cig0, :, map(range -> length(range), indices[1:3])..., 1)

            x0 = repeat(x0, outer=[1, 1, 1, 1, length(indices[4])])
            cig0 = repeat(cig0, outer=[1, 1, 1, 1, length(indices[4])])

            # Concat along dimension 1
            data = cat(x, x0, cig0, dims=1)
            flip && (data = reverse(data, dims=1))
        end

        # data_channels * nx * ny * nz * nt * n = data_channels * nx * nz * nh * 1 * n
        return data
    end

    function read_x_tensor(file_name, key, indices)
        requested = length(indices[4])
        @assert (requested - nvalid) % (total - nvalid) == 0

        nrounds = (requested - nvalid) ÷ (total - nvalid)
        augmented_data = []

        samples_per_round = (requested - nvalid) ÷ nrounds

        for round in 1:nrounds
            Random.seed!(round % 2)
            
            init_index = rand(1:samples_per_round)
            new_indices = [indices[1:3]..., 1:samples_per_round]
    
            data = read_x_tensor_helper(file_name, key, new_indices, flip=(round <= nrounds ÷ 2), init_index=init_index)

            if round == 1
                augmented_data = data
            else
                augmented_data = cat(augmented_data, data, dims=ndims(augmented_data))
            end
        end

        Random.seed!(1 % 2) # Simulate round 1
        init_index = rand(1:samples_per_round+nvalid)

        new_indices = [indices[1:3]..., 1:samples_per_round+nvalid]
        validation = read_x_tensor_helper(file_name, key, new_indices, flip=false, init_index=init_index)

        augmented_data = cat(augmented_data, validation[:, :, :, :, end-nvalid+1:end], dims=ndims(augmented_data))
        return augmented_data
    end
    
    function read_y_tensor_helper(file_name, key, indices; flip=false)
        data = nothing
        h5open(file_name, "r") do file
            cigs_data = file[key]
            data = cigs_data[offset_start:offset_start+offsets-1, indices[1], indices[2], 1, indices[5]] # first dim is offsets as channel, dim 4 which is t = 1:1
        end

        # channels * nx * ny * nz * nt * n = channels * nx * nz * nh * 1 * n
        data = reshape(data, :, map(range -> length(range), indices[1:5])...)
        flip && (data = reverse(data, dims=2))
        return data
    end

    function read_y_tensor(file_name, key, indices)
        requested = length(indices[5])
        @assert (requested - nvalid) % (total - nvalid) == 0

        nrounds = (requested - nvalid) ÷ (total - nvalid)
        augmented_data = []

        samples_per_round = (requested - nvalid) ÷ nrounds

        for round in 1:nrounds
            Random.seed!(round % 2)
            
            init_index = rand(1:samples_per_round)

            new_indices = [indices[1:4]..., 1:samples_per_round]
            data = read_y_tensor_helper(file_name, key, new_indices, flip=(round <= nrounds ÷ 2))

            if round == 1
                augmented_data = data
            else
                augmented_data = cat(augmented_data, data, dims=ndims(augmented_data))
            end
        end

        Random.seed!(1 % 2) # Simulate round 1
        init_index = rand(1:samples_per_round+nvalid)

        new_indices = [indices[1:4]..., 1:samples_per_round+nvalid]
        validation = read_y_tensor_helper(file_name, key, new_indices, flip=false)

        augmented_data = cat(augmented_data, validation[:, :, :, :, :, end-nvalid+1:end], dims=ndims(augmented_data))
        return augmented_data
    end

    dataConfig = DFNO_3D.DataConfig(modelConfig=modelConfig, 
                                    ntrain=ntrain, 
                                    nvalid=nvalid, 
                                    x_file=path,
                                    y_file=path,
                                    x_key=["xs", "cigs"],
                                    y_key="cigs")

    return DFNO_3D.loadDistData(dataConfig, dist_read_x_tensor=read_x_tensor_helper, dist_read_y_tensor=read_y_tensor_helper)
end

modelConfig = DFNO_3D.ModelConfig(nc_in=nc_in, nc_lift=nc_lift, nc_out=nc_out, nx=nx, ny=nz, nz=nh, nt=1, mx=mx, my=mz, mz=mh, mt=1, nblocks=nblocks, partition=partition, dtype=Float32, relu01=false)
_, _, x_valid, y_valid = read_velocity_cigs_offsets_as_nc_old(dataset_path, modelConfig, ntrain=ntrain, nvalid=nvalid)

@info "Loaded data..."

model = DFNO_3D.Model(modelConfig)
θ = DFNO_3D.initModel(model)

DFNO_3D.loadWeights!(θ, filename, "θ_save", partition)
labels = @strdict nx nz nblocks mx mz mh ntrain nvalid nc_lift offset_is_nz

y_predict = DFNO_3D.forward(model, θ, x_valid) |> cpu
plot_cig_eval(modelConfig, x_valid, y_valid, y_predict, use_nz=use_nz, additional=labels)

MPI.Finalize()
