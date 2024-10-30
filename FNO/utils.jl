module Utils

include("../config.jl")

using HDF5
using DrWatson
using PyPlot
using Images
using ArgParse
using Statistics
using LinearAlgebra
using ParametricDFNOs.DFNO_3D
using FFTW
using Random
using .Config

export create_wavelet, create_geometry, generate_noise, parse_commandline, ContJitter, plot_velocity_model, plot_cig, read_velocity_cigs_offsets_as_nc, read_velocity_cigs_offsets_as_nz, plot_cig_eval, plot_cig_eval_wrapper, plotLoss, plot_validation_cig, get_energy_cig, plot_optimize_eval, plot_optimize_loss, plot_control_histogram, plot_velocity_cigs, plot_rankings, plot_cig_diffs

function plot_rankings(true_ranking, test_ranking)
    fig = figure(figsize=(8, 12))

    PyPlot.rc("figure", titlesize=8)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
    PyPlot.rc("axes", labelsize=8)
    PyPlot.rc("axes", titlesize=8)

    subplot(1,1,1)
    plot(true_ranking, test_ranking, "o-")
    xlabel("true ranking")
    ylabel("test ranking")
    title("Comparison of rankings")
    tight_layout();

    savefig(joinpath("plots", "DFNO_3D", "ranking_comparison.png"), bbox_inches="tight", dpi=300)
    close(fig);
end

function plot_control_histogram(hist1, hist2; additional=Dict{String,Any}())

    PyPlot.rc("figure", titlesize=10)
    PyPlot.rc("font", family="serif"); 
    PyPlot.rc("xtick", labelsize=10); 
    PyPlot.rc("ytick", labelsize=10)
    PyPlot.rc("axes", labelsize=10)
    PyPlot.rc("axes", titlesize=10)

    fig, axs = plt.subplots(2, 2)

    # Plot histogram 1
    axs[1, 1].hist(hist1, bins=20, color="blue", alpha=0.7)
    axs[1, 1].set_title("Histogram of True CIGs")
    axs[1, 1].set_xlabel("Value")
    axs[1, 1].set_ylabel("Frequency")

    # Plot histogram 2
    axs[1, 2].hist(hist2, bins=20, color="green", alpha=0.7)
    axs[1, 2].set_title("Histogram of Predicted CIGs")
    axs[1, 2].set_xlabel("Value")
    axs[1, 2].set_ylabel("Frequency")

    # Plot combined histograms
    axs[2, 1].hist([hist1, hist2], bins=20, color=["blue", "green"], alpha=0.7, label=["hist1", "hist2"])
    axs[2, 1].set_title("Combined Histogram")
    axs[2, 1].set_xlabel("Value")
    axs[2, 1].set_ylabel("Frequency")
    axs[2, 1].legend()

    axs[2, 2].hist([hist1, hist2], bins=20, color=["blue", "green"], alpha=0.7, stacked=true)
    axs[2, 2].set_title("Stacked Histogram")
    axs[2, 2].set_xlabel("Value")
    axs[2, 2].set_ylabel("Frequency")

    tight_layout()

    savefig(joinpath("plots", "DFNO_3D", savename(additional; digits=6) * "_DFNO_CIG_histogram.png"), bbox_inches="tight", dpi=300)
    close(fig)
end

function plot_cig_diffs(modelConfig, y, ŷ; trainConfig=nothing, use_nz=false, additional=Dict{String,Any}())
    params = Config.get_parameters()

    offset_start = params["offset_start"]
    offset_end = params["offset_end"]
    d = params["d"]
    n = [modelConfig.nx, modelConfig.ny]
    n_offsets = params["n_offsets"]
    down_rate = params["down_rate"]

    # Downsample
    d = d.*down_rate

    y_plot = reshape(y, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))
    y_predict = reshape(ŷ, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))

    num_samples = size(y_plot, 6)
    
    PyPlot.rc("figure", titlesize=40)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=40); PyPlot.rc("ytick", labelsize=40)
    PyPlot.rc("axes", labelsize=40)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=40)     # Default fontsize for titles

    fig, axs = subplots(num_samples*2, 6, figsize=(50, num_samples*10), gridspec_kw = Dict("width_ratios" => [4, 1, 4, 1, 4, 1], "height_ratios" => vcat([[1, 3] for i in 1:num_samples]...)))

    for i in 1:num_samples
        output_CIG = use_nz ? y_predict[1, 1, :, :, :, i] : permutedims(y_predict[:, 1, :, :, 1, i], [2, 3, 1])
        true_CIG = use_nz ? y_plot[1, 1, :, :, :, i] : permutedims(y_plot[:, 1, :, :, 1, i], [2, 3, 1])

        plot_cig_helper(true_CIG, n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 1:2])
        plot_cig_helper(output_CIG, n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 3:4])
        plot_cig_helper(5f0 .* abs.(true_CIG - output_CIG), n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 5:6])

        if i == 1
            axs[1, 1].set_title("True CIG")
            axs[1, 3].set_title("Predicted CIG")
            axs[1, 5].set_title("5X Diff")
        end
    end

    suptitle("Model and CIG Comparison", fontsize=20)
    
    figname = _getFigname(trainConfig, additional)
    tight_layout()

    savefig(joinpath("plots", "DFNO_3D", savename(figname; digits=6) * "_DFNO_CIG_fitting.png"), bbox_inches="tight", dpi=300)
    close(fig)
end

function plot_velocity_cigs(modelConfig, x, y; additional=Dict{String,Any}())

end

function plot_cig_eval_wrapper(use_nz; labels=Dict{String,Any}())
    function _wrapper(modelConfig, x_plot, y_plot, y_predict; trainConfig, additional=labels)
        return plot_cig_eval(modelConfig, x_plot, y_plot, y_predict, use_nz=use_nz, trainConfig=trainConfig, additional=additional)
    end
    return _wrapper
end

function plot_cig_eval(modelConfig, x_plot, y_plot, y_predict; use_nz=false, trainConfig=nothing, additional=Dict{String,Any}())
    params = Config.get_parameters()

    offset_start = params["offset_start"]
    offset_end = params["offset_end"]
    d = params["d"]
    n = [modelConfig.nx, modelConfig.ny]
    n_offsets = params["n_offsets"]
    down_rate = params["down_rate"]

    # Downsample
    d = d.*down_rate

    # Reshape the data to fit the model configuration
    x_plot = reshape(x_plot, (modelConfig.nc_in, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))
    y_plot = reshape(y_plot, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))
    y_predict = reshape(y_predict, (modelConfig.nc_out, modelConfig.nt, modelConfig.nx, modelConfig.ny, modelConfig.nz, :))

    num_samples = size(x_plot, 6)
    
    PyPlot.rc("figure", titlesize=40)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=40); PyPlot.rc("ytick", labelsize=40)
    PyPlot.rc("axes", labelsize=40)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=40)     # Default fontsize for titles

    fig, axs = subplots(num_samples*2, 13, figsize=(110, num_samples*10), gridspec_kw = Dict("width_ratios" => [4, 4, 1, 4, 4, 1, 4, 1, 4, 1, 4, 1, 4], "height_ratios" => vcat([[1, 3] for i in 1:num_samples]...)))

    for i in 1:num_samples
        perturbed_model = x_plot[1, 1, :, :, 1, i]
        init_background_model = x_plot[2, 1, :, :, 1, i]
        input_CIG = use_nz ? x_plot[3, 1, :, :, :, i] : permutedims(x_plot[3:(3+n_offsets-1), 1, :, :, 1, i], [2, 3, 1])
        output_CIG = use_nz ? y_predict[1, 1, :, :, :, i] : permutedims(y_predict[:, 1, :, :, 1, i], [2, 3, 1])
        true_CIG = use_nz ? y_plot[1, 1, :, :, :, i] : permutedims(y_plot[:, 1, :, :, 1, i], [2, 3, 1])

        axs[i*2-1, 1].set_visible(false)
        axs[i*2-1, 4].set_visible(false)
        axs[i*2-1, 13].set_visible(false)

        plot_velocity_model_helper(init_background_model, n, d, axs[i*2, 1])
        plot_velocity_model_helper(perturbed_model, n, d, axs[i*2, 4])

        plot_cig_helper(input_CIG, n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 2:3])
        plot_cig_helper(true_CIG, n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 5:6])
        plot_cig_helper(output_CIG, n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 7:8])
        plot_cig_helper(5f0 .* abs.(true_CIG - output_CIG), n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 9:10])
        plot_cig_helper(abs.(fftshift(fft(true_CIG - output_CIG))), n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 11:12])

        plot_energy_vs_offset_helper(true_CIG, output_CIG, n, d, offset_start, offset_end, n_offsets, axs[i*2, 13])

        # Label columns for the first sample
        if i == 1
            axs[1, 1].set_title("Init velocity model")
            axs[1, 2].set_title("Init CIG")
            axs[1, 4].set_title("Smoothed velocity model")
            axs[1, 5].set_title("True CIG")
            axs[1, 7].set_title("Predicted CIG")
            axs[1, 9].set_title("5X Diff")
            axs[1, 11].set_title("FFT of 1X difference")
            axs[1, 13].set_title("% Energy Distribution")
        end
    end

    suptitle("Model and CIG Comparison", fontsize=20)
    
    figname = _getFigname(trainConfig, additional)
    tight_layout()

    savefig(joinpath("plots", "DFNO_3D", savename(figname; digits=6) * "_DFNO_CIG_fitting.png"), bbox_inches="tight", dpi=300)
    close(fig)
end

# NOTE: Make sure input has offsets as first dimension. Choice due to channel being the first output of DFNO
function get_energy_cig(n, n_offsets, cig; dist=false)
    cig = reshape(cig, n_offsets, :)
    center_offset = n_offsets ÷ 2 + 1

    @assert n_offsets % 2 == 1
    range = n_offsets - center_offset

    energies = [sum(abs2, fft(permutedims(cig[center_offset-distance:center_offset+distance, :], [2, 1]), [2])) for distance in 0:range]

    if dist
        reduce = ParReduce(eltype(energies))
        energies = reduce(energies)
    end

    return (energies ./ energies[end]) # Last contains the total energy
end

# NOTE: All plot functions will receive offsets as last dimension. It is their responsibility to permute
function plot_energy_vs_offset_helper(true_CIG, output_CIG, n, d, offset_start, offset_end, n_offsets, ax)

    true_CIG = permutedims(true_CIG, [3, 1, 2])
    output_CIG = permutedims(output_CIG, [3, 1, 2])

    true_energy = get_energy_cig(n, n_offsets, true_CIG)
    output_energy = get_energy_cig(n, n_offsets, output_CIG)

    x_values = LinRange(0, offset_end, length(true_energy))
    x_ticks = [x_values[1], x_values[Int(ceil(length(x_values) / 2))], x_values[end]]

    ax.plot(x_values, true_energy, label="True Energy", color="blue")
    ax.plot(x_values, output_energy, label="Output Energy", color="red")
    ax.set_xlabel("Distance from 0 offset [m]")
    ax.set_ylabel("% of Energy")
    ax.set_xticks(x_ticks)
    ax.legend()
end

function plot_velocity_model_helper(x, n, d, ax)
    # Assume that vmin and vmax are computed similarly to the plot_cig_helper function
    vmin = quantile(vec(x), 0.05)  # 5th percentile
    vmax = quantile(vec(x), 0.95)  # 95th percentile
    extentfull = (0f0, (n[1]-1)*d[1], (n[2]-1)*d[2], 0f0)
    cax = ax.imshow(x', vmin=vmin, vmax=vmax, extent=extentfull, aspect="auto")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
end

function plot_cig_helper(cig, n, d, offset_start, offset_end, n_offsets, axs)
    y = reshape(cig, n[1], n[2], n_offsets)
    
    ### X, Z position in km
    xpos = 3.6f3
    zpos = 2.7f3
    xgrid = Int(round(xpos / d[1]))
    zgrid = Int(round(zpos / d[2]))
    
    # Adjust the spacing between the plots
    subplots_adjust(hspace=0.0, wspace=0.0)
    
    vmin1, vmax1 = (-1, 1) .* quantile(abs.(vec(y[:,zgrid,:,1])), 0.99)
    vmin2, vmax2 = (-1, 1) .* quantile(abs.(vec(y[:,:,div(n_offsets,2)+1,1])), 0.88)
    vmin3, vmax3 = (-1, 1) .* quantile(abs.(vec(y[xgrid,:,:,1])), 0.999)
    sca(axs[1, 1])
    
    # Top left subplot
    axs[1, 1].imshow(y[:,zgrid,:,1]', aspect="auto", cmap="gray", interpolation="none",vmin=vmin1, vmax=vmax1,
        extent=(0f0, (n[1]-1)*d[1], offset_start, offset_end))
    axs[1, 1].set_ylabel("Offset [m]", fontsize=40)
    axs[1, 1].set_xticklabels([])
    axs[1, 1].set_xlabel("")
    hlines(y=0, colors=:b, xmin=0, xmax=(n[1]-1)*d[1], linewidth=3)
    vlines(x=xpos, colors=:b, ymin=offset_start, ymax=offset_end, linewidth=3)
    
    # Bottom left subplot
    sca(axs[2, 1])
    axs[2, 1].imshow(y[:,:,div(n_offsets,2)+1,1]', aspect="auto", cmap="gray", interpolation="none",vmin=vmin2, vmax=vmax2,
    extent=(0f0, (n[1]-1)*d[1], (n[2]-1)*d[2], 0f0))
    axs[2, 1].set_xlabel("X [m]", fontsize=40)
    axs[2, 1].set_ylabel("Z [m]", fontsize=40)
    axs[2, 1].set_xticks([0, 2000, 4000, 6000])
    axs[2, 1].set_xticklabels(["0", "2000", "4000", "6000"])
    axs[2, 1].set_yticks([1000, 2000, 3000])
    axs[2, 1].set_yticklabels(["1000", "2000", "3000"])
    
    # axs[2, 2].get_shared_x_axes().join(axs[1, 1], axs[2, 1])
    vlines(x=xpos, colors=:b, ymin=0, ymax=(n[2]-1)*d[2], linewidth=3)
    hlines(y=zpos, colors=:b, xmin=0, xmax=(n[1]-1)*d[1], linewidth=3)
    
    # Top right subplot
    axs[1, 2].set_visible(false)
    
    # Bottom right subplot
    sca(axs[2, 2])
    axs[2, 2].imshow(y[xgrid,:,:,1], aspect="auto", cmap="gray", interpolation="none",vmin=vmin3, vmax=vmax3,
    extent=(offset_start, offset_end, (n[2]-1)*d[2], 0f0))
    axs[2, 2].set_xlabel("Offset [m]", fontsize=40)
    # Share y-axis with bottom left
    # axs[2, 2].get_shared_y_axes().join(axs[2, 2], axs[2, 1])
    axs[2, 2].set_yticklabels([])
    axs[2, 2].set_ylabel("")
    vlines(x=0, colors=:b, ymin=0, ymax=(n[2]-1)*d[2], linewidth=3)
    hlines(y=zpos, colors=:b, xmin=offset_end, xmax=offset_start, linewidth=3)
end

function _getFigname(config, additional::Dict)
    isnothing(config) && return additional

    nbatch = config.nbatch
    epochs = config.epochs
    ntrain = size(config.x_train, 3)
    nvalid = size(config.x_valid, 3)
    
    figname = @strdict nbatch epochs ntrain nvalid
    return merge(additional, figname)
end

function read_velocity_cigs_offsets_as_nc(x_path::String, y_path::String, modelConfig::DFNO_3D.ModelConfig; ntrain::Int, nvalid::Int)

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
                                    x_file=x_path,
                                    y_file=y_path,
                                    x_key=["x0", "CIG0"],
                                    y_key="CIGs")

    return DFNO_3D.loadDistData(dataConfig, dist_read_x_tensor=read_x_tensor_helper, dist_read_y_tensor=read_y_tensor_helper)
end

function read_velocity_cigs_offsets_as_nz(path::String, modelConfig::DFNO_3D.ModelConfig; ntrain::Int, nvalid::Int)

    params = Config.get_parameters()
    offset_start = params["read_offset_start"]

    # Assumption that x is (nx, nz, 1, n). x0 is (nx, nz). CIG0 is (nh, nx, nz). CIG is (nh, nx, nz, 1, n)
    function read_x_tensor(file_name, key, indices)
        data = nothing
        h5open(file_name, "r") do file
            # Size to augment
            target_zeros = zeros(modelConfig.dtype, 1, map(range -> length(range), indices[1:4])...)

            x_data = file[key[1]]
            cigs_data = file[key[2]]

            # Read proper indices of x and x0. NOTE: Disclude 3 because no z = h = offset = 1 for background
            x = x_data[indices[1], indices[2], 1, indices[4]]
            x0 = x_data[indices[1], indices[2], 1, 1] # Use the first x for init model for now
            cig0 = cigs_data[indices[3] .+ (offset_start - 1), indices[1], indices[2], 1, 1] # Use the first cig for init cig for now
            cig0 = permutedims(cig0, [2, 3, 1])

            # Reshape to prepare for augmentation
            x = reshape(x, 1, length(indices[1]), length(indices[2]), 1, length(indices[4]))
            x0 = reshape(x0, 1, length(indices[1]), length(indices[2]), 1, 1)
            cig0 = reshape(cig0, 1, length(indices[1]), length(indices[2]), length(indices[3]), 1)

            # Augment to full size
            x = target_zeros .+ x
            x0 = target_zeros .+ x0
            cig0 = target_zeros .+ cig0

            # Concat along dimension 1
            data = cat(x, x0, cig0, dims=1)
        end

        # data_channels * nx * ny * nz * nt * n = data_channels * nx * nz * nh * 1 * n
        return data
    end
    
    function read_y_tensor(file_name, key, indices)
        data = nothing
        h5open(file_name, "r") do file
            cigs_data = file[key]
            cigs = cigs_data[indices[3] .+ (offset_start - 1), indices[1], indices[2], indices[4], indices[5]] # dim 4 which is t = 1:1
            data = permutedims(cigs, [2, 3, 1, 4, 5])
        end

        # channels * nx * ny * nz * nt * n = channels * nx * nz * nh * 1 * n
        return reshape(data, 1, size(data)...)
    end

    dataConfig = DFNO_3D.DataConfig(modelConfig=modelConfig, 
                                    ntrain=ntrain, 
                                    nvalid=nvalid, 
                                    x_file=path,
                                    y_file=path,
                                    x_key=["xs", "cigs"],
                                    y_key="cigs")

    return DFNO_3D.loadDistData(dataConfig, dist_read_x_tensor=read_x_tensor, dist_read_y_tensor=read_y_tensor)
end

function create_wavelet(timeD, dtD, f0)
    return ricker_wavelet(timeD, dtD, f0)
end

function create_geometry(n, d, nsrc, nxrec, dtD, timeD)
    xrec = range(0f0, stop=(n[1]-1)*d[1], length=nxrec)
    yrec = 0f0
    zrec = range(d[1], stop=d[1], length=nxrec)
    recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

    ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
    zsrc = convertToCell(range(d[1], stop=d[1], length=nsrc))
    xsrc = convertToCell(ContJitter((n[1]-1)*d[1], nsrc))
    srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

    return recGeometry, srcGeometry
end

function generate_noise(d_obs, nsrc, snr, q)
    noise_ = deepcopy(d_obs)
    for l = 1:nsrc
        noise_.data[l] = randn(Float32, size(d_obs.data[l]))
        noise_.data[l] = real.(ifft(fft(noise_.data[l]).*fft(q.data[1])))
    end
    noise_ = noise_/norm(noise_) * norm(d_obs) * 10f0^(-snr/20f0)
    return noise_
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--startidx"
            help = "Start index"
            arg_type = Int
            default = 1
        "--endidx"
            help = "End index"
            arg_type = Int
            default = 1000
        "--n_offsets"
            help = "num of offsets"
            arg_type = Int
            default = 51
        "--offset_start"
            help = "start of offset"
            arg_type = Float32
            default = -500f0
        "--offset_end"
            help = "end of offset"
            arg_type = Float32
            default = 500f0
        "--keep_offset_num"
            help = "keep how many offset during training"
            arg_type = Int
            default = 51
    end
    return parse_args(s)
end

# function de_z_shape_simple(G::NetworkGlow, X::AbstractArray{T, N}) where {T, N}
#     G.split_scales && (Z_save = array_of_array(X, max(G.L-1,1)))

#     logdet_ = 0
#     for i=1:G.L
#         (G.split_scales) && (X = G.squeezer.forward(X))
#         if G.split_scales && (i < G.L || i == 1)    # don't split after last iteration
#             X, Z = tensor_split(X)
#             Z_save[i] = Z
#             G.Z_dims[i] = collect(size(Z))
#         end
#     end
#     G.split_scales && (X = cat_states(Z_save, X))

#     return X
# end

# function z_shape_simple(G::NetworkGlow, ZX_test::AbstractArray{T, N}) where {T, N}
#     Z_save, ZX = split_states(ZX_test[:], G.Z_dims)
#     for i=G.L:-1:1
#         if i < G.L
#             ZX = tensor_cat(ZX, Z_save[i])
#         end
#         ZX = G.squeezer.inverse(ZX) 
#     end
#     ZX
# end

function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2

    return interval_center .+ randomshift
end

function plot_velocity_model(x, n, d, filename)
    vmin = quantile(vec(x), 0.05)  # 5th percentile
    vmax = quantile(vec(x), 0.95)  # 95th percentile

    fig, ax = subplots(figsize=(20,12)) 
    extentfull = (0f0, (n[1]-1)*d[1], (n[end]-1)*d[end], 0f0)
    cax = ax.imshow(x', vmin=vmin, vmax=vmax, extent=extentfull, aspect=0.45*(extentfull[2]-extentfull[1])/(extentfull[3]-extentfull[4]))
    ax.set_xlabel("X [m]", fontsize=40)
    ax.set_ylabel("Z [m]", fontsize=40)
    savefig(filename, bbox_inches="tight", dpi=300)
    close(fig)
end

function plot_cig(cig, n, d, offset_start, offset_end, n_offsets, filename)
    y = reshape(permutedims(cig, [2, 3, 1]), n[1], n[2], n_offsets, 1)

    PyPlot.rc("figure", titlesize=40)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=40); PyPlot.rc("ytick", labelsize=40)
    PyPlot.rc("axes", labelsize=40)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=40)     # Default fontsize for titles
    
    ### X, Z position in km
    xpos = 3.6f3
    zpos = 2.7f3
    xgrid = Int(round(xpos / d[1]))
    zgrid = Int(round(zpos / d[2]))
    
    # Create a figure and a 2x2 grid of subplots
    fig, axs = subplots(2, 2, figsize=(20,12), gridspec_kw = Dict("width_ratios" => [3, 1], "height_ratios" => [1, 3]))
    
    # Adjust the spacing between the plots
    subplots_adjust(hspace=0.0, wspace=0.0)
    
    vmin1, vmax1 = (-1, 1) .* quantile(abs.(vec(y[:,zgrid,:,1])), 0.99)
    vmin2, vmax2 = (-1, 1) .* quantile(abs.(vec(y[:,:,div(n_offsets,2)+1,1])), 0.88)
    vmin3, vmax3 = (-1, 1) .* quantile(abs.(vec(y[xgrid,:,:,1])), 0.999)
    sca(axs[1, 1])
    
    # Top left subplot
    axs[1, 1].imshow(y[:,zgrid,:,1]', aspect="auto", cmap="gray", interpolation="none",vmin=vmin1, vmax=vmax1,
        extent=(0f0, (n[1]-1)*d[1], offset_start, offset_end))
    axs[1, 1].set_ylabel("Offset [m]", fontsize=40)
    axs[1, 1].set_xticklabels([])
    axs[1, 1].set_xlabel("")
    hlines(y=0, colors=:b, xmin=0, xmax=(n[1]-1)*d[1], linewidth=3)
    vlines(x=xpos, colors=:b, ymin=offset_start, ymax=offset_end, linewidth=3)
    
    # Bottom left subplot
    sca(axs[2, 1])
    axs[2, 1].imshow(y[:,:,div(n_offsets,2)+1,1]', aspect="auto", cmap="gray", interpolation="none",vmin=vmin2, vmax=vmax2,
    extent=(0f0, (n[1]-1)*d[1], (n[2]-1)*d[2], 0f0))
    axs[2, 1].set_xlabel("X [m]", fontsize=40)
    axs[2, 1].set_ylabel("Z [m]", fontsize=40)
    axs[2, 1].set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    axs[2, 1].set_xticklabels(["0", "1000", "2000", "3000", "4000", "5000"])
    axs[2, 1].set_yticks([1000, 2000, 3000])
    axs[2, 1].set_yticklabels(["1000", "2000", "3000"])
    
    axs[2, 2].get_shared_x_axes().join(axs[1, 1], axs[2, 1])
    vlines(x=xpos, colors=:b, ymin=0, ymax=(n[2]-1)*d[2], linewidth=3)
    hlines(y=zpos, colors=:b, xmin=0, xmax=(n[1]-1)*d[1], linewidth=3)
    
    # Top right subplot
    axs[1, 2].set_visible(false)
    
    # Bottom right subplot
    sca(axs[2, 2])
    axs[2, 2].imshow(y[xgrid,:,:,1], aspect="auto", cmap="gray", interpolation="none",vmin=vmin3, vmax=vmax3,
    extent=(offset_start, offset_end, (n[2]-1)*d[2], 0f0))
    axs[2, 2].set_xlabel("Offset [m]", fontsize=40)
    # Share y-axis with bottom left
    axs[2, 2].get_shared_y_axes().join(axs[2, 2], axs[2, 1])
    axs[2, 2].set_yticklabels([])
    axs[2, 2].set_ylabel("")
    vlines(x=0, colors=:b, ymin=0, ymax=(n[2]-1)*d[2], linewidth=3)
    hlines(y=zpos, colors=:b, xmin=offset_end, xmax=offset_start, linewidth=3)
    
    # Remove the space between subplots and hide the spines
    for ax in reshape(axs, :)
        for spine in ["top", "right", "bottom", "left"]
            ax.spines[spine].set_visible(false)
        end
    end
    
    savefig(filename, bbox_inches="tight", dpi=300);
    close(fig)
end

function plotLoss(ep, Loss, Loss_valid, trainConfig::DFNO_3D.TrainConfig; additional=Dict())

    ntrain = size(trainConfig.x_train, 3)
    nbatches = Int(ntrain/trainConfig.nbatch)

    loss_train = Loss[1:ep*nbatches]
    loss_valid = Loss_valid[1:ep]
    fig = figure(figsize=(20, 12))

    PyPlot.rc("figure", titlesize=8)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
    PyPlot.rc("axes", labelsize=8)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=8)     # Default fontsize for titles

    subplot(1,3,1)
    plot(loss_train)
    xlabel("batch iterations")
    ylabel("loss")
    title("training loss at epoch $ep")
    subplot(1,3,2)
    plot(1:nbatches:nbatches*ep, loss_valid);
    xlabel("batch iterations")
    ylabel("loss")
    title("validation loss at epoch $ep")
    subplot(1,3,3)
    plot(loss_train);
    plot(1:nbatches:nbatches*ep, loss_valid); 
    xlabel("batch iterations")
    ylabel("loss")
    title("Objective function at epoch $ep")
    legend(["training", "validation"])
    tight_layout();

    figname = _getFigname(trainConfig, additional)

    savefig(joinpath("plots", "DFNO_3D", savename(figname; digits=6) * "_DFNO_CIG_loss.png"), bbox_inches="tight", dpi=300)
    close(fig);
end

function plot_optimize_loss(Loss, labels)

    fig = figure(figsize=(8, 12))

    PyPlot.rc("figure", titlesize=8)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
    PyPlot.rc("axes", labelsize=8)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=8)     # Default fontsize for titles

    subplot(1,1,1)
    plot(Loss)
    xlabel("iterations")
    ylabel("loss")
    title("loss after $(length(Loss)) iterations")
    tight_layout();

    savefig(joinpath("plots", "DFNO_3D", savename(labels; digits=6) * "_DFNO_CIG_OPT_loss.png"), bbox_inches="tight", dpi=300)
    close(fig);
end

function plot_optimize_eval(modelConfig, start_velocity, start_cig, end_velocity, end_cig; trainConfig=nothing, additional=Dict{String,Any}())
    params = Config.get_parameters()

    offset_start = params["offset_start"]
    offset_end = params["offset_end"]
    d = params["d"]
    n = [modelConfig.nx, modelConfig.ny]
    n_offsets = params["n_offsets"]
    down_rate = params["down_rate"]

    # Downsample
    d = d.*down_rate

    # Reshape the data to fit the model configuration
    start_velocity = reshape(start_velocity, (modelConfig.nx, modelConfig.ny, :))
    end_velocity = reshape(end_velocity, (modelConfig.nx, modelConfig.ny, :))

    start_cig = reshape(start_cig, (modelConfig.nc_out, modelConfig.nx, modelConfig.ny, :))
    end_cig = reshape(end_cig, (modelConfig.nc_out, modelConfig.nx, modelConfig.ny, :))

    num_samples = size(start_velocity, 3)
    
    PyPlot.rc("figure", titlesize=40)
    PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=40); PyPlot.rc("ytick", labelsize=40)
    PyPlot.rc("axes", labelsize=40)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=40)     # Default fontsize for titles

    fig, axs = subplots(num_samples*2, 12, figsize=(105, num_samples*10), gridspec_kw = Dict("width_ratios" => [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 1, 4], "height_ratios" => vcat([[1, 3] for i in 1:num_samples]...)))

    for i in 1:num_samples
        start_velocity_sample = start_velocity[:, :, i]
        end_velocity_sample = end_velocity[:, :, i]

        start_cig_sample = permutedims(start_cig[:, :, :, i], [2, 3, 1])
        end_cig_sample = permutedims(end_cig[:, :, :, i], [2, 3, 1])

        axs[i*2-1, 1].set_visible(false)
        axs[i*2-1, 4].set_visible(false)
        axs[i*2-1, 7].set_visible(false)
        axs[i*2-1, 12].set_visible(false)

        plot_velocity_model_helper(start_velocity_sample, n, d, axs[i*2, 1])
        plot_velocity_model_helper(end_velocity_sample, n, d, axs[i*2, 4])
        plot_velocity_model_helper(abs.(start_velocity_sample - end_velocity_sample), n, d, axs[i*2, 7])

        plot_cig_helper(start_cig_sample, n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 2:3])
        plot_cig_helper(end_cig_sample, n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 5:6])
        plot_cig_helper(5f0 .* abs.(start_cig_sample - end_cig_sample), n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 8:9])
        plot_cig_helper(abs.(fftshift(fft(start_cig_sample - end_cig_sample, [3]))), n, d, offset_start, offset_end, n_offsets, axs[i*2-1:i*2, 10:11])

        plot_energy_vs_offset_helper(start_cig_sample, end_cig_sample, n, d, offset_start, offset_end, n_offsets, axs[i*2, 12])

        # Label columns for the first sample
        if i == 1
            axs[2, 1].set_title("Init velocity model")
            axs[1, 2].set_title("Target CIG")
            axs[2, 4].set_title("Opt velocity model")
            axs[1, 5].set_title("Opt CIG")
            axs[2, 7].set_title("1X Diff velocity models")
            axs[1, 8].set_title("5X Diff CIG")
            axs[1, 10].set_title("FFT 1X CIG Diff")
            axs[2, 12].set_title("% Energy vs Offsets")
        end
    end

    suptitle("Optimization Comparison", fontsize=20)
    tight_layout()

    savefig(joinpath("plots", "DFNO_3D", savename(additional; digits=6) * "_DFNO_FIT_CIG_OPT_fitting.png"), bbox_inches="tight", dpi=300)
    close(fig)
end

end
