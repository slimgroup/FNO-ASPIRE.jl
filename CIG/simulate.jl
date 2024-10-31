
# import Pkg; Pkg.instantiate()
using DrWatson
using JLD2, JUDI, SegyIO, ImageGather
using ArgParse
using Statistics, Images
using FFTW
using LinearAlgebra
using Augmentor
try
    @eval using PyPlot  # Try to load PyPlot the first time
catch e
    @warn "Failed to load PyPlot on the first attempt: $e"
    @info "Retrying to load PyPlot..."
    @eval using PyPlot  # Retry loading PyPlot
end
using PyPlot, SlimPlotting
using Random

seed = 1
Random.seed!(seed)

include("../config.jl")

j, n_samples, rank, comm_size = parse.(Int, ARGS[1:4])

# Calculate the number of samples each process should handle
samples_per_process = div(n_samples, comm_size)
remainder = mod(n_samples, comm_size)

# Determine the start and end indices for each rank
if rank < remainder
    n_start = rank * (samples_per_process + 1)
    n_end = n_start + samples_per_process
else
    n_start = rank * samples_per_process + remainder
    n_end = n_start + samples_per_process - 1
end

n_start = n_start + 1
n_end = n_end + 1

println("Process $rank handling range: $n_start to $n_end")

PyPlot.rcdefaults()

params = Config.get_parameters()
n_offsets = params["n_offsets"]
offset_start = params["offset_start"]
offset_end = params["offset_end"]
f0 = params["f0"]
timeD = params["timeD"]
timeR = params["timeR"]
TD = params["TD"]
dtD = params["dtD"]
dtS = params["dtS"]
nbl = params["nbl"]
down_rate = params["down_rate"]

plot_path = joinpath("plots", "CIG")

function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2
    return interval_center .+ randomshift
end

offsetrange = range(offset_start, stop=offset_end, length=n_offsets)
wavelet = ricker_wavelet(TD, dtS, f0)
wavelet = filter_data(wavelet, dtS; fmin=3f0, fmax=Inf)
d = (12.5f0, 12.5f0)
o = (0f0, 0f0)

# use no salt (after smoothing) as the background
@load "data/posteriors_iteration_j=$(j-1).jld2" x0
x_no_salt = x0

# use salt (after smoothing) as the target
@load "data/initial.jld2" x
x_salt = x

# Down Sampled velocity models in scripts/prepare_training.jl
n = (size(x_salt)[1], size(x_salt)[2])
f0 = f0 / down_rate
d = d .* down_rate

# Setup model structure
nsrc = 16    # number of sources
model = Model(n, d, o, (1f0./imresize(x_salt[:,:,1,1], n)).^2f0; nb=nbl)
nxrec = n[1]
xrec = range(0f0, stop=(n[1]-1)*d[1], length=nxrec)
yrec = 0f0 # WE have to set the y coordiante to zero (or any number) for 2D modeling
zrec = range(d[1], stop=d[1], length=nxrec)
# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)
wb = 16
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range((wb-1)*d[1], stop=(wb-1)*d[1], length=nsrc))
snr = 12f0

# Setup operators
gaussian = 20

for i in n_start:n_end
    filename = "data/$j/sample_$i.jld2"
    if isfile(filename)
        @info "Skipping sample $i as file $filename already exists."
        continue  # Skip this iteration and move to the next one
    end

    Base.flush(Base.stdout)
    @info "sample $i out of $(size(x_salt)[end]) samples"
    # call function that generate background
    x_back = x_no_salt[:,:,1,i];
    x_salt_i= x_salt[:,:,1,i]

    # add water bottom
    nwb = 15
    x_back[:, 1:nwb] .= 1.48
    x_salt_i[:, 1:nwb] .= 1.48

    # Set up source structure
    xsrc = convertToCell(ContJitter((n[1]-1)*d[1], nsrc))
    srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)
    q = judiVector(srcGeometry, wavelet)
    opt = Options(isic=true)
    F = judiModeling(model, srcGeometry, recGeometry, options=opt)
    @time d_obs = F(1f0./x_salt_i.^2f0) * q
    J = judiExtendedJacobian(F(1f0./x_back.^2f0), q, offsetrange)
    d_obs0 = F(1f0./x_back.^2f0) * q
    noise_ = deepcopy(d_obs)
    for l = 1:nsrc
        noise_.data[l] = randn(Float32, size(d_obs.data[l]))
        noise_.data[l] = real.(ifft(fft(noise_.data[l]).*fft(q.data[1])))
    end
    noise_ = noise_/norm(noise_) * norm(d_obs) * 10f0^(-snr/20f0)
    @time rtm = J' * (d_obs0 - (d_obs + noise_))

    # save_dict = @strdict f0 dtD dtS nbl timeD timeR nsrc nxrec n d o q d_obs i rtm snr offset_start offset_end n_offsets
    # @tagsave(
    #         joinpath(joinpath(plot_path, "cig"), savename(save_dict, "jld2"; digits=6)),
    #         save_dict;
    #         safe=true
    #     );

    rtm[:, :, 1:20] .= 0;
    for z = 1:n[2]
        rtm[:,:,z] .*= z * d[2]
    end

    JLD2.@save filename rtm

    if false
        plot_velocity(x_back', d, perc=95, vmax=4.8, aspect=true, name="idx=$i"); 
        savefig(joinpath(plot_path, "x_back_idx=$(i)"), bbox_inches="tight", dpi=300);
        plot_velocity(x_salt_i', d, perc=95, vmax=4.8, aspect=true, name="idx=$i"); 
        savefig(joinpath(plot_path, "x_target_idx=$(i)"), bbox_inches="tight", dpi=300);
        rtm[:, :, 1:20] .= 0;
        for z = 1:n[2]
            rtm[:,:,z] .*= z * d[2]
        end

        cig_img_suffix = "_gaussian=$(gaussian)"
        cig_img_fname = "cig_idx=$(i)" * cig_img_suffix
        plot_cig(rtm, plot_path, cig_img_fname)
    end
end
