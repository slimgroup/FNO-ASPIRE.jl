function posterior_sampler(G, y, x; device=gpu, num_samples=1, batch_size=16)
  size_x = size(x)
    # make samples from posterior for train sample
  X_forward = randn(Float32, size_x[1:end-1]...,batch_size) |> device
    Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
    _, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat); #needs to set the proper sizes here

    X_post_train = zeros(Float32, size_x[1:end-1]...,num_samples)
    for i in 1:div(num_samples, batch_size)
      ZX_noise_i = randn(Float32, size_x[1:end-1]...,batch_size)|> device

      X_post_train[:,:,:, (i-1)*batch_size+1 : i*batch_size] = G.inverse(
        ZX_noise_i,
        Zy_fixed_train
      ) |> cpu;
  end
  X_post_train
end

function get_cm_l2_ssim(G, X_batch, Y_batch; device=gpu, num_samples=1)
        # needs to be towards target so that it generalizes accross iteration
        num_test = size(Y_batch)[end]
        l2_total = 0
        ssim_total = 0
        std_total = 0
        #get cm for each element in batch
        for i in 1:num_test
            y   = Y_batch[:,:,:,i:i]
            x   = X_batch[:,:,:,i:i]

            X_post = posterior_sampler(G, y, x; device=device, num_samples=num_samples, batch_size=batch_size)
            x_hat =   mean(X_post; dims=4)[:,:,1,1]
            x_gt =  (x[:,:,1,1]) |> cpu
            X_post_std = std(X_post, dims=4)
            ssim_total += assess_ssim(x_hat, x_gt)
            l2_total   += sqrt(mean((x_hat - x_gt).^2))
            std_total  += sqrt(mean(X_post_std.^2))
        end
    return l2_total / num_test, ssim_total / num_test, std_total / num_test
end

function get_loss(G, X_batch, Y_batch; device=gpu, batch_size=16)
    l2_total = 0
    logdet_total = 0
    num_batches = div(size(Y_batch)[end], batch_size)
    for i in 1:num_batches
        x_i = X_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size]
        y_i = Y_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size]

        x_i .+= noise_lev_x*randn(Float32, size(x_i)); 
        y_i .+= noise_lev_y*randn(Float32, size(y_i)); 
        Zx, Zy, lgdet = G.forward(x_i|> device, y_i|> device) |> cpu;
        l2_total     += norm(Zx)^2 / (N*batch_size)
        logdet_total += lgdet / N
    end

    return l2_total / (num_batches), logdet_total / (num_batches)
end



# # load synthoseis data for training
# using NPZ

# # Function to read all .npy files in a directory and store in one array
# function load_npy(directory)
#     # Get a list of .npy files in the directory
#     npy_files = filter(f -> endswith(f, ".npy"), readdir(directory; join=true))

#     data_array = zeros(256, 256, 1, size(npy_files)[1])
#     for (i, file) in enumerate(npy_files)
#         # println("Reading file: $file")
#         data = npzread(file)
#         data_array[:, :, 1, i] = data
#     end

#     return data_array
# end


function load_compass()
    repeat_train = Int(num_train/800)
    
    data_path = "/slimdata/rafaeldata/fwiuq_eod/rtms_oed.jld2" 
    v_train = JLD2.jldopen(data_path, "r")["m_train"][:, :, :, 1:800];
    v_train = cat([v_train for _ in 1:repeat_train]..., dims=4)
    v_test = JLD2.jldopen(data_path, "r")["m_train"][:, :, :, 851:1000]
    v_train = cat(v_train, v_test, dims=4)
    return v_train
end

function load_compass_all()
    data_path = "/slimdata/rafaeldata/fwiuq_eod/rtms_oed.jld2"
    v_train = JLD2.jldopen(data_path, "r")["m_train"][:, :, :, 1:1000];
    return v_train
end

function load_synthoseis_data()
    directory = "/slimdata/tunadata/ambient_synthetic_test/synthoseis/synthoseis_data/3209_synthetic_velocities_256x256"
    v_train = load_npy(directory);
    v_train ./= 1000   # The resulting velocity is closer to compass model and the noise level takes best effect
    # transpose velocity to make them consistent with compass model
    v_train = permutedims(v_train, (2, 1, 3, 4)) 

    return v_train
end

function load_synthoseis_w_wb_data(; nwb=20)
    m_train = load_synthoseis_data()
    tmp = zeros(size(m_train)[1], size(m_train)[2] + nwb, size(m_train)[3], size(m_train)[4])
    tmp[:, nwb + 1:end, :, :] = m_train
    tmp[:, 1:nwb, :, :] .= 1.48
    m_train = deepcopy(tmp)

    return m_train
end

function load_synthoseis_all_bkg(gaussian; nwb=20)
    m_train = load_synthoseis_data()
    m_back_all = zeros(size(m_train)[1], size(m_train)[2] + nwb, size(m_train)[3], size(m_train)[4])
    m_back_all[:, 1:nwb, :, :] .= 1.48
    for i in 1:size(m_train)[end]
        x = m_train[:, :, 1, i]
        # pert = 40
        # pl = ElasticDistortion(pert, pert);
        # xnew = augment(x, pl);
        # no perturbation
        xnew = x
        m_back = 1f0./Float32.(imfilter(1f0./xnew, Kernel.gaussian(gaussian)));
        
        m_back_all[:, nwb+1:end, 1, i] = m_back
    end

    return m_back_all
end

function load_rand_prior_bkg()
    # random walk. Draw from the fixed region
    prior_vel = gen_prior_vel(vel_all)
    m_back = 1f0./Float32.(imfilter(1f0./prior_vel, Kernel.gaussian(10)))
    plot_vel_random(m_back, i, plot_path)
end

function background_1d_average()
    m_mean = mean(m_train, dims=4)[:,:,1,1]
    wb = maximum(find_water_bottom(m_mean.-minimum(m_mean)))
    m0 = deepcopy(m_mean)
    m0[:,wb+1:end] .= 1f0./Float32.(imfilter(1f0./m_mean[:,wb+1:end], Kernel.gaussian(10)))
    return m0
end

function background_1d_gradient()
    m_1d_gradient = reshape(repeat(range(minimum(m_train), stop=maximum(m_train), length=n[2]), inner=n[1]), n)
    m0 = 1f0./Float32.(imfilter(1f0./m_1d_gradient, Kernel.gaussian(10)))
    return m0
end

function load_y()
    n_tot_sample = num_train + num_test
    m0_train = deepcopy(m_train)
    grad_train = zeros(Float32, size(m_train, 1), size(m_train, 2), keep_offset_num, n_tot_sample)
    keep_offset_idx = div(n_offsets,2)+1-div(keep_offset_num, 2):div(n_offsets,2)+1+div(keep_offset_num, 2)

    for i = 1:num_train
        # misc_dict = @strdict f0 dtD dtS nbl timeD timeR nsrc nxrec n d o i snr offset_start offset_end n_offsets
        if rtm_type == "ext-rtm"
            if i <= 800
                misc_dict = @strdict f0 dtD dtS nbl timeD timeR nsrc nxrec n d o i snr offset_start offset_end n_offsets
                grad_train[:,:,1:keep_offset_num,i] = permutedims(JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", cig_fname * "-1"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"][keep_offset_idx, :, :], [2, 3, 1])
                # grad_train[:,:,keep_offset_num+1:end,i] = permutedims(JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", cig_fname * "-4"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"][keep_offset_idx, :, :], [2, 3, 1])

                # gen_ext_rtm.jl saves rtm with rtm.data, which has only 2 dimensions. size = (512, 256).
                # rtm = JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", rtm_fname * "-1"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"]
                # rtm = reshape(rtm, (size(rtm)[1], size(rtm)[2], 1, 1))
                # grad_train[:,:,:,i] = rtm
            elseif 801 <= i <= 1600
                i -= 800
                misc_dict = @strdict f0 dtD dtS nbl timeD timeR nsrc nxrec n d o i snr offset_start offset_end n_offsets
                grad_train[:,:,:,i+800] = permutedims(JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", cig_fname * "-1"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"][keep_offset_idx, :, :], [2, 3, 1])
                # rtm = JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", rtm_fname * "-2"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"]
                # rtm = reshape(rtm, (size(rtm)[1], size(rtm)[2], 1, 1))
                # i += 800
                # grad_train[:,:,:,i] = rtm
            elseif 1601 <= i <= 2400
                i -= 1600
                misc_dict = @strdict f0 dtD dtS nbl timeD timeR nsrc nxrec n d o i snr offset_start offset_end n_offsets
                grad_train[:,:,:,i+1600] = permutedims(JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", cig_fname * "-1"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"][keep_offset_idx, :, :], [2, 3, 1])
                # rtm = JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", rtm_fname * "-1"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"]
                # rtm = reshape(rtm, (size(rtm)[1], size(rtm)[2], 1, 1))
                # i += 1600
                # grad_train[:,:,:,i] = rtm
            elseif 2401 <= i <= 3200
                i -= 2400
                misc_dict = @strdict f0 dtD dtS nbl timeD timeR nsrc nxrec n d o i snr offset_start offset_end n_offsets
                grad_train[:,:,:,i+2400] = permutedims(JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", cig_fname * "-4"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"][keep_offset_idx, :, :], [2, 3, 1])
                # rtm = JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", rtm_fname * "-1"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"]
                # rtm = reshape(rtm, (size(rtm)[1], size(rtm)[2], 1, 1))
                # i += 2400
                # grad_train[:,:,:,i] = rtm
            end
        else
            load_grad = JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", rtm_fname), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"]
            for k = 1:size(grad_train, 3)
                grad_train[:,:,k,i] = load_grad
            end
        end
        if background_type == "1d-average" || background_type == "1d-gradient"
            m0_train[:,:,1,i] .= m0
        end
    end

    # append test samples to the end of `grad_train`
    for (idx, i) in enumerate(851:1000)
        misc_dict = @strdict f0 dtD dtS nbl timeD timeR nsrc nxrec n d o i snr offset_start offset_end n_offsets
        if rtm_type == "ext-rtm"
            grad_train[:,:,1:keep_offset_num,num_train+idx] = permutedims(JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", cig_fname * "-1"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"][keep_offset_idx, :, :], [2, 3, 1])
            # grad_train[:,:,keep_offset_num+1:end,num_train+idx] = permutedims(JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", cig_fname * "-4"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"][keep_offset_idx, :, :], [2, 3, 1])
    
            # grad_train[:,:,keep_offset_num+1:end,num_train+idx] = permutedims(JLD2.jldopen(joinpath("/slimdata/zyin62/francis_data/plots/FWIUQ/gen_ext-rtm-1d-average", savename(misc_dict, "jld2"; digits=6)), "r")["rtm"][keep_offset_idx, :, :], [2, 3, 1])
            # rtm = JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", rtm_fname * "-1"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"]
            # rtm = reshape(rtm, (size(rtm)[1], size(rtm)[2], 1, 1))
            # grad_train[:,:,:,num_train+idx] = rtm
        end
    end

    # for synthoseis cig data, depth scaling has been done in gen_cig.jl
    # for z = 1:n[2]
    #     grad_train[:,z,:,:] .*= z * d[2]
    #     if z <= wb
    #         # grad_train[:,z,:,:] .= 0 # no mute
    #     end
    # end

    println("finish loading training and testing data")
    return grad_train
end


function load_y_test(cig_fname)
    grad_train = zeros(Float32, size(m_train, 1), size(m_train, 2), keep_offset_num, num_test)
    keep_offset_idx = div(n_offsets,2)+1-div(keep_offset_num, 2):div(n_offsets,2)+1+div(keep_offset_num, 2)
    
    # append test samples to the end of `grad_train`
    for (idx, i) in enumerate(start_test_idx:start_test_idx+149)
        misc_dict = @strdict f0 dtD dtS nbl timeD timeR nsrc nxrec n d o i snr offset_start offset_end n_offsets
        if rtm_type == "ext-rtm"
            grad_train[:,:,:,idx] = permutedims(JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", cig_fname * "-1"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"][keep_offset_idx, :, :], [2, 3, 1])
    
            # rtm = JLD2.jldopen(joinpath(joinpath("/slimdata/yunlindata/plots/FWIUQ", rtm_fname * "-1"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"]
            # rtm = reshape(rtm, (size(rtm)[1], size(rtm)[2], 1, 1))
            # grad_train[:,:,1,num_train+idx] = rtm
        end
    end

    println("finish loading grad_train")
    
    if !synthoseis
        for z = 1:n[2]
            grad_train[:,z,:,:] .*= z * d[2]
            if z <= wb
                #grad_train[:,z,:,:] .= 0
            end
        end
    end
    
    # normalize rtms
    grad_train ./= quantile(abs.(vec(grad_train[:,:,:,end-149:end])),0.9999);
     
    return grad_train
end 



function load_trained_network(net_path, chan_obs)
    # n_x, n_y, chan_target, n_test = size(X_test)
    # N = n_x*n_y*chan_target
    chan_cond = 1

    # Summary network parametrs
    unet_lev = 4
    unet = Chain(Unet(chan_obs, chan_cond, unet_lev)|> device);
    trainmode!(unet, true); 
    unet = FluxBlock(unet);
    
    # Create conditional network
    # L = 3
    # K = 9
    # n_hidden = 64
    # low = 0.5f0
    
    Random.seed!(123);
    
    unet_lev = BSON.load(net_path)["unet_lev"];
    n_hidden = BSON.load(net_path)["n_hidden"];
    L = BSON.load(net_path)["L"];
    K = BSON.load(net_path)["K"];
    
    unet = Unet(chan_obs,1,unet_lev);
    trainmode!(unet, true);
    unet = FluxBlock(Chain(unet)) |> device;
    
    cond_net = NetworkConditionalGlow(1, 1, n_hidden,  L, K;  freeze_conv=true, split_scales=true, activation=SigmoidLayer(low=0.5f0,high=1.0f0)) |> device;
    G        = SummarizedNet(cond_net, unet)
    
    Params = BSON.load(net_path)["Params"]; 
    noise_lev_x = BSON.load(net_path)["noise_lev_x"]; 
    set_params!(G,Params)
    
    # Load in unet summary net
    G.sum_net.model = BSON.load(net_path)["unet_model"]; 
    G = G |> device;

    return G
end













# end
