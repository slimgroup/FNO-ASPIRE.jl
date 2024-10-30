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

# Training hyperparameters
T = Float32
device = gpu
plot_path = "plots"
weights_path = "weights"

PyPlot.rc("font", family="serif");

lr           = 8f-4
clipnorm_val = 3f0
noise_lev_x  = 0.1f0
noise_lev_init = deepcopy(noise_lev_x)
noise_lev_y  = 0.0 

offset_start = 1
keep_offset_num = 21
d = (12.5f0, 12.5f0)

save_every   = 10
plot_every   = 10
n_condmean   = 12
num_post_samples = 64

batch_size, n_epochs, j, ntrain, ntest = 8, 2, 1, 16, 16
batch_size, n_epochs, j, ntrain, ntest = parse.(Int, ARGS[1:5])

@load "data/initial.jld2" x
@load "data/cigs_iteration_j=$j.jld2" CIGs

weights_path = joinpath(weights_path, "$j", "CNF")
plot_path = joinpath(plot_path, "$j", "CNF")

isdir(weights_path) || mkpath(weights_path)
isdir(plot_path) || mkpath(plot_path)

target_train = x[:, :, :, 1:ntrain]
target_test = x[:, :, :, ntrain+1:ntrain+ntest]
Y_train = CIGs[:, :, :, 1:ntrain]
Y_test = CIGs[:, :, :, ntrain+1:ntrain+ntest]

n = (size(target_train)[1], size(target_train)[2])
for z = 1:n[2]
	Y_train[:,z,:,:] .*= z * d[2]
	Y_test[:,z,:,:] .*= z * d[2]
end

# normalize CIGs
max_y = quantile(abs.(vec(Y_train)),0.9999);
Y_train ./= max_y;
Y_test ./= max_y;

n_x, n_y, chan_target, n_train = size(target_train)
n_train = size(target_train)[end]
println("n_train: ", n_train)
N = n_x*n_y*chan_target
chan_obs   = size(Y_train)[end-1]
chan_cond = 1

X_train  = target_train
X_test   = target_test 

vmax_v = maximum(X_train)
vmin_v = minimum(X_train)

n_batches    = cld(n_train, batch_size)-1
n_train_safe = batch_size*n_batches

# Summary network parametrs
unet_lev = 4
unet = Chain(Unet(chan_obs, chan_cond, unet_lev)|> device);

trainmode!(unet, true); 
unet = FluxBlock(unet);

# Create conditional network
L = 3
K = 9
n_hidden = 64
low = 0.5f0

cond_net = NetworkConditionalGlow(chan_target, chan_cond, n_hidden,  L, K;  split_scales=true, activation=SigmoidLayer(low=low,high=1.0f0)) |> device;
G        = SummarizedNet(cond_net, unet)

# Optimizer
opt = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr))

# Training logs 
loss      = []; logdet_train = []; ssim      = []; l2_cm      = []; std_train = [];
loss_test = []; logdet_test  = []; ssim_test = []; l2_cm_test = []; std_test = [];

noise_lev_x_min =  1f-3
noise_decay_per_epochs = div(n_epochs-50, Int(floor(log(noise_lev_x_min/noise_lev_init)/log(1f0/1.2f0))+1))

println("start training")

for e=1:n_epochs # epoch loop
	idx_e = reshape(randperm(n_train)[1:n_train_safe], batch_size, n_batches) 

	if (e >= 30) && (e <= n_epochs-20) && (mod(e,noise_decay_per_epochs) == 0)
        global noise_lev_x /= 1.2f0
        global noise_lev_x = max(noise_lev_x, noise_lev_x_min)
    end

	for b = 1:n_batches # batch loop
		X = X_train[:, :, :, idx_e[:,b]];
		Y = Y_train[:, :, :, idx_e[:,b]];

		for i in 1:batch_size # quick data augmentation to prevent overfitting
			if rand() > 0.5
				X[:,:,:,i:i] = X[end:-1:1,:,:,i:i]
				Y[:,:,:,i:i] = Y[end:-1:1,:,:,i:i]
			end
		end

		X .+= noise_lev_x*randn(Float32, size(X)); # noises not related to inverse problem 
		Y .+= noise_lev_y*randn(Float32, size(Y))
		Y = Y |> device;
		
		Zx, Zy, lgdet = G.forward(X |> device, Y)

		# Loss function is l2 norm 
		append!(loss, norm(Zx)^2 / (N*batch_size))  # normalize by image size and batch size
		append!(logdet_train, -lgdet / N) # logdet is internally normalized by batch size

		# Set gradients of flow and summary network
		dx, x, dy = G.backward(Zx / batch_size, Zx, Zy; Y_save = Y)

		for p in get_params(G) 
			Flux.update!(opt,p.data,p.grad)
		end; clear_grad!(G)

		print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
			"; f l2 = ",  loss[end], 
			"; lgdet = ", logdet_train[end], "; f = ", loss[end] + logdet_train[end], "\n")
		Base.flush(Base.stdout)
	end
	
	if(mod(e,plot_every)==0) 
		#get loss of training objective on test set corresponds to mutual information between summary statistic and x
		@time l2_test_val, lgdet_test_val  = get_loss(G, X_test, Y_test; device=device, batch_size=batch_size)
		append!(logdet_test, -lgdet_test_val)
		append!(loss_test, l2_test_val)

		# get conditional mean metrics over training batch
		@time cm_l2_train, cm_ssim_train, cm_std_train = get_cm_l2_ssim(G, X_train[:,:,:,1:n_condmean], Y_train[:,:,:,1:n_condmean]; device=device, num_samples=num_post_samples)
		append!(ssim, cm_ssim_train)
		append!(l2_cm, cm_l2_train)
		append!(std_train, cm_std_train) 

		# get conditional mean metrics over testing batch
		@time cm_l2_test, cm_ssim_test, cm_std_test = get_cm_l2_ssim(G, X_test[:,:,:,1:n_condmean], Y_test[:,:,:,1:n_condmean]; device=device, num_samples=num_post_samples)
		append!(ssim_test, cm_ssim_test)
		append!(l2_cm_test, cm_l2_test)
		append!(std_test, cm_std_test)

		for (test_x, test_y, file_str) in [[X_train, Y_train, "train"], [X_test, Y_test, "test"]]
			num_cols = 7
			plots_len = 2
			all_sampls = size(test_x)[end]-1
			fig = figure(figsize=(25, 5)); 
			for (i,ind) in enumerate((1:div(all_sampls,3):all_sampls)[1:plots_len])
				x = test_x[:,:,:,ind:ind] 
				y = test_y[:,:,:,ind:ind]
				y .+= noise_lev_y*randn(Float32, size(y));

				# make samples from posterior for train sample 
				X_post = posterior_sampler(G,  y, x; device=device, num_samples=num_post_samples,batch_size=batch_size)|> cpu
				X_post_mean = mean(X_post,dims=4)
				X_post_std  = std(X_post, dims=4)

				x_hat =  X_post_mean[:,:,1,1]
				x_gt =   x[:,:,1,1]
				error_mean = abs.(x_hat-x_gt)

				ssim_i = round(assess_ssim(x_hat, x_gt),digits=2)
				rmse_i = round(sqrt(mean(error_mean.^2)),digits=4)
				stdtotal_i = round(sqrt(mean(X_post_std.^2)),digits=4)

				# It's wired but if not train_rtm then comment out these three lines otherwise keep_offset_num not defined below 
				# if train_rtm
				#     keep_offset_num = 1
				# end

				y_plot = y[:,:,div(keep_offset_num, 2)+1, 1]
				# make RTM looks better
				y_plot[:, 1:37] = zeros(size(y_plot)[1], 37)
				a = quantile(abs.(vec(y_plot)), 98/100)

				# if synthoseis
				#     global y_plot = y_plot'
				#     # global X_post = X_post'
				#     global x_gt = x_gt'
				#     global x_hat = x_hat'
				#     global error_mean = error_mean'
				#     # global X_post_std = X_post_std'
				# end

				subplot(plots_len,num_cols,(i-1)*num_cols+1); imshow(y_plot', vmin=-a,vmax=a,interpolation="none", cmap="gray")
				axis("off"); title("Migration");#colorbar(fraction=0.046, pad=0.04);

				subplot(plots_len,num_cols,(i-1)*num_cols+2); imshow(X_post[:,:,1,1]', vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="cet_rainbow4")
				axis("off"); title("Posterior sample") #colorbar(fraction=0.046, pad=0.04);
				
				subplot(plots_len,num_cols,(i-1)*num_cols+3); imshow(X_post[:,:,1,2]', vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="cet_rainbow4")
				axis("off");title("Posterior sample")  #colorbar(fraction=0.046, pad=0.04);title("Posterior sample")

				subplot(plots_len,num_cols,(i-1)*num_cols+4); imshow(x_gt',  vmin=vmin_v,vmax=vmax_v,  interpolation="none", cmap="cet_rainbow4")
				axis("off"); title(L"Reference $\mathbf{x^{*}}$") ; #colorbar(fraction=0.046, pad=0.04)

				subplot(plots_len,num_cols,(i-1)*num_cols+5); imshow(x_hat' ,  vmin=vmin_v,vmax=vmax_v,  interpolation="none", cmap="cet_rainbow4")
				axis("off"); title("Posterior mean | SSIM="*string(ssim_i)) ; #colorbar(fraction=0.046, pad=0.04)

				subplot(plots_len,num_cols,(i-1)*num_cols+6); imshow(error_mean' , vmin=0,vmax=0.42, interpolation="none", cmap="magma")
				axis("off");title("Error | RMSE="*string(rmse_i)) ;# cb = colorbar(fraction=0.046, pad=0.04)

				subplot(plots_len,num_cols,(i-1)*num_cols+7); imshow(X_post_std[:,:,1,1]', vmin=0,vmax=0.42,interpolation="none", cmap="magma")
				axis("off"); title("Posterior variance | RMS point-wise std="*string(stdtotal_i)) ;#cb =colorbar(fraction=0.046, pad=0.04)
			end
			tight_layout()
			fig_name = @strdict chan_obs noise_lev_x noise_lev_init n_train e offset_start # offset_end n_offsets keep_offset_num
			safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_"*file_str*".png"), fig); close(fig)
		end
			
		############# Training metric logs
		if e != plot_every
			sum_train = loss + logdet_train
			sum_test = loss_test + logdet_test

			fig = figure("training logs ", figsize=(10,12))
			subplot(6,1,1); title("L2 Term: train="*string(loss[end])*" test="*string(loss_test[end]))
			plot(range(0f0, 1f0, length=length(loss)), loss, label="train");
			plot(range(0f0, 1f0, length=length(loss_test)),loss_test, label="test"); 
			axhline(y=1,color="red",linestyle="--",label="Normal Noise")
			ylim(bottom=0.,top=1.5)
			xlabel("Parameter Update"); legend();

			subplot(6,1,2); title("Logdet Term: train="*string(logdet_train[end])*" test="*string(logdet_test[end]))
			plot(range(0f0, 1f0, length=length(logdet_train)),logdet_train);
			plot(range(0f0, 1f0, length=length(logdet_test)),logdet_test);
			xlabel("Parameter Update") ;

			subplot(6,1,3); title("Total Objective: train="*string(sum_train[end])*" test="*string(sum_test[end]))
			plot(range(0f0, 1f0, length=length(sum_train)),sum_train); 
			plot(range(0f0, 1f0, length=length(sum_test)),sum_test); 
			xlabel("Parameter Update") ;

			subplot(6,1,4); title("SSIM train=$(ssim[end]) test=$(ssim_test[end])")
			plot(range(0f0, 1f0, length=length(ssim)),ssim); 
			plot(range(0f0, 1f0, length=length(ssim_test)),ssim_test); 
			xlabel("Parameter Update") 

			subplot(6,1,5); title("RMSE train=$(l2_cm[end]) test=$(l2_cm_test[end])")
			plot(range(0f0, 1f0, length=length(l2_cm)),l2_cm); 
			plot(range(0f0, 1f0, length=length(l2_cm_test)),l2_cm_test); 
			xlabel("Parameter Update") 

			subplot(6,1,6); title("RMS pointwise STD train=$(std_train[end]) test=$(std_test[end])")
			plot(range(0f0, 1f0, length=length(std_train)),std_train); 
			plot(range(0f0, 1f0, length=length(std_test)),std_test); 
			xlabel("Parameter Update")

			tight_layout()
			fig_name = @strdict chan_obs noise_lev_x noise_lev_init n_train e offset_start # offset_end n_offsets keep_offset_num
			safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
		end

	end

	if(mod(e,save_every)==0) 
		unet_model = G.sum_net.model|> cpu;
		G_save = deepcopy(G);
		reset!(G_save.sum_net); # clear params to not save twice
		Params = get_params(G_save) |> cpu;
		save_dict = @strdict chan_obs unet_lev unet_model n_train e noise_lev_x noise_lev_init lr n_hidden L K Params loss logdet_train l2_cm ssim loss_test logdet_test l2_cm_test ssim_test batch_size offset_start # offset_end n_offsets keep_offset_num; 
		@tagsave(
			joinpath(weights_path, savename(save_dict, "bson"; digits=6)),
			save_dict;
			safe=true
		);
	end
end
