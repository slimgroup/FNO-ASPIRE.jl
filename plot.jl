function plot_metrics(x_gt, x_hat_fno_data, x_hat_true_data; n, d, title="")
    """
    Compare the posterior means of each iteration (both using FNO CIG and analytical CIG) with the ground truth velocity
    Assume `x_hat_fno_data` and `x_hat_true_dat` have shape (nx, nz, n_iter)
    """
    n_iter = size(x_hat_fno_data, 3)
    extentfull = (0f0, (n[1]-1)*d[1], (n[end]-1)*d[end], 0f0)

    fig, axes = subplots(2, n_iter + 1, figsize=((n_iter + 1)*9, 12),)
    
    # plot ground truth in the last column
    for i in 1:2
        ax = axes[i, n_iter + 1]
        vmin = quantile(vec(x_gt), 0.05)
        vmax = quantile(vec(x_gt), 0.95)
        cax = ax.imshow(x_gt', vmin=vmin, vmax=vmax, cmap="cet_rainbow4", extent=extentfull, aspect=0.45 * (extentfull[2] - extentfull[1]) / (extentfull[3] - extentfull[4]))
        ax.set_xlabel("X [m]", fontsize=14)
        ax.set_ylabel("Z [m]", fontsize=14)
        ax.set_title("Ground Truth", fontsize=16)
    end

    # plot x_hat_fno_data in the first row
    for j in 1:n_iter
        ax = axes[1, j]
        x = x_hat_fno_data[:, :, j]
        vmin = quantile(vec(x), 0.05)
        vmax = quantile(vec(x), 0.95)
        cax = ax.imshow(x', vmin=vmin, vmax=vmax, cmap="cet_rainbow4", extent=extentfull, aspect=0.45 * (extentfull[2] - extentfull[1]) / (extentfull[3] - extentfull[4]))
        ax.set_xlabel("X [m]", fontsize=14)
        ax.set_ylabel("Z [m]", fontsize=14)
        ax.set_title("Use FNO CIG Iter $j", fontsize=16)
    end

    # plot x_hat_true_data in the second row
    for j in 1:n_iter
        ax = axes[2, j]
        x = x_hat_true_data[:, :, j]
        vmin = quantile(vec(x), 0.05)
        vmax = quantile(vec(x), 0.95)
        cax = ax.imshow(x', vmin=vmin, vmax=vmax, cmap="cet_rainbow4", extent=extentfull, aspect=0.45 * (extentfull[2] - extentfull[1]) / (extentfull[3] - extentfull[4]))
        ax.set_xlabel("X [m]", fontsize=14)
        ax.set_ylabel("Z [m]", fontsize=14)
        ax.set_title("Use analytical CIG Iter $j", fontsize=16)
    end

    # Add a main title for the figure
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    savefig("plots/result_inference.png", bbox_inches="tight", dpi=300)
    close(fig)
end


function plot_cig_metric(cig_fno, cig_analytic, plot_path, params; fs=40)
    """
    A similar function as `plot_metrics`. This function compares FNO CIG and analytical CIG across multiple iterations.
    The 3rd row is the 5X difference between FNO CIG and analytical CIG.
    Assume `cig_fno`, `cig_analytic` both have shape (nx, nz, n_offset, n_iter)
    """

    n_iter = size(cig_fno, 4)  # Get the number of iterations

    fig2, ax2 = subplots(3, n_iter, figsize=(15 * n_iter, 25))   # need to name it fig2, ax2. Otherwise fig, ax will mix with those defined in plot_cig

    for i in 1:n_iter
        cig_fno_i = cig_fno[:, :, :, i]
        cig_analytic_i = cig_analytic[:, :, :, i]
        cig_dif_i = abs.(cig_fno_i - cig_analytic_i) .* 5
    
        # plot FNO CIG
        cig_img_fname = "cig_fno_iter_$i.png"
        plot_cig(cig_fno_i, plot_path, cig_img_fname, params)
        img1 = imread(joinpath(plot_path, cig_img_fname))
        ax2[1, i].imshow(img1)
        ax2[1, i].axis("off")  # Hide axes for a cleaner look
        ax2[1, i].set_title("FNO predicted CIG - Iter $i", fontsize=fs)
    
        # plot analytical CIG
        cig_img_fname = "cig_analytic_iter_$i.png"
        plot_cig(cig_analytic_i, plot_path, cig_img_fname, params)
        img2 = imread(joinpath(plot_path, cig_img_fname))
        ax2[2, i].imshow(img2)
        ax2[2, i].axis("off")
        ax2[2, i].set_title("Analytical CIG - Iter $i", fontsize=fs)
    
        # plot (FNO CIG - analytical CIG) X 5
        cig_img_fname = "cig_dif_iter_$i.png"
        plot_cig(cig_dif_i, plot_path, cig_img_fname, params)
        img3 = imread(joinpath(plot_path, cig_img_fname))
        ax2[3, i].imshow(img3)
        ax2[3, i].axis("off")
        ax2[3, i].set_title("5X Difference - Iter $i", fontsize=fs)
    end

    fig2.tight_layout()

    fig2.savefig(joinpath(plot_path, "cig_compare_all_iters.png"))
end


using PyCall
function make_movie(X_post, plot_path; movie_suffix="X_post")
    """
    Make a movie of posterior samples
    size(X_post) = (nx, nz, 1, n_post) 
    """
    down_rate = params["down_rate"]
    d = params["d"] .* down_rate

    nx = params["nx"] ÷ down_rate
    nz = params["nz"] ÷ down_rate

    n = (nx, nz)

    # Update function to change the image per frame
    function update_frame(frame)
        frame_i = X_post[:, :, 1, frame]
        img.set_data(frame_i')  # Update the image data
        ax.set_title("Frame $frame")  # Update title with frame number
        return img
    end

    fig, ax = plt.subplots()

    # Initial plot (first image)
    frame_1 = X_post[:, :, 1, 1]
    # img = ax.imshow(frame_1', cmap="cet_rainbow4", vmin=minimum(m_train), vmax=maximum(m_train), extent=[0, size(X_post, 1) * d[1], size(X_post, 2) * d[2], 0])
    img = ax.imshow(frame_1', cmap="cet_rainbow4", vmax=4.5, extent=[0, size(X_post, 1) * d[1], size(X_post, 2) * d[2], 0])
    ax.set_title("Frame 1")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")

    # Create the animation
    animation = pyimport("matplotlib.animation")
    n_frames = size(X_post)[end]
    anim = animation.FuncAnimation(fig, update_frame, frames=1:n_frames, interval=200)

    # Save as a video or GIF
    anim.save(joinpath(plot_path, "movie_$(movie_suffix).gif"), writer="ffmpeg")
end


function make_movie_X_post_w_CIG(X_post, plot_path, cig; movie_suffix="X_post")
    """
    Make a movie of posterior samples with a static image on the left.
    size(X_post) = (nx, nz, 1, n_post) 
    size(cig) = (nx, nz, n_offset)
    """
    down_rate = params["down_rate"]
    d = params["d"] .* down_rate

    nx = params["nx"] ÷ down_rate
    nz = params["nz"] ÷ down_rate

    n = (nx, nz)

    # Update function to change the image per frame
    function update_frame(frame)
        frame_i = X_post[:, :, 1, frame]
        img.set_data(frame_i')  # Update the image data on the right
        ax[2].set_title("Frame $frame")  # Update title with frame number
        return img
    end

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Load and plot CIG
    cig_img_fname = "cig.png"
    plot_cig(cig, plot_path, cig_img_fname, params)
    img = imread(joinpath(plot_path, cig_img_fname))

    ax[1].imshow(img)
    ax[1].axis("off")  # Hide axes for a cleaner look
    ax[1].set_title("input CIG")

    # Plot the first frame of the movie
    frame_1 = X_post[:, :, 1, 1]
    img = ax[2].imshow(frame_1', cmap="cet_rainbow4", vmax=4.5, extent=[0, size(X_post, 1) * d[1], size(X_post, 2) * d[2], 0])
    ax[2].set_title("Frame 1")
    ax[2].set_xlabel("X [m]")
    ax[2].set_ylabel("Z [m]")

    # Create the animation
    animation = pyimport("matplotlib.animation")
    n_frames = size(X_post)[end]
    anim = animation.FuncAnimation(fig, update_frame, frames=1:n_frames, interval=200)

    fig.tight_layout()

    # Save as a video or GIF
    anim.save(joinpath(plot_path, "movie_$(movie_suffix).gif"), writer="ffmpeg")
end

