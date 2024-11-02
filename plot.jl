function plot_metrics(x_gt, x_hat_fno_data, x_hat_true_data; n, d, title="")
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
        ax.set_title("FNO CIG Iter $j", fontsize=16)
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
        ax.set_title("Analytical CIG Iter $j", fontsize=16)
    end

    # Add a main title for the figure
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    savefig("result_inference.png", bbox_inches="tight", dpi=300)
    close(fig)
end
