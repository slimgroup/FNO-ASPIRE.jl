include("simulate.jl")

# Training ASPIRE

# TEMP
N = 800

grid = (256, 128)
offsets = 21
N = 10
J = 2

sim = TrueSimulator(...) # TODO: YUNLIN
CNFs = [init_cnf(j) for j=1:J] # TODO: RICHARD

# TODO: Draw N samples: x # TODO: YUNLIN
x = zeros(N, grid...)

# TODO: Simulate x -> y
y = zeros(N, grid...)
for i = 1:N
    y = simulate(x[i]) # TODO: YUNLIN
end

# TODO: Init N Fiducials: x0 # TODO: YUNLIN
x0 = zeros(N, grid...)

for j = 1:J
    # TODO: Calculate summary statistic yhat for all observations y around fiducials x0 # TODO: YUNLIN
    yhat = zeros(N, grid..., offsets)
    for i = 1:N
        yhat[i] = compute_CIG(sim, x0[i], y[i])
    end

    # TODO: Train CNF on (x, yhat) # TODO: RICHARD
    train_cnf!(CNF[j], x, yhat)

    # TODO: Update fiducials x0 by averaging posterior samples conditioned on yhats # TODO: RICHARD
    for i = 1:N
        x0[i] = posterior_mean(CNF[j], yhat[i])
    end
end

# TODO: save CNFs TODO: RICHARD
