using JLD2

T = Float64
J = 3

nz = 128
nx = 256
N = 800

offsets = 21

x = rand(T, nx, nz, 1, N)
y = rand(T, nx, nz, N)

@JLD2.save "data/initial.jld2" x y

for j in 0:J
    x0 = rand(T, nx, nz, 1, N)
    CIGs = rand(T, offsets, nx, nz, 1, N)
    CIG0 = CIGs[:, :, :, 1:1, 1:1]

    @JLD2.save "data/posteriors_iteration_j=$j.jld2" x0 CIG0
    if j > 0
        @JLD2.save "data/cigs_iteration_j=$j.jld2" CIGs
    end
end

# BEFORE ANYTHING

# posteriors_iteration_0.jld2
x0 -> (nx, nz, N)

# initial.jld2 # make one example save to initial.jld2. size (1, nx, nz)
x  -> (nx, nz, N)
y  -> (??????, N)

# CIG SUMMARY STATISTIC SIMULATION

# cigs_iteration_j=$j.jld2 # save in a separate file 
CIGs -> (nx, nz, offsets, N)

# CNF POST TRAINING POSTERIORS

# posteriors_iteration_1.jld2
x0 -> (nx, nz, N)
