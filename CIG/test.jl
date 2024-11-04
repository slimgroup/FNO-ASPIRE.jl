using DrWatson
using Random
using JLD2

j, n_samples, n_procs = parse.(Int, ARGS[1:3])
temp_dir = "data/temp"

random_number = rand(1:10000)
data = @strdict random_number
file_path = joinpath(temp_dir, savename(data, "jld2"; digits=6))
@JLD2.save file_path random_number

println("Process with random number $random_number saved as $file_path")

while length(readdir(temp_dir)) < n_procs
    sleep(0.1)
end

numbers = []
for file in readdir(temp_dir)
    @load joinpath(temp_dir, file) random_number
    push!(numbers, random_number)
end

sorted_numbers = sort(numbers)
rank = findfirst(x -> x == random_number, sorted_numbers) - 1 

# Calculate the number of samples each process should handle
samples_per_process = div(n_samples, n_procs)
remainder = mod(n_samples, n_procs)

# Determine the start and end indices for this process
function get_sample_range(rank, samples_per_process, remainder)
    if rank < remainder
        n_start = rank * (samples_per_process + 1)
        n_end = n_start + samples_per_process
    else
        n_start = rank * samples_per_process + remainder
        n_end = n_start + samples_per_process - 1
    end
    return n_start + 1, n_end + 1
end

n_start, n_end = get_sample_range(rank, samples_per_process, remainder)
println("Process rank $rank handling range: $n_start to $n_end")

# # Clean up temporary directory (optional, if needed)
# rm(temp_dir, force=true)
