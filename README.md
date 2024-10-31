## TODO:

1. Make Inference Loop
2. Clean Plotting Code
3. Run everything

## Training Routine

1. Make `initial.jld2` and `posteriors_iteration_j=0.jld2`
```
julia --project=CIG/ scripts/prepare_training.jl
```

For each j:

2. Run generate CIGs for j -> `cigs_iteration_j=j.jld2`

```
sh scripts/launch_cigs.sh j nsamples nprocs
```

Collect CIGs when batch jobs finish
```
salloc --nodes=1 --constraint=gpu --gpus=1 --qos=interactive --time=04:00:00 --account=m3863_g --ntasks=1 --gpus-per-task=1 --gpu-bind=none
srun julia --project=CIG/ CIG/collect_cigs.jl j nsamples
```

3. Train CNF and FNO parallely

```
sh scripts/train_FNO.sh ntrain j epochs
sh scripts/train_CNF.sh ntrain j epochs
```

Clean Directories # TODO: Fix inside `ParametricDFNOs.jl`

```
sh scripts/clean_dir.sh j
```

4. Run Posterior mean function -> `posteriors_iteration_j=j.jld2`

```
salloc --nodes=1 --constraint=gpu --gpus=1 --qos=interactive --time=04:00:00 --account=m3863_g --ntasks=1 --gpus-per-task=1 --gpu-bind=none
srun julia --project=CNF/ CNF/update_fiducials.jl j nsamples ntrain epochs
```

## Inference Routine

1. Make `initial.jld2` and `posteriors_iteration_j=0.jld2` (Inside test data directory)

```
sh scripts/preprare_for_testing.sh nsamples 1 (offset, number of samples to test)
```

2. Run Inference using Trained Networks

```
srun julia --project=. inference.jl 0 (1/0 dont-use/use FNO)
```

Internally, For each j:
- Run generate CIGs for j -> `cigs_iteration_j=j.jld2` (Using Trained FNO / Using True Simulator)
- Run Posterior mean function -> `posteriors_iteration_j=j.jld2`
- Plot refined posterior and reference for each iteration


## Example run:

```
julia --project=CIG/ scripts/prepare_training.jl
sh scripts/launch_cigs.sh 1 850 34

salloc --nodes=1 --constraint=gpu --gpus=1 --qos=interactive --time=04:00:00 --account=m3863_g --ntasks=1 --gpus-per-task=1 --gpu-bind=none
srun julia --project=CIG/ CIG/collect_cigs.jl 1 850

salloc --nodes=1 --constraint=gpu --gpus=1 --qos=interactive --time=04:00:00 --account=m3863_g --ntasks=1 --g
pus-per-task=1 --gpu-bind=none
srun julia --project=FNO/ FNO/train.jl 2 2 0 2 2
srun julia --project=CNF/ CNF/train.jl 2 2 0 2 2

sh scripts/train_FNO.sh 800 1 80
sh scripts/train_CNF.sh 800 1 70

sh scripts/update_fiducial 1 850
```
