#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --job-name=FNO_ntrain=${1}_iteration_j=${2}_epochs=${3}
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --account=m3863_g

nvidia-smi
export SLURM_CPU_BIND="cores"
export PATH=$PATH:$HOME/.julia/bin
export DFNO_3D_GPU=1
# export LD_LIBRARY_PATH=
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0
module load cudnn/8.9.3_cuda12 julia/1.8.5

srun julia --project=FNO/ FNO/train.jl 2 ${3} ${2} ${1} 2

exit 0
EOT
