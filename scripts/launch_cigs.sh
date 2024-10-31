#!/bin/bash

# Check if three arguments (j, N and P) are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <j> <N> <P>"
    exit 1
fi

j=$1      # iteration index
N=$2      # number of samples
P=$3      # number of processors

for (( i=0; i<P; i++ ))
do
    sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --job-name=GEN_CIG_nsample=${i}_iteration_j=${j}
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=01:10:00
#SBATCH --account=m3863_g

nvidia-smi
export SLURM_CPU_BIND="cores"
export PATH=\$PATH:\$HOME/.julia/bin
export DFNO_3D_GPU=1
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0

export JULIA_NUM_THREADS=$SLURM_CPUS_ON_NODE
export DEVITO_LANGUAGE=openacc
export DEVITO_ARCH=nvc
export DEVITO_PLATFORM=nvidiaX

module load cudnn/8.9.3_cuda12 julia/1.8.5

srun julia --project=CIG/ CIG/simulate.jl ${j} ${N} ${i} ${P}

exit 0
EOT
done
