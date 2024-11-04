nvidia-smi
export SLURM_CPU_BIND="cores"
export PATH=$PATH:$HOME/.julia/bin
export DFNO_3D_GPU=1
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0

export JULIA_NUM_THREADS=$SLURM_CPUS_ON_NODE
export DEVITO_LANGUAGE=openacc
export DEVITO_ARCH=nvc
export DEVITO_PLATFORM=nvidiaX

# export LD_LIBRARY_PATH=

module load cudnn/8.9.3_cuda12 julia/1.8.5
