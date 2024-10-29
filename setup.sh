nvidia-smi
export SLURM_CPU_BIND="cores"
export PATH=$PATH:$HOME/.julia/bin
export DFNO_3D_GPU=1
# export LD_LIBRARY_PATH=
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0
module load cudnn/8.9.3_cuda12 julia/1.8.5
