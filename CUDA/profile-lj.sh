#!/bin/bash

N=$1
THREADS=$2
OUT_SUFFIX=$3
N_STEPS=$4

sbatch <<EOT
#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj-profile-${N}-${THREADS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=out/lj_profile-${N}-${THREADS}-${OUT_SUFFIX}.log

# LOAD MODULES
module load CUDA

# BUILD
make

# RUN WITH NSIGHT COMPUTE PROFILING
ncu --set full \
    --export out/ncu-${N}-${THREADS}-${OUT_SUFFIX} \
    srun ./lj.out ${N} ${N_STEPS} ${THREADS}

EOT