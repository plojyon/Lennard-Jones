#!/bin/bash

N=$1
THREADS=$2
OUT_SUFFIX=$3
N_STEPS=$4

sbatch <<EOT
#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones-${N}-${THREADS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=out/lj_out-${N}-${THREADS}-${OUT_SUFFIX}.log

#LOAD MODULES 
module load CUDA

#BUILD
make N=${N} THREADS=${THREADS} N_STEPS=${N_STEPS}

#RUN
srun ./lj.out
EOT