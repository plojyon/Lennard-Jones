#!/bin/bash

N=$1
OUT_SUFFIX=$2
N_STEPS=$3

sbatch <<EOT
#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones-reference-${N}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=out_reference/lj_out-${N}-${THREADS}-${OUT_SUFFIX}.log

#LOAD MODULES 
module load CUDA

#BUILD
make

#RUN
srun ./lj.out ${N} ${N_STEPS}
EOT