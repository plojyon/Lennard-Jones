#!/bin/bash

N=$1
THREADS=$2
OUT_FOLDER=$3

sbatch <<EOT
#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones-${N}-${THREADS}
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=${OUT_FOLDER}/lj_out-${N}-${THREADS}_%j.log

#LOAD MODULES 
module load CUDA

#BUILD
make PARAMETER_N=${N} PARAMETER_THREADS=${THREADS}

#RUN
srun ./lj.out
EOT