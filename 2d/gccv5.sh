#!/bin/bash
#SBATCH -J cross
#SBATCH -p v5_192
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./runlog/%j.out


#module load gcc/8.3.0
#module load gcc/7.4.0
module load gcc/9.1.0
module li
date
which gcc

srun --cpu-bind=v,map_cpu:0 ./test.sh
#srun ./test.sh
