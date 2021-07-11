#!/bin/bash
#SBATCH -J cross
#SBATCH -p v3_128
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./runlog/%j.out

module load intel/19.0.3
module li
date
which icc

srun --cpu-bind=v,map_cpu:0 ./test.sh
