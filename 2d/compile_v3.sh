#!/bin/bash
#SBATCH -J compile
#SBATCH -p v3_128
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./compile-log/%j.out
###SBATCH --cpu-bind=verbose

#module load gcc/8.3.0
#module load gcc/7.4.0
#module load gcc/9.1.0
#module load intel/19.0.3
#module li
date
which icc
which gcc

make clean
make
echo "Compile finished!"
#srun --cpu-bind=v,map_cpu:0 ./test.sh

