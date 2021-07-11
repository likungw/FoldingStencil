#!/bin/bash
#SBATCH -p v3_128
##SBATCH -p v3_test

srun -p v3_128 -N 1 -n 1 sleep 3600
