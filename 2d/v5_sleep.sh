#!/bin/bash
#SBATCH -p v5_192

srun -p v5_192 -N 1 -n 1 sleep 3600
