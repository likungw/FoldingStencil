#!/bin/bash
# test fot vectime
#$SLURM_CPU_BIND=map_cpu:1
export OMP_NUM_THREADS=1
echo $SLURM_CLUSTER_NAME
echo $SLURM_CPU_BIND_VERBOSE
echo $SLURM_CPU_BIND_TYPE
echo $SLURM_CPU_BIND_list
echo $SLURM_CPUS_PER_TASK
echo $SLURM_NTASKS_PER_CORE
echo $OMP_NUM_THREADS


echo "40x40"     && ./exe_2d5p 40   40   1000   40 40 1  >> perf.txt
echo "60x60"     && ./exe_2d5p 60   60   1000   60   60  1  >> perf.txt
echo "80x80"     && ./exe_2d5p 80   80   1000   80   80     1  >> perf.txt
echo "100x100"   && ./exe_2d5p 100  100  1000   100  100    1  >> perf.txt
echo "200x200"   && ./exe_2d5p 200  200  1000   200  200    1  >> perf.txt
echo "400x400"   && ./exe_2d5p 400  400  1000   400  400    1  >> perf.txt
echo "1000x1000" && ./exe_2d5p 1000 1000 1000   1000 1000   1  >> perf.txt
echo "2000x2000" && ./exe_2d5p 2000 2000 1000   2000 2000   1  >> perf.txt
echo "4000x4000" && ./exe_2d5p 4000 4000 1000   4000 4000    1  >> perf.txt

echo "2d test done!"
