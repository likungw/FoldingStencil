echo $SLURM_CLUSTER_NAME
echo $SLURM_CPU_BIND_VERBOSE
echo $SLURM_CPU_BIND_TYPE
echo $SLURM_CPU_BIND_list
export OMP_NUM_THREADS=24
echo "cpus/task:"   $SLURM_CPUS_PER_TASK
echo "ntasks/node:" $SLURM_NTASKS_PER_CORE
echo "omp_threads:" $OMP_NUM_THREADS

echo "Bx: 20000  tb: 7000  threads: 2  " &&  ./exe_1d5p 10240000 10000 20000  7000  2   >> perf_vec4_blk.txt
./exe_1d5p 2560000 1000 2000   999
echo 4.1
./exe_1d5p 5120000 1000 5120000   2
echo 4.2
./exe_1d5p 10240000 1000 10240000   2
echo 4.3
./exe_1d5p 20480000 1000 20480000   2
