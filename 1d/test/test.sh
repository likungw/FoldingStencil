echo $SLURM_CLUSTER_NAME
echo $SLURM_CPU_BIND_VERBOSE
echo $SLURM_CPU_BIND_TYPE
echo $SLURM_CPU_BIND_list
export OMP_NUM_THREADS=24
echo "cpus/task:"   $SLURM_CPUS_PER_TASK
echo "ntasks/node:" $SLURM_NTASKS_PER_CORE
echo "omp_threads:" $OMP_NUM_THREADS
./exe_1d3p 10240000 1000 1000   400
