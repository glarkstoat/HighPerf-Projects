+##!/bin/bash
#$ -cwd
#$ -N C-kernel_mpi
#$ -o $JOB_ID.$JOB_NAME.out
#$ -j yes
#$ -pe orte 1
echo "Got $NSLOTS slots." > $JOB_ID.$JOB_NAME.out
echo "Date start: " `date` >> $JOB_ID.$JOB_NAME.out
/opt/global/MPI/OpenMPI/OpenMPI-1.6.5/bin/mpirun -np 16 ./C-kernel_mpi 8192 200
echo "Date end: " `date` >> $JOB_ID.$JOB_NAME.out
