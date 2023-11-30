EXECS = C-kernel_mpi jobscript
MPICC? = mpicc

all: ${EXECS}

C-kernel_mpi: C-kernel_mpi.c
	${MPICC?} -o C-kernel_mpi -O3 C-kernel_mpi.c

jobscript: jobscript.sh
	qsub jobscript.sh

make clean:
	rm *.out  
