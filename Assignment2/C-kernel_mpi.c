/*
Parallel MPI program using a combination of Scatter/Gather and Send/Receive
to maximize performance. Divides up the work equally among the processes and
carrier out different stencil operations on a given matrix.
Due to the dependence between the operations they were divided into two phases.
First the row- and then the column operations.

Important constraint:
Choose the amount of processes such that N % size = 0.
*/

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define MASTER 0	// Master process diving up the work among the slaves
typedef double cell_type;    // element type of matrix
typedef long result_type;    // type for correctness check

int main(int argc, char* argv[])
{
	// VARIABLES ARE INITIALIZED
	int rank, size, i, j, iter;
	double start, finish; // For time measurement

	// INITIALIZE MPI COMMANDS
	MPI_Init(&argc, &argv); // Initialize MPI
	MPI_Comm_size(MPI_COMM_WORLD, &size); // Amount of processes involved
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Process ID of active process

	if (argc <= 2) {
		if (rank == MASTER) {
			printf("Usage: program <argument1> <argument2> \n");
			MPI_Finalize();
			return EXIT_FAILURE;
		}
	}

	int N = atoi(argv[1]);
	int STEPS = atoi(argv[2]);

	if (rank == MASTER) {
		printf("N: %d, STEPS: %d \n", N, STEPS);
	}

	int count = N / size; // Amount of work for each process
	if (N % size != 0 && rank == MASTER) { // Check size constraint
		printf("Constraint violated....\n");
		printf("Provide a number of processes (size), such that %d %% size = 0 \n", N);
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// INITIALIZE DATA TYPES
	MPI_Datatype colT, subT; // Derived data types for column operations
	MPI_Type_vector(N, count, N, MPI_DOUBLE, &colT); // Define Vector of column elements
	MPI_Type_create_resized(colT, 0, count * sizeof(double), &subT); // Make Vector contiguous
	MPI_Type_commit(&subT); // Commit to custom data type

	cell_type arr[N][N];
	result_type res = 0;
	cell_type col[N][count], row[count][N]; // Define arrays for rows and columns for every process

	//  INITIALIZE MATRIX
	if (rank == MASTER) {

		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++)
			{
				arr[i][j] = (i * 2 + j * 3) / 3;
			}

		for (i = 0; i < N; i++)
		{
			arr[0][i] = 9.50;
			arr[N - 1][i] = 99.50;
			arr[i][0] = 999.50;
			arr[i][N - 1] = 9999.50;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime(); // Starts the timer

	// STARTS THE ITERATIONS ACORDING TO HOW MANY STEPS WERE DEFINED
	for (iter = 1; iter <= STEPS; iter++) {
		// --------------------------------------- PHASE 1 ---------------------------------------------
		// Row operations are conducted
		// Master divides the work equally among slaves via Scatter

		// Scatter the the rows of the matrix to the processes. Each process reveices N*count rows.
		// The values are written into the buffer row for every process.
		MPI_Scatter(&(arr[0][0]), N * count, MPI_DOUBLE, &(row[0][0]), N * count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

		// Combination of Forward & Backward written into one single for-loop

		if (rank == MASTER) { // Necessary to ensure, that the first row is not overwritten,
						// in order to stay compliant with the computation in C-kernel.c
			for (i = 1; i < count; i++)
				for (j = 1; j <= N - 2; j++)
				{
					row[i][j] = (row[i][j] + row[i][j - 1]) / 2 +
						(row[i][j] / 3 + row[i][j - 1] / 3) / N -
						(row[i][j] / 5 + row[i][j - 1] / 5) / N;
				}

			for (i = 1; i < count; i++)
				for (j = N - 2; j >= 1; j--)
				{
					row[i][j] = (row[i][j] + row[i][j + 1]) / 2 +
						(row[i][j] / 3 + row[i][j + 1] / 3) / N -
						(row[i][j] / 5 + row[i][j + 1] / 5) / N;
				}
		}

		else if (rank == size - 1) { // Necessary to ensure, that the last row is not overwritten,
									// in order to stay compliant with the computation in C-kernel.c
			for (i = 0; i < count - 1; i++)
				for (j = 1; j <= N - 2; j++)
				{
					row[i][j] = (row[i][j] + row[i][j - 1]) / 2 +
						(row[i][j] / 3 + row[i][j - 1] / 3) / N -
						(row[i][j] / 5 + row[i][j - 1] / 5) / N;
				}

			for (i = 0; i < count - 1; i++)
				for (j = N - 2; j >= 1; j--)
				{
					row[i][j] = (row[i][j] + row[i][j + 1]) / 2 +
						(row[i][j] / 3 + row[i][j + 1] / 3) / N -
						(row[i][j] / 5 + row[i][j + 1] / 5) / N;
				}
		}

		else { // All other processes conduct their operation on all available rows
			for (i = 0; i < count; i++)
				for (j = 1; j <= N - 2; j++)
				{
					row[i][j] = (row[i][j] + row[i][j - 1]) / 2 +
						(row[i][j] / 3 + row[i][j - 1] / 3) / N -
						(row[i][j] / 5 + row[i][j - 1] / 5) / N;
				}

			for (i = 0; i < count; i++)
				for (j = N - 2; j >= 1; j--)
				{
					row[i][j] = (row[i][j] + row[i][j + 1]) / 2 +
						(row[i][j] / 3 + row[i][j + 1] / 3) / N -
						(row[i][j] / 5 + row[i][j + 1] / 5) / N;
				}
		}

		// Gathers all the rows of the individual processes and overwrites the rows of the array in the MASTER
		MPI_Gather(&(row[0][0]), N * count, MPI_DOUBLE, &(arr[0][0]), N * count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

		// --------------------------------------- PHASE 2 ---------------------------------------------

		// Column operations are conducted
		// Master divides the work equally among slaves via Scatter

		// Scatter the columns of the matrix to the processes. Each process reveices N*count columns.
		// The values are written into the buffer col for every process.
		MPI_Scatter(&(arr[0][0]), 1, subT, &(col[0][0]), N * count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

		// Combination of Upward & Downward written into one single for-loop

		if (rank == MASTER) { // Necessary to ensure, that the first column is not overwritten,
						// in order to stay compliant with the computation in C-kernel.c
			for (i = 1; i <= N - 2; i++)
				for (j = 1; j < count; j++)
				{
					arr[i][j] = (arr[i][j] + arr[i - 1][j]) / 2 +
						(arr[i][j] / 3 + arr[i - 1][j] / 3) / N -
						(arr[i][j] / 5 + arr[i - 1][j] / 5) / N;
				}

			for (i = N - 2; i >= 1; i--)
				for (j = 1; j < count; j++)
				{
					arr[i][j] = (arr[i][j] + arr[i + 1][j]) / 2 +
						(arr[i][j] / 3 + arr[i + 1][j] / 3) / N -
						(arr[i][j] / 5 + arr[i + 1][j] / 5) / N;
				}

		}

		else if (rank == (size - 1)) { // Necessary to ensure, that the last column is not overwritten,
										// in order to stay compliant with the computation in C-kernel.c
			for (i = 1; i <= N - 2; i++)
				for (j = 0; j < count - 1; j++)
				{
					col[i][j] = (col[i][j] + col[i - 1][j]) / 2 +
						(col[i][j] / 3 + col[i - 1][j] / 3) / N -
						(col[i][j] / 5 + col[i - 1][j] / 5) / N;
				}

			for (i = N - 2; i >= 1; i--)
				for (j = 0; j < count - 1; j++)
				{
					col[i][j] = (col[i][j] + col[i + 1][j]) / 2 +
						(col[i][j] / 3 + col[i + 1][j] / 3) / N -
						(col[i][j] / 5 + col[i + 1][j] / 5) / N;
				}
		}

		else { // All other processes conduct their operation on all available columns
			for (i = 1; i <= N - 2; i++)
				for (j = 0; j < count; j++)
				{
					col[i][j] = (col[i][j] + col[i - 1][j]) / 2 +
						(col[i][j] / 3 + col[i - 1][j] / 3) / N -
						(col[i][j] / 5 + col[i - 1][j] / 5) / N;
				}

			for (i = N - 2; i >= 1; i--)
				for (j = 0; j < count; j++)
				{
					col[i][j] = (col[i][j] + col[i + 1][j]) / 2 +
						(col[i][j] / 3 + col[i + 1][j] / 3) / N -
						(col[i][j] / 5 + col[i + 1][j] / 5) / N;
				}
		}

		if (rank > 0) { // All slaves send their computed columns to the master
			MPI_Send(&(col[0][0]), N * count, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
		}

		else if (rank == MASTER) { // Master receives the columns from the slaves and overwrites the columns of the array
			for (i = 1; i < size; i++) {
				MPI_Recv(&(arr[0][count * i]), 1, subT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	finish = MPI_Wtime(); // Ends the time measurement
	printf("Execution time [s]: %f from proc: %d \n", finish - start, rank); // Prints the elapsed time of each process

	// CALCULATE CHECKSUM
	if (rank == MASTER) {
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++)
			{
				res += arr[i][j];
			}
		}
		printf("Checksum: %f\n", (double)res);
	}

	// CLEAN UP
	MPI_Finalize();
	return 0;
}
