#include <stdio.h>
#include <time.h>

#define N 4            // number of rows/columns
#define STEPS 1            // number of iterations
typedef double cell_type;    // element type of matrix
typedef long result_type;    // type for correctness check

cell_type arr[N][N];
result_type res = 0;

int main(int argc, char* argv[])
{
    int i, j, iter;
    clock_t start, end;
    double cpu_time_used;

    // initialization of arr
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
    start = clock();
    for (iter = 1; iter <= STEPS; iter++)
    {
        // Phase 1: forward/backward along rows
        // forward
        for (i = 1; i <= N - 2; i++)
            for (j = 1; j <= N - 2; j++)
            {
                arr[i][j] = (arr[i][j] + arr[i][j - 1]) / 2 +
                    (arr[i][j] / 3 + arr[i][j - 1] / 3) / N -
                    (arr[i][j] / 5 + arr[i][j - 1] / 5) / N;
            }

        // backward
        for (i = 1; i <= N - 2; i++)
            for (j = N - 2; j >= 1; j--)
            {
                arr[i][j] = (arr[i][j] + arr[i][j + 1]) / 2 +
                    (arr[i][j] / 3 + arr[i][j + 1] / 3) / N -
                    (arr[i][j] / 5 + arr[i][j + 1] / 5) / N;
            }


        // Phase 2: downward/upward along columns
        // downward
        for (i = 1; i <= N - 2; i++)
            for (j = 1; j <= N - 2; j++)
            {
                arr[i][j] = (arr[i][j] + arr[i - 1][j]) / 2 +
                    (arr[i][j] / 3 + arr[i - 1][j] / 3) / N -
                    (arr[i][j] / 5 + arr[i - 1][j] / 5) / N;
            }

        // upward
        for (i = N - 2; i >= 1; i--)
            for (j = 1; j <= N - 2; j++)
            {
                arr[i][j] = (arr[i][j] + arr[i + 1][j]) / 2 +
                    (arr[i][j] / 3 + arr[i + 1][j] / 3) / N -
                    (arr[i][j] / 5 + arr[i + 1][j] / 5) / N;
            }

    }
    end = clock();
    cpu_time_used = (((double)end - (double)start)) / CLOCKS_PER_SEC;
    printf("time: %f \n", cpu_time_used);

    /* perform the correctness check */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            res += arr[i][j];
        }

    printf("Checksum: %f\n", (double)res);

}
