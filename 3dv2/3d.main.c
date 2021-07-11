#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include "3d.defines.h"

int main(int argc, char *argv[])
{
	struct timeval start, end;
	printf("KKK!!\n");
	long int i, j, k;
	if (argc != 8)
	{
		printf("usage: %s <NX> <NY> <NZ> <T> <Bx> <By> <tb>\n", argv[0]);
		return 0;
	}
	int NX = atoi(argv[1]);
	int NY = atoi(argv[2]);
	int NZ = atoi(argv[3]);
	int T = atoi(argv[4]);
	int Bx = atoi(argv[5]);
	int By = atoi(argv[6]);
	int tb = atoi(argv[7]);

	if (Bx < (2 * XSLOPE + 1) || By < (2 * YSLOPE + 1) || Bx > NX || By > NY || tb > min(((Bx - 1) / 2) / XSLOPE, ((By - 1) / 2) / YSLOPE))
	{
		return 0;
	}
	double ****A = (double ****)malloc(sizeof(double ***) * 2);
	double ****B = (double ****)malloc(sizeof(double ***) * 2);
	for (i = 0; i < 2; i++)
	{
		A[i] = (double ***)malloc(sizeof(double **) * (NX + 2 * XSLOPE));
		B[i] = (double ***)malloc(sizeof(double **) * (NX + 2 * XSLOPE));
	}
	for (i = 0; i < 2; i++)
	{
		for (j = 0; j < (NX + 2 * XSLOPE); j++)
		{
			A[i][j] = (double **)malloc(sizeof(double *) * (NY + 2 * YSLOPE));
			B[i][j] = (double **)malloc(sizeof(double *) * (NY + 2 * YSLOPE));
		}
	}
	for (i = 0; i < 2; i++)
	{
		for (j = 0; j < (NX + 2 * XSLOPE); j++)
		{
			for (k = 0; k < (NY + 2 * YSLOPE); k++)
			{
				A[i][j][k] = (double *)malloc(sizeof(double) * (NZ + 2 * ZSLOPE));
				B[i][j][k] = (double *)malloc(sizeof(double) * (NZ + 2 * ZSLOPE));
			}
		}
	}

	for (i = 0; i < NX + 2 * XSLOPE; i++)
	{
		for (j = 0; j < NY + 2 * YSLOPE; j++)
		{
			for (k = 0; k < NZ + 2 * ZSLOPE; k++)
			{
				// modified
				srand(time(NULL));
				A[0][i][j][k] = 1.0 * (rand() % 1024);
				// A[0][i][j][k] = i * 2 + j / 2 + k;
				A[1][i][j][k] = 0;
				B[0][i][j][k] = A[0][i][j][k];
				B[1][i][j][k] = 0;
			}
		}
	}

	ompp_3d(B, NX, NY, NZ, T, Bx, By, tb);
	//one_tile_3d(A, NX, NY, NZ, T, Bx, By, tb);
	//one_tile_3d_32(A, NX, NY, NZ, T, Bx, By, tb);
	one_tile_3d27p(A, NX, NY, NZ, T, Bx, By, tb);

	return 0;
}
