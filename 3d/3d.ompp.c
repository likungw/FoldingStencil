#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include "3d.defines.h"
#include <omp.h>

void ompp_3d(double ****A,double ****Y,double ****Z, int NX, int NY, int NZ, int T, int Bx, int By, int tb)
{
	struct timeval start, end;
	long int t, i, j, k;

	int bx = Bx - 2 * (tb * XSLOPE);
	int by = By - 2 * (tb * YSLOPE);

	int ix = Bx + bx;
	int iy = By + by;

	int xnb0 = ceild(NX, ix);
	int ynb0 = ceild(NY, iy);
	int xnb11 = ceild(NX - ix / 2 + 1, ix) + 1;
	int ynb12 = ceild(NY - iy / 2 + 1, iy) + 1;
	int xnb12 = xnb0;
	int ynb11 = ynb0;
	int xnb2 = max(xnb11, xnb0);
	int ynb2 = max(ynb12, ynb0);

	int nb1[2] = {xnb12 * ynb12, xnb11 * ynb11};
	int nb02[2] = {xnb2 * ynb2, xnb0 * ynb0};
	int xnb1[2] = {xnb12, xnb11};
	int xnb02[2] = {xnb2, xnb0};

	int xleft02[2] = {XSLOPE - bx, XSLOPE + (Bx - bx) / 2};
	int ybottom02[2] = {YSLOPE - by, YSLOPE + (By - by) / 2};
	int xleft11[2] = {XSLOPE + (Bx - bx) / 2, XSLOPE - bx};
	int ybottom11[2] = {YSLOPE - (By + by) / 2, YSLOPE};
	int xleft12[2] = {XSLOPE - (Bx + bx) / 2, XSLOPE};
	int ybottom12[2] = {YSLOPE + (By - by) / 2, YSLOPE - by};

	int level = 1;
	int tt, n;
	int x, y, z;
	int ymin, ymax;
	int xmin, xmax;

	gettimeofday(&start, 0);

	for (tt = -tb; tt < T; tt += tb)
	{
#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, t, x, y)
		for (n = 0; n < nb02[level]; n++)
		{

			for (t = max(tt, 0); t < min(tt + 2 * tb, T); t++)
			{

				xmin = max(XSLOPE, xleft02[level] + (n % xnb02[level]) * ix - tb * XSLOPE + myabs(t + 1, tt + tb) * XSLOPE);
				xmax = min(NX + XSLOPE, xleft02[level] + (n % xnb02[level]) * ix + bx + tb * XSLOPE - myabs(t + 1, tt + tb) * XSLOPE);
				ymin = max(YSLOPE, ybottom02[level] + (n / xnb02[level]) * iy - tb * YSLOPE + myabs(t + 1, tt + tb) * YSLOPE);
				ymax = min(NY + YSLOPE, ybottom02[level] + (n / xnb02[level]) * iy + by + tb * YSLOPE - myabs(t + 1, tt + tb) * YSLOPE);

				for (x = xmin; x < xmax; x++)
				{
					for (y = ymin; y < ymax; y++)
					{
#pragma ivdep
#pragma vector always
						for (z = ZSLOPE; z < NZ + ZSLOPE; z++)
						{
							three_kernel(A,Y,Z);
						}
					}
				}
			}
		}

#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, t, x, y)
		for (n = 0; n < nb1[0] + nb1[1]; n++)
		{

			for (t = tt + tb; t < min(tt + 2 * tb, T); t++)
			{
				if (n < nb1[level])
				{
					xmin = max(XSLOPE, xleft11[level] + (n % xnb1[level]) * ix - (t + 1 - tt - tb) * XSLOPE);
					xmax = min(NX + XSLOPE, xleft11[level] + (n % xnb1[level]) * ix + bx + (t + 1 - tt - tb) * XSLOPE);
					ymin = max(YSLOPE, ybottom11[level] + (n / xnb1[level]) * iy + (t + 1 - tt - tb) * YSLOPE);
					ymax = min(NY + YSLOPE, ybottom11[level] + (n / xnb1[level]) * iy + By - (t + 1 - tt - tb) * YSLOPE);
				}
				else
				{

					xmin = max(XSLOPE, xleft12[level] + ((n - nb1[level]) % xnb1[1 - level]) * ix + (t + 1 - tt - tb) * XSLOPE);
					xmax = min(NX + XSLOPE, xleft12[level] + ((n - nb1[level]) % xnb1[1 - level]) * ix + Bx - (t + 1 - tt - tb) * XSLOPE);
					ymin = max(YSLOPE, ybottom12[level] + ((n - nb1[level]) / xnb1[1 - level]) * iy - (t + 1 - tt - tb) * YSLOPE);
					ymax = min(NY + YSLOPE, ybottom12[level] + ((n - nb1[level]) / xnb1[1 - level]) * iy + by + (t + 1 - tt - tb) * YSLOPE);
				}
				for (x = xmin; x < xmax; x++)
				{
					for (y = ymin; y < ymax; y++)
					{
#pragma ivdep
#pragma vector always
						for (z = ZSLOPE; z < NZ + ZSLOPE; z++)
						{
							three_kernel(A,Y,Z);
						}
					}
				}
			}
		}
		level = 1 - level;
	}

	gettimeofday(&end, 0);

	printf("OMPP MStencil/s = %f\n", ((double)NX * NY * NZ * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000L);

#ifdef CHECK
	for (t = 0; t < T; t++)
	{
		for (x = XSLOPE; x < NX + XSLOPE; x++)
		{
			for (y = YSLOPE; y < NY + YSLOPE; y++)
			{
				for (z = ZSLOPE; z < NZ + ZSLOPE; z++)
				{
					kernel(B);
				}
			}
		}
	}
	for (i = XSLOPE; i < NX + XSLOPE; i++)
	{
		for (j = YSLOPE; j < NY + YSLOPE; j++)
		{
			for (k = ZSLOPE; k < NZ + ZSLOPE; k++)
			{
				if (myabs(A[T % 2][i][j][k], B[T % 2][i][j][k]) > TOLERANCE)
					printf("Naive[%d][%d][%d] = %f, Check() = %f: FAILED!\n", i, j, k, B[T % 2][i][j][k], A[T % 2][i][j][k]);
			}
		}
	}
#endif
}
