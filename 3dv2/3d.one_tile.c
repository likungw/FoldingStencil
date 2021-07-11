#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include "3d.defines.h"
#include <omp.h>

#define CHECK_ONE_TILE_3D

void one_tile_3d(double ****A, int NX, int NY, int NZ, int T, int Bx, int By, int tb)
{
	struct timeval start, end;
	long int t, i, j, k;

	double ****C = (double ****)malloc(sizeof(double ***) * 2);
	double ****C_check = (double ****)malloc(sizeof(double ***) * 2);
	for (i = 0; i < 2; i++)
	{
		C[i] = (double ***)malloc(sizeof(double **) * (NX + 2 * XSLOPE));
		C_check[i] = (double ***)malloc(sizeof(double **) * (NX + 2 * XSLOPE));
		for (j = 0; j < (NX + 2 * XSLOPE); j++)
		{
			C[i][j] = (double **)malloc(sizeof(double *) * (NY + 2 * YSLOPE));
			C_check[i][j] = (double **)malloc(sizeof(double *) * (NY + 2 * YSLOPE));
			for (k = 0; k < (NY + 2 * YSLOPE); k++)
			{
				C[i][j][k] = (double *)malloc(sizeof(double) * (NZ + 2 * ZSLOPE));
				C_check[i][j][k] = (double *)malloc(sizeof(double) * (NZ + 2 * ZSLOPE));
			}
		}
	}
	for (i = 0; i < NX + 2 * XSLOPE; i++)
	{
		for (j = 0; j < NY + 2 * YSLOPE; j++)
		{
			for (k = 0; k < NZ + 2 * ZSLOPE; k++)
			{
				C[0][i][j][k] = A[0][i][j][k];
				C[1][i][j][k] = 0;
				C_check[0][i][j][k] = A[0][i][j][k];
				C_check[1][i][j][k] = 0;
			}
		}
	}

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

	vec ww, kk;
	vallset(ww, GW);
	vallset(kk, LK);
	int zmod = NZ % veclen8;
	int zup = NZ - zmod;

	gettimeofday(&start, 0);

	for (tt = -tb; tt < T; tt += tb)
	{
#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, n, t, x, y, z) firstprivate(ww, kk, zmod, zup)
		for (n = 0; n < nb02[level]; n++)
		{
			vec v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, gw;
			vec v1_front, v1_back, v1_up, v1_down, v2_front, v2_back, v2_up, v2_down;
			vec v3_front, v3_back, v3_up, v3_down, v4_front, v4_back, v4_up, v4_down;
			vec v5_front, v5_back, v5_up, v5_down, v6_front, v6_back, v6_up, v6_down;
			vec v7_front, v7_back, v7_up, v7_down, v8_front, v8_back, v8_up, v8_down;
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
						for (z = ZSLOPE; z < zup; z += veclen8)
						{
							// load
							setload(v1_front, v2_front, v3_front, v4_front, v5_front, v6_front, v7_front, v8_front, C[t % 2][x][y + 1][z]);
							setload(v1_back, v2_back, v3_back, v4_back, v5_back, v6_back, v7_back, v8_back, C[t % 2][x][y - 1][z]);
							setload(v1_up, v2_up, v3_up, v4_up, v5_up, v6_up, v7_up, v8_up, C[t % 2][x + 1][y][z]);
							setload(v1_down, v2_down, v3_down, v4_down, v5_down, v6_down, v7_down, v8_down, C[t % 2][x - 1][y][z]);
							if (z == ZSLOPE)
							{
								vallset(v0, C[t % 2][x][y][z - 1]);
								setloadw(v1, v2, v3, v4, v5, v6, v7, v8, gw, C[t % 2][x][y][z]);
							}
							else
							{
								setload(v2, v3, v4, v5, v6, v7, v8, gw, C[t % 2][x][y][z + veclen]);
							}

							// transpose
							if (t == 0)
							{
								trans(v1_front, v2_front, v3_front, v4_front, v5_front, v6_front, v7_front, v8_front);
								trans(v1_back, v2_back, v3_back, v4_back, v5_back, v6_back, v7_back, v8_back);
								trans(v1_up, v2_up, v3_up, v4_up, v5_up, v6_up, v7_up, v8_up);
								trans(v1_down, v2_down, v3_down, v4_down, v5_down, v6_down, v7_down, v8_down);
								trans(v1, v2, v3, v4, v5, v6, v7, v8);
								if (x == XSLOPE)
								{
									setstore(v1_down, v2_down, v3_down, v4_down, v5_down, v6_down, v7_down, v8_down, C[0][x - 1][y][z]);
								}
								else if (x == NX)
								{
									setstore(v1_up, v2_up, v3_up, v4_up, v5_up, v6_up, v7_up, v8_up, C[0][x + 1][y][z]);
								}
								if (y == YSLOPE)
								{
									setstore(v1_back, v2_back, v3_back, v4_back, v5_back, v6_back, v7_back, v8_back, C[0][x][y - 1][z]);
								}
								else if (y == NY)
								{
									setstore(v1_front, v2_front, v3_front, v4_front, v5_front, v6_front, v7_front, v8_front, C[0][x][y + 1][z]);
								}
							}

							// permute
							shuffle_headd(v8, v0);
							shuffle_tail(v1, gw, v9);

							// compute. results are stored in v0, v1, v2, v3
							computed_7p(v0, v1, v2, v1_front, v1_back, v1_up, v1_down, ww, kk);
							computed_7p(v1, v2, v3, v2_front, v2_back, v2_up, v2_down, ww, kk);
							computed_7p(v2, v3, v4, v3_front, v3_back, v3_up, v3_down, ww, kk);
							computed_7p(v3, v4, v5, v4_front, v4_back, v4_up, v4_down, ww, kk);
							computed_7p(v4, v5, v6, v5_front, v5_back, v5_up, v5_down, ww, kk);
							computed_7p(v5, v6, v7, v6_front, v6_back, v6_up, v6_down, ww, kk);
							computed_7p(v6, v7, v8, v7_front, v7_back, v7_up, v7_down, ww, kk);
							computed_7p(v7, v8, v9, v8_front, v8_back, v8_up, v8_down, ww, kk);

							// transpose back
							if (t == (T - 1))
							{
								trans(v0, v1, v2, v3, v4, v5, v6, v7);
							}

							// store
							setstore(v0, v1, v2, v3, v4, v5, v6, v7, C[(t + 1) % 2][x][y][z]);

							v0 = v8;
							v1 = gw;
						}

#pragma ivdep
#pragma vector always
						for (z = zup + 1; z < NZ + ZSLOPE; z++)
						{
							kernel(C);
						}
					}
				}
			}
		}

#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, n, t, x, y, z) firstprivate(ww, kk, zmod, zup)
		for (n = 0; n < nb1[0] + nb1[1]; n++)
		{
			vec v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, gw;
			vec v1_front, v1_back, v1_up, v1_down, v2_front, v2_back, v2_up, v2_down;
			vec v3_front, v3_back, v3_up, v3_down, v4_front, v4_back, v4_up, v4_down;
			vec v5_front, v5_back, v5_up, v5_down, v6_front, v6_back, v6_up, v6_down;
			vec v7_front, v7_back, v7_up, v7_down, v8_front, v8_back, v8_up, v8_down;
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
						for (z = ZSLOPE; z < zup; z += veclen8)
						{
							// load
							setload(v1_front, v2_front, v3_front, v4_front, v5_front, v6_front, v7_front, v8_front, C[t % 2][x][y + 1][z]);
							setload(v1_back, v2_back, v3_back, v4_back, v5_back, v6_back, v7_back, v8_back, C[t % 2][x][y - 1][z]);
							setload(v1_up, v2_up, v3_up, v4_up, v5_up, v6_up, v7_up, v8_up, C[t % 2][x + 1][y][z]);
							setload(v1_down, v2_down, v3_down, v4_down, v5_down, v6_down, v7_down, v8_down, C[t % 2][x - 1][y][z]);
							if (z == ZSLOPE)
							{
								vallset(v0, C[t % 2][x][y][z - 1]);
								setloadw(v1, v2, v3, v4, v5, v6, v7, v8, gw, C[t % 2][x][y][z]);
							}
							else
							{
								setload(v2, v3, v4, v5, v6, v7, v8, gw, C[t % 2][x][y][z + veclen]);
							}

							// transpose
							if (t == 0)
							{
								trans(v1_front, v2_front, v3_front, v4_front, v5_front, v6_front, v7_front, v8_front);
								trans(v1_back, v2_back, v3_back, v4_back, v5_back, v6_back, v7_back, v8_back);
								trans(v1_up, v2_up, v3_up, v4_up, v5_up, v6_up, v7_up, v8_up);
								trans(v1_down, v2_down, v3_down, v4_down, v5_down, v6_down, v7_down, v8_down);
								trans(v1, v2, v3, v4, v5, v6, v7, v8);
								if (x == XSLOPE)
								{
									setstore(v1_down, v2_down, v3_down, v4_down, v5_down, v6_down, v7_down, v8_down, C[0][x - 1][y][z]);
								}
								else if (x == NX)
								{
									setstore(v1_up, v2_up, v3_up, v4_up, v5_up, v6_up, v7_up, v8_up, C[0][x + 1][y][z]);
								}
								if (y == YSLOPE)
								{
									setstore(v1_back, v2_back, v3_back, v4_back, v5_back, v6_back, v7_back, v8_back, C[0][x][y - 1][z]);
								}
								else if (y == NY)
								{
									setstore(v1_front, v2_front, v3_front, v4_front, v5_front, v6_front, v7_front, v8_front, C[0][x][y + 1][z]);
								}
							}

							// permute
							shuffle_headd(v8, v0);
							shuffle_tail(v1, gw, v9);

							// compute. results are stored in v0, v1, v2, v3
							computed_7p(v0, v1, v2, v1_front, v1_back, v1_up, v1_down, ww, kk);
							computed_7p(v1, v2, v3, v2_front, v2_back, v2_up, v2_down, ww, kk);
							computed_7p(v2, v3, v4, v3_front, v3_back, v3_up, v3_down, ww, kk);
							computed_7p(v3, v4, v5, v4_front, v4_back, v4_up, v4_down, ww, kk);
							computed_7p(v4, v5, v6, v5_front, v5_back, v5_up, v5_down, ww, kk);
							computed_7p(v5, v6, v7, v6_front, v6_back, v6_up, v6_down, ww, kk);
							computed_7p(v6, v7, v8, v7_front, v7_back, v7_up, v7_down, ww, kk);
							computed_7p(v7, v8, v9, v8_front, v8_back, v8_up, v8_down, ww, kk);

							// transpose back
							if (t == (T - 1))
							{
								trans(v0, v1, v2, v3, v4, v5, v6, v7);
							}

							// store
							setstore(v0, v1, v2, v3, v4, v5, v6, v7, C[(t + 1) % 2][x][y][z]);

							v0 = v8;
							v1 = gw;
						}

#pragma ivdep
#pragma vector always
						for (z = zup + 1; z < NZ + ZSLOPE; z++)
						{
							kernel(C);
						}
					}
				}
			}
		}
		level = 1 - level;
	}

	gettimeofday(&end, 0);

	printf("ONE_TILE_3D MStencil/s = %f\n", ((double)NX * NY * NZ * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000L);

#ifdef CHECK_ONE_TILE_3D
	for (t = 0; t < T; t++)
	{
		for (x = XSLOPE; x < NX + XSLOPE; x++)
		{
			for (y = YSLOPE; y < NY + YSLOPE; y++)
			{
				for (z = ZSLOPE; z < NZ + ZSLOPE; z++)
				{
					kernel(C_check);
				}
			}
		}
	}
	int my_check_flag = 1;
	for (i = XSLOPE; i < NX + XSLOPE; i++)
	{
		for (j = YSLOPE; j < NY + YSLOPE; j++)
		{
			for (k = ZSLOPE; k < NZ + ZSLOPE; k++)
			{
				if (myabs(C[T % 2][i][j][k], C_check[T % 2][i][j][k]) > TOLERANCE)
				{
					printf("Diff[%ld][%ld][%ld] = %lf, Now = %lf, Check = %lf: FAILED!\n", i, j, k, C[T % 2][i][j][k] - C_check[T % 2][i][j][k], C[T % 2][i][j][k], C_check[T % 2][i][j][k]);
					my_check_flag = 0;
				}
			}
		}
	}
	if (my_check_flag)
	{
		printf("CHECK CORRECT!\n");
	}
#endif
}
