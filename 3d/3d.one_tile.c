#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include "3d.defines.h"
//#include <omp.h>

//#define CHECK_ONE_TILE_3D

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
	int zmod = NZ % veclen4;
	int zup = NZ - zmod;

	gettimeofday(&start, 0);
#pragma omp parallel for private(z) collapse(2)
	for (x = 0; x < NX + 2 * XSLOPE; x++)
	{
		for (y = 0; y < NY + 2 * YSLOPE; y++)
		{
			vec V1, V2, V3, V4;
			for (z = ZSLOPE; z <= NZ + ZSLOPE - veclen4; z += veclen4)
			{
				setload(V1, V2, V3, V4, C[0][x][y][z]);
				trans(V1, V2, V3, V4);
				setstore(V1, V2, V3, V4, C[0][x][y][z]);
			}
		}
	}

	for (tt = -tb; tt < T; tt += tb)
	{
#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, n, t, x, y, z) firstprivate(ww, kk, zmod, zup)
		for (n = 0; n < nb02[level]; n++)
		{
			vec v0, v1, v2, v3, v4, v5, gw;
			vec v_front, v_back, v_up, v_down;
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
						vallset(v0, C[t % 2][x][y][0]);
						vload(v1, C[t % 2][x][y][ZSLOPE]);
						for (z = ZSLOPE; z < zup; z += veclen4)
						{
							// load

							setload(v2, v3, v4, gw, C[t % 2][x][y][z + veclen]);

							// permute
							shuffle_headd(v4, v0);
							shuffle_tail(v1, gw, v5);

							vload(v_front, C[t % 2][x][y + 1][z]);
							vload(v_back, C[t % 2][x][y - 1][z]);
							vload(v_up, C[t % 2][x + 1][y][z]);
							vload(v_down, C[t % 2][x - 1][y][z]);

							computed_7p(v0, v1, v2, v_front, v_back, v_up, v_down, ww, kk);

							vload(v_front, C[t % 2][x][y + 1][z + veclen]);
							vload(v_back, C[t % 2][x][y - 1][z + veclen]);
							vload(v_up, C[t % 2][x + 1][y][z + veclen]);
							vload(v_down, C[t % 2][x - 1][y][z + veclen]);

							computed_7p(v1, v2, v3, v_front, v_back, v_up, v_down, ww, kk);

							vload(v_front, C[t % 2][x][y + 1][z + 2 * veclen]);
							vload(v_back, C[t % 2][x][y - 1][z + 2 * veclen]);
							vload(v_up, C[t % 2][x + 1][y][z + 2 * veclen]);
							vload(v_down, C[t % 2][x - 1][y][z + 2 * veclen]);

							computed_7p(v2, v3, v4, v_front, v_back, v_up, v_down, ww, kk);

							vload(v_front, C[t % 2][x][y + 1][z + 3 * veclen]);
							vload(v_back, C[t % 2][x][y - 1][z + 3 * veclen]);
							vload(v_up, C[t % 2][x + 1][y][z + 3 * veclen]);
							vload(v_down, C[t % 2][x - 1][y][z + 3 * veclen]);

							computed_7p(v3, v4, v5, v_front, v_back, v_up, v_down, ww, kk);

							// transpose back
							if (t == (T - 1))
							{
								trans(v0, v1, v2, v3);
							}

							// store
							setstore(v0, v1, v2, v3, C[(t + 1) % 2][x][y][z]);

							v0 = v4;
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
			vec v0, v1, v2, v3, v4, v5, gw;
			vec v_front, v_back, v_up, v_down;
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
						vallset(v0, C[t % 2][x][y][0]);
						vload(v1, C[t % 2][x][y][ZSLOPE]);

						for (z = ZSLOPE; z < zup; z += veclen4)
						{
							setload(v2, v3, v4, gw, C[t % 2][x][y][z + veclen]);

							// permute
							shuffle_headd(v4, v0);
							shuffle_tail(v1, gw, v5);

							vload(v_front, C[t % 2][x][y + 1][z]);
							vload(v_back, C[t % 2][x][y - 1][z]);
							vload(v_up, C[t % 2][x + 1][y][z]);
							vload(v_down, C[t % 2][x - 1][y][z]);

							computed_7p(v0, v1, v2, v_front, v_back, v_up, v_down, ww, kk);

							vload(v_front, C[t % 2][x][y + 1][z + veclen]);
							vload(v_back, C[t % 2][x][y - 1][z + veclen]);
							vload(v_up, C[t % 2][x + 1][y][z + veclen]);
							vload(v_down, C[t % 2][x - 1][y][z + veclen]);

							computed_7p(v1, v2, v3, v_front, v_back, v_up, v_down, ww, kk);

							vload(v_front, C[t % 2][x][y + 1][z + 2 * veclen]);
							vload(v_back, C[t % 2][x][y - 1][z + 2 * veclen]);
							vload(v_up, C[t % 2][x + 1][y][z + 2 * veclen]);
							vload(v_down, C[t % 2][x - 1][y][z + 2 * veclen]);

							computed_7p(v2, v3, v4, v_front, v_back, v_up, v_down, ww, kk);

							vload(v_front, C[t % 2][x][y + 1][z + 3 * veclen]);
							vload(v_back, C[t % 2][x][y - 1][z + 3 * veclen]);
							vload(v_up, C[t % 2][x + 1][y][z + 3 * veclen]);
							vload(v_down, C[t % 2][x - 1][y][z + 3 * veclen]);

							computed_7p(v3, v4, v5, v_front, v_back, v_up, v_down, ww, kk);

							// transpose back
							if (t == (T - 1))
							{
								trans(v0, v1, v2, v3);
							}

							// store
							setstore(v0, v1, v2, v3, C[(t + 1) % 2][x][y][z]);

							v0 = v4;
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

	printf("Our 2steps MStencil/s = %f\n", ((double)NX * NY * NZ * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000L);

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
