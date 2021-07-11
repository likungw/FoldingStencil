#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
//#include <avx2intrin.h>
#include "2d.defines.h"
#include <omp.h>

//#define CHECK_ONE_TILE_2D

void tile_2d2s(double ***A, int NX, int NY, int T, int Bx, int By, int tb)
{
	int i, j;
	// A is for initialization, B is for check
	double ***B = (double ***)malloc(sizeof(double **) * 2);
	double ***C = (double ***)malloc(sizeof(double **) * 2);
	for (i = 0; i < 2; i++)
	{
		B[i] = (double **)malloc(sizeof(double *) * (NX + 2 * XSLOPE));
		C[i] = (double **)malloc(sizeof(double *) * (NX + 2 * XSLOPE));
	}
	for (i = 0; i < 2; i++)
	{
		for (j = 0; j < (NX + 2 * XSLOPE); j++)
		{
			B[i][j] = (double *)malloc(sizeof(double) * (NY + 2 * YSLOPE));
			C[i][j] = (double *)malloc(sizeof(double) * (NY + 2 * YSLOPE));
		}
	}
	for (i = 0; i < (NX + 2 * XSLOPE); i++)
	{
		for (j = 0; j < (NY + 2 * YSLOPE); j++)
		{
			B[0][i][j] = A[0][i][j];
			B[1][i][j] = 0;
			C[0][i][j] = A[0][i][j];
			C[1][i][j] = 0;
		}
	}

	int bx = Bx - 2 * (tb * XSLOPE);
	int by = By - 2 * (tb * YSLOPE);

	int ix = Bx + bx;
	int iy = By + by;

	int xnb0 = ceild(NX - bx, ix);
	int ynb0 = ceild(NY - by, iy);
	int xnb11 = ceild(NX + (Bx - bx) / 2, ix);
	int ynb12 = ceild(NY + (By - by) / 2, iy);
	int ynb11 = ynb0;
	int xnb12 = xnb0;
	int xnb2 = xnb11;
	int ynb2 = ynb12;

	int nb1[2] = {xnb11 * ynb11, xnb12 * ynb12};
	int nb02[2] = {xnb0 * ynb0, xnb2 * ynb2};
	int xnb1[2] = {xnb11, xnb12};
	int xnb02[2] = {xnb0, xnb2};

	int xleft02[2] = {XSLOPE + bx, XSLOPE - (Bx - bx) / 2};   // the start x dimension of the first B11 block is bx
	int ybottom02[2] = {YSLOPE + by, YSLOPE - (By - by) / 2}; // the start y dimension of the first B11 block is by
	int xleft11[2] = {XSLOPE, XSLOPE + ix / 2};
	int ybottom11[2] = {YSLOPE + by, YSLOPE - (By - by) / 2};
	int xleft12[2] = {XSLOPE + bx, XSLOPE - (Bx - bx) / 2};
	int ybottom12[2] = {YSLOPE, YSLOPE + iy / 2};

	int level = 0;
	int t, tt, n;
	int x, y;
	int xmin, xmax;
	register int ymin, ymax;

	vec ww, kk, gw;
	vec v11, v12, v13, v14, v20, v21, v22, v23, v24, v25, gw2, v30, v31, v32, v33, v34, v35, gw3;
	vec v40, v41, v42, v43, v44, v45, gw4, v50, v51, v52, v53, v54, v55, gw5, v61, v62, v63, v64;
	vallset(ww, GW);
	vallset(kk, LK);
	int xmod, xup, ymod, yup;

	struct timeval start, end;
	gettimeofday(&start, 0);
	for (tt = -tb; tt < T; tt += tb)
	{
// B0, B2
#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, xmod, xup, ymod, yup, t, x, y, n)
		for (n = 0; n < nb02[level]; n++)
		{
			for (t = max(tt, 0); t < min(tt + 2 * tb, T); t+=2)
			{
				xmin = max(XSLOPE, xleft02[level] + (n % xnb02[level]) * ix + myabs(t + 1, tt + tb) * XSLOPE);
				xmax = min(NX + XSLOPE, xleft02[level] + (n % xnb02[level]) * ix + Bx - myabs(t + 1, tt + tb) * XSLOPE);
				ymin = max(YSLOPE, ybottom02[level] + (n / xnb02[level]) * iy + myabs(t + 1, tt + tb) * YSLOPE);
				ymax = min(NY + YSLOPE, ybottom02[level] + (n / xnb02[level]) * iy + By - myabs(t + 1, tt + tb) * YSLOPE);

				xmod = (xmax - xmin) % 4;
				xup = xmax - xmod;
				ymod = (ymax - ymin) % veclen4;
				yup = ymax - ymod;

				for (x = xmin; x < xmax; x ++)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load
						setload(v11, v12, v13, v14, C[t % 2][x - 1][y]);						
						if (y == ymin)
						{
							vallset(v20, C[t % 2][x][y - 1]);
							setload5(v21, v22, v23, v24, gw2, C[t % 2][x][y]);
						}
						else
						{
							setload(v22, v23, v24, gw2, C[t % 2][x][y + veclen]);
						}
						setload(v31, v32, v33, v34, C[t % 2][x + 1][y]);

						// transpose
						//if(tt==-tb){
						trans(v11, v12, v13, v14);
						trans(v21, v22, v23, v24);
						trans(v31, v32, v33, v34);
						//}

						// permute
						shuffle_headd(v24, v20);
						shuffle_tail(v21, gw2, v25);

						// results are stored in v11, v12, v13, v14
						compute2_5p(v11, v20, v21, v22, v31, ww, kk);
						compute2_5p(v12, v21, v22, v23, v32, ww, kk);
						compute2_5p(v13, v22, v23, v24, v33, ww, kk);
						compute2_5p(v14, v23, v24, v25, v34, ww, kk);

						setload(v41, v42, v43, v44, C[(t+1) % 2][x - 1][y]);
						setload(v51, v52, v53, v54, C[(t+1) % 2][x    ][y]);
						setstore(v11, v12, v13, v14, C[(t + 1) % 2][x+1][y]);

						//trans(v41, v42, v43, v44);

						compute2_up(v11, v12, v13, v14, v41, v42, v43, v44, kk);
						compute2_middle(v51, v11, v11, v12, kk);
						compute2_middle(v52, v11, v12, v13, kk);
						compute2_middle(v53, v12, v13, v14, kk);
						compute2_middle(v54, v13, v14, v14, kk);

						// transpose back
						//if(tt==T-tb){
						//trans(v51, v52, v53, v54);
						trans(v11, v12, v13, v14);
						//}

						// store						
						setstore(v11, v12, v13, v14, C[(t + 1) % 2][x-1][y]);
						setstore(v51, v52, v53, v54, C[(t + 1) % 2][x][y]);

						v20 = v24;
						v21 = gw2;
					}
					#pragma ivdep
					#pragma vector always
					for (y = yup; y < ymax; y++)
					{
						kernel(C);
					}
				}
			}
		}

// B11, B12
#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, xmod, xup, ymod, yup, t, x, y, n)
		for (n = 0; n < nb1[0] + nb1[1]; n++)
		{
			for (t = max(tt, 0); t < min(tt + 2 * tb, T); t+=2)
			{
				xmin = max(XSLOPE, xleft02[level] + (n % xnb02[level]) * ix + myabs(t + 1, tt + tb) * XSLOPE);
				xmax = min(NX + XSLOPE, xleft02[level] + (n % xnb02[level]) * ix + Bx - myabs(t + 1, tt + tb) * XSLOPE);
				ymin = max(YSLOPE, ybottom02[level] + (n / xnb02[level]) * iy + myabs(t + 1, tt + tb) * YSLOPE);
				ymax = min(NY + YSLOPE, ybottom02[level] + (n / xnb02[level]) * iy + By - myabs(t + 1, tt + tb) * YSLOPE);

				xmod = (xmax - xmin) % 4;
				xup = xmax - xmod;
				ymod = (ymax - ymin) % veclen4;
				yup = ymax - ymod;

				for (x = xmin; x < xmax; x ++)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load
						setload(v11, v12, v13, v14, C[t % 2][x - 1][y]);						
						if (y == ymin)
						{
							vallset(v20, C[t % 2][x][y - 1]);
							setload5(v21, v22, v23, v24, gw2, C[t % 2][x][y]);
						}
						else
						{
							setload(v22, v23, v24, gw2, C[t % 2][x][y + veclen]);
						}
						setload(v31, v32, v33, v34, C[t % 2][x + 1][y]);

						// transpose
						//if(tt==-tb){
						trans(v11, v12, v13, v14);
						trans(v21, v22, v23, v24);
						trans(v31, v32, v33, v34);
						//}

						// permute
						shuffle_headd(v24, v20);
						shuffle_tail(v21, gw2, v25);

						// results are stored in v11, v12, v13, v14
						compute2_5p(v11, v20, v21, v22, v31, ww, kk);
						compute2_5p(v12, v21, v22, v23, v32, ww, kk);
						compute2_5p(v13, v22, v23, v24, v33, ww, kk);
						compute2_5p(v14, v23, v24, v25, v34, ww, kk);

						setload(v41, v42, v43, v44, C[(t+1) % 2][x - 1][y]);
						setload(v51, v52, v53, v54, C[(t+1) % 2][x    ][y]);
						setstore(v11, v12, v13, v14, C[(t + 1) % 2][x+1][y]);

						//trans(v41, v42, v43, v44);

						compute2_up(v11, v12, v13, v14, v41, v42, v43, v44, kk);
						compute2_middle(v51, v11, v11, v12, kk);
						compute2_middle(v52, v11, v12, v13, kk);
						compute2_middle(v53, v12, v13, v14, kk);
						compute2_middle(v54, v13, v14, v14, kk);

						// transpose back
						//if(tt==T-tb){
						//trans(v51, v52, v53, v54);
						trans(v11, v12, v13, v14);
						//}

						// store						
						setstore(v11, v12, v13, v14, C[(t + 1) % 2][x-1][y]);
						setstore(v51, v52, v53, v54, C[(t + 1) % 2][x][y]);

						v20 = v24;
						v21 = gw2;
					}
					#pragma ivdep
					#pragma vector always
					for (y = yup; y < ymax; y++)
					{
						kernel(C);
					}
				}
			}
		}
		level = 1 - level;
	}

	gettimeofday(&end, 0);

	printf("2D2S_TILE_2D GStencil/s = %f\n", ((double)NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

#ifdef CHECK_ONE_TILE_2D
	for (t = 0; t < T; t++)
	{
		for (x = XSLOPE; x < NX + XSLOPE; x++)
		{
			for (y = YSLOPE; y < NY + YSLOPE; y++)
			{
				kernel(B);
			}
		}
	}
	check_flag = 1;
	for (i = XSLOPE; i < NX + XSLOPE; i++)
	{
		for (j = YSLOPE; j < NY + YSLOPE; j++)
		{
			if (myabs(C[T % 2][i][j], B[T % 2][i][j]) > TOLERANCE)
			{
				printf("Diff[%d][%d] = %lf, Now = %lf, Check = %lf: FAILED!\n", i, j, C[T % 2][i][j] - B[T % 2][i][j], C[T % 2][i][j], B[T % 2][i][j]);
				check_flag = 0;
			}
		}
	}
	if (check_flag)
	{
		printf("CHECK CORRECT!\n");
	}
#endif
	return;
}
