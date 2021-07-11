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

void our_2d2s5p(double ***A, int NX, int NY, int T, int Bx, int By, int tb)
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
	vec v40, v41, v42, v43, v44, v45, gw4, v50, v51, v52, v53, v54, v55, gw5;
	vec v60, v61, v62, v63, v64, v65, gw6, v71, v72, v73, v74, v2_21, v2_22, v2_23, v2_24;
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

				for (x = xmin; x < xup; x += 1)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load
						setload(v11, v12, v13, v14, C[t % 2][x - 1][y]);
						if(x == xmin){
							if (y == ymin)
							{
								vallset(v20, C[t % 2][x][y - 1]);
								setload5(v21, v22, v23, v24, gw2, C[t % 2][x][y]);
								vallset(v30, C[t % 2][x + 1][y - 1]);
								setload5(v31, v32, v33, v34, gw3, C[t % 2][x + 1][y]);
								vallset(v40, C[t % 2][x + 2][y - 1]);
								setload5(v41, v42, v43, v44, gw4, C[t % 2][x + 2][y]);
							}
							else
							{
								setload(v22, v23, v24, gw2, C[t % 2][x][y + veclen]);
								setload(v32, v33, v34, gw3, C[t % 2][x + 1][y + veclen]);
								setload(v42, v43, v44, gw4, C[t % 2][x + 2][y + veclen]);
							}
							setload(v51, v52, v53, v54, C[t % 2][x + 3][y]);
						}else
						{
						
							setload(v51, v52, v53, v54, C[t % 2][x + 0][y]);
						}
						

						// transpose
						if(tt<10*tb){
						
						trans(v51, v52, v53, v54);
						}
						

						// permute
						if(x ==xmin){
							shuffle_headd(v24, v20);
							shuffle_tail(v21, gw2, v25);
							shuffle_headd(v34, v30);
							shuffle_tail(v31, gw3, v35);
							shuffle_headd(v44, v40);
							shuffle_tail(v41, gw4, v45);
							

							// results are stored in v11, v12, v13, v14
							compute2_5p(v11, v20, v21, v22, v31, ww, kk);
							compute2_5p(v12, v21, v22, v23, v32, ww, kk);
							compute2_5p(v13, v22, v23, v24, v33, ww, kk);
							compute2_5p(v14, v23, v24, v25, v34, ww, kk);

							v20 = v24;
							compute2_5p(v21, v30, v31, v32, v41, ww, kk);
							compute2_5p(v22, v31, v32, v33, v42, ww, kk);
							compute2_5p(v23, v32, v33, v34, v43, ww, kk);
							compute2_5p(v24, v33, v34, v35, v44, ww, kk);

							v30 = v34;
							compute2_5p(v31, v40, v41, v42, v51, ww, kk);
							compute2_5p(v32, v41, v42, v43, v52, ww, kk);
							compute2_5p(v33, v42, v43, v44, v53, ww, kk);
							compute2_5p(v34, v43, v44, v45, v54, ww, kk);

							shuffle_headd(v34, v30);
							shuffle_tail(v31, gw3, v35);
							compute2_5p(v2_21, v30, v31, v32, v41, ww, kk);
							compute2_5p(v2_22, v31, v32, v33, v42, ww, kk);
							compute2_5p(v2_23, v32, v33, v34, v43, ww, kk);
							compute2_5p(v2_24, v33, v34, v35, v44, ww, kk);
							// transpose back
							if(tt<10*tb){ 
							trans(v2_21, v2_22, v2_23, v2_24);
							}
							setstore(v2_21, v2_22, v2_23, v2_24, C[(t /2) % 2][x+1 ][y]);

							v21 = gw2;
							v31 = gw3;
							v41 = gw4;
							v50 = v54;
							v51 = gw5;
						}
						else
						{
							shuffle_headd(v44, v40);
							shuffle_tail(v41, gw4, v45);
							compute2_5p(v31, v40, v41, v42, v51, ww, kk);
							compute2_5p(v32, v41, v42, v43, v52, ww, kk);
							compute2_5p(v33, v42, v43, v44, v53, ww, kk);
							compute2_5p(v34, v43, v44, v45, v54, ww, kk);

							shuffle_headd(v34, v30);
							shuffle_tail(v31, gw3, v35);
							compute2_5p(v2_21, v30, v31, v32, v41, ww, kk);
							compute2_5p(v2_22, v31, v32, v33, v42, ww, kk);
							compute2_5p(v2_23, v32, v33, v34, v43, ww, kk);
							compute2_5p(v2_24, v33, v34, v35, v44, ww, kk);
							

							if(tt<10*tb){ 
							trans(v2_21, v2_22, v2_23, v2_24);
							}
							setstore(v2_21, v2_22, v2_23, v2_24, C[(t /2) % 2][x +1][y]);
						}
						
					}
				}
				

				// deal with the rest elements of each row
				for (x = xmin; x < xup; x++)
				{
					#pragma ivdep
					#pragma vector always
					for (y = yup; y < ymax; y++)
					{
						kernel_5p(C);
					}
				}

				// deal with the rest rows
				for (x = xup; x < xmax; x++)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load
						setload(v11, v12, v13, v14, C[t % 2][x - 1][y]);
						if (y == ymin)
						{
							vallset(v20, C[t % 2][x][y - 1]);
							setload5(v21, v22, v23, v24, gw, C[t % 2][x][y]);
						}
						else
						{
							setload(v22, v23, v24, gw, C[t % 2][x][y + veclen]);
						}
						setload(v31, v32, v33, v34, C[t % 2][x + 1][y]);

						// transpose
						if(tt<10*tb){
						trans(v11, v12, v13, v14);
						trans(v21, v22, v23, v24);
						trans(v31, v32, v33, v34);
						}
						// permute
						shuffle_headd(v24, v20);
						shuffle_tail(v21, gw, v25);

						// compute. result stored in v20, v21, v22, v23
						compute_5p(v11, v20, v21, v22, v31, ww, kk);
						compute_5p(v12, v21, v22, v23, v32, ww, kk);
						compute_5p(v13, v22, v23, v24, v33, ww, kk);
						compute_5p(v14, v23, v24, v25, v34, ww, kk);

						// transpose back
						if(tt<10*tb)
						trans(v20, v21, v22, v23);

						// store
						setstore(v20, v21, v22, v23, C[(t /2) % 2][x][y]);

						v20 = v24;
						v21 = gw;
					}

					#pragma ivdep
					#pragma vector always
					for (y = yup; y < ymax; y++)
					{
						kernel_5p(C);
					}
				}
			}
		}
		
// B11, B12
#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, xmod, xup, ymod, yup, t, x, y, n)
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

				xmod = (xmax - xmin) % 4;
				xup = xmax - xmod;
				ymod = (ymax - ymin) % veclen4;
				yup = ymax - ymod;
for (x = xmin; x < xup; x += 1)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load
						setload(v11, v12, v13, v14, C[t % 2][x - 1][y]);
						if(x == xmin){
							if (y == ymin)
							{
								vallset(v20, C[t % 2][x][y - 1]);
								setload5(v21, v22, v23, v24, gw2, C[t % 2][x][y]);
								vallset(v30, C[t % 2][x + 1][y - 1]);
								setload5(v31, v32, v33, v34, gw3, C[t % 2][x + 1][y]);
								vallset(v40, C[t % 2][x + 2][y - 1]);
								setload5(v41, v42, v43, v44, gw4, C[t % 2][x + 2][y]);
							}
							else
							{
								setload(v22, v23, v24, gw2, C[t % 2][x][y + veclen]);
								setload(v32, v33, v34, gw3, C[t % 2][x + 1][y + veclen]);
								setload(v42, v43, v44, gw4, C[t % 2][x + 2][y + veclen]);
							}
							setload(v51, v52, v53, v54, C[t % 2][x + 3][y]);
						}else
						{
						
							setload(v51, v52, v53, v54, C[t % 2][x + 0][y]);
						}
						

						// transpose
						if(tt<10*tb){
						
						trans(v51, v52, v53, v54);
						}
						

						// permute
						if(x ==xmin){
							shuffle_headd(v24, v20);
							shuffle_tail(v21, gw2, v25);
							shuffle_headd(v34, v30);
							shuffle_tail(v31, gw3, v35);
							shuffle_headd(v44, v40);
							shuffle_tail(v41, gw4, v45);
							

							// results are stored in v11, v12, v13, v14
							compute2_5p(v11, v20, v21, v22, v31, ww, kk);
							compute2_5p(v12, v21, v22, v23, v32, ww, kk);
							compute2_5p(v13, v22, v23, v24, v33, ww, kk);
							compute2_5p(v14, v23, v24, v25, v34, ww, kk);

							v20 = v24;
							compute2_5p(v21, v30, v31, v32, v41, ww, kk);
							compute2_5p(v22, v31, v32, v33, v42, ww, kk);
							compute2_5p(v23, v32, v33, v34, v43, ww, kk);
							compute2_5p(v24, v33, v34, v35, v44, ww, kk);

							v30 = v34;
							compute2_5p(v31, v40, v41, v42, v51, ww, kk);
							compute2_5p(v32, v41, v42, v43, v52, ww, kk);
							compute2_5p(v33, v42, v43, v44, v53, ww, kk);
							compute2_5p(v34, v43, v44, v45, v54, ww, kk);

							shuffle_headd(v34, v30);
							shuffle_tail(v31, gw3, v35);
							compute2_5p(v2_21, v30, v31, v32, v41, ww, kk);
							compute2_5p(v2_22, v31, v32, v33, v42, ww, kk);
							compute2_5p(v2_23, v32, v33, v34, v43, ww, kk);
							compute2_5p(v2_24, v33, v34, v35, v44, ww, kk);
							// transpose back
							if(tt<10*tb){ 
							trans(v2_21, v2_22, v2_23, v2_24);
							}
							setstore(v2_21, v2_22, v2_23, v2_24, C[(t /2) % 2][x+1 ][y]);

							v21 = gw2;
							v31 = gw3;
							v41 = gw4;
							v50 = v54;
							v51 = gw5;
						}
						else
						{
							shuffle_headd(v44, v40);
							shuffle_tail(v41, gw4, v45);
							compute2_5p(v31, v40, v41, v42, v51, ww, kk);
							compute2_5p(v32, v41, v42, v43, v52, ww, kk);
							compute2_5p(v33, v42, v43, v44, v53, ww, kk);
							compute2_5p(v34, v43, v44, v45, v54, ww, kk);

							shuffle_headd(v34, v30);
							shuffle_tail(v31, gw3, v35);
							compute2_5p(v2_21, v30, v31, v32, v41, ww, kk);
							compute2_5p(v2_22, v31, v32, v33, v42, ww, kk);
							compute2_5p(v2_23, v32, v33, v34, v43, ww, kk);
							compute2_5p(v2_24, v33, v34, v35, v44, ww, kk);
							

							if(tt<10*tb){ 
							trans(v2_21, v2_22, v2_23, v2_24);
							}
							setstore(v2_21, v2_22, v2_23, v2_24, C[(t /2) % 2][x +1][y]);
						}
						
					}
				}
				

				// deal with the rest elements of each row
				for (x = xmin; x < xup; x++)
				{
					#pragma ivdep
					#pragma vector always
					for (y = yup; y < ymax; y++)
					{
						kernel_5p(C);
					}
				}

				// deal with the rest rows
				for (x = xup; x < xmax; x++)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load
						setload(v11, v12, v13, v14, C[t % 2][x - 1][y]);
						if (y == ymin)
						{
							vallset(v20, C[t % 2][x][y - 1]);
							setload5(v21, v22, v23, v24, gw, C[t % 2][x][y]);
						}
						else
						{
							setload(v22, v23, v24, gw, C[t % 2][x][y + veclen]);
						}
						setload(v31, v32, v33, v34, C[t % 2][x + 1][y]);

						// transpose
						if(tt<10*tb){
						trans(v11, v12, v13, v14);
						trans(v21, v22, v23, v24);
						trans(v31, v32, v33, v34);
						}
						// permute
						shuffle_headd(v24, v20);
						shuffle_tail(v21, gw, v25);

						// compute. result stored in v20, v21, v22, v23
						compute_5p(v11, v20, v21, v22, v31, ww, kk);
						compute_5p(v12, v21, v22, v23, v32, ww, kk);
						compute_5p(v13, v22, v23, v24, v33, ww, kk);
						compute_5p(v14, v23, v24, v25, v34, ww, kk);

						// transpose back
						if(tt<10*tb)
						trans(v20, v21, v22, v23);

						// store
						setstore(v20, v21, v22, v23, C[(t /2) % 2][x][y]);

						v20 = v24;
						v21 = gw;
					}

					#pragma ivdep
					#pragma vector always
					for (y = yup; y < ymax; y++)
					{
						kernel_5p(C);
					}
				}
			}
		}
		level = 1 - level;
	}

	gettimeofday(&end, 0);

	printf("our_2d2s5p GStencil/s = %f\n", ((double)NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

#ifdef CHECK_ONE_TILE_2D
	for (t = 0; t < T; t++)
	{
		for (x = XSLOPE; x < NX + XSLOPE; x++)
		{
			for (y = YSLOPE; y < NY + YSLOPE; y++)
			{
				kernel_5p(B);
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
