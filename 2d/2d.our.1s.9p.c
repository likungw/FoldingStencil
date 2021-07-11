#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include "2d.defines.h"
#include <omp.h>
//#define CHECK_ONE_TILE_2D9P

void our_2d1s9p(double ***A, int NX, int NY, int T, int Bx, int By, int tb)
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

	int xleft02[2] = {XSLOPE + bx, XSLOPE - (Bx - bx) / 2};	  // the start x dimension of the first B11 block is bx
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

	vec ww, kk;
	vallset(ww, GW);
	vallset(kk, LK);
	int xmod, xup, ymod, yup;

	struct timeval start, end;
	gettimeofday(&start, 0);
	for (tt = -tb; tt < T; tt += tb)
	{
// B0, B2
#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, xmod, xup, ymod, yup, t, x, y, n) firstprivate(ww, kk)
		for (n = 0; n < nb02[level]; n++)
		{
			vec v10, v11, v12, v13, v14, v15, gw1, v20, v21, v22, v23, v24, v25, gw2, v30, v31, v32, v33, v34, v35, gw3;
			vec v40, v41, v42, v43, v44, v45, gw4, v50, v51, v52, v53, v54, v55, gw5, v60, v61, v62, v63, v64, v65, gw6;
			for (t = max(tt, 0); t < min(tt + 2 * tb, T); t++)
			{
				xmin = max(XSLOPE, xleft02[level] + (n % xnb02[level]) * ix + myabs(t + 1, tt + tb) * XSLOPE);
				xmax = min(NX + XSLOPE, xleft02[level] + (n % xnb02[level]) * ix + Bx - myabs(t + 1, tt + tb) * XSLOPE);
				ymin = max(YSLOPE, ybottom02[level] + (n / xnb02[level]) * iy + myabs(t + 1, tt + tb) * YSLOPE);
				ymax = min(NY + YSLOPE, ybottom02[level] + (n / xnb02[level]) * iy + By - myabs(t + 1, tt + tb) * YSLOPE);

				xmod = (xmax - xmin) % 4;
				xup = xmax - xmod;
				ymod = (ymax - ymin) % veclen4;
				yup = ymax - ymod;

				for (x = xmin; x < xup; x += 4)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load
						if (y == ymin)
						{
							vallset(v10, C[t % 2][x - 1][y - 1]);
							setload5(v11, v12, v13, v14, gw1, C[t % 2][x - 1][y]);
							vallset(v20, C[t % 2][x][y - 1]);
							setload5(v21, v22, v23, v24, gw2, C[t % 2][x][y]);
							vallset(v30, C[t % 2][x + 1][y - 1]);
							setload5(v31, v32, v33, v34, gw3, C[t % 2][x + 1][y]);
							vallset(v40, C[t % 2][x + 2][y - 1]);
							setload5(v41, v42, v43, v44, gw4, C[t % 2][x + 2][y]);
							vallset(v50, C[t % 2][x + 3][y - 1]);
							setload5(v51, v52, v53, v54, gw5, C[t % 2][x + 3][y]);
							vallset(v60, C[t % 2][x + 4][y - 1]);
							setload5(v61, v62, v63, v64, gw6, C[t % 2][x + 4][y]);
						}
						else
						{
							setload(v12, v13, v14, gw1, C[t % 2][x - 1][y + veclen]);
							setload(v22, v23, v24, gw2, C[t % 2][x][y + veclen]);
							setload(v32, v33, v34, gw3, C[t % 2][x + 1][y + veclen]);
							setload(v42, v43, v44, gw4, C[t % 2][x + 2][y + veclen]);
							setload(v52, v53, v54, gw5, C[t % 2][x + 3][y + veclen]);
							setload(v62, v63, v64, gw6, C[t % 2][x + 4][y + veclen]);
						}

						// transpose
						trans(v11, v12, v13, v14);
						trans(v21, v22, v23, v24);
						trans(v31, v32, v33, v34);
						trans(v41, v42, v43, v44);
						trans(v51, v52, v53, v54);
						trans(v61, v62, v63, v64);

						// permute
						shuffle_headd(v14, v10);
						shuffle_tail(v11, gw1, v15);
						shuffle_headd(v24, v20);
						shuffle_tail(v21, gw2, v25);
						shuffle_headd(v34, v30);
						shuffle_tail(v31, gw3, v35);
						shuffle_headd(v44, v40);
						shuffle_tail(v41, gw4, v45);
						shuffle_headd(v54, v50);
						shuffle_tail(v51, gw5, v55);
						shuffle_headd(v64, v60);
						shuffle_tail(v61, gw6, v65);

						// results are stored in v10, v11, v12, v13
						compute2_9p(v10, v11, v12, v20, v21, v22, v30, v31, v32, ww, kk);
						compute2_9p(v11, v12, v13, v21, v22, v23, v31, v32, v33, ww, kk);
						compute2_9p(v12, v13, v14, v22, v23, v24, v32, v33, v34, ww, kk);
						compute2_9p(v13, v14, v15, v23, v24, v25, v33, v34, v35, ww, kk);

						compute2_9p(v20, v21, v22, v30, v31, v32, v40, v41, v42, ww, kk);
						compute2_9p(v21, v22, v23, v31, v32, v33, v41, v42, v43, ww, kk);
						compute2_9p(v22, v23, v24, v32, v33, v34, v42, v43, v44, ww, kk);
						compute2_9p(v23, v24, v25, v33, v34, v35, v43, v44, v45, ww, kk);

						compute2_9p(v30, v31, v32, v40, v41, v42, v50, v51, v52, ww, kk);
						compute2_9p(v31, v32, v33, v41, v42, v43, v51, v52, v53, ww, kk);
						compute2_9p(v32, v33, v34, v42, v43, v44, v52, v53, v54, ww, kk);
						compute2_9p(v33, v34, v35, v43, v44, v45, v53, v54, v55, ww, kk);

						compute2_9p(v40, v41, v42, v50, v51, v52, v60, v61, v62, ww, kk);
						compute2_9p(v41, v42, v43, v51, v52, v53, v61, v62, v63, ww, kk);
						compute2_9p(v42, v43, v44, v52, v53, v54, v62, v63, v64, ww, kk);
						compute2_9p(v43, v44, v45, v53, v54, v55, v63, v64, v65, ww, kk);

						// transpose back
						trans(v10, v11, v12, v13);
						trans(v20, v21, v22, v23);
						trans(v30, v31, v32, v33);
						trans(v40, v41, v42, v43);

						// store
						setstore(v10, v11, v12, v13, C[(t + 1) % 2][x][y]);
						setstore(v20, v21, v22, v23, C[(t + 1) % 2][x + 1][y]);
						setstore(v30, v31, v32, v33, C[(t + 1) % 2][x + 2][y]);
						setstore(v40, v41, v42, v43, C[(t + 1) % 2][x + 3][y]);

						v10 = v14;
						v11 = gw1;
						v20 = v24;
						v21 = gw2;
						v30 = v34;
						v31 = gw3;
						v40 = v44;
						v41 = gw4;
						v50 = v54;
						v51 = gw5;
						v60 = v64;
						v61 = gw6;
					}
				}

				// deal with the rest elements of each row
				for (x = xmin; x < xup; x++)
				{
#pragma ivdep
#pragma vector always
					for (y = yup; y < ymax; y++)
					{
						kernel_9p(C);
					}
				}

				// deal with the rest rows
				for (x = xup; x < xmax; x++)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load
						if (y == ymin)
						{
							vallset(v10, C[t % 2][x - 1][y - 1]);
							setload5(v11, v12, v13, v14, gw1, C[t % 2][x - 1][y]);
							vallset(v20, C[t % 2][x][y - 1]);
							setload5(v21, v22, v23, v24, gw2, C[t % 2][x][y]);
							vallset(v30, C[t % 2][x + 1][y - 1]);
							setload5(v31, v32, v33, v34, gw3, C[t % 2][x + 1][y]);
						}
						else
						{
							setload(v12, v13, v14, gw1, C[t % 2][x - 1][y + veclen]);
							setload(v22, v23, v24, gw2, C[t % 2][x][y + veclen]);
							setload(v32, v33, v34, gw3, C[t % 2][x + 1][y + veclen]);
						}

						// transpose
						trans(v11, v12, v13, v14);
						trans(v21, v22, v23, v24);
						trans(v31, v32, v33, v34);

						// permute
						shuffle_headd(v14, v10);
						shuffle_tail(v11, gw1, v15);
						shuffle_headd(v24, v20);
						shuffle_tail(v21, gw2, v25);
						shuffle_headd(v34, v30);
						shuffle_tail(v31, gw3, v35);

						// compute. results are stored in v20, v21, v22, v23
						compute_9p(v10, v11, v12, v20, v21, v22, v30, v31, v32, ww, kk);
						compute_9p(v11, v12, v13, v21, v22, v23, v31, v32, v33, ww, kk);
						compute_9p(v12, v13, v14, v22, v23, v24, v32, v33, v34, ww, kk);
						compute_9p(v13, v14, v15, v23, v24, v25, v33, v34, v35, ww, kk);

						// transpose back
						trans(v20, v21, v22, v23);

						// store
						setstore(v20, v21, v22, v23, C[(t + 1) % 2][x][y]);

						v10 = v14;
						v11 = gw1;
						v20 = v24;
						v21 = gw2;
						v30 = v34;
						v31 = gw3;
					}

#pragma ivdep
#pragma vector always
					for (y = yup; y < ymax; y++)
					{
						kernel_9p(C);
					}
				}
			}
		}

// B11, B12
#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, xmod, xup, ymod, yup, t, x, y, n) firstprivate(ww, kk)
		for (n = 0; n < nb1[0] + nb1[1]; n++)
		{
			vec v10, v11, v12, v13, v14, v15, gw1, v20, v21, v22, v23, v24, v25, gw2, v30, v31, v32, v33, v34, v35, gw3;
			vec v40, v41, v42, v43, v44, v45, gw4, v50, v51, v52, v53, v54, v55, gw5, v60, v61, v62, v63, v64, v65, gw6;
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

				for (x = xmin; x < xup; x += 4)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load
						if (y == ymin)
						{
							vallset(v10, C[t % 2][x - 1][y - 1]);
							setload5(v11, v12, v13, v14, gw1, C[t % 2][x - 1][y]);
							vallset(v20, C[t % 2][x][y - 1]);
							setload5(v21, v22, v23, v24, gw2, C[t % 2][x][y]);
							vallset(v30, C[t % 2][x + 1][y - 1]);
							setload5(v31, v32, v33, v34, gw3, C[t % 2][x + 1][y]);
							vallset(v40, C[t % 2][x + 2][y - 1]);
							setload5(v41, v42, v43, v44, gw4, C[t % 2][x + 2][y]);
							vallset(v50, C[t % 2][x + 3][y - 1]);
							setload5(v51, v52, v53, v54, gw5, C[t % 2][x + 3][y]);
							vallset(v60, C[t % 2][x + 4][y - 1]);
							setload5(v61, v62, v63, v64, gw6, C[t % 2][x + 4][y]);
						}
						else
						{
							setload(v12, v13, v14, gw1, C[t % 2][x - 1][y + veclen]);
							setload(v22, v23, v24, gw2, C[t % 2][x][y + veclen]);
							setload(v32, v33, v34, gw3, C[t % 2][x + 1][y + veclen]);
							setload(v42, v43, v44, gw4, C[t % 2][x + 2][y + veclen]);
							setload(v52, v53, v54, gw5, C[t % 2][x + 3][y + veclen]);
							setload(v62, v63, v64, gw6, C[t % 2][x + 4][y + veclen]);
						}

						// transpose
						trans(v11, v12, v13, v14);
						trans(v21, v22, v23, v24);
						trans(v31, v32, v33, v34);
						trans(v41, v42, v43, v44);
						trans(v51, v52, v53, v54);
						trans(v61, v62, v63, v64);

						// permute
						shuffle_headd(v14, v10);
						shuffle_tail(v11, gw1, v15);
						shuffle_headd(v24, v20);
						shuffle_tail(v21, gw2, v25);
						shuffle_headd(v34, v30);
						shuffle_tail(v31, gw3, v35);
						shuffle_headd(v44, v40);
						shuffle_tail(v41, gw4, v45);
						shuffle_headd(v54, v50);
						shuffle_tail(v51, gw5, v55);
						shuffle_headd(v64, v60);
						shuffle_tail(v61, gw6, v65);

						// results are stored in v10, v11, v12, v13
						compute2_9p(v10, v11, v12, v20, v21, v22, v30, v31, v32, ww, kk);
						compute2_9p(v11, v12, v13, v21, v22, v23, v31, v32, v33, ww, kk);
						compute2_9p(v12, v13, v14, v22, v23, v24, v32, v33, v34, ww, kk);
						compute2_9p(v13, v14, v15, v23, v24, v25, v33, v34, v35, ww, kk);

						compute2_9p(v20, v21, v22, v30, v31, v32, v40, v41, v42, ww, kk);
						compute2_9p(v21, v22, v23, v31, v32, v33, v41, v42, v43, ww, kk);
						compute2_9p(v22, v23, v24, v32, v33, v34, v42, v43, v44, ww, kk);
						compute2_9p(v23, v24, v25, v33, v34, v35, v43, v44, v45, ww, kk);

						compute2_9p(v30, v31, v32, v40, v41, v42, v50, v51, v52, ww, kk);
						compute2_9p(v31, v32, v33, v41, v42, v43, v51, v52, v53, ww, kk);
						compute2_9p(v32, v33, v34, v42, v43, v44, v52, v53, v54, ww, kk);
						compute2_9p(v33, v34, v35, v43, v44, v45, v53, v54, v55, ww, kk);

						compute2_9p(v40, v41, v42, v50, v51, v52, v60, v61, v62, ww, kk);
						compute2_9p(v41, v42, v43, v51, v52, v53, v61, v62, v63, ww, kk);
						compute2_9p(v42, v43, v44, v52, v53, v54, v62, v63, v64, ww, kk);
						compute2_9p(v43, v44, v45, v53, v54, v55, v63, v64, v65, ww, kk);

						// transpose back
						trans(v10, v11, v12, v13);
						trans(v20, v21, v22, v23);
						trans(v30, v31, v32, v33);
						trans(v40, v41, v42, v43);

						// store
						setstore(v10, v11, v12, v13, C[(t + 1) % 2][x][y]);
						setstore(v20, v21, v22, v23, C[(t + 1) % 2][x + 1][y]);
						setstore(v30, v31, v32, v33, C[(t + 1) % 2][x + 2][y]);
						setstore(v40, v41, v42, v43, C[(t + 1) % 2][x + 3][y]);

						v10 = v14;
						v11 = gw1;
						v20 = v24;
						v21 = gw2;
						v30 = v34;
						v31 = gw3;
						v40 = v44;
						v41 = gw4;
						v50 = v54;
						v51 = gw5;
						v60 = v64;
						v61 = gw6;
					}
				}

				// deal with the rest elements of each row
				for (x = xmin; x < xup; x++)
				{
#pragma ivdep
#pragma vector always
					for (y = yup; y < ymax; y++)
					{
						kernel_9p(C);
					}
				}

				// deal with the rest rows
				for (x = xup; x < xmax; x++)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load
						if (y == ymin)
						{
							vallset(v10, C[t % 2][x - 1][y - 1]);
							setload5(v11, v12, v13, v14, gw1, C[t % 2][x - 1][y]);
							vallset(v20, C[t % 2][x][y - 1]);
							setload5(v21, v22, v23, v24, gw2, C[t % 2][x][y]);
							vallset(v30, C[t % 2][x + 1][y - 1]);
							setload5(v31, v32, v33, v34, gw3, C[t % 2][x + 1][y]);
						}
						else
						{
							setload(v12, v13, v14, gw1, C[t % 2][x - 1][y + veclen]);
							setload(v22, v23, v24, gw2, C[t % 2][x][y + veclen]);
							setload(v32, v33, v34, gw3, C[t % 2][x + 1][y + veclen]);
						}

						// transpose
						trans(v11, v12, v13, v14);
						trans(v21, v22, v23, v24);
						trans(v31, v32, v33, v34);

						// permute
						shuffle_headd(v14, v10);
						shuffle_tail(v11, gw1, v15);
						shuffle_headd(v24, v20);
						shuffle_tail(v21, gw2, v25);
						shuffle_headd(v34, v30);
						shuffle_tail(v31, gw3, v35);

						// compute. results are stored in v20, v21, v22, v23
						compute_9p(v10, v11, v12, v20, v21, v22, v30, v31, v32, ww, kk);
						compute_9p(v11, v12, v13, v21, v22, v23, v31, v32, v33, ww, kk);
						compute_9p(v12, v13, v14, v22, v23, v24, v32, v33, v34, ww, kk);
						compute_9p(v13, v14, v15, v23, v24, v25, v33, v34, v35, ww, kk);

						// transpose back
						trans(v20, v21, v22, v23);

						// store
						setstore(v20, v21, v22, v23, C[(t + 1) % 2][x][y]);

						v10 = v14;
						v11 = gw1;
						v20 = v24;
						v21 = gw2;
						v30 = v34;
						v31 = gw3;
					}

#pragma ivdep
#pragma vector always
					for (y = yup; y < ymax; y++)
					{
						kernel_9p(C);
					}
				}
			}
		}
		level = 1 - level;
	}

	gettimeofday(&end, 0);

	printf("our_2d1s9p GStencil/s = %f\n", ((double)NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

#ifdef CHECK_ONE_TILE_2D9P
	for (t = 0; t < T; t++)
	{
		for (x = XSLOPE; x < NX + XSLOPE; x++)
		{
			for (y = YSLOPE; y < NY + YSLOPE; y++)
			{
				kernel_9p(B);
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
