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


void cross_2d1sgol(double ***A, int NX, int NY, int T, int Bx, int By, int tb)
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
	vec vc_h1, vc_h2, vc_h3, vc_h4, vl, vr, vt, vb, vc_l;
	vec vc_v1, vc_v2, vc_v3, vc_v4;
	vec vn_h1, vn_h2, vn_h3, vn_h4;
	vec vn_v1, vn_v2, vn_v3, vn_v4; 
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

				for (x = xmin; x < xmax; x ++)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load top
						vload(vt, C[t % 2][x - 1][y+veclen3+veclen4]);
						if (y == ymin)
						{							
							setload(vc_h1, vc_h2, vc_h3, vc_h4, C[t % 2][x][y]);
							//vloadset(vl, C[t % 2][x][y], C[t % 2][x + 1][y], C[t % 2][x + 2][y], C[t % 2][x + 3][y]);
							vload(vl, C[t % 2][x - 1][y]);
						} 

						//load next
						setload(vn_h1, vn_h2, vn_h3, vn_h4, C[t % 2][x][y+veclen4]);

						//load bottom						
						vload(vb, C[t % 2][x + 1][y+veclen4]);

						if (y == ymin){
							//current vertical copy construct
							vc_v1 =  vc_h1;
							vc_v2 =  vc_h2;
							vc_v3 =  vc_h3;
							vc_v4 =  vc_h4;
						}				

						//next vertical copy construct
						vn_v1 =  vn_h1;
						vn_v2 =  vn_h2;
						vn_v3 =  vn_h3;
						vn_v4 =  vn_h4;		

						//next vertical comp.
						cross_comp_h6(vt, vn_h1, vn_h2, vn_h3, vn_h4, vb);

						//transpose next
						trans(vt, vn_h1, vn_h2, vn_h3);						
						
						//current horizontal comp.
						cross_comp_v6(vl, vc_v1, vc_v2, vc_v3, vc_v4, vt);

						//transpose back
						trans(vl, vc_v1, vc_v2, vc_v3);

						//combine comp.
						cobine_comp_hv4( vl, vc_v1, vc_v2, vc_v3, vc_h1, vc_h2, vc_h3, vc_h4, kk);

						//judge.
						vb2s23(vc_h1, vl);
						vb2s23(vc_h2, vc_v1);
						vb2s23(vc_h3, vc_v2);
						vb2s23(vc_h4, vc_v3);
		
						//store
						setstore(vc_h1, vc_h2, vc_h3, vc_h4, C[(t + 1) % 2][x][y-veclen4]);

						//reserve copy
						vl = vc_v4;
						vc_h1 = vn_v1;
						vc_h2 = vn_v2;
						vc_h3 = vn_v3;
						vc_h4 = vn_v4;
						vc_v1 = vt;
						vc_v2 = vn_h1;
						vc_v3 = vn_h2;
						vc_v4 = vn_h3;
					}
					#pragma ivdep
					#pragma vector always
					for (; y < ymax; y++)
					{
						//kernel_gol(C);
					}
				}
			}
		}//printf("KKK!\n");

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

				for (x = xmin; x < xmax; x ++)
				{
					for (y = ymin; y < yup; y += veclen4)
					{
						// load top
						vload(vt, C[t % 2][x - 1][y+veclen3+veclen4]);
						if (y == ymin)
						{							
							setload(vc_h1, vc_h2, vc_h3, vc_h4, C[t % 2][x][y]);
							//vloadset(vl, C[t % 2][x][y], C[t % 2][x + 1][y], C[t % 2][x + 2][y], C[t % 2][x + 3][y]);
							vload(vl, C[t % 2][x - 1][y]);
						} 
//next vertical copy construct
						vn_v1 =  vn_h1;
						vn_v2 =  vn_h2;
						vn_v3 =  vn_h3;
						vn_v4 =  vn_h4;		

						//next vertical comp.
						cross_comp_h6(vt, vn_h1, vn_h2, vn_h3, vn_h4, vb);

						//transpose next
						trans(vt, vn_h1, vn_h2, vn_h3);						
						
						//current horizontal comp.
						cross_comp_v6(vl, vc_v1, vc_v2, vc_v3, vc_v4, vt);

						//transpose back
						trans(vl, vc_v1, vc_v2, vc_v3);

						//combine comp.
						cobine_comp_hv4( vl, vc_v1, vc_v2, vc_v3, vc_h1, vc_h2, vc_h3, vc_h4, kk);

						//judge.
						vb2s23(vc_h1, vl);
						vb2s23(vc_h2, vc_v1);
						vb2s23(vc_h3, vc_v2);
						vb2s23(vc_h4, vc_v3);
		
						//store
						setstore(vc_h1, vc_h2, vc_h3, vc_h4, C[(t + 1) % 2][x][y-veclen4]);

						//reserve copy
						vl = vc_v4;
						vc_h1 = vn_v1;
						vc_h2 = vn_v2;
						vc_h3 = vn_v3;
						vc_h4 = vn_v4;
						vc_v1 = vt;
						vc_v2 = vn_h1;
						vc_v3 = vn_h2;
						vc_v4 = vn_h3;
					}
					#pragma ivdep
					#pragma vector always
					for (; y < ymax; y++)
					{
						//kernel_gol(C);
					}
				}
			}
		}
		level = 1 - level;
	}

	gettimeofday(&end, 0);

	printf("cross_2d1sgol GStencil/s = %f\n", ((double)NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

#ifdef CHECK_ONE_TILE_2D
	for (t = 0; t < T; t++)
	{
		for (x = XSLOPE; x < NX + XSLOPE; x++)
		{
			for (y = YSLOPE; y < NY + YSLOPE; y++)
			{
				kernel_gol(B);
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
