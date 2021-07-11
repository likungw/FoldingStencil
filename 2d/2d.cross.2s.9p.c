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

void cross_2d2s9p(double ***A, int NX, int NY, int T, int Bx, int By, int tb)
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
	int dif;

	vec ww, kk, gw;
	vec vc_h1, vc_h2, vc_h3, vc_h4, vl_1, vt_1, vb_1;
	vec vl_2, vt_2, vb_2;
	vec vln_2,vln_1;
	vec vn_h1, vn_h2, vn_h3, vn_h4;
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
			dif = 0;
			for (t = max(tt, 0); t < min(tt + 2 * tb, T); t+=2)
			{
				xmin = max(   XSLOPE,   xleft02[level] + (n%xnb02[level]) * ix      + myabs(t+dif+1,tt+tb) * XSLOPE);
				xmax = min(NX+XSLOPE,   xleft02[level] + (n%xnb02[level]) * ix + Bx - myabs(t+dif+1,tt+tb) * XSLOPE);
				ymin = max(   YSLOPE, ybottom02[level] + (n/xnb02[level]) * iy      + myabs(t+dif+1,tt+tb) * YSLOPE);
				ymax = min(NY+YSLOPE, ybottom02[level] + (n/xnb02[level]) * iy + By - myabs(t+dif+1,tt+tb) * YSLOPE);
				//printf("1 -- t %d xmin %d xmax %d ymin %d ymax %d \n",tt ,xmin ,xmax ,ymin, ymax );
				//dif++;
				for (x = xmin; x < xmax-veclen; x+=veclen)
				{
					for (y = ymin; y < ymax-veclen; y += veclen)
					{						
						if (y == ymin)
						{						
							vloadset(vl_1, C[((t/2) % 2)][x + 0][y-1], C[((t/2) % 2)][x + 1][y-1], C[((t/2) % 2)][x + 2][y-1], C[((t/2) % 2)][x + 3][y-1]);
							if(ymin < 2){
								vl_2 = vl_1;
							}
							else
							{
								vloadset(vl_2, C[((t/2) % 2)][x + 0][y-2], C[((t/2) % 2)][x + 1][y-2], C[((t/2) % 2)][x + 2][y-2], C[((t/2) % 2)][x + 3][y-2]); 
							}
								
							// load current top.
							vload(vt_1, C[((t/2) % 2)][x - 1][y]);
							if(x < 2) {
								vt_2 = vt_1;
							}
							else
							{
								vload(vt_2, C[((t/2) % 2)][x - 2][y]); 
							}

							//squareload(vc_h1, vc_h2, vc_h3, vc_h4, C[((t/2) % 2)][x + 0][y], NY);
							vload(vc_h1, C[((t/2) % 2)][x+0][y]);
							vload(vc_h2, C[((t/2) % 2)][x+1][y]);
							vload(vc_h3, C[((t/2) % 2)][x+2][y]);
							vload(vc_h4, C[((t/2) % 2)][x+3][y]);

							//load current bottom.						
							vload(vb_1, C[((t/2) % 2)][x + 4][y]);
							vload(vb_2, C[((t/2) % 2)][x + 5][y]);

							//vertical comp.
							cross_comp_p9vertical(vc_h1, vc_h2, vc_h3, vc_h4, vt_2, vt_1, vc_h1, vc_h2, vc_h3, vc_h4, vb_1, vb_2);

							//transpose.
							trans(vc_h1, vc_h2, vc_h3, vc_h4);	
							//squarestore(vc_h1, vc_h2, vc_h3, vc_h4, C[(t/2 + 1) % 2][x][y], NY);							
							vln_1 = vc_h3;
							vln_2 = vc_h4;
						}
						else
						{
							// load top.
							vload(vt_1, C[((t/2) % 2)][x - 1][y]);
							if(x < 2) {
								vt_2 = vt_1;
							}else
							{
								vload(vt_2, C[((t/2) % 2)][x - 2][y]); 
							}

							//load next.
							//squareload(vn_h1, vn_h2, vn_h3, vn_h4, C[((t/2) % 2)][x][y+veclen],NY);
							vload(vn_h1, C[((t/2) % 2)][x+0][y]);
							vload(vn_h2, C[((t/2) % 2)][x+1][y]);
							vload(vn_h3, C[((t/2) % 2)][x+2][y]);
							vload(vn_h4, C[((t/2) % 2)][x+3][y]);

							//load bottom.						 		
							vload(vb_1, C[((t/2) % 2)][x + 4][y]);
							vload(vb_2, C[((t/2) % 2)][x + 5][y]);
							
							//vertical comp.
							cross_comp_p9vertical(vn_h1, vn_h2, vn_h3, vn_h4, vt_2, vt_1, vn_h1, vn_h2, vn_h3, vn_h4, vb_1, vb_2);

							//transpose next.
							trans(vn_h1, vn_h2, vn_h3, vn_h4);	

							//shifts reuseing.
							vln_1 = vc_h3;
							vln_2 = vc_h4;

							//horizontal comp.
							cross_comp_p9vertical(vc_h1, vc_h2, vc_h3, vc_h4, vl_2, vl_1, vc_h1, vc_h2, vc_h3, vc_h4, vn_h1, vn_h2);						

							//results
							cross_comp_s2p9_m4(vc_h1, vc_h2, vc_h3, vc_h4, kk);

							//trans back (optional)
							trans(vc_h1, vc_h2, vc_h3, vc_h4);

							//store
							//squarestore(vc_h1, vc_h2, vc_h3, vc_h4, C[(t/2 + 1) % 2][x][y], NY);
							vstore(C[((t/2) % 2)][x+0][y - veclen],vc_h1);
							vstore(C[((t/2) % 2)][x+1][y - veclen],vc_h2);
							vstore(C[((t/2) % 2)][x+2][y - veclen],vc_h3);
							vstore(C[((t/2) % 2)][x+3][y - veclen],vc_h4);

							//loop vl copy
							
							vl_1 = vln_1;
							vl_2 = vln_2;
							vc_h1 = vn_h1;
							vc_h2 = vn_h2;
							vc_h3 = vn_h3;
							vc_h4 = vn_h4;
						}
									
					}
					if((ymax<NY-1)&&(xmax<NX-1)&&(xmin>1)){
						#pragma ivdep
						#pragma vector always
						for (; y < ymax; y++)
						{
							kernel_2s9p(C);
						}
					}			
				}
				
				if((ymax<NY-1)&&(xmax<NX-1)&&(ymin>1)){
					for(; x < xmax; x++) {
					#pragma ivdep
					#pragma vector always
						for(y = ymin; y < ymax; y++) {
							kernel_2s9p(C);
						}
					}	
				}		
			}
		}
//printf("kk! \n");
// B11, B12
#pragma omp parallel for schedule(dynamic) private(xmin, xmax, ymin, ymax, xmod, xup, ymod, yup, t, x, y, n)
		for (n = 0; n < nb1[0] + nb1[1]; n++)
		{
			dif = 0;
			for (t = tt + tb; t < min(tt + 2 * tb, T); t+=2)
			{
				if(n < nb1[level]) {
					xmin = max(     XSLOPE,   xleft11[level] + (n%xnb1[level]) * ix       - (t+dif+1-tt-tb) * XSLOPE);
					xmax = min(NX + XSLOPE,   xleft11[level] + (n%xnb1[level]) * ix  + bx + (t+dif+1-tt-tb) * XSLOPE);
					ymin = max(     YSLOPE, ybottom11[level] + (n/xnb1[level]) * iy       + (t+dif+1-tt-tb) * YSLOPE);
					ymax = min(NY + YSLOPE, ybottom11[level] + (n/xnb1[level]) * iy  + By - (t+dif+1-tt-tb) * YSLOPE);
				}
				else {
					xmin = max(     XSLOPE,   xleft12[level] + ((n-nb1[level])%xnb1[1-level]) * ix      + (t+dif+1-tt-tb) * XSLOPE);
					xmax = min(NX + XSLOPE,   xleft12[level] + ((n-nb1[level])%xnb1[1-level]) * ix + Bx - (t+dif+1-tt-tb) * XSLOPE);
					ymin = max(     YSLOPE, ybottom12[level] + ((n-nb1[level])/xnb1[1-level]) * iy      - (t+dif+1-tt-tb) * YSLOPE);
					ymax = min(NY + YSLOPE, ybottom12[level] + ((n-nb1[level])/xnb1[1-level]) * iy + by + (t+dif+1-tt-tb) * YSLOPE);
				} 
				for (x = xmin; x < xmax-veclen; x+=veclen)
				{
					for (y = ymin; y < ymax-veclen; y += veclen)
					{						
						if (y == ymin)
						{						
							vloadset(vl_1, C[((t/2) % 2)][x + 0][y-1], C[((t/2) % 2)][x + 1][y-1], C[((t/2) % 2)][x + 2][y-1], C[((t/2) % 2)][x + 3][y-1]);
							if(ymin < 2){
								vl_2 = vl_1;
							}
							else
							{
								vloadset(vl_2, C[((t/2) % 2)][x + 0][y-2], C[((t/2) % 2)][x + 1][y-2], C[((t/2) % 2)][x + 2][y-2], C[((t/2) % 2)][x + 3][y-2]); 
							}
								
							// load current top.
							vload(vt_1, C[((t/2) % 2)][x - 1][y]);
							if(x < 2) {
								vt_2 = vt_1;
							}
							else
							{
								vload(vt_2, C[((t/2) % 2)][x - 2][y]); 
							}

							//squareload(vc_h1, vc_h2, vc_h3, vc_h4, C[((t/2) % 2)][x + 0][y], NY);
							vload(vc_h1, C[((t/2) % 2)][x+0][y]);
							vload(vc_h2, C[((t/2) % 2)][x+1][y]);
							vload(vc_h3, C[((t/2) % 2)][x+2][y]);
							vload(vc_h4, C[((t/2) % 2)][x+3][y]);

							//load current bottom.						
							vload(vb_1, C[((t/2) % 2)][x + 4][y]);
							vload(vb_2, C[((t/2) % 2)][x + 5][y]);

							//vertical comp.
							cross_comp_p9vertical(vc_h1, vc_h2, vc_h3, vc_h4, vt_2, vt_1, vc_h1, vc_h2, vc_h3, vc_h4, vb_1, vb_2);

							//transpose.
							trans(vc_h1, vc_h2, vc_h3, vc_h4);	
							//squarestore(vc_h1, vc_h2, vc_h3, vc_h4, C[(t/2 + 1) % 2][x][y], NY);							
							vln_1 = vc_h3;
							vln_2 = vc_h4;
						}
						else
						{
							// load top.
							vload(vt_1, C[((t/2) % 2)][x - 1][y]);
							if(x < 2) {
								vt_2 = vt_1;
							}else
							{
								vload(vt_2, C[((t/2) % 2)][x - 2][y]); 
							}

							//load next.
							//squareload(vn_h1, vn_h2, vn_h3, vn_h4, C[((t/2) % 2)][x][y+veclen],NY);
							vload(vn_h1, C[((t/2) % 2)][x+0][y]);
							vload(vn_h2, C[((t/2) % 2)][x+1][y]);
							vload(vn_h3, C[((t/2) % 2)][x+2][y]);
							vload(vn_h4, C[((t/2) % 2)][x+3][y]);

							//load bottom.						 		
							vload(vb_1, C[((t/2) % 2)][x + 4][y]);
							vload(vb_2, C[((t/2) % 2)][x + 5][y]);
							
							//vertical comp.
							cross_comp_p9vertical(vn_h1, vn_h2, vn_h3, vn_h4, vt_2, vt_1, vn_h1, vn_h2, vn_h3, vn_h4, vb_1, vb_2);

							//transpose next.
							trans(vn_h1, vn_h2, vn_h3, vn_h4);	

							//shifts reuseing.
							vln_1 = vc_h3;
							vln_2 = vc_h4;

							//horizontal comp.
							cross_comp_p9vertical(vc_h1, vc_h2, vc_h3, vc_h4, vl_2, vl_1, vc_h1, vc_h2, vc_h3, vc_h4, vn_h1, vn_h2);						

							//results
							cross_comp_s2p9_m4(vc_h1, vc_h2, vc_h3, vc_h4, kk);

							//trans back (optional)
							trans(vc_h1, vc_h2, vc_h3, vc_h4);	

							//store
							//squarestore(vc_h1, vc_h2, vc_h3, vc_h4, C[(t/2 + 1) % 2][x][y], NY);
							vstore(C[((t/2) % 2)][x+0][y - veclen],vc_h1);
							vstore(C[((t/2) % 2)][x+1][y - veclen],vc_h2);
							vstore(C[((t/2) % 2)][x+2][y - veclen],vc_h3);
							vstore(C[((t/2) % 2)][x+3][y - veclen],vc_h4);

							//loop vl copy
							
							vl_1 = vln_1;
							vl_2 = vln_2;
							vc_h1 = vn_h1;
							vc_h2 = vn_h2;
							vc_h3 = vn_h3;
							vc_h4 = vn_h4;
						}
									
					}
					if((ymax<NY-1)&&(xmax<NX-1)&&(xmin>1)){
						#pragma ivdep
						#pragma vector always
						for (; y < ymax; y++)
						{
							kernel_2s9p(C);
						}
					}			
				}
				
				if((ymax<NY-1)&&(xmax<NX-1)&&(ymin>1)){
					for(; x < xmax; x++) {
					#pragma ivdep
					#pragma vector always
						for(y = ymin; y < ymax; y++) {
							kernel_2s9p(C);
						}
					}	
				}		
			}
		}
		level = 1 - level;
	}

	gettimeofday(&end, 0);

	printf("cross_2d2s9p GStencil/s = %f\n", ((double)NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

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
