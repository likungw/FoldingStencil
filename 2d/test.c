#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include "2d.defines.h"
#define NX 40
#define NY 40
#define T 1000
void test() {
	vec vc_h1, vc_h2, vc_h3, vc_h4;
	vec vn_h1, vn_h2, vn_h3, vn_h4;
	int i,j;
	int x , y;
	struct timeval start, end;
	struct timeval start2, end2;
	long int t;

	double **A = (double **)malloc(sizeof(double *)*NX);
	double **B = (double **)malloc(sizeof(double *)*NX);
	double **C = (double **)malloc(sizeof(double *)*NX);

	double ***D = (double ***)malloc(sizeof(double **) * 2);
	double ***E = (double ***)malloc(sizeof(double **) * 2);
	for (i = 0; i < 2; i++)
	{
		D[i] = (double **)malloc(sizeof(double *) * (NX ));
		E[i] = (double **)malloc(sizeof(double *) * (NX ));
	}
	for (i = 0; i < 2; i++)
	{
		for (j = 0; j < (NX + 2 * XSLOPE); j++)
		{
			D[i][j] = (double *)malloc(sizeof(double) * (NY));
			E[i][j] = (double *)malloc(sizeof(double) * (NY));
		}
	}
	
	for (i = 0; i < (NX); i++)
	{
		for (j = 0; j < (NY); j++)
		{
			D[0][i][j] = i*j+j;
			D[1][i][j] = 0;
			E[0][i][j] = D[0][i][j];
			E[1][i][j] = 0;
			//printf("dfsfsd\n");
		}
	}

	for (i = 0; i < NX; i++)
	{
		A[i] = (double *)malloc(sizeof(double) * (NY));
		B[i] = (double *)malloc(sizeof(double) * (NY));
		C[i] = (double *)malloc(sizeof(double) * (NY));
	}
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			A[i][j] = j;
			B[i][j] = A[i][j];
			C[i][j] = A[i][j];
			//printf("CHECK %d,%d, value: %g!\n",i,j,A[i][j]);
	
		}
	}

	gettimeofday(&start,0);
	
	for (t = 0; t < T; t++){
		for (x = 0; x < NX; x++)
		{
			for (y = 0; y < NY; y += veclen)
			{

				
				
				vload(vc_h1, D[t%2][x + 0][y+ 0 ]);
				
				//vload(vc_h2, D[t%2][x + 0][y+ veclen ]);
				//vload(vc_h3, D[t%2][x + 0][y+ veclen2 ]);
				//vload(vc_h4, D[t%2][x + 0][y+ veclen3 ]);
				
				vc_h1 = _mm256_fmadd_pd(vc_h1,_mm256_set1_pd(2.0),vc_h1);
				//vc_h2 = _mm256_fmadd_pd(vc_h2,_mm256_set1_pd(2.0),vc_h2);
				//vc_h3 = _mm256_fmadd_pd(vc_h3,_mm256_set1_pd(2.0),vc_h3);
				//vc_h4 = _mm256_fmadd_pd(vc_h4,_mm256_set1_pd(2.0),vc_h4);
				//vc_h1 = vc_h1* vc_h1*vc_h1*vc_h1+vc_h1;
				//vc_h2 = vc_h2* vc_h2*vc_h2*vc_h2+vc_h2;
				//vc_h3 = vc_h3* vc_h3*vc_h3*vc_h3+vc_h3;
				//vc_h4 = vc_h4* vc_h4*vc_h4*vc_h4+vc_h4;
				
				//vload(vn_h1, A[x + 0][y+veclen+1]);
				//vload(vn_h2, A[x + 1][y+veclen+1]);
				//vload(vn_h3, A[x + 2][y+veclen+1]);
				//vload(vn_h4, A[x + 3][y+veclen+1]);

				//vn_h1 = gwpair(vc_h1,vc_h2);
				//vn_h1 = kwpair(vn_h1);
				//vn_h2 = gwpair(vc_h2,vc_h3);
				//vn_h2 = kwpair(vn_h2);
				//vn_h3 = gwpair(vc_h3,vc_h4);
				//vn_h3 = kwpair(vn_h3);
				//vn_h4 = gwpair(vc_h4,vc_h1);
				//vn_h4 = kwpair(vn_h4);
			
				
				//vc_h1 = _mm256_fmadd_pd(vc_h1,_mm256_set1_pd(2.0),vn_h1);
				//vc_h2 = _mm256_fmadd_pd(vc_h2,_mm256_set1_pd(2.0),vn_h2);
				//vc_h3 = _mm256_fmadd_pd(vc_h3,_mm256_set1_pd(2.0),vn_h3);
				//vc_h4 = _mm256_fmadd_pd(vc_h4,_mm256_set1_pd(2.0),vn_h4);
				vstore(D[(t+1)%2][x + 0][y],vc_h1);
				//vstore(D[(t+1)%2][x + 0][y+veclen],vc_h2);
				//vstore(D[(t+1)%2][x + 0][y+veclen2],vc_h3);
				//vstore(D[(t+1)%2][x + 0][y+veclen3],vc_h4);
				//vstore(A[x + 0][y+1],vn_h1);
				//vstore(A[x + 1][y+1],vn_h2);
				//vstore(A[x + 2][y+1],vn_h3);
				//vstore(A[x + 3][y+1],vn_h4);
				//printf("ww! %g %g %g %g \n",D0(vc_h1),D1(vc_h1),D2(vc_h1),D3(vc_h1));
				//printf("ww! %g %g %g %g \n",D0(vc_h2),D1(vc_h2),D2(vc_h2),D3(vc_h2));
				//printf("ww! %g %g %g %g \n",D0(vc_h3),D1(vc_h3),D2(vc_h3),D3(vc_h3));
				//printf("ww! %g %g %g %g \n",D0(vc_h4),D1(vc_h4),D2(vc_h4),D3(vc_h4));
			}
		}
	}
	/*
	for (t = 0; t < T; t++){
		for (x = 0; x < NX; x++)
		{
			for (y = 0; y < NY/veclen; y+=veclen)
			{
				vload(vc_h1, A[x][y + 0*NY/veclen]);
				vload(vc_h2, A[x][y + 1*NY/veclen]);
				vload(vc_h3, A[x][y + 2*NY/veclen]);
				vload(vc_h4, A[x][y + 3*NY/veclen]);
				//printf("ww! %g %g %g %g \n",D0(vc_h1),D1(vc_h1),D2(vc_h1),D3(vc_h1));
				//printf("ww! %g %g %g %g \n",D0(vc_h2),D1(vc_h2),D2(vc_h2),D3(vc_h2));
				//printf("ww! %g %g %g %g \n",D0(vc_h3),D1(vc_h3),D2(vc_h3),D3(vc_h3));
				//printf("ww! %g %g %g %g \n",D0(vc_h4),D1(vc_h4),D2(vc_h4),D3(vc_h4));
				trans2d(vc_h1, vc_h2, vc_h3, vc_h4);
				//vc_h1 =  vc_h1 * vc_h2 + vc_h3;
				//vc_h2 =  vc_h2 * vc_h3 + vc_h4;
				//vc_h3 =  vc_h3 * vc_h4 + vc_h1;
				//vc_h4 =  vc_h4 * vc_h1 + vc_h2;
				//printf("ww! %g %g %g %g \n",D0(vc_h1),D1(vc_h1),D2(vc_h1),D3(vc_h1));
				//printf("ww! %g %g %g %g \n",D0(vc_h2),D1(vc_h2),D2(vc_h2),D3(vc_h2));
				//printf("ww! %g %g %g %g \n",D0(vc_h3),D1(vc_h3),D2(vc_h3),D3(vc_h3));
				//printf("ww! %g %g %g %g \n",D0(vc_h4),D1(vc_h4),D2(vc_h4),D3(vc_h4));
				vstore(B[x][y*veclen],vc_h1);
				vstore(B[x][y*veclen+veclen],vc_h2);
				vstore(B[x][y*veclen+2*veclen],vc_h3);
				vstore(B[x][y*veclen+3*veclen],vc_h4);
								
			}
		}
	}*/

	gettimeofday(&end,0);

	printf("TEST GStencil/s = %f\n", ((double)NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
	

/*
	for (t = 0; t < T; t++){
		for (x = 0; x < NX; x++)
		{
			for (y = 0; y < NY/veclen; y++)
			{
				C[x][y*veclen] = A[x][y];				
			}
			for (; y < 2 * NY/veclen; y++)
			{
				C[x][(y-NY/veclen)*veclen+1] = A[x][y];				
			}
			for (; y < 3 * NY/veclen; y++)
			{
				C[x][(y-2*NY/veclen)*veclen+2] = A[x][y];				
			}
			for (; y < 4 * NY/veclen; y++)
			{
				C[x][(y-3*NY/veclen)*veclen+3] = A[x][y];				
			}
		}
	}*/
	for (t = 0; t < T; t++){
		for (y = 0; y < NY; y+=veclen)
		{
			for (x = 0; x < NX; x += veclen)
			{
				vload(vc_h1, D[t%2]    [x + 0 ][y]);
				vload(vc_h2, D[t%2]    [x + 1  ][y] );
				vload(vc_h3, D[(t+0)%2][x + 2 ][y]);
				vload(vc_h4, D[(t+0)%2][x + 3 ][y]);
				
				vc_h1 = _mm256_fmadd_pd(vc_h1,_mm256_set1_pd(2.0),vc_h1);
				vc_h2 = _mm256_fmadd_pd(vc_h2,_mm256_set1_pd(2.0),vc_h2);
				vc_h3 = _mm256_fmadd_pd(vc_h3,_mm256_set1_pd(2.0),vc_h3);
				vc_h4 = _mm256_fmadd_pd(vc_h4,_mm256_set1_pd(2.0),vc_h4);
				
		
				vstore(D[(t+1)%2][x][y],vc_h1);
				vstore(D[(t+1)%2][x+1 ][y],vc_h2);
				vstore(D[(t+1)%2][x+2][y],vc_h3);
				vstore(D[(t+1)%2][x+3][y],vc_h4);

				

			}
		}
	}
	gettimeofday(&end2,0);

	/*
	for (t = 0; t < T; t++){
		for (x = 0; x < NX; x++)
		{
			for (y = 0; y < NY; y++)
			{
				double diff = B[x][y] - C[x][y];
				if(diff > 0.5){
					printf("Error %g %g\n",B[x][y] , C[x][y]);
				}
								
			}
		}
	}*/
	printf("TEST2 GStencil/s = %f\n", ((double)NX * NY * T) / (double)(end2.tv_sec - end.tv_sec + (end2.tv_usec - end.tv_usec) * 1.0e-6) / 1000000000L);

}
