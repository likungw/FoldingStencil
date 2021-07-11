#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
#include "defines.h"

void one_step(double** A, double** B, int N, int T, int Bx, int tb){
	double** C = (double**)malloc(sizeof(double*) * 2);
	long int i;
	for (i = 0; i < 2; i++) {
		C[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
	}
	for (i = 0; i < N+2*XSLOPE; i++) {
		C[0][i] = A[0][i];
		C[1][i] = 0;
	}
	int bx = Bx - 2 * tb * XSLOPE;
	int ix = Bx + bx;   // ix is even
	int nb0[2] = { myfloor(N-Bx,ix), myfloor(N-Bx,ix) + 1 };
	
	int nrestpoints = N % ix;
	int bx_first_B1 = (Bx + nrestpoints)/2;
	int bx_last_B1  = (Bx + nrestpoints) - bx_first_B1;

	int xright[2] = {bx_first_B1 + Bx + XSLOPE,  bx_first_B1 + (Bx - bx)/2 + XSLOPE};

	int level = 0;
	int x, xx, t, tt;
 
	register int xmin, xmax;
	vec ww,kk;
	vec v0,v1,v2,v3,v4,v5,gw;
	vallset(ww,GW);
	vallset(kk,LK);
	int xmod = N%veclen4;
	int xup = N - xmod ;
	struct timeval start, end;
	gettimeofday(&start, 0);	
	for (t = 0; t < T; t++) {
		vec v0,v1,v2,v3,v4,v5,gw;
		if(t == 0){
			vloadset(v0,C[0][veclen3],C[0][veclen2],C[0][veclen],C[0][0]);
		}
		else{
			vloadset(v0,C[0][veclen3+3],C[0][veclen3+2],C[0][veclen3+1],C[0][0]);
		}	
		setloadw(v1,v2,v3,v4,gw,C[0][XSLOPE]);
		if(t == 0){
			trans(v1,v2,v3,v4);
		}
		shuffle_tail(v1,gw,v5);
		setcompute(v0,v1,v2,v3,v4,v5,ww,kk);
		if(t == T-1){
			trans(v0,v1,v2,v3);
		}
		setstore(v0,v1,v2,v3,C[0][XSLOPE]);
		v0 = v4;
		v1 = gw;			
		for (x = XSLOPE+veclen4; x < xup; x+=veclen4) {
			setload(v2,v3,v4,gw,C[0][x+veclen]); 
			if(t == 0){
				trans(v1,v2,v3,v4);
			}
			shuffle_headd(v4,v0);
			shuffle_tail(v1,gw,v5);

			setcompute(v0,v1,v2,v3,v4,v5,ww,kk);
			//printf("WW1! %g %g %g %g \n",D0(v2),D1(v2),D2(v2),D3(v2));
			if(t == T-1){
				trans(v0,v1,v2,v3);
			}
			setstore(v0,v1,v2,v3,C[0][x]);
			v0 = v4;
			v1 = gw;
		}
		for (x = xup+1; x < N + XSLOPE; x++) {
			kernel(C);
		}
	}
	gettimeofday(&end, 0);
	//check(B,C,N,T);
	printf("one_step GStencil/s = %f\n",((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
}

void halfpipe(double** A, double** B, int N, int T, int Bx, int tb){
	double** C = (double**)malloc(sizeof(double*) * 2);
	long int i;
	int remain = N%veclen4;

	for (i = 0; i < 2; i++) {
		C[i] = (double*)malloc(sizeof(double) * ((N+2*veclen4-remain) + 2 * XSLOPE));
	}
	for (i = 0; i < N+2*XSLOPE; i++) {
		C[0][i] = A[0][i];
		C[1][i] = 0;
	}
	for (i = N+2*XSLOPE; i < (N+2*veclen4-remain); i++) {
		C[0][i] = 0;
		C[1][i] = 0;
	}
	int bx = Bx - 2 * tb * XSLOPE;
	int ix = Bx + bx;   // ix is even
	int nb0[2] = { myfloor(N-Bx,ix), myfloor(N-Bx,ix) + 1 };
	
	int nrestpoints = N % ix;
	int bx_first_B1 = (Bx + nrestpoints)/2;
	int bx_last_B1  = (Bx + nrestpoints) - bx_first_B1;

	int xright[2] = {bx_first_B1 + Bx + XSLOPE,  bx_first_B1 + (Bx - bx)/2 + XSLOPE};

	int level = 0;
	int x, xx, t, tt;
 
	register int xmin, xmax;
	vec ww,kk;
	vec v0,v1,v2,v3,v4,v5,gw;
	vallset(ww,GW);
	vallset(kk,LK);
	int xmod = N%veclen4;
	int xup = N - xmod ;
	int tailsize = veclen4+veclen+xmod+1;
	double tail[tailsize];
	for (int m =0;m<tailsize;m++){
		tail[m] = C[t%2][N-tailsize+2+m];
	}
	struct timeval start, end;
	gettimeofday(&start, 0);
	int cw = 1;	
	for (t = 0; t < T; t+=4) {
		
		vec v0,v1,v2,v3,v4,v5;
		vec w1,w2,w3,w4;
		vec k1,k2,k3,k4;
		
		setload(v1,v2,v3,v4,C[1-cw][XSLOPE]);
		if(t == 0){
			trans(v1,v2,v3,v4);
		}
		w1 = v1;
		w2 = v2;
		w3 = v3;
		w4 = v4;
		vec zero;
		vallset(zero,0.0);
		shuffle_head(v4,zero,v0);
		v5=_mm256_permute4x64_pd(v1,57);
		setcompute(v0,v1,v2,v3,v4,v5,ww,kk);
		//vallset(zero,0.0);
		shuffle_head(v3,zero,v4);
		v5=_mm256_permute4x64_pd(v0,57);
		setcompute(v4,v0,v1,v2,v3,v5,ww,kk);
		//vallset(zero,0.0);
		shuffle_head(v2,zero,v3);
		v5=_mm256_permute4x64_pd(v4,57);
		setcompute(v3,v4,v0,v1,v2,v5,ww,kk);
		//vallset(zero,0.0);
		shuffle_head(v1,zero,v2);
		v5=_mm256_permute4x64_pd(v3,57);
		setcompute(v2,v3,v4,v0,v1,v5,ww,kk);

		for (x = XSLOPE+veclen4; x < N+2*veclen4-remain; x+=veclen4) {
			setload(k1,k2,k3,k4,C[1-cw][x]); 
			if(t == 0){
				trans(k1,k2,k3,k4);
			}
			
			w1 = _mm256_blend_pd(w1, k1, 3);
			w2 = _mm256_blend_pd(w2, k2, 3);
			w3 = _mm256_blend_pd(w3, k3, 3);
			w4 = _mm256_blend_pd(w4, k4, 3);
			
			v1 = _mm256_permute4x64_pd(w4,147);
			v5 = _mm256_permute4x64_pd(w1,57);

			setcompute(v1,w1,w2,w3,w4,v5,ww,kk);
			
			w4 = _mm256_permute4x64_pd(w3,147);
			v5 = _mm256_permute4x64_pd(v1,57);

			setcompute(w4,v1,w1,w2,w3,v5,ww,kk);

			w3 = _mm256_permute4x64_pd(w2,147);
			v5 = _mm256_permute4x64_pd(w4,57);
			setcompute(w3,w4,v1,w1,w2,v5,ww,kk);

			w2 = _mm256_permute4x64_pd(w1,147);
			v5 = _mm256_permute4x64_pd(w3,57);
			setcompute(w2,w3,w4,v1,w1,v5,ww,kk);
			//printf("WW5! %g %g %g %g \n",D0(v1),D1(v1),D2(v1),D3(v1));
			v2 = _mm256_blend_pd(w2, v2, 7);
			v3 = _mm256_blend_pd(w3, v3, 7);
			v4 = _mm256_blend_pd(w4, v4, 7);
			v0 = _mm256_blend_pd(v1, v0, 7);
			w1 = v1;
			
			if(t == T-4){
				trans(v2,v3,v4,v0);
			}
			
			setstore(v2,v3,v4,v0,C[1-cw][x-veclen4]);			
			v1 = k1;
			v2 = k2;
			v3 = k3;
			v4 = k4;
			v0=_mm256_permute4x64_pd(v4,147);
			v5=_mm256_permute4x64_pd(v1,57);
			setcompute(v0,v1,v2,v3,v4,v5,ww,kk);
			v4=_mm256_permute4x64_pd(v3,147);
			v5=_mm256_permute4x64_pd(v0,57);
			setcompute(v4,v0,v1,v2,v3,v5,ww,kk);
			v3=_mm256_permute4x64_pd(v2,147);
			v5=_mm256_permute4x64_pd(v4,57);
			setcompute(v3,v4,v0,v1,v2,v5,ww,kk);
			v2=_mm256_permute4x64_pd(v1,147);
			v5=_mm256_permute4x64_pd(v3,57);
			setcompute(v2,v3,v4,v0,v1,v5,ww,kk);
			v2 = _mm256_blend_pd(v2, w2, 1);
			v3 = _mm256_blend_pd(v3, w3, 1);
			v4 = _mm256_blend_pd(v4, w4, 1);
			v0 = _mm256_blend_pd(v0, w1, 1);
			w1 = k1;
			w2 = k2;
			w3 = k3;
			w4 = k4;
			//printf("WW1! %g %g %g %g \n",D0(v2),D1(v2),D2(v2),D3(v2));
	
		}
		//cw = 1 - cw;
		
	}
	gettimeofday(&end, 0);
	//check(B,C,N,T);
	printf("HalfPIPE GStencil/s = %f\n",((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
}

