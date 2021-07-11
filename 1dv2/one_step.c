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
	
	__m512d ww,kk;
	__m512d v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,gw;
	vallset(ww,GW);
	vallset(kk,LK);
	int xmod = N%veclen8;
	int xup = N - xmod ;
	struct timeval start, end;
	gettimeofday(&start, 0);	
	for (t = 0; t < T; t++) {
		if(t == 0){
			vloadset(v0,C[0][veclen7],C[0][veclen6],C[0][veclen5],C[0][veclen4],C[0][veclen3],C[0][veclen2],C[0][veclen],C[0][0]);
		}
		else{
			vloadset(v0,C[0][veclen7+7],C[0][veclen7+6],C[0][veclen7+5],C[0][veclen7+4],C[0][veclen7+3],C[0][veclen7+2],C[0][veclen7+1],C[0][0]);
		}	
		setloadw(v1,v2,v3,v4,v5,v6,v7,v8,gw,C[0][XSLOPE]);
		if(t == 0){
			trans(v1,v2,v3,v4,v5,v6,v7,v8);
		}
		shuffle_tail(v1,gw,v9);
		setcompute(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,ww,kk);
		if(t == T-1){
			trans(v0,v1,v2,v3,v4,v5,v6,v7);
		}
		setstore(v0,v1,v2,v3,v4,v5,v6,v7,C[0][XSLOPE]);
		v0 = v8;
		v1 = gw;			
		for (x = XSLOPE+veclen8; x < xup; x+=veclen8) {
			setload(v2,v3,v4,v5,v6,v7,v8,gw,C[0][x+veclen]); 
			if(t == 0){
				trans(v1,v2,v3,v4,v5,v6,v7,v8);
			}
			shuffle_headd(v4,v0);
			shuffle_tail(v1,gw,v9);

			setcompute(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,ww,kk);
			//printf("WW1! %g %g %g %g \n",D0(v2),D1(v2),D2(v2),D3(v2));
			if(t == T-1){
				trans(v0,v1,v2,v3,v4,v5,v6,v7);
			}
			setstore(v0,v1,v2,v3,v4,v5,v6,v7,C[0][x]);
			v0 = v8;
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
