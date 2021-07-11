#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
#include "defines.h"

void ompp_apop(double** A, double** B, int N, int T, int Bx, int tb){

	struct timeval start, end;
	gettimeofday(&start, 0);
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
	for (tt = -tb; tt < T ;  tt += tb ){
	#pragma omp parallel for private(xmin,xmax,t,x) 
		for(xx = 0; xx <nb0[level]; xx++) {
			for(t= max(tt, 0) ; t <min( tt + 2*tb,  T); t++){
				xmin = (level == 1 && xx == 0) ?             XSLOPE : (xright[level] - Bx + xx*ix + myabs((tt+tb),(t+1))*XSLOPE);
				xmax = (level == 1 && xx == nb0[1] -1) ? N + XSLOPE : (xright[level]      + xx*ix - myabs((tt+tb),(t+1))*XSLOPE);
				#pragma ivdep
				#pragma vector always
				for(x = xmin; x < xmax; x++){
					kernel_apop(A,B);
				}
				
			}
		}
		level = 1 - level;
	}
	gettimeofday(&end, 0);
	//check(A,A,N,T);
	printf("ompp_apop GStencil/s = %f\n",((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
}