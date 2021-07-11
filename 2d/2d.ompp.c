#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h> 
#include "2d.defines.h"

void ompp_2d(double*** A, int NX, int NY, int T, int Bx, int By, int tb){
	struct timeval start, end;
	long int t;

	int bx = Bx-2*(tb*XSLOPE);
	int by = By-2*(tb*YSLOPE);

	int ix = Bx+bx;
	int iy = By+by; // ix and iy are even.

	int xnb0  = ceild(NX-bx,ix);
	int ynb0  = ceild(NY-by,iy);
	int xnb11 = ceild(NX+(Bx-bx)/2,ix);
	int ynb12 = ceild(NY+(By-by)/2,iy);
	int ynb11 = ynb0;
	int xnb12 = xnb0;
	int xnb2  = xnb11;
	int ynb2  = ynb12;

	int nb1[2]   = {xnb11 * ynb11, xnb12 * ynb12};
	int nb02[2]  = {xnb0 * ynb0, xnb2 * ynb2};
	int xnb1[2]  = {xnb11, xnb12};
	int xnb02[2] = {xnb0, xnb2};

	int xleft02[2]   = {XSLOPE + bx, XSLOPE - (Bx-bx)/2}; // the start x dimension of the first B11 block is bx
	int ybottom02[2] = {YSLOPE + by, YSLOPE - (By-by)/2}; // the start y dimension of the first B11 block is by
	int xleft11[2]   = {XSLOPE,		 XSLOPE + ix/2};
	int ybottom11[2] = {YSLOPE + by, YSLOPE - (By-by)/2};
	int xleft12[2]   = {XSLOPE + bx, XSLOPE - (Bx-bx)/2};
	int ybottom12[2] = {YSLOPE,		 YSLOPE + iy/2};

	// printf("%d\t%d\n", NX - xnb0 * ix, NY - ynb0 * iy);
	int level = 0;
	int tt,n;
	int x, y;
	register int ymin, ymax;
	int xmin,xmax;
	int dif;

	gettimeofday(&start,0);

	for(tt = -tb; tt < T; tt += tb){
#pragma omp parallel for schedule(dynamic)  private(xmin,xmax,ymin,ymax,t,x,y)
		for(n = 0; n < nb02[level]; n++){
 dif = 0;
			for(t = max(tt,0); t < min(tt + 2*tb, T); t+=1) { 

				xmin = max(   XSLOPE,   xleft02[level] + (n%xnb02[level]) * ix      + myabs(t+dif+1,tt+tb) * XSLOPE);
				xmax = min(NX+XSLOPE,   xleft02[level] + (n%xnb02[level]) * ix + Bx - myabs(t+dif+1,tt+tb) * XSLOPE);
				ymin = max(   YSLOPE, ybottom02[level] + (n/xnb02[level]) * iy      + myabs(t+dif+1,tt+tb) * YSLOPE);
				ymax = min(NY+YSLOPE, ybottom02[level] + (n/xnb02[level]) * iy + By - myabs(t+dif+1,tt+tb) * YSLOPE);
				//printf("1 -- t %d xmin %d xmax %d ymin %d ymax %d \n",tt ,xmin ,xmax ,ymin, ymax );
				//dif++;

				for(x = xmin; x < xmax; x++) {
#pragma ivdep
#pragma vector always
					for(y = ymin; y < ymax; y++) {
						kernel(A);
					}
				}
			}
		}

#pragma omp parallel for schedule(dynamic)  private(xmin,xmax,ymin,ymax,t,x,y)
		
		for(n = 0; n < nb1[0] + nb1[1]; n++) {
dif = 0;
			for(t = tt + tb; t < min(tt + 2*tb, T); t+=1) {
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
				//dif++;
				//printf("2 -- t %d xmin %d xmax %d ymin %d ymax %d \n",tt ,xmin ,xmax ,ymin, ymax );

				for(x = xmin; x < xmax; x++) {
#pragma ivdep
#pragma vector always
					for(y = ymin; y < ymax; y++) {
						kernel(A);
					}
				}
			}
		}
		level = 1 - level;
	}

	gettimeofday(&end,0);

	printf("OMPP GStencil/s = %f\n", ((double)NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

}
