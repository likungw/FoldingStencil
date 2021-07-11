#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h> 
#include "defines.h"

void two_steps(double** A, int N, int T, int Bx, int tb){
	tb =1;
	double** B = (double**)malloc(sizeof(double*) * 2);
	long int i;
	for (i = 0; i < 2; i++) {
		B[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
	}
	for (i = 0; i < N+2*XSLOPE; i++) {
		B[0][i] = A[0][i];
		B[1][i] = 0;
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
	int xup;
	vec ww,kk;
	
	vallset(ww,GW);
	vallset(kk,LK);
	struct timeval start, end;
	
	int slevel = T%len;
	gettimeofday(&start, 0);
	for (tt = -tb; tt < T ;  tt += tb ){
		#pragma omp parallel for private(xmin,xmax,t,x,xup,xx,ww,kk)
		for(xx = 0; xx <nb0[level]; xx++) {
			vec v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,gw;
			int ini = max(tt, 0) ;
			for(t= max(tt, 0) ; t <min( tt + 2*tb,  T); t++){
				xmin = (level == 1 && xx == 0) ?             XSLOPE : (xright[level] - Bx + xx*ix + myabs((tt+tb),(t+1))*XSLOPE);
				xmax = (level == 1 && xx == nb0[1] -1) ? N + XSLOPE : (xright[level]      + xx*ix - myabs((tt+tb),(t+1))*XSLOPE);
				//printf("xmin %d xmax %d  \n",xmin,xmax);
				int xmod = (xmax - xmin)%veclen8;
				xup = xmax - xmod;
				if((tt == -tb)||(tt == T-tb)){
					//int wei = tt + slevel*tb;
					vloadset(v0,B[t%2][veclen7+xmin-1],B[t%2][veclen6+xmin-1],B[t%2][veclen5+xmin-1],B[t%2][veclen4+xmin-1],B[t%2][veclen3+xmin-1],B[t%2][veclen2+xmin-1],B[t%2][veclen+xmin-1],B[t%2][xmin-1]);
					setloadw(v1,v2,v3,v4,v5,v6,v7,v8,gw,B[t%2][xmin]);
					if(tt <= 0){
						trans(v1,v2,v3,v4,v5,v6,v7,v8);
					}
					shuffle_tail(v1,gw,v9);
					setcompute(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,ww,kk);
					if(tt > 0){
						trans(v0,v1,v2,v3,v4,v5,v6,v7);
					}
					setstore(v0,v1,v2,v3,v4,v5,v6,v7,B[(t+1)%2][xmin]);						
					v0 = v8;
					v1 = gw;
					for (x = xmin+veclen8; x < xup; x+=veclen8) {
						setload(v2,v3,v4,v5,v6,v7,v8,gw,B[t%2][x+veclen]); 							
						if(tt <= 0){
							trans(v1,v2,v3,v4,v5,v6,v7,v8);
						}							
						shuffle_headd(v8,v0);
						shuffle_tail(v1,gw,v9);
						setcompute(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,ww,kk);
						if(tt > 0){
							trans(v0,v1,v2,v3,v4,v5,v6,v7);
						}
						setstore(v0,v1,v2,v3,v4,v5,v6,v7,B[(t+1)%2][x]);
						v0 = v8;
						v1 = gw;
					}
					for (x = xup; x < xmax; x++) {
						kernel(B);
					}
				}
				else{
					vec va40, vb10, va11, va71,va81, va41, va12, va22, va32, va42, va52, va62, va72, va82, zero;
					double left1,left2,right1,right2;
					for (x = xmin; x < xup; x+=veclen8) {
						if(x == xmin){
							vloadset(v0,B[1-level][veclen7+xmin-1],B[1-level][veclen6+xmin-1],B[1-level][veclen5+xmin-1],B[1-level][veclen4+xmin-1],B[1-level][veclen3+xmin-1],B[1-level][veclen2+xmin-1],B[1-level][veclen+xmin-1],B[1-level][xmin-1]);
							setloadw(v1,v2,v3,v4,v5,v6,v7,v8,gw,B[1-level][xmin]);
						}
						else{								
							setload(v2,v3,v4,v5,v6,v7,v8,gw,B[1-level][x+veclen]);
						}
						if(tt == 0){
							trans(v1,v2,v3,v4,v5,v6,v7,v8);
						}
						if(x > xmin){
							shuffle_headd(v8,v0);
						}
						shuffle_tail(v1,gw,v9);
						setcompute(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,ww,kk);								
						if(x == xmin){
							if((xx == 0)&&(level == 1)){	
								left1 = D0(v0);							
								vallset(zero,0.0);
								shuffle_headd(v7,zero);	
							}
							else{
								left1 = D0(v0);							
								vallset(zero,B[1-level][xmin-2]);
								shuffle_headd(v7,zero);	
							}							
						}
						else{
							shuffle_tail(va11,v0,va11);							
							compute(va71,va82,va11,ww,kk);
							if(tt == T-2*tb){
								trans(va12,va22,va32,va42,va52,va62,va72,va82);
							}
							setstore(va12,va22,va32,va42,va52,va62,va72,va82,B[1-level][x-veclen8]);
							shuffle_head(v7,va81,zero);																	
						}
						va11 = v0;
						va71 = v6;
						va82 = v7;
						va81 = v7;
						compute(zero,v0,v1,ww,kk);
						compute(v0,v1,v2,ww,kk);
						compute(v1,v2,v3,ww,kk);
						compute(v2,v3,v4,ww,kk);
						compute(v3,v4,v5,ww,kk);
						compute(v4,v5,v6,ww,kk);
						compute(v5,v6,v7,ww,kk);
						va12 = v0;
						va22 = v1;
						va32 = v2;
						va42 = v3;
						va52 = v4;
						va62 = v5;
						va72 = v6; 
						v0 = v8;
						v1 = gw;
						if(x == xmin){	
							left2 = D0(va12);									
						}
					}
					double now,last; 
					last =  B[1-level][xup-1];
					for (x = xup; x < xmax; x++) {
						now = B[1-level][x];
						B[1-level][x] = LK * ((B[1-level][x+1] + GW * now) + last);
						last = now;
					}
					last = D3(va42);
					right1 =  B[1-level][xmax-1];
					vallset(gw,B[1-level][xup]);
					shuffle_tail(va11,gw,va11);
					compute(va71,va82,va11,ww,kk);
					if(tt == T-2*tb){
						trans(va12,va22,va32,va42,va52,va62,va72,va82);
					}
					setstore(va12,va22,va32,va42,va52,va62,va72,va82,B[1-level][xup-veclen8]);
					for (x = xup; x < xmax-1; x++) {
						now = B[1-level][x];
						B[1-level][x] = LK * ((B[1-level][x+1] + GW * now) + last);
						last = now;
					}
					B[1-level][xmax-1] = LK * ((B[1-level][xmax+1] + GW * B[1-level][xmax-1]) + last);
					right2 = B[1-level][xmax-1];
					if((level == 1)&&(xx == 0)){
						B[level][xmax-1] = right1;
						B[level][xmax-2] = right2;
					}
					else if((level == 1)&&(xx == nb0[level]-1)){
						B[level][xmin] = left1;
						B[level][xmin-1] = left2;
					}
					else{
						B[level][xmax-1] = right1;
						B[level][xmax-2] = right2;
						B[level][xmin] = left1;
						B[level][xmin-1] = left2;
					}	
					t++;					
				}					
			}
		}
		level = 1 - level;
	}
	gettimeofday(&end, 0);
	//check(A,B,N,T);
	printf("two_steps GStencil/s = %f\n",((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
}


