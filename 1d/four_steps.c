#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
#include "defines.h"

void four_steps(double** A, double** R, int N, int T, int Bx, int tb){
    tb = 2;
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
    vec ww,kk;
	vallset(ww,GW);
	vallset(kk,LK);
	struct timeval start, end;
	
	int slevel = T%len;
    
	gettimeofday(&start, 0);
    for (tt = -tb; tt < T ;  tt += tb ){
		#pragma omp parallel for private(xmin,xmax,t,x,xx,ww,kk)
		for(xx = 0; xx <nb0[level]; xx++) {
			vec v0,v1,v2,v3,v4,v5,gw;
			int ini = max(tt, 0) ;
			for(t= max(tt, 0) ; t <min( tt + 2*tb,  T); t++){
				xmin = (level == 1 && xx == 0) ?             XSLOPE : (xright[level] - Bx + xx*ix + myabs((tt+tb),(t+1))*XSLOPE);
				xmax = (level == 1 && xx == nb0[1] -1) ? N + XSLOPE : (xright[level]      + xx*ix - myabs((tt+tb),(t+1))*XSLOPE);
				//printf("xmin %d xmax %d  \n",xmin,xmax);
				int xmod = (xmax - xmin)%veclen4;
				int xup = xmax - xmod;
				if((tt == -tb)||(tt == T-tb)){
					//int wei = tt + slevel*tb;
					vec va40, va11, va31, va41, va12, va22, va32, va42, zero;
					for (x = xmin; x < xup; x+=veclen4) {
						if(x == xmin){
							//Could it be optimized ?
							B[t%2][xmin-2] = 0.25 * ((B[t%2][xmin] + B[t%2][xmin-1]) + B[t%2][xmin-2]);
							vloadset(v0,B[t%2][veclen3+xmin-1],B[t%2][veclen2+xmin-1],B[t%2][veclen+xmin-1],B[t%2][xmin-1]);
							setloadw(v1,v2,v3,v4,gw,B[t%2][xmin]);
						}
						else{
							setload(v2,v3,v4,gw,B[1-level][x+veclen]);
						}
						if(tt == 0){
							trans(v1,v2,v3,v4);
						}
						if(x > xmin){
							shuffle_headd(v4,v0);
						}
						shuffle_tail(v1,gw,v5);
						setcompute(v0,v1,v2,v3,v4,v5,ww,kk);
						if(x == xmin){
							if((xx == 0)&&(level == 1)){															
								vallset(zero,0.0);
								shuffle_headd(v3,zero);	
							}
							else{//??						
								vallset(zero,B[1-level][xmin-2]);
								shuffle_headd(v3,zero);	
							}							
						}
						else{
							shuffle_tail(va11,v0,va11);							
							compute(va31,va42,va11,ww,kk);
							if(tt == T-2*tb){
								trans(va12,va22,va32,va42);
							}
							setstore(va12,va22,va32,va42,B[1-level][x-veclen4]);
							shuffle_head(v3,va41,zero);																	
						}
						va11 = v0;
						va31 = v2;
						va41 = v3;
						va42 = v3;
						compute(zero,v0,v1,ww,kk);
						compute(v0,v1,v2,ww,kk);
						compute(v1,v2,v3,ww,kk);
						va12 = v0;
						va22 = v1;
						va32 = v2;
						v0 = v4;
						v1 = gw;
					}
					double now,last; 
					last =  B[1-level][xup-1];
					for (x = xup; x < xmax; x++) {
						now = B[1-level][x];
						B[1-level][x] = LK * ((B[1-level][x+1] +  now) + last);
						last = now;
					}
					if(tt < 0){
						B[1-level][xmax+1] = LK * ((B[1-level][xmax+1] +  B[1-level][xmax]) + last);
					}
					last = D3(va42);
					
					vallset(gw,B[1-level][xup]);
					shuffle_tail(va11,gw,va11);
					compute(va31,va42,va11,ww,kk);
					if(tt == T-2*tb){
						trans(va12,va22,va32,va42);
					}
					setstore(va12,va22,va32,va42,B[1-level][xup-veclen4]);
					for (x = xup; x < xmax-1; x++) {
						now = B[1-level][x];
						B[1-level][x] = LK * ((B[1-level][x+1] +  now) + last);
						last = now;
					}
					B[1-level][xmax-1] = LK * ((B[1-level][xmax+1] +  B[1-level][xmax-1]) + last);
					
					t++;
				}
				else{
					vec w1,w2,w3,w4,w5;
					vec k0,k1,k2,k3,k4,k5;
					vec w40,w41,w31,w11,k42;
					vec va40, va11, va31, va41, va12, va22, va32, va42, zero;
					vec vb40, vb11, vb31, vb41, vb12, vb22, vb32, vb42, bzero;
					double left1,left2,left3,left4;
					double right1,right2,right3,right4;
					for (x = xmin; x < xup; x+=veclen4) {
						if(x == xmin){
							if((tt==0)&&(xx>0)){
								B[1-level][xmin-6] = LK * ((B[t%2][xmin-2] +  B[t%2][xmin-3]) + B[t%2][xmin-4]);
								B[1-level][xmin-5] = LK * ((B[t%2][xmin-1] +  B[t%2][xmin-2]) + B[t%2][xmin-3]);
								B[1-level][xmin-2] = LK * ((B[t%2][xmin  ] +  B[t%2][xmin-1]) + B[t%2][xmin-2]);
								B[1-level][xmin-4] = LK * ((B[t%2][xmin+1] +  B[t%2][xmin  ]) + B[t%2][xmin-1]);
								B[1-level][xmin-3] = LK * ((B[t%2][xmin+2] +  B[t%2][xmin+1]) + B[t%2][xmin  ]);
								B[1-level][xmin-7] = LK * ((B[t%2][xmin-2] +  B[t%2][xmin-5]) + B[t%2][xmin-6]);								
								B[1-level][xmin-6] = LK * ((B[t%2][xmin-3] +  B[t%2][xmin-4]) + B[t%2][xmin-2]);
								B[1-level][xmin-3] = LK * ((B[t%2][xmin-4] +  B[t%2][xmin-2]) + B[t%2][xmin-5]);
								B[1-level][xmin-4] = LK * ((B[t%2][xmin-6] +  B[t%2][xmin-3]) + B[t%2][xmin-7]);
							}							
							vloadset(v0,B[1-level][veclen3+xmin-1],B[1-level][veclen2+xmin-1],B[1-level][veclen+xmin-1],B[1-level][xmin-1]);
							setload(v1,v2,v3,v4,B[1-level][xmin]);
							setloadw(w1,w2,w3,w4,gw,B[1-level][xmin]);
						}
						else{								
							setload(v2,v3,v4,gw,B[1-level][x+veclen]);
						}
						if(tt == 0){
							trans(v1,v2,v3,v4);
							if(x==xmin){
								trans(w1,w2,w3,w4);
							}
						}
						if(x > xmin){
							shuffle_headd(v4,v0);
							shuffle_tail(v1,gw,v5);
						}else{
							shuffle_head(w4,v4,zero);	
							shuffle_tail(v1,w1,v5);
							shuffle_tail(w1,gw,w5);
						}						
						setcompute(v0,v1,v2,v3,v4,v5,ww,kk);							
						if(x == xmin){							
							setcompute(zero,w1,w2,w3,w4,w5,ww,kk);	
							w40 = w4;
							w41 = w3;
							w11 = zero;
							w31 = w2;
							if((xx == 0)&&(level == 1)){	
								//left1 = D0(v0);							
								vallset(v4,0.0);									
							}
							else{
								//left1 = D0(v0);							
								vallset(v4,B[1-level][xmin-2]);								
							}			
							shuffle_headd(v3,v4);			
							shuffle_head(w3,v3,w4);	
							shuffle_tail(v0,zero,v5);
							setcompute(v4,v0,v1,v2,v3,v5,ww,kk);								
							computed(w4,zero,w1,ww,kk);
							computed(zero,w1,w2,ww,kk);
							computed(w1,w2,w3,ww,kk);
							k42 = v2;
							if((xx == 0)&&(level == 1)){	
								//left1 = D0(v0);							
								vallset(v3,0.0);									
							}
							else{
								//left1 = D0(v0);							
								vallset(v3,B[1-level][xmin-3]);								
							}
							shuffle_headd(v2,v3);
							shuffle_tail(v4,w4,v5);
							setcompute(v3,v4,v0,v1,v2,v5,ww,kk);
							k1 = v3;
							k2 = v4;
							k3 = v0;
							k4 = v1;
							w1 = w4;
							w4 = w3;
							w2 = zero;
							w3 = w1;
							v0 = w40;
							v1 = gw;
						}
						else{
							shuffle_head(v3,w41,v4);
							shuffle_tail(w11,v0,w11);
							computed(w31,w41,w11,ww,kk);
							k42 = w31;
							shuffle_head(w41,k42,w4);
							shuffle_tail(w1,v4,w5);	
							setcompute(w4,w1,w2,w3,w31,w5,ww,kk);
							w31 = v2;	
							w40 = v4;
							w41 = v3;
							w11 = v0;							
							computed(v4,v0,v1,ww,kk);
							computed(v0,v1,v2,ww,kk);
							computed(v1,v2,v3,ww,kk);																				
							
							
							if(x == xmin+veclen4*2){
								if((xx == 0)&&(level == 1)){	
									//left1 = D0(v0);							
									vallset(k0,0.0);									
								}
								else{
									//left1 = D0(v0);							
									vallset(k0,B[1-level][xmin-4]);								
								}
							}
							shuffle_head(k4,k0,k0);
							shuffle_tail(k1,w3,k5);
							setcompute(k0,k1,k2,k3,k4,k5,ww,kk);	
							if(tt == T-2*tb){
								trans(k0,k1,k2,k3);
							}
							setstore(k0,k1,k2,k3,B[1-level][x-veclen4*2]);
							k0 = k4;
							k1 = w3;
							k2 = w4;
							k3 = zero;
							k4 = w1;
							w1 = v4;
							w2 = v0;
							w3 = v1;
							w4 = v2;
							v0 = w40;
							v1 = gw;
							
						}
						if(x == xmin){	
							left2 = D0(va12);
							t+=veclen4;									
						}
					}
					double upleft0,upleft1,upleft2,upleft3;
					upleft0 = B[1-level][xup-1];
					
					B[1-level][xup-7] = LK * ((B[1-level][xup+3] +  B[1-level][xup+2]) + B[1-level][xup+1]);
					B[1-level][xup-6] = LK * ((B[1-level][xup+2] +  B[1-level][xup+1]) + B[1-level][xup  ]);
					B[1-level][xup-5] = LK * ((B[1-level][xup+1] +  B[1-level][xup  ]) + B[1-level][xup-1]);
					B[1-level][xup-4] = LK * ((B[1-level][xup  ] +  B[1-level][xup-1]) + B[1-level][xup-2]);
					B[1-level][xup-3] = LK * ((B[1-level][xup-1] +  B[1-level][xup-2]) + B[1-level][xup-3]);
					B[1-level][xup-1] = LK * ((B[1-level][xup-5] +  B[1-level][xup-6]) + B[1-level][xup-7]);								
					B[1-level][xup-2] = LK * ((B[1-level][xup-4] +  B[1-level][xup-5]) + B[1-level][xup-6]);
					B[1-level][xup-3] = LK * ((B[1-level][xup-3] +  B[1-level][xup-4]) + B[1-level][xup-5]);
					B[1-level][xup-1] = LK * ((B[1-level][xup-1] +  B[1-level][xup-2]) + B[1-level][xup-3]);
					
					double now,last; 
					last =  upleft0;
					for (x = xup; x < xmax+veclen; x++) {
						now = B[1-level][x];
						B[1-level][x] = LK * ((B[1-level][x+1] +  now) + last);
						last = now;
					}
                    right1 =  B[1-level][xmax-1];

                    upleft1 = D3(w41);
                    vallset(gw,B[1-level][xup]);
					shuffle_tail(w11,gw,w5);
					compute(w31,w41,w11,ww,kk);
					last = upleft1;	
                    for (x = xup; x < xmax+veclen; x++) {
						now = B[1-level][x];
						B[1-level][x] = LK * ((B[1-level][x+1] +  now) + last);
						last = now;
					}
                    right1 =  B[1-level][xmax-1];				
					

                    upleft2 = D3(w41);
					shuffle_head(w41,k42,w4);
                    vallset(gw,B[1-level][xup]);
					shuffle_tail(w11,gw,w5);
					shuffle_tail(w1,v4,w5);	
					setcompute(w4,w1,w2,w3,w31,w5,ww,kk);
                    
					shuffle_head(k4,k0,k0);
					shuffle_tail(k1,w3,k5);
					setcompute(k0,k1,k2,k3,k4,k5,ww,kk);
					if(tt == T-2*tb){
						trans(k0,k1,k2,k3);
					}
					setstore(k0,k1,k2,k3,B[1-level][x-veclen4]);
					shuffle_head(w3,k4,w31);
					shuffle_tail(k1,w3,k5);
                    setcompute(w31,w4,w1,w2,w3,k5,ww,kk);
                    if(tt == T-2*tb){
						trans(k0,k1,k2,k3);
					}
					setstore(w31,w4,w1,w2,B[1-level][x]);


					for (x = xup; x < xmax-1; x++) {
						now = B[1-level][x];
						B[1-level][x] = LK * ((B[1-level][x+1] +  now) + last);
						last = now;
					}
					B[1-level][xmax-1] = LK * ((B[1-level][xmax+1] +  B[1-level][xmax-1]) + last);
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
					t+=3;					
				}					
			}
		}
		level = 1 - level;
	}
	gettimeofday(&end, 0);
	//check(A,B,N,T);
	printf("four_steps GStencil/s = %f\n",((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
}