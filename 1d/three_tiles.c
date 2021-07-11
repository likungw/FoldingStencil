#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
#include "defines.h"

void four_tiles(double** A, double** R, int N, int T, int Bx, int tb){
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
	double aw,bw;
	gettimeofday(&start, 0);
	int tbs = tb/2;
	vec v0,v1,v2,v3,v4,v5,v6,v7,v8;
	vec w0,w1,w2,w3,w4,w5;
	vec pre1,pre2,pre3,pre4,post1,post2,post3,post4;
	for (tt = -tb; tt < T ;  tt += tb ){
		//#pragma omp parallel for private(xmin,xmax,t,x)
		for(xx = 0; xx <nb0[level]; xx++) {
			int xmin, xmax;
			double left,right;			
			for(t= max(tt, 0) ; t <min( tt + 2*tb,  T); t+=2){
				xmin = (level == 1 && xx == 0) ?             XSLOPE : (xright[level] - Bx + xx*ix + myabs((tt+tb),(t+1))*XSLOPE);
				xmax = (level == 1 && xx == nb0[1] -1) ? N + XSLOPE : (xright[level]      + xx*ix - myabs((tt+tb),(t+1))*XSLOPE);
				//printf("xmin %d xmax %d  \n",xmin,xmax);
				int xmod = (xmax - xmin)%veclen4;
				int xup = xmax - xmod;
				int xb,xe;			
				if((tt == -tb)||(t >= tt + tb)){					
					xe = xmax - ((xmax-xmin-2)%veclen4) - 2;
					for(x = xmin; x < xe; x+=veclen4){
						if(x == xmin){					
							setloadw(v1,v2,v3,v4,v5,B[(t/2)%2][x-1]);
						}
						else{
							setload(v2,v3,v4,v5,B[(t/2)%2][x-1+veclen]);
						}												
						trans(v1,v2,v3,v4);	
						post1 = tail1(v1,v5);
						post2 = tail2(v2,v5);
						setcompute(v1,v2,v3,v4,post1,post2,ww,kk); 						
						if(x > xmin){
							post1 = tail1(w1,v1);
							post2 = tail1(w2,v2);
							post3 = tail1(w3,v3);
							post4 = tail1(w4,v4);
							if(x == 1 + veclen4){
								vec tmp = w1 + w2;
								setcompute8(w1,w2,w3,w4,post1,post2,post3,post4,ww,kk);
								tmp = tmp + w1;
								vstore(B[((t/2)+1)%2][1],tmp);
								tmp = tmp + w2;
								vstore(B[((t/2)+1)%2][2],tmp);
							}
							else{//To be verified!
								vstore(B[((t/2)+1)%2][x],v1);
								B[(t/2)%2][xmin+1] = D0(v2);
							}
							setcompute8(w1,w2,w3,w4,post1,post2,post3,post4,ww,kk);
							setcompute(w1,w2,w3,w4,post1,post2,ww,kk);							
							trans(w1,w2,w3,w4);	
							setstore(w1,w2,w3,w4,B[((t/2)+1)%2][x+2-veclen4]);
						}
						w1 = v1;
						w2 = v2;
						w3 = v3;
						w4 = v4;
						v1 = v5;						
					}								
					for(x = xe; x < xmax; x++){
						B[((t/2)+1)%2][x] = LK * ((B[(t/2)%2][x+1] + B[(t/2)%2][x]) + B[(t/2)%2][x-1]);
						//printf("tt5 %d t %d xe %d www!x %d: %g \n",tt,t,xe,x,B[((t/2)+1)%2][x]);
					}
					B[(t/2)%2][xmax-2] = B[((t/2)+1)%2][xmax-2];
					//double fix = B[(t/2)%2][xmax-1];
					B[((t/2)+1)%2][xe - 2] = D3(v3);
					B[((t/2)+1)%2][xe - 1] = D3(v4);
					double last = B[((t/2)+1)%2][xe];					
					vload(v1,B[((t/2)+1)%2][xe]);
					post1 = tail1(w1,v1);
					post2 = tail2(w2,v1);
					setcompute(w1,w2,w3,w4,post1,post2,ww,kk);
					trans(w1,w2,w3,w4);	
					setstore(w1,w2,w3,w4,B[((t/2)+1)%2][xe+1-veclen4]);					
					for (x = xe + 1; x < xmax-1; x++) {
						double now = B[((t/2)+1)%2][x];
						B[((t/2)+1)%2][x] = LK * ((B[((t/2)+1)%2][x+1] + now) + last);
						last = now;
					}
					if(xmax >= N){
						B[((t/2)+1)%2][x] = LK * ((B[((t/2)+1)%2][xmax-1] + B[((t/2)+1)%2][xmax]) + last);
					}
					
				}
				else{
					xe = xmax + 1 - ((xmax-xmin+2)%veclen4) ;
					for(x = xmin; x < xe; x+=veclen4){
						if(x == xmin){					
							setloadw(v0,v1,v2,v3,v4,B[(t/2)%2][x-3]);
							pre2 = v0;
							pre1 = v0;
						}
						else{
							setload(v1,v2,v3,v4,B[(t/2)%2][x+1]);
						}						
						trans(v1,v2,v3,v4);						
						pre1 = head1(v4,pre1);
						if(x == xmin ){
							pre2 = head2(v3,pre2);
						}
						else{
							pre2 = head1(v3,pre2);
						}		
						setcompute(pre2,pre1,v1,v2,v3,v4,ww,kk);
						if(x == xmin ){
							pre3 = headnum(v2,B[((t/2)+1)%2][xmin-1]);
							pre4 = headnum(v1,B[(t/2)%2][xmin-2]);
						}
						else{
							pre3 = head1(v2,pre3);
							pre4 = head1(v1,pre4);
						}
						setcompute(pre4,pre3,pre2,pre1,v1,v2,ww,kk);
						if(x == 1){
							pre4 = boundary(pre4);
						}
					
						trans(pre4,pre3,pre2,pre1);	
						setstore(pre4,pre3,pre2,pre1,B[((t/2)+1)%2][x-1]);
						pre1 = v4;
						pre2 = v3;
						pre3 = v2;
						pre4 = v1;						
					}
					for(x = xe ; x < xmax; x++){
						B[((t/2)+1)%2][x] = LK  * ((B[(t/2)%2][x+1] + B[(t/2)%2][x]) + B[(t/2)%2][x-1]);
					}
					double last = D3(v1);
					for (x = xe ; x < xmax; x++) {
						double now = B[((t/2)+1)%2][x];
						B[((t/2)+1)%2][x] = LK * ((B[((t/2)+1)%2][x+1] + now) + last);
						last = now;
					}
					if(x<N){
						B[((t/2)+1)%2][x] = LK * ((B[(t/2)%2][xmax+1] + B[((t/2)+1)%2][x]) + last);
					}
				}					
			}
		}
		level = 1 - level;
	}
	gettimeofday(&end, 0);
	check(R,B,N,T);
	printf("Tiles 2 GStencil/s = %f\n",((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
}


