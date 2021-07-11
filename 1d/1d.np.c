#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
#include "defines.h"

int main(int argc, char * argv[]) {

	struct timeval start, end;
	printf("KKK!!\n");
	long int  i;
	int N  = atoi(argv[1]);
	int T  = atoi(argv[2]);
	int Bx = atoi(argv[3]);
	int tb = atoi(argv[4]);

	if(Bx<(2*XSLOPE+1) || Bx>N || tb>(((Bx-1)/2)/XSLOPE)){ 
		return 0;
	}
	double** A = (double**)malloc(sizeof(double*) * 2);
	double** B = (double**)malloc(sizeof(double*) * 2);
	double** C = (double**)malloc(sizeof(double*) * 3);
	double** D = (double**)malloc(sizeof(double*) * 1);
	for (i = 0; i < 2; i++) {
		A[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
		B[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
	}
	for (i = 0; i < 3; i++) {
		C[i] = (double*)malloc(sizeof(double) * (N + 2 ));
	}
	for (i = 0; i < 1; i++) {
		D[i] = (double*)malloc(sizeof(double) * (3* N + 6 ));
	}

	srand(100);
	A[0][0] = 0; 
	for (i = 1; i < N+2*XSLOPE; i++) {
		A[0][i] = i;
		A[1][i] = 0;
	}
	A[0][N+XSLOPE] = 0; 
	for (i = 0; i < N+2*XSLOPE; i++) {
		B[0][i] = A[0][i];
		B[1][i] = 0;
	}    
	for (i = 0; i < N+2*XSLOPE; i++) {
		C[0][i] = i;
		C[1][i] = i;
		C[2][i] = i;
	}   
	for (i = 0; i < 3*N+2*3; i++) {
		D[0][i] = i;
	}   
 
	//redun_load(A, N, T);
	//regis_mov(A, N, T);
	ompp(B, N, T, Bx, tb);
	ompp_apop(A, C, N, T, Bx, tb);
	our_1d1sapop(A, D, N, T, Bx, tb);
	//our_1d1s3p(A, B, N, T, Bx, tb);
	//cross_1d2s3p(A, B, N, T, Bx, tb);
	//cross_1d3s3p(A, B, N, T, Bx, tb);
	//cross_1d4s3p(A, B, N, T, Bx, tb);
	//ompp_apop(A, C, N, T, Bx, tb);
	//redun_load(A, N, T);
	//regis_mov(A, N, T);
	//one_step(A, B, N, T, Bx, tb);
	//two_steps(A, N, T, Bx, tb);
	//four_steps(A, B, N, T, Bx, tb);
	//halfpipe(A, B, N, T, Bx, tb);
	//fold_1d4s5p(A, B, N, T, Bx, tb);
	//cross_1d4s5p(A, B, N, T, Bx, tb);
	//cross_1d2s3p(A, B, N, T, Bx, tb);
	//cross_1d3s3p(A, B, N, T, Bx, tb);
	//cross_1d4s3p(A, B, N, T, Bx, tb);
	//cross_1d2s5p(A, B, N, T, Bx, tb);
	//our_1d1sapop(A, D, N, T, Bx, tb);
	//our_1d1s3p(A, B, N, T, Bx, tb);
	//our_1d2s3ph(A, B, N, T, Bx, tb);
	//our_1d1s5p(A, B, N, T, Bx, tb);
	//our_1d2s5p(A, B, N, T, Bx, tb);
	//dlt(A,B,N,T);
	//dlt_unaligned(A, N, T);
	//dlt_aligned(A, N, T);
	//dlt_aligned_unroll(A, N, T);
	//dlt_tblock2(A, N, T);

	return 0;
}





