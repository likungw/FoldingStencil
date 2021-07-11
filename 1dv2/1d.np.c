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
	for (i = 0; i < 2; i++) {
		A[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
		B[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
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
 
	ompp(B, N, T, Bx, tb);
	//redun_load(A, N, T);
	//regis_mov(A, N, T);
	one_step(A, B, N, T, Bx, tb);
	two_steps(A, N, T, Bx, tb);
	four_steps(A, B, N, T, Bx, tb);
	//halfpipe(A, B, N, T, Bx, tb);
	one_tile(A, B, N, T, Bx, tb);
	two_tiles(A, B, N, T, Bx, tb);
	//dlt(A,B,N,T);

	return 0;
}





