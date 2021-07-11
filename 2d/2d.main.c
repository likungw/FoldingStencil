#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
//#include <avx2intrin.h>
#include "2d.defines.h"

int main(int argc, char* argv[]) {
	struct timeval start, end;
	printf("Start!!\n");
	long int i, j;
	if (argc != 7) {
		printf("usage: %s <NX> <NY> <T> <Bx> <By> <tb>\n", argv[0]);
		return 0;
	}
	int NX = atoi(argv[1]);
	int NY = atoi(argv[2]);
	int T = atoi(argv[3]);
	int Bx = atoi(argv[4]);
	int By = atoi(argv[5]);
	int tb = atoi(argv[6]);

	if (Bx<(2 * XSLOPE + 1) || By<(2 * YSLOPE + 1) || Bx>NX || By>NY || tb > min(((Bx - 1) / 2) / XSLOPE, ((By - 1) / 2) / YSLOPE)) {
		return 0;
	}
	double*** A = (double***)malloc(sizeof(double**) * 2);
	double*** B = (double***)malloc(sizeof(double**) * 2);
	for (i = 0; i < 2; i++) {
		A[i] = (double**)malloc(sizeof(double*) * (NX + 2 * XSLOPE));
		B[i] = (double**)malloc(sizeof(double*) * (NX + 2 * XSLOPE));
	}
	for (i = 0; i < 2; i++) {
		for (j = 0; j < (NX + 2 * XSLOPE); j++) {
			A[i][j] = (double*)malloc(sizeof(double) * (NY + 2 * YSLOPE));
			B[i][j] = (double*)malloc(sizeof(double) * (NY + 2 * YSLOPE));
		}
	}

	for (i = 0; i < NX + 2 * XSLOPE; i++) {
		for (j = 0; j < NY + 2 * YSLOPE; j++) {
			// modified
			srand(time(NULL));
			// A[0][i][j] = 1.0 * (rand() % 1024);
			A[0][i][j] = i * 2 + j / 2;
			A[1][i][j] = 0;
		}
	}
	for (i = 0; i < NX + 2 * XSLOPE; i++) {
		for (j = 0; j < NY + 2 * YSLOPE; j++) {
			B[0][i][j] = A[0][i][j];
			B[1][i][j] = 0;
		}
	}

	
	//ompp_2d(B, NX, NY, T, Bx, By, tb);
	//ompp_2d9w(B, NX, NY, T, Bx, By, tb);
	//two_tile_2d(A, NX, NY, T, Bx, By, tb);
	//tile_2d2s(A, NX, NY, T, Bx, By, tb); 
	allpipe_2d(A, B, NX, NY, T, Bx, By, tb);
	cross_2d1s9p(A, NX, NY, T, Bx, By, tb);
	cross_2d2s9p(A, NX, NY, T, Bx, By, tb);
	//allpipe_2d(A, B, NX, NY, T, Bx, By, tb);
	//two_steps_2d(A, B, NX, NY, T, Bx, By, tb);
	
	//our_2d1s5p(A, NX, NY, T, Bx, By, tb);
	//our_2d1s9p(A, NX, NY, T, Bx, By, tb);
	//cross_2d1s5p(A, NX, NY, T, Bx, By, tb);
	//sc_2d1sgol(A, NX, NY, T, Bx, By, tb);
	//cross_2d1sgol(A, NX, NY, T, Bx, By, tb);
	//cross_2d2s5p(A, NX, NY, T, Bx, By, tb); 
	//cross_2d2s9w(A, NX, NY, T, Bx, By, tb);
	//test();

	//cross_2d2sgol(A, NX, NY, T, Bx, By, tb);

	return 0;
}
