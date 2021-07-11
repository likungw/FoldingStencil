#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
#include "defines.h"

// redundant loads stencil
void redun_load(double** A, int N, int T) {
	double** D = (double**)malloc(sizeof(double*) * 2);
	long int i;
	for (i = 0; i < 2; i++) {
		D[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
	}
	for (i = 0; i < N+2*XSLOPE; i++) {
		D[0][i] = A[0][i];
		D[1][i] = 0;
	}
	struct timeval start, end;
	vec ww, kk;
	int t, x, xmod, xup, check_flag;
	
	vallset(ww, GW);
	vallset(kk, LK);
	xmod = N % veclen4;
	xup = N - xmod;
	// compute
	gettimeofday(&start, 0);
	for (t = 0; t < T; t++) {
		vec v1_up, v1, v1_down, v2_up, v2, v2_down, v3_up, v3, v3_down, v4_up, v4, v4_down;
		for (x = XSLOPE; x < xup; x += veclen4) {
			setload(v1_up, v2_up, v3_up, v4_up, D[t % 2][x] - 1);
			setload(v1, v2, v3, v4, D[t % 2][x]);
			setload(v1_down, v2_down, v3_down, v4_down, D[t % 2][x] + 1);
			redun_compute(v1_up, v1, v1_down, v2_up, v2, v2_down, v3_up, v3, v3_down, v4_up, v4, v4_down, ww, kk);
			setstore(v1, v2, v3, v4, D[(t + 1) % 2][x]);
		}
		for (x = xup + 1; x < N + XSLOPE; x++) {
			kernel(D);
		}
	}
	gettimeofday(&end, 0);
	printf("Reload GStencil/s = %lf\n", ((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
	return;
}

// register movement stencil
void regis_mov(double** A, int N, int T) {
	long int i;
	double** E = (double**)malloc(sizeof(double*) * 2);
	for (i = 0; i < 2; i++) {
		E[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
	}
	for (i = 0; i < N+2*XSLOPE; i++) {
		E[0][i] = A[0][i];
		E[1][i] = 0;
	}
	struct timeval start, end;
	vec ww, kk;
	int t, x, xmod, xup, check_flag;
	vallset(ww, GW);
	vallset(kk, LK);
	xmod = N % veclen4;
	xup = N - xmod;

	// compute
	gettimeofday(&start, 0);
	for (t = 0; t < T; t++) {
		vec v1, v1_tmp, v2, v2_tmp, v3, v3_tmp, v4, v4_tmp, vbound;
		for (x = XSLOPE; x < xup; x += veclen4) {
			setload(v1, v2, v3, v4, E[t % 2][x]);
			vloadset(vbound, E[t % 2][x + veclen4], 0, 0, E[t % 2][x - 1]);
			if (t == 0) {
				trans(v1, v2, v3, v4);
			}
			v1_tmp = v1;
			v2_tmp = v2;
			v3_tmp = v3;
			v4_tmp = v4;
			regis_compute(v1, v2, v3, ww, kk);  // update v2
			regis_compute(v2_tmp, v3, v4, ww, kk);  // update v3

			v1 = shift_left(v1);
			v1 = blend(v1, vbound, 0b1000);  // after shift and blend, v1: 4 8 12 16
			regis_compute(v3_tmp, v4, v1, ww, kk);  // update v4

			v4_tmp = shift_right(v4_tmp);
			v4_tmp = blend(v4_tmp, vbound, 0b0001);  // v4: -1 3 7 11
			v1 = shift_right(v1);
			v1 = blend(v1, v1_tmp, 0b0001);  // v1: 0 4 8 12
			regis_compute(v4_tmp, v1, v2_tmp, ww, kk);  // update v1

			if (t == T - 1) {
				trans(v1, v2, v3, v4);
			}
			setstore(v1, v2, v3, v4, E[(t + 1) % 2][x]);
		}
		for (x = xup + 1; x < N + XSLOPE; x++) {
			kernel(E);
		}
	}
	gettimeofday(&end, 0);
	printf("Remov GStencil/s = %lf\n", ((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
	return;
}
