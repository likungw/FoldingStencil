/*
 * @Author: Yue Yue
 * @Date: 2020-02-24 18:11:19
 * @LastEditTime: 2020-03-03 11:32:54
 * @Description: 2d5p allpipe stencil version 2
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h> 
#include "2d.defines.h"

void allpipe_2d(double*** A, double*** B, int NX, int NY, int T, int Bx, int By, int tb) {
	// A is the initial array, B is the benchmark array
	// Bx, By, tb are not needed for now
	long int i, j;
	double*** C = (double***)malloc(sizeof(double**) * 2);
	for (i = 0; i < 2; i++) {
		C[i] = (double**)malloc(sizeof(double*) * (NX + 2 * XSLOPE));
	}
	for (i = 0; i < 2; i++) {
		for (j = 0; j < (NX + 2 * XSLOPE); j++) {
			C[i][j] = (double*)malloc(sizeof(double) * (NY + 2 * YSLOPE));
		}
	}
	for (i = 0; i < (NX + 2 * XSLOPE); i++) {
		for (j = 0; j < (NY + 2 * YSLOPE); j++) {
			C[0][i][j] = A[0][i][j];
			C[1][i][j] = 0;
		}
	}

	int x, y, t;
	vec ww, kk, vbound, gw;
	vec v11, v12, v13, v14, v20, v21, v22, v23, v24, v25, v31, v32, v33, v34;
	vallset(ww, GW);
	vallset(kk, LK);
	int ymod = NY % veclen4;
	int yup = NY - ymod;
	struct timeval start, end;
	gettimeofday(&start, 0);
	for (t = 0; t < T; t++) {
		for (x = XSLOPE; x < NX + XSLOPE; x++) {
			for (y = YSLOPE; y <= yup; y += veclen4) {
				// load
				setload(v11, v12, v13, v14, C[t % 2][x - 1][y]);
				vallset(v20, C[t % 2][x][y - 1]);
				setload5(v21, v22, v23, v24, gw, C[t % 2][x][y]);
				setload(v31, v32, v33, v34, C[t % 2][x + 1][y]);

				// transpose
				if (t == 0) {
					trans(v11, v12, v13, v14);
					trans(v21, v22, v23, v24);
					trans(v31, v32, v33, v34);
				}

				// permute
				shuffle_headd(v24, v20);
				shuffle_tail(v21, gw, v25);

				// compute. result stored in v20, v21, v22, v23
				compute_5p(v11, v20, v21, v22, v31, ww, kk);
				compute_5p(v12, v21, v22, v23, v32, ww, kk);
				compute_5p(v13, v22, v23, v24, v33, ww, kk);
				compute_5p(v14, v23, v24, v25, v34, ww, kk);

				// transpose back
				if (t == (T - 1)) {
					trans(v20, v21, v22, v23);
				}

				// store
				setstore(v20, v21, v22, v23, C[(t + 1) % 2][x][y]);

				v20 = v24;
				v21 = gw;
			}

			for (y = yup + 1; y < NY + YSLOPE; y++) {
				kernel(C);
			}
		}
	}

	gettimeofday(&end, 0);
	printf("ALLPIPE_2D GStencil/s = %f\n", ((double)NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
 
	// free
	for (i = 0; i < 2; i++) {
		for (j = 0; j < (NX + 2 * XSLOPE); j++) {
			free(C[i][j]);
		}
	}
	for (i = 0; i < 2; i++) {
		free(C[i]);
	}
	free(C);
}
