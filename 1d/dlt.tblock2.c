#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include "defines.h"

#define T_BLOCK 2  // temporal block size

void dlt_tblock2(double** A, int N, int T) {
	struct timeval start, end;
	int i, j;

	int N_tmp;
	if (N % veclen != 0) {
		N_tmp = N + veclen - N % veclen;
	}
	else {
		N_tmp = N;
	}
	// memalign: the memory address will be a multiple of 32 bytes
	double(*dlt_A)[N_tmp + 2 * veclen] = (double(*)[N_tmp + 2 * veclen])malloc(sizeof(double) * (N_tmp + 2 * veclen) * 2);
	// dlt_B is for validating
	double(*dlt_B)[N + 2 * XSLOPE] = (double(*)[N + 2 * XSLOPE])malloc(sizeof(double) * (N + 2 * XSLOPE) * 2);
	for (i = 0; i < N + 2 * veclen; i++) {
		dlt_A[0][i] = 0;
		dlt_A[1][i] = 1.0 * (i - veclen + XSLOPE) - 1.0;
	}
	for (i = 0; i < N + 2 * XSLOPE; i++) {
		dlt_B[0][i] = 1.0 * i - 1.0;
		dlt_B[1][i] = 0;
	}

	int x, t;
	if (N < 20) {
		gettimeofday(&start, 0);
		for (t = 0; t < T; t++) {
			for (x = XSLOPE; x < N + XSLOPE; x++) {
				kernel(dlt_A);
			}
		}
		gettimeofday(&end, 0);
		printf("Naive/s = %f\n", ((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
		return;
	}

	// Perform Data Layout Transformation
	gettimeofday(&start, 0);
	int num_row = veclen;
	int num_per_row = N / veclen;
	int num_rest = N % veclen;
	int num_dlt = N - num_rest;
	for (i = 0; i < num_row; i++) {
		for (j = 0; j < num_per_row; j++) {
			dlt_A[0][num_row * j + i + veclen] = dlt_A[1][num_per_row * i + j + veclen];
		}
	}
	// assign the rest elements
	for (i = num_dlt; i < (N + XSLOPE); i++) {
		dlt_A[0][i + veclen] = dlt_A[1][i + veclen];
	}
	// be consistent with dlt_B[1]
	dlt_A[1][N + veclen] = 0;

	// Perform Stencil Computation using vectorization
	__m256d gw_vec = _mm256_set1_pd(GW);
	__m256d lk_vec = _mm256_set1_pd(LK);
	__m256d vec_zero = _mm256_set1_pd(0);
	__m256d vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
	__m256d vec1_1, vec2_1, vec3_1, vec4_1, vec5_1, vec6_1;
	__m256d vec3_tmp1, vec3_tmp2, bound_vec1, bound_vec3, bound_vec7;
	num_row = num_per_row;
	num_per_row = veclen;
	int rest_row = (num_row - 1) % 4;
	int rest_row_begin = num_row - 1 - rest_row;
	int rest_time = T % T_BLOCK;
	for (t = 0; t < (T - rest_time); t += T_BLOCK) {
		// least load
		vec1 = _mm256_load_pd(&dlt_A[t % 2][num_dlt]);
		vec2 = _mm256_load_pd(&dlt_A[t % 2][veclen]);

		bound_vec1 = _mm256_set1_pd(dlt_A[(t + 1) % 2][veclen - XSLOPE]);
		vec1 = blend(vec1, bound_vec1, 0b1000);
		vec1 = shift_right(vec1);
		vec3_tmp1 = vec2;

		for (i = 0; i < (num_row - 4); i += 4) {
			// load
			if (i == 0) {
				vec0 = _mm256_load_pd(&dlt_A[t % 2][num_dlt - veclen]);
				vec0 = shift_right(vec0);
			}
			vec3 = _mm256_load_pd(&dlt_A[t % 2][(i + 2) * veclen]);
			vec4 = _mm256_load_pd(&dlt_A[t % 2][(i + 3) * veclen]);
			vec5 = _mm256_load_pd(&dlt_A[t % 2][(i + 4) * veclen]);
			vec6 = _mm256_load_pd(&dlt_A[t % 2][(i + 5) * veclen]);
			if (rest_row == 0 && i == (num_row - 5)) {
				bound_vec7 = _mm256_set1_pd(dlt_A[t % 2][num_dlt + veclen]);
				vec7 = blend(vec3_tmp1, bound_vec7, 0b0001);
				vec7 = shift_left(vec7);
			}
			else {
				vec7 = _mm256_load_pd(&dlt_A[t % 2][(i + 6) * veclen]);
			}

			// 1st update. result stored in vec1_1 to vec6_1
			compute2(vec0, vec1, vec2, vec1_1, gw_vec, lk_vec);
			compute2(vec1, vec2, vec3, vec2_1, gw_vec, lk_vec);
			compute2(vec2, vec3, vec4, vec3_1, gw_vec, lk_vec);
			compute2(vec3, vec4, vec5, vec4_1, gw_vec, lk_vec);
			compute2(vec4, vec5, vec6, vec5_1, gw_vec, lk_vec);
			compute2(vec5, vec6, vec7, vec6_1, gw_vec, lk_vec);

			if (i == 0) {
				vec1_1 = blend(vec1_1, vec_zero, 0b0001);
				vec3_tmp2 = vec2_1;
			}

			// 2nd update. result stored in vec1_1 to vec4_1
			setcompute(vec1_1, vec2_1, vec3_1, vec4_1, vec5_1, vec6_1, gw_vec, lk_vec);

			_mm256_store_pd(&dlt_A[t % 2][(i + 1) * veclen], vec1_1);
			_mm256_store_pd(&dlt_A[t % 2][(i + 2) * veclen], vec2_1);
			_mm256_store_pd(&dlt_A[t % 2][(i + 3) * veclen], vec3_1);
			_mm256_store_pd(&dlt_A[t % 2][(i + 4) * veclen], vec4_1);

			vec0 = vec4;
			vec1 = vec5;
			vec2 = vec6;
		}

		int tt;
		// boundary cases
		for (tt = 0; tt < T_BLOCK; tt++) {
			// case 1: the rest rows except the last row
			if (rest_row != 0) {
				for (i = rest_row_begin; i < (num_row - 1); i++) {
					if (i == rest_row_begin) {
						if (tt == 0) {
							// vec5_1 of the previous group is vec1_1 of the current group
							_mm256_store_pd(&dlt_A[(tt + 1) % 2][i * veclen], vec5_1);
						}
						if (tt == 1) {
							// load intermediate results
							vec1 = _mm256_load_pd(&dlt_A[tt % 2][i * veclen]);
							vec2 = _mm256_load_pd(&dlt_A[tt % 2][(i + 1) * veclen]);
						}
					}
					vec3 = _mm256_load_pd(&dlt_A[tt % 2][(i + 2) * veclen]);
					compute2(vec1, vec2, vec3, vec2_1, gw_vec, lk_vec);
					_mm256_store_pd(&dlt_A[(tt + 1) % 2][(i + 1) * veclen], vec2_1);
					vec1 = vec2;
					vec2 = vec3;
				}
			}

			// case 2: the last row (i == num_row - 1)
			if (tt == 0) {
				// the first 4 elements that have not been updated
				vec3 = vec3_tmp1;
			}
			if (tt == 1) {
				// the first 4 elements that have been updated once
				vec3 = vec3_tmp2;
			}
			bound_vec3 = _mm256_set1_pd(dlt_A[tt % 2][num_dlt + veclen]);
			vec3 = blend(vec3, bound_vec3, 0b0001);
			vec3 = shift_left(vec3);
			if (num_rest != 0) {
				// update dlt_A[num_dlt + veclen] for next iteration
				x = num_dlt + veclen;
				kernel(dlt_A);
			}
			compute2(vec1, vec2, vec3, vec2_1, gw_vec, lk_vec);
			_mm256_store_pd(&dlt_A[(tt + 1) % 2][(i + 1) * veclen], vec2_1);
			if (rest_row == 0) {
				vec1 = vec5_1;
				vec2 = vec2_1;
			}

			// case 3: the rest elements
			for (x = (num_dlt + veclen); x < (N + veclen); x++) {
				dlt_A[(tt + 1) % 2][x] = LK * (dlt_A[tt % 2][x + 1] + GW * dlt_A[tt % 2][x] + dlt_A[tt % 2][x - 1]);
			}
		}
	}

	// rest time
	for (t = (T - rest_time); t < T; t ++) {
		vec1 = _mm256_load_pd(&dlt_A[t % 2][num_dlt]);
		vec2 = _mm256_load_pd(&dlt_A[t % 2][veclen]);
		bound_vec1 = _mm256_set1_pd(dlt_A[(t + 1) % 2][veclen - XSLOPE]);
		vec1 = blend(vec1, bound_vec1, 0b1000);
		vec1 = shift_right(vec1);
		
		for (i = 0; i < (num_row - 1); i++) {
			vec3 = _mm256_load_pd(&dlt_A[t % 2][(i + 2) * veclen]);
			compute2(vec1, vec2, vec3, vec2_1, gw_vec, lk_vec);
			_mm256_store_pd(&dlt_A[(t + 1) % 2][(i + 1) * veclen], vec2_1);
			vec1 = vec2;
			vec2 = vec3;
		}

		vec3 = _mm256_load_pd(&dlt_A[t % 2][veclen]);
		bound_vec3 = _mm256_set1_pd(dlt_A[t % 2][num_dlt + veclen]);
		vec3 = blend(vec3, bound_vec3, 0b0001);
		vec3 = shift_left(vec3);
		if (num_rest != 0) {
			x = num_dlt + veclen;
			kernel(dlt_A);
		}
		compute2(vec1, vec2, vec3, vec2_1, gw_vec, lk_vec);
		_mm256_store_pd(&dlt_A[(t + 1) % 2][(i + 1) * veclen], vec2_1);

		for (x = (num_dlt + veclen + 1); x < (N + veclen); x++) {
			kernel(dlt_A);
		}
	}

	// Transform back for validating
	for (i = 0; i < num_row; i++) {
		for (j = 0; j < num_per_row; j++) {
			dlt_A[(T + 1) % 2][num_row * j + i + veclen] = dlt_A[T % 2][num_per_row * i + j + veclen];
		}
	}
	// assign the rest elements
	for (i = num_dlt; i < N; i++) {
		dlt_A[(T + 1) % 2][i + veclen] = dlt_A[T % 2][i + veclen];
	}
	gettimeofday(&end, 0);
	printf("TBLOCK_2 DLT/s = %f\n", ((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

	// check correctness
	for (t = 0; t < T; t++) {
		for (x = XSLOPE; x < N + XSLOPE; x++) {
			kernel(dlt_B);
		}
	}
	int my_check_flag = 1;
	for (i = XSLOPE; i < N + XSLOPE; i++) {
		if (myabs(dlt_A[(T + 1) % 2][i + veclen - XSLOPE], dlt_B[T % 2][i]) > TOLERANCE) {
			printf("Naive[%d] = %f, Check = %f: FAILED!\n", i, dlt_A[(T + 1) % 2][i + veclen - XSLOPE], dlt_B[T % 2][i]);
			my_check_flag = 0;
		}
	}
	if (my_check_flag) {
		printf("CHECK CORRECT!\n");
	}
	return;
}
