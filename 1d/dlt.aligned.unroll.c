#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include "defines.h"


// aligned and unrolling dlt
void dlt_aligned_unroll(double** A, int N, int T) {
	struct timeval start, end;
	struct timeval start_1, end_1, start_2, end_2;
	int i, j;

	int N_tmp;
	if (N % veclen != 0) {
		N_tmp = N + veclen - N % veclen;
	}
	else {
		N_tmp = N;
	}
	// memalign: the memory address will be a multiple of 32 bytes
	double(*dlt_A)[N_tmp + 2 * veclen] = (double(*)[N_tmp + 2 * veclen])malloc( sizeof(double) * (N_tmp + 2 * veclen) * 2);
	// dlt_B is for validating
	double(*dlt_B)[N + 2 * XSLOPE] = (double(*)[N + 2 * XSLOPE])malloc(sizeof(double) * (N + 2 * XSLOPE) * 2);
	for (i = 0; i < N + 2 * veclen; i++) {
		dlt_A[0][i] = 0;
		dlt_A[1][i] = A[0][i - veclen + XSLOPE];
	}
	for (i = 0; i < N + 2 * XSLOPE; i++) {
		dlt_B[0][i] = A[0][i];
		dlt_B[1][i] = 0;
	}

	// Perform Data Layout Transformation
	gettimeofday(&start, 0);
	gettimeofday(&start_1, 0);
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
	gettimeofday(&end_1, 0);

	// Perform Stencil Computation using vectorization
	__m256d gw_vec = _mm256_set1_pd(GW);
	__m256d lk_vec = _mm256_set1_pd(LK);
	__m256d result_vec, vec1, vec2, vec3, vec4, vec5, vec6, bound_vec1, bound_vec3;
	num_row = num_per_row;
	num_per_row = veclen;
	int rest_row = (num_row - 1) % 4;
	int rest_row_begin = num_row - 1 - rest_row;
	int x, t;
	for (t = 0; t < T; t++) {
		// least load
		vec1 = _mm256_load_pd(&dlt_A[t % 2][num_dlt]);
		bound_vec1 = _mm256_set1_pd(dlt_A[(t + 1) % 2][veclen - XSLOPE]);
		vec2 = _mm256_load_pd(&dlt_A[t % 2][veclen]);
		// permute for boundary computation
		vec1 = blend(vec1, bound_vec1, 0b1000);
		vec1 = _mm256_permute4x64_pd(vec1, 0b10010011);  // 0 1 2 3 --> 3 0 1 2
		/* method 2:
		vec1 = _mm256_set_pd(dlt_A[t % 2][num_dlt + 2], dlt_A[t % 2][num_dlt + 1], dlt_A[t % 2][num_dlt], dlt_A[(t + 1) % 2][veclen - XSLOPE]);*/

		for (i = 0; i < (num_row - 4); i += 4) {
			// loop unrolling
			vec3 = _mm256_load_pd(&dlt_A[t % 2][(i + 2) * veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec2, vec1);
			result_vec = _mm256_add_pd(result_vec, vec3);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_store_pd(&dlt_A[(t + 1) % 2][(i + 1) * veclen], result_vec);

			vec4 = _mm256_load_pd(&dlt_A[t % 2][(i + 3) * veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec3, vec2);
			result_vec = _mm256_add_pd(result_vec, vec4);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_store_pd(&dlt_A[(t + 1) % 2][(i + 2) * veclen], result_vec);

			vec5 = _mm256_load_pd(&dlt_A[t % 2][(i + 4) * veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec4, vec3);
			result_vec = _mm256_add_pd(result_vec, vec5);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_store_pd(&dlt_A[(t + 1) % 2][(i + 3) * veclen], result_vec);

			vec6 = _mm256_load_pd(&dlt_A[t % 2][(i + 5) * veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec5, vec4);
			result_vec = _mm256_add_pd(result_vec, vec6);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_store_pd(&dlt_A[(t + 1) % 2][(i + 4) * veclen], result_vec);

			vec1 = vec5;
			vec2 = vec6;
		}

		// deal with the rest rows
		for (i = rest_row_begin; i < (num_row - 1); i++) {
			vec3 = _mm256_load_pd(&dlt_A[t % 2][(i + 2) * veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec2, vec1);
			result_vec = _mm256_add_pd(result_vec, vec3);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_store_pd(&dlt_A[(t + 1) % 2][(i + 1) * veclen], result_vec);
			vec1 = vec2;
			vec2 = vec3;
		}

		// boundary case: i == num_row - 1
		vec3 = _mm256_load_pd(&dlt_A[t % 2][veclen]);
		bound_vec3 = _mm256_set1_pd(dlt_A[t % 2][num_dlt + veclen]);
		vec3 = blend(vec3, bound_vec3, 0b0001);
		vec3 = _mm256_permute4x64_pd(vec3, 0b00111001);  // 0 1 2 3 --> 1 2 3 0
		/* method 2:
		vec3 = _mm256_set_pd(dlt_A[t % 2][num_dlt + veclen], dlt_A[t % 2][veclen + 3], dlt_A[t % 2][veclen + 2], dlt_A[t % 2][veclen + 1]);*/
		if (num_rest != 0) {
			// update dlt_A[num_dlt + veclen] for next iteration
			x = num_dlt + veclen;
			kernel(dlt_A);
		}
		// compute
		result_vec = _mm256_fmadd_pd(gw_vec, vec2, vec1);
		result_vec = _mm256_add_pd(result_vec, vec3);
		result_vec = _mm256_mul_pd(result_vec, lk_vec);
		_mm256_store_pd(&dlt_A[(t + 1) % 2][(i + 1) * veclen], result_vec);

		// compute the rest elements
		for (x = (num_dlt + veclen + 1); x < (N + veclen); x++) {
			kernel(dlt_A);
		}
	}
	gettimeofday(&start_2, 0);

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
	gettimeofday(&end_2, 0);
	gettimeofday(&end, 0);
	printf("aligned & unrolling DLT/s = %f\n", ((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);
	printf("aligned & unrolling DLT/s = %f\n", ((double)N * T) / (double)(start_2.tv_sec - end_1.tv_sec + (start_2.tv_usec - end_1.tv_usec) * 1.0e-6) / 1000000000L);
	printf("Ratio: %g\n", (((double)(end_2.tv_sec - start_2.tv_sec + (end_2.tv_usec - start_2.tv_usec) * 1.0e-6) / 1000000000L)+((double)(end_1.tv_sec - start_1.tv_sec + (end_1.tv_usec - start_1.tv_usec) * 1.0e-6) / 1000000000L))/((double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L));
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
