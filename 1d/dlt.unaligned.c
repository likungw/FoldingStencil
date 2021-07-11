#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include "defines.h"

// unaligned dlt
void dlt_unaligned(double** A, int N, int T) {
	struct timeval start, end;
	int i, j;

	double** dlt_A = (double**)malloc(sizeof(double*) * 2);
	// dlt_B is for validating
	double** dlt_B = (double**)malloc(sizeof(double*) * 2);
	for (i = 0; i < 2; i++) {
		dlt_A[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
		dlt_B[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
	}
	for (i = 0; i < N + 2 * XSLOPE; i++) {
		dlt_A[0][i] = 0;
		dlt_A[1][i] = A[0][i];
		dlt_B[0][i] = A[0][i];
		dlt_B[1][i] = 0;
	}

	// Perform Data Layout Transformation
	gettimeofday(&start, 0);
	int num_row = veclen;
	int num_per_row = N / veclen;
	int num_rest = N % veclen;
	int num_dlt = N - num_rest;
	for (i = 0; i < num_row; i++) {
		for (j = 0; j < num_per_row; j++) {
			dlt_A[0][num_row * j + i + XSLOPE] = dlt_A[1][num_per_row * i + j + XSLOPE];
		}
	}
	// assign the rest elements
	for (i = num_dlt; i < (N + XSLOPE); i++) {
		dlt_A[0][i + XSLOPE] = dlt_A[1][i + XSLOPE];
	}
	// be consistent with dlt_B[1]
	dlt_A[1][N + XSLOPE] = 0;

	// Perform Stencil Computation using vectorization
	__m256d gw_vec = _mm256_set1_pd(GW);
	__m256d lk_vec = _mm256_set1_pd(LK);
	__m256d result_vec, vec1, vec2, vec3, vec4, vec5, vec6, bound_vec1, bound_vec3;
	num_row = num_per_row;
	num_per_row = veclen;
	int x, t;
	for (t = 0; t < T; t++) {
		// least load
		vec1 = _mm256_loadu_pd(&dlt_A[t % 2][num_dlt - veclen + XSLOPE]);
		bound_vec1 = _mm256_set1_pd(dlt_A[(t + 1) % 2][0]);
		vec2 = _mm256_loadu_pd(&dlt_A[t % 2][XSLOPE]);
		// permute for boundary computation
		vec1 = blend(vec1, bound_vec1, 0b1000);
		vec1 = _mm256_permute4x64_pd(vec1, 0b10010011);  // 0 1 2 3 --> 3 0 1 2

		for (i = 0; i < (num_row - 1); i++) {
			vec3 = _mm256_loadu_pd(&dlt_A[t % 2][(i + 1) * veclen + XSLOPE]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec2, vec1);
			result_vec = _mm256_add_pd(result_vec, vec3);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_storeu_pd(&dlt_A[(t + 1) % 2][i * veclen + XSLOPE], result_vec);

			vec1 = vec2;
			vec2 = vec3;
		}

		// boundary case: i == num_row - 1
		vec3 = _mm256_loadu_pd(&dlt_A[t % 2][XSLOPE]);
		bound_vec3 = _mm256_set1_pd(dlt_A[t % 2][num_dlt + XSLOPE]);
		if (num_rest != 0) {
			// update dlt_A[num_dlt + XSLOPE] for next iteration
			x = num_dlt + XSLOPE;
			kernel(dlt_A);
		}
		vec3 = blend(vec3, bound_vec3, 0b0001);
		vec3 = _mm256_permute4x64_pd(vec3, 0b00111001);  // 0 1 2 3 --> 1 2 3 0
		// compute
		result_vec = _mm256_fmadd_pd(gw_vec, vec2, vec1);
		result_vec = _mm256_add_pd(result_vec, vec3);
		result_vec = _mm256_mul_pd(result_vec, lk_vec);
		_mm256_storeu_pd(&dlt_A[(t + 1) % 2][i * veclen + XSLOPE], result_vec);

		// compute the rest elements
		for (x = (num_dlt + XSLOPE + 1); x < (N + XSLOPE); x++) {
			kernel(dlt_A);
		}
	}

	// Transform back for validating
	for (i = 0; i < num_row; i++) {
		for (j = 0; j < num_per_row; j++) {
			dlt_A[(T + 1) % 2][num_row * j + i + XSLOPE] = dlt_A[T % 2][num_per_row * i + j + XSLOPE];
		}
	}
	// assign the rest elements
	for (i = num_dlt; i < N; i++) {
		dlt_A[(T + 1) % 2][i + XSLOPE] = dlt_A[T % 2][i + XSLOPE];
	}
	gettimeofday(&end, 0);
	printf("unaligned DLT/s = %f\n", ((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

	// check correctness
	for (t = 0; t < T; t++) {
		for (x = XSLOPE; x < N + XSLOPE; x++) {
			kernel(dlt_B);
		}
	}
	int my_check_flag = 1;
	for (i = XSLOPE; i < N + XSLOPE; i++) {
		if (myabs(dlt_A[(T + 1) % 2][i], dlt_B[T % 2][i]) > TOLERANCE) {
			printf("Naive[%d] = %f, Check = %f: FAILED!\n", i, dlt_A[(T + 1) % 2][i], dlt_B[T % 2][i]);
			my_check_flag = 0;
		}
	}
	if (my_check_flag) {
		printf("CHECK CORRECT!\n");
	}
	return;
}
