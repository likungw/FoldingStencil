#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
#include "defines.h"

#ifdef DLT
	//allocate data aligned on a 32-byte boundary for better vectorization performance
	//double (*dlt_A)[N+2*veclen] = (double (*)[N+2*veclen])memalign(32, sizeof(double)*(N+2*veclen)*2);

	//allocate for validating
	//double (*dlt_B)[N+2*XSLOPE] = (double (*)[N+2*XSLOPE])malloc(sizeof(double)*(N+2*XSLOPE)*2);
#endif

void dlt(double** A, double** B, int N, int T){
	struct timeval start, end;
	double** dlt_A = (double**)malloc(sizeof(double*) * 2);
	double** dlt_B = (double**)malloc(sizeof(double*) * 2);
	long int i;
	for (i = 0; i < 2; i++) {
		dlt_A[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
		dlt_B[i] = (double*)malloc(sizeof(double) * (N + 2 * XSLOPE));
	}
	for (i = 0; i < N+2*XSLOPE; i++) {
		B[0][i] = A[0][i];
		B[1][i] = 0;
	}
	
	dlt_A[1][0] = 0;
	dlt_A[0][0] = -1.0;
	int count = 1;
	for (i = veclen; i < N + veclen; i++) {
		dlt_A[1][i] = 1.0 * count -1.0;
		dlt_A[0][i] = 0;
		count++;
	}
	dlt_A[1][N+2*veclen-1] = 0;
	dlt_A[0][N+2*veclen-1] = 1.0 * count -1.0;


	for (i = 0; i < N+2*XSLOPE; i++) {
		dlt_B[0][i] = 1.0 * i-1.0;
		dlt_B[1][i] = 0;
	}

	//Perform Data Layout Transform
	gettimeofday(&start, 0);	
	int num_row = veclen; 
	int num_per_row = N / veclen;
	for (int i = 0; i < num_row; i++) {
		for (int j = 0; j < num_per_row; j++) {
			dlt_A[0][num_row * j + i + veclen]  = dlt_A[1][num_per_row * i + j + veclen];
		}
	}
	//boundary initialization
	dlt_A[0][1] = dlt_A[0][N];
	dlt_A[0][2] = dlt_A[0][N + 1];
	dlt_A[0][3] = dlt_A[0][N + 2];
	dlt_A[0][N + veclen] = dlt_A[0][veclen + 1];
	dlt_A[0][N + veclen + 1] = dlt_A[0][veclen + 2];
	dlt_A[0][N + veclen + 2] = dlt_A[0][veclen + 3];

	//Perform Stencil Computation using vectorization
	__m256d gw_vec = _mm256_set1_pd(GW);
	__m256d lk_vec = _mm256_set1_pd(LK);
	__m256d result_vec, vec1, vec2, vec3, vec4, vec5, vec6;
	num_row = N / veclen;
	num_per_row = veclen;
	int x, t;
	for (t = 0; t < T; t++) {
		/*

		for (i = 0; i < num_row; i+=4) {
			vec1 = _mm256_loadu_pd(&dlt_A[t % 2][ i*veclen ]);
			vec2 = _mm256_loadu_pd(&dlt_A[t % 2][(i+1)*veclen]);
			vec3 = _mm256_loadu_pd(&dlt_A[t % 2][(i+2)*veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec2,vec1);
			result_vec = _mm256_add_pd(result_vec, vec3);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_storeu_pd(&dlt_A[(t+1) % 2][(i+1)*veclen], result_vec);
			vec4 = _mm256_loadu_pd(&dlt_A[t % 2][(i+3)*veclen ]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec3,vec2);
			result_vec = _mm256_add_pd(result_vec, vec4);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_storeu_pd(&dlt_A[(t+1) % 2][(i+2)*veclen], result_vec);			
			vec5 = _mm256_loadu_pd(&dlt_A[t % 2][(i+4)*veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec4,vec3);
			result_vec = _mm256_add_pd(result_vec, vec5);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_storeu_pd(&dlt_A[(t+1) % 2][(i+3)*veclen], result_vec);
			vec6 = _mm256_loadu_pd(&dlt_A[t % 2][(i+5)*veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec5,vec4);
			result_vec = _mm256_add_pd(result_vec, vec6);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_storeu_pd(&dlt_A[(t+1) % 2][(i+4)*veclen], result_vec);
		
		}*/
		/*
		for (i = 0; i < num_row; i++) {
			vec1 = _mm256_load_pd(&dlt_A[t % 2][ i*veclen ]);
			vec2 = _mm256_load_pd(&dlt_A[t % 2][(i+1)*veclen]);
			vec3 = _mm256_load_pd(&dlt_A[t % 2][(i+2)*veclen]);

			result_vec = _mm256_mul_pd(gw_vec, vec2);
			result_vec = _mm256_add_pd(result_vec, vec1);
			result_vec = _mm256_add_pd(result_vec, vec3);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);

			_mm256_store_pd(&dlt_A[(t+1) % 2][(i+1)*veclen], result_vec);
		}*/
		i = 0;
		vec1 = _mm256_loadu_pd(&dlt_A[t % 2][ i*veclen ]);
		vec2 = _mm256_loadu_pd(&dlt_A[t % 2][(i+1)*veclen]);
		for (i = 0; i < num_row; i+=4) {
			//vec1 = _mm256_loadu_pd(&dlt_A[t % 2][ i*veclen ]);
			//vec2 = _mm256_loadu_pd(&dlt_A[t % 2][(i+1)*veclen]);
			vec3 = _mm256_loadu_pd(&dlt_A[t % 2][(i+2)*veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec2,vec1);
			result_vec = _mm256_add_pd(result_vec, vec3);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_storeu_pd(&dlt_A[(t+1) % 2][(i+1)*veclen], result_vec);
			vec4 = _mm256_loadu_pd(&dlt_A[t % 2][(i+3)*veclen ]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec3,vec2);
			result_vec = _mm256_add_pd(result_vec, vec4);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_storeu_pd(&dlt_A[(t+1) % 2][(i+2)*veclen], result_vec);			
			vec5 = _mm256_loadu_pd(&dlt_A[t % 2][(i+4)*veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec4,vec3);
			result_vec = _mm256_add_pd(result_vec, vec5);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_storeu_pd(&dlt_A[(t+1) % 2][(i+3)*veclen], result_vec);
			vec6 = _mm256_loadu_pd(&dlt_A[t % 2][(i+5)*veclen]);
			result_vec = _mm256_fmadd_pd(gw_vec, vec5,vec4);
			result_vec = _mm256_add_pd(result_vec, vec6);
			result_vec = _mm256_mul_pd(result_vec, lk_vec);
			_mm256_storeu_pd(&dlt_A[(t+1) % 2][(i+4)*veclen], result_vec);
			vec1 = vec5;
			vec2 = vec6;
		
		}

		//boundary initialization for next round
		dlt_A[(t+1)%2][1] = dlt_A[(t+1)%2][N];
		dlt_A[(t+1)%2][2] = dlt_A[(t+1)%2][N + 1];
		dlt_A[(t+1)%2][3] = dlt_A[(t+1)%2][N + 2];
		dlt_A[(t+1)%2][N + veclen] = dlt_A[(t+1)%2][veclen + 1];
		dlt_A[(t+1)%2][N + veclen + 1] = dlt_A[(t+1)%2][veclen + 2];
		dlt_A[(t+1)%2][N + veclen + 2] = dlt_A[(t+1)%2][veclen + 3];
	}
	
	//Transform back for validating
	for (int i = 0; i < num_row; i++) {
		for (int j = 0; j < num_per_row; j++) {
			dlt_A[(T+1)%2][num_row * j + i + veclen]  = dlt_A[T%2][num_per_row * i + j + veclen];
		}
	}
	gettimeofday(&end, 0);
	printf("DLT/s = %f\n",((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

/*
	//check correctness
	for (t = 0; t < T; t++) {
		for (x = XSLOPE; x < N + XSLOPE; x++) {
			kernel(dlt_B);
		}
	}
	int my_check_flag =1;
	for (i = XSLOPE; i < N + XSLOPE; i++) {
		if(myabs(dlt_A[(T+1)%2][i + veclen - XSLOPE], dlt_B[T%2][i]) > TOLERANCE){
			printf("Naive[%ld] = %f, Check = %f: FAILED!\n", i, dlt_B[T%2][i], dlt_A[(T+1)%2][i+veclen-XSLOPE]);
			my_check_flag = 0; }
	}
	if(my_check_flag){
		printf("CHECK CORRECT!\n");
	}*/
}