#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
#include "defines.h"



void check(double** A, double** B, int N, int T){
	check_flag = 1;
	int i;
	for (i = XSLOPE; i < N + XSLOPE; i++) {
		if (myabs(A[T % 2][i], B[T % 2][i]) > TOLERANCE) {
			printf("Diff[%d] = %lf, Now = %lf, Check = %lf: FAILED!\n", i, B[T % 2][i] - A[T % 2][i], B[T % 2][i], A[T % 2][i]);
			check_flag = 0;
		}
	}
	if (check_flag) {
		printf("CHECK CORRECT!\n");
	}
}