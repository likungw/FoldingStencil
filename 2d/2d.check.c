#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include "2d.defines.h"

void check_2d(double*** A, double*** B, int NX, int NY, int T) {
	check_flag = 1;
	for (int i = XSLOPE; i < NX + XSLOPE; i++) {
		for (int j = YSLOPE; j < NY + YSLOPE; j++) {
			if (myabs(A[T % 2][i][j], B[T % 2][i][j]) > TOLERANCE) {
				printf("Diff[%d][%d] = %lf, Now = %lf, Check = %lf: FAILED!\n", i, j, B[T % 2][i][j] - A[T % 2][i][j], B[T % 2][i][j], A[T % 2][i][j]);
				check_flag = 0;
			}
		}
	}
	if (check_flag) {
		printf("CHECK CORRECT!\n");
	}
}
