#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
#include "defines.h"

// redundant loads stencil
void redun_load(double **A, int N, int T)
{
	long int i;
	double **D = (double **)malloc(sizeof(double *) * 2);
	double **D_check = (double **)malloc(sizeof(double *) * 2);
	for (i = 0; i < 2; i++)
	{
		D[i] = (double *)malloc(sizeof(double) * (N + 2 * XSLOPE));
		D_check[i] = (double *)malloc(sizeof(double) * (N + 2 * XSLOPE));
	}
	for (i = 0; i < N + 2 * XSLOPE; i++)
	{
		D[0][i] = A[0][i];
		D[1][i] = 0;
		D_check[0][i] = A[0][i];
		D_check[1][i] = 0;
	}

	struct timeval start, end;
	__m512d ww, kk;
	__m512d v1_up, v1, v1_down, v2_up, v2, v2_down, v3_up, v3, v3_down, v4_up, v4, v4_down;
	__m512d v5_up, v5, v5_down, v6_up, v6, v6_down, v7_up, v7, v7_down, v8_up, v8, v8_down;
	int t, x, xmod, xup, check_flag;
	vallset(ww, GW);
	vallset(kk, LK);
	xmod = N % veclen8;
	xup = N - xmod;

	gettimeofday(&start, 0);
	for (t = 0; t < T; t++)
	{
		for (x = XSLOPE; x < xup; x += veclen8)
		{
			setload(v1_up, v2_up, v3_up, v4_up, v5_up, v6_up, v7_up, v8_up, D[t % 2][x - 1]);
			setload(v1, v2, v3, v4, v5, v6, v7, v8, D[t % 2][x]);
			setload(v1_down, v2_down, v3_down, v4_down, v5_down, v6_down, v7_down, v8_down, D[t % 2][x + 1]);

			redun_compute(v1_up, v1, v1_down, v2_up, v2, v2_down, v3_up, v3, v3_down, v4_up, v4, v4_down, ww, kk);
			redun_compute(v5_up, v5, v5_down, v6_up, v6, v6_down, v7_up, v7, v7_down, v8_up, v8, v8_down, ww, kk);

			setstore(v1, v2, v3, v4, v5, v6, v7, v8, D[(t + 1) % 2][x]);
		}

		for (x = xup + 1; x < N + XSLOPE; x++)
		{
			kernel(D);
		}
	}
	gettimeofday(&end, 0);
	printf("Reload GStencil/s = %lf\n", ((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

	for (t = 0; t < T; t++)
	{
		for (x = XSLOPE; x < N + XSLOPE; x++)
		{
			kernel(D_check);
		}
	}
	check_flag = 1;
	for (i = XSLOPE; i < N + XSLOPE; i++)
	{
		if (myabs(D[T % 2][i], D_check[T % 2][i]) > TOLERANCE)
		{
			printf("Diff[%d] = %lf, Now = %lf, Check = %lf: FAILED!\n", i, D[T % 2][i] - D_check[T % 2][i], D[T % 2][i], D_check[T % 2][i]);
			check_flag = 0;
		}
	}
	if (check_flag)
	{
		printf("CHECK CORRECT!\n");
	}
	return;
}

// register movement stencil
void regis_mov(double **A, int N, int T)
{
	long int i, j;
	double **E = (double **)malloc(sizeof(double *) * 2);
	double **E_check = (double **)malloc(sizeof(double *) * 2);
	for (i = 0; i < 2; i++)
	{
		E[i] = (double *)malloc(sizeof(double) * (N + 2 * XSLOPE));
		E_check[i] = (double *)malloc(sizeof(double) * (N + 2 * XSLOPE));
	}
	for (i = 0; i < N + 2 * XSLOPE; i++)
	{
		E[0][i] = A[0][i];
		E[1][i] = 0;
		E_check[0][i] = A[0][i];
		E_check[1][i] = 0;
	}

	struct timeval start, end;
	vec ww, kk;
	vec vbound, v1, v1_tmp1, v1_tmp2, v2, v2_tmp, v3, v4, v5, v6, v7, v8;
	int t, x, xmod, xup, check_flag;
	vallset(ww, GW);
	vallset(kk, LK);
	xmod = N % veclen8;
	xup = N - xmod;

	gettimeofday(&start, 0);
	for (t = 0; t < T; t++)
	{
		for (x = XSLOPE; x < xup; x += veclen8)
		{
			setload(v1, v2, v3, v4, v5, v6, v7, v8, E[t % 2][x]);
			vloadset(vbound, E[t % 2][x + veclen8], 0, 0, 0, 0, 0, 0, E[t % 2][x - 1]); // -1 ... 64

			if (t == 0)
			{
				trans(v1, v2, v3, v4, v5, v6, v7, v8);
			}
			v1_tmp1 = v1;
			v1_tmp2 = v1;
			v2_tmp = v2;

			// update 6 rows except the top and the bottom. results are stored in v1 to v6
			setcompute8(v1, v2, v3, v4, v5, v6, v7, v8, ww, kk);

			shuffle_tail3(v1_tmp1, vbound);	// v1_tmp: v1[1] v1[2] ... v1[7] vbound[7]
			computed(v7, v8, v1_tmp1, ww, kk); // update v8

			shuffle_head4(v8, vbound);			   // v8: vbound[0] v8[0] v8[1] ... v8[6]
			computed(v8, v1_tmp2, v2_tmp, ww, kk); // update v1

			if (t == T - 1)
			{
				trans(v8, v1, v2, v3, v4, v5, v6, v7);
			}

			setstore(v8, v1, v2, v3, v4, v5, v6, v7, E[(t + 1) % 2][x]);
		}

		for (x = xup + 1; x < N + XSLOPE; x++)
		{
			kernel(E);
		}
	}
	gettimeofday(&end, 0);
	printf("Remov GStencil/s = %lf\n", ((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

	for (t = 0; t < T; t++)
	{
		for (x = XSLOPE; x < N + XSLOPE; x++)
		{
			kernel(E_check);
		}
	}
	check_flag = 1;
	for (i = XSLOPE; i < N + XSLOPE; i++)
	{
		if (myabs(E[T % 2][i], E_check[T % 2][i]) > TOLERANCE)
		{
			printf("Diff[%d] = %lf, Now = %lf, Check = %lf: FAILED!\n", i, E[T % 2][i] - E_check[T % 2][i], E[T % 2][i], E_check[T % 2][i]);
			check_flag = 0;
		}
	}
	if (check_flag)
	{
		printf("CHECK CORRECT!\n");
	}
	return;
}
