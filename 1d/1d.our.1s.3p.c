#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <malloc.h>
#include <immintrin.h>
#include "defines.h"

void our_1d1s3p(double **A, double **R, int N, int T, int Bx, int tb)
{
	// A is for initialization, R is for check
	long int i;
	double **B = (double **)malloc(sizeof(double *) * 2);
	for (i = 0; i < 2; i++)
	{
		B[i] = (double *)malloc(sizeof(double) * (N + 2 * XSLOPE));
	}
	for (i = 0; i < N + 2 * XSLOPE; i++)
	{
		B[0][i] = A[0][i];
		B[1][i] = 0;
	}

	int bx = Bx - 2 * tb * XSLOPE;
	int ix = Bx + bx;
	int nb0[2] = {myfloor(N - Bx, ix), myfloor(N - Bx, ix) + 1};

	int nrestpoints = N % ix;
	int bx_first_B1 = (Bx + nrestpoints) / 2;
	int bx_last_B1 = (Bx + nrestpoints) - bx_first_B1;
	int xright[2] = {bx_first_B1 + Bx + XSLOPE, bx_first_B1 + (Bx - bx) / 2 + XSLOPE};

	int level = 0; // two types of blocks: B0, B1
	int x, xx, t, tt;
	register int xmin, xmax;
	int xmod, xup;

	vec ww, kk;
	
	vallset(ww, GW);
	vallset(kk, LK);

	struct timeval start, end;
	gettimeofday(&start, 0);
	for (tt = -tb; tt < T; tt += tb)
	{
		#pragma omp parallel for private(xmin, xmax, xup,xmod,t, x,xx,ww, kk) 
		for (xx = 0; xx < nb0[level]; xx++)
		{
			vec v1, v2, v3, v4, v5;
			vec post1, post2;
			for (t = max(tt, 0); t < min(tt + 2 * tb, T); t++)
			{
				xmin = (level == 1 && xx == 0) ? XSLOPE : (xright[level] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
				xmax = (level == 1 && xx == nb0[1] - 1) ? N + XSLOPE : (xright[level] + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
				xmod = (xmax - xmin) % veclen4;
				xup = xmax - xmod;

				for (x = xmin; x < xup; x += veclen4)
				{
					if (x == xmin)
					{
						setloadw(v1, v2, v3, v4, v5, B[t % 2][x - 1]);
					}
					else
					{
						setload(v2, v3, v4, v5, B[t % 2][x - 1 + veclen]);
					}

					// TODO: do not transpose every time
					if(t==0)
						trans(v1, v2, v3, v4);

					shuffle_tail(v1, v5,post1);
					shuffle_tail(v2, v5,post2);

					// results are stored in v1 v2 v3 v4
					setcompute2(v1, v2, v3, v4, post1, post2, ww, kk);

					if(t==T-1)
					trans(v1, v2, v3, v4);
					setstore(v1, v2, v3, v4, B[(t + 1) % 2][x]);

					v1 = v5;
				}

				// deal with the rest elements
				#pragma ivdep
				#pragma vector always
				for (x = xup; x < xmax; x++)
				{
					kernel(B);
				}
			}
		}
		level = 1 - level;
	}

	gettimeofday(&end, 0);
	printf("our_1d1s3p GStencil/s = %f\n", ((double)N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

	//check(R, B, N, T);
}
