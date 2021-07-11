
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))
#define myabs(x,y)  ((x) > (y)? ((x)-(y)) : ((y)-(x))) 
#define myceil(x,y)  (int)ceil(((double)x)/((double)y)) // if x and y are integers, myceil(x,y) = (x-1)/y + 1
#define myfloor(x,y)  (int)floor(((double)x)/((double)y)) // if x and y are integers, myceil(x,y) = (x-1)/y + 1

typedef __m256d vec;
#define len 2
#define veclen 4
#define veclen2 8
#define veclen3 12
#define veclen4 16
#define veclen5 20
#define veclen6 24

//#define GW 2.1
//#define LK 0.244
#define GW 1
#define LK 1
#define TOLERANCE 0.0001
int check_flag; 

#define DI(v) ((double*)(&(v)))
#define D0(v) (DI(v))[0]
#define D1(v) (DI(v))[1]
#define D2(v) (DI(v))[2]
#define D3(v) (DI(v))[3]

#define vload(a,b) {a=_mm256_loadu_pd((&b));}
#define vload1(a,b) {a=_mm256_loadu_pd((&b)+veclen);}
#define vload2(a,b) {a=_mm256_loadu_pd((&b)+veclen2);}
#define vload3(a,b) {a=_mm256_loadu_pd((&b)+veclen3);}
#define vload4(a,b) {a=_mm256_loadu_pd((&b)+veclen4);}
#define vload5(a,b) {a=_mm256_loadu_pd((&b)+veclen5);}
#define vload6(a,b) {a=_mm256_loadu_pd((&b)+veclen6);}
#define vloadset(a,b,c,d,e) {a=_mm256_set_pd(b,c,d,e);}
#define vallset(a,b) {a = _mm256_set1_pd(b);}
#define setloadw(a,b,c,d,e,f) {vload(a,f);vload1(b,f);vload2(c,f);vload3(d,f);vload4(e,f);}
#define setload(a,b,c,d,e) {vload(a,e);vload1(b,e);vload2(c,e);vload3(d,e);}
#define setload6(a,b,c,d,e,f,g) {vload(a,g);vload1(b,g);vload2(c,g);vload3(d,g);vload4(e,g);vload5(f,g);}
#define setload7(a,b,c,d,e,f,g,h) {vload(a,h);vload1(b,h);vload2(c,h);vload3(d,h);vload4(e,h);vload5(f,h);vload6(g,h);}

#define vstore(a,b) {_mm256_storeu_pd((&a),b);}
#define vstore1(a,b) {_mm256_storeu_pd((&a)+4,b);}
#define vstore2(a,b) {_mm256_storeu_pd((&a)+8,b);}
#define vstore3(a,b) {_mm256_storeu_pd((&a)+12,b);} 
#define setstore(a,b,c,d,e) {vstore(e,a);vstore1(e,b);vstore2(e,c);vstore3(e,d);}


#define kwpair(a) _mm256_permute4x64_pd(a,0b01001110) //	a0 a1 a2 a3 --> a2 a3 a0 a1
#define gwpair(a,b) _mm256_blend_pd(a,b,0b1100) //	--> a0 a1 c0 c1
#define kwpermute(a,b) _mm256_permute2f128_pd(a,b,0b00010011)
#define gwpermute(a,b) _mm256_permute2f128_pd(a,b,0b00000010)
#define trans(a,b,c,d)	{vec l=kwpermute(a,b); vec k=kwpermute(c,d); vec g=gwpermute(a,b); vec w=gwpermute(c,d);\
							a=_mm256_unpackhi_pd(k,l); b=_mm256_unpacklo_pd(k,l);\
						    d=_mm256_unpacklo_pd(w,g); c=_mm256_unpackhi_pd(w,g);}

#define compute(up, v, down, ww, kk) {v=_mm256_add_pd(v,down);v=_mm256_add_pd(up,v);v=_mm256_mul_pd(v,kk);}
#define computed(up, v, down, ww, kk) {up=_mm256_mul_pd(_mm256_add_pd(up,_mm256_add_pd(v,down)),kk);}
#define compute2(up, v, down, result, ww, kk) {result=_mm256_fmadd_pd(v,ww,down);result=_mm256_add_pd(up,result);result=_mm256_mul_pd(result,kk);}

#define fold_cp_s4p5(v1,v2,v3,v4,v5,v6){v6=_mm256_add_pd(v5,_mm256_fmadd_pd(v4,v4,_mm256_fmadd_pd(v3,v3,_mm256_fmadd_pd(v2,v2,v1))));}

#define fold_cp_s4p5n2(v1,v2,v3,v4,v5){v5=_mm256_fmadd_pd(v4,v4,_mm256_fmadd_pd(v3,v3,_mm256_fmadd_pd(v2,v2,_mm256_fmadd_pd(v1,v1,v5))));}

#define cross_cp_s2p5(v0,v1,v2,v3,v4,v5, v6, v7,v8,ww){v0 =_mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(v7,v7,_mm256_fmadd_pd(v6,v6,_mm256_fmadd_pd(v5,v5,_mm256_fmadd_pd(v4,v4,_mm256_fmadd_pd(v3,v3,_mm256_fmadd_pd(v2,v2,_mm256_fmadd_pd(v1,v1,v0))))))),v8),ww);}

#define cross_cp_s2p3(v0,v1,v2,v3,v4,ww){v0=_mm256_mul_pd(_mm256_add_pd(v4,_mm256_fmadd_pd(v3,v3,_mm256_fmadd_pd(v2,v2,_mm256_fmadd_pd(v1,v1,v0)))),ww);}

#define cross_cp_s3p3(v0,v1,v2,v3,v4,v5,v6,ww){v0=_mm256_mul_pd(_mm256_add_pd(v6,_mm256_fmadd_pd(v5,v5,_mm256_fmadd_pd(v4,v4,_mm256_fmadd_pd(v3,v3,_mm256_fmadd_pd(v2,v2,_mm256_fmadd_pd(v1,v1,v0)))))),ww);}

#define cross_cp_s4p3(v0,v1,v2,v3,v4,v5,v6,v7,v8,ww){v0=_mm256_mul_pd(_mm256_add_pd(v8,_mm256_fmadd_pd(v7,v7,_mm256_fmadd_pd(v6,v6,_mm256_fmadd_pd(v5,v5,_mm256_fmadd_pd(v4,v4,_mm256_fmadd_pd(v3,v3,_mm256_fmadd_pd(v2,v2,_mm256_fmadd_pd(v1,v1,v0)))))))),ww);}


#define cross_cp_s4p5(v1, v2, v3, v4, v5, v6, v7,v8,v9,v10,v11,v12,v13, v14, v15, v16,v17, ww){v1 =_mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(v16,v16,_mm256_fmadd_pd(v15,v15,_mm256_fmadd_pd(v14,v14,_mm256_fmadd_pd(v13,v13,_mm256_fmadd_pd(v12,v12,_mm256_fmadd_pd(v11,v11,_mm256_fmadd_pd(v10,v10,_mm256_fmadd_pd(v9,v9,_mm256_fmadd_pd(v8,v8,_mm256_fmadd_pd(v7,v7,_mm256_fmadd_pd(v6,v6,_mm256_fmadd_pd(v5,v5,_mm256_fmadd_pd(v4,v4,_mm256_fmadd_pd(v3,v3,_mm256_fmadd_pd(v2,v2,v1))))))))))))))),v17),ww);}


#define setcompute(v0,v1,v2,v3,v4,v5,ww,kk) 	{v0=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v1,v0),v2),kk);\
												v1=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v2,v1),v3),kk);\
												v2=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v3,v2),v4),kk);\
												v3=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v4,v3),v5),kk);}
#define setcompute2(v0,v1,v2,v3,v4,v5,ww,kk) 	{v0=_mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(v1,ww,v0),v2),kk);\
												v1=_mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(v2,ww,v1),v3),kk);\
												v2=_mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(v3,ww,v2),v4),kk);\
												v3=_mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(v4,ww,v3),v5),kk);}
												
#define setcompute_5p(v0,v1,v2,v3,v4,v5,v6,v7,ww,kk) 	{v0=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v0,v1),v2),v3),v4),kk);\
													 v1=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v1,v2),v3),v4),v5),kk);\
													 v2=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v2,v3),v4),v5),v6),kk);\
													 v3=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v3,v4),v5),v6),v7),kk);}		

#define setcompute8(v0,v1,v2,v3,v4,v5,v6,v7,ww,kk) 	{v0=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v1,v0),v2),kk);\
													 v1=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v2,v1),v3),kk);\
													 v2=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v3,v2),v4),kk);\
													 v3=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v4,v3),v5),kk);\
													 v4=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v5,v4),v6),kk);\
													 v5=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v6,v5),v7),kk);}													

#define redun_compute(v1_up, v1, v1_down, v2_up, v2, v2_down, v3_up, v3, v3_down, v4_up, v4, v4_down, ww, kk) {\
					v1=_mm256_fmadd_pd(v1,ww,v1_down);v1=_mm256_add_pd(v1_up,v1);v1=_mm256_mul_pd(v1,kk);\
					v2=_mm256_fmadd_pd(v2,ww,v2_down);v2=_mm256_add_pd(v2_up,v2);v2=_mm256_mul_pd(v2,kk);\
					v3=_mm256_fmadd_pd(v3,ww,v3_down);v3=_mm256_add_pd(v3_up,v3);v3=_mm256_mul_pd(v3,kk);\
					v4=_mm256_fmadd_pd(v4,ww,v4_down);v4=_mm256_add_pd(v4_up,v4);v4=_mm256_mul_pd(v4,kk);}

#define regis_compute(v1_up, v1, v1_down, ww, kk) {\
	v1=_mm256_fmadd_pd(v1,ww,v1_down);v1=_mm256_add_pd(v1_up,v1);v1=_mm256_mul_pd(v1,kk);}

#define shift_left(a) _mm256_permute4x64_pd(a,0b00111001) //	a0 a1 a2 a3 --> a1 a2 a3 a0
#define shift_right(a) _mm256_permute4x64_pd(a,0b10010011) //	a0 a1 a2 a3 --> a3 a0 a1 a2
#define blend(a,b,c) _mm256_blend_pd(a, b, c)

#define shuffle_head(a,b,c) {c = _mm256_blend_pd(a, b, 8); c = _mm256_permute4x64_pd(c,147);}
#define shuffle_headd(a,b) {b = _mm256_blend_pd(a, b, 8); b = _mm256_permute4x64_pd(b,147);}
#define shuffle_head2(a,b,c) {b = _mm256_permute4x64_pd(b,128);c = _mm256_blend_pd(a, b, 8); c = _mm256_permute4x64_pd(c,147);}
#define shuffle_head3(a,b,c) {b = _mm256_permute4x64_pd(b,64);c = _mm256_blend_pd(a, b, 8); c = _mm256_permute4x64_pd(c,147);}

#define shuffle_tail(a,b,c) {c = _mm256_blend_pd(a, b, 1); c = _mm256_permute4x64_pd(c,57);}
#define shuffle_tail2(a,b,c) {b = _mm256_permute4x64_pd(b,1); c = _mm256_blend_pd(a, b, 1); c = _mm256_permute4x64_pd(c,57);}
#define shuffle_tail3(a,b,c) {b = _mm256_permute4x64_pd(b,2);c = _mm256_blend_pd(a, b, 1); c = _mm256_permute4x64_pd(c,57);}

#define shuffle_boundary(a) {vec b = _mm256_set1_pd(0); a = _mm256_blend_pd(a, b, 1); }
#define shuffle_headnum(a,b,c) {vec d = _mm256_set1_pd(b); c = _mm256_blend_pd(a, d, 8); c = _mm256_permute4x64_pd(c,147);}

//#define point 5
#if !defined(point)
#define point 3
#endif

#if point == 3
#define  kernel(A) A[(t+1)%2][x] = LK * (A[t%2][x+1] + (A[t%2][x] + A[t%2][x-1]))
#define  kernel_apop(A,B) A[(t+1)%2][x] =  (B[0][x]*A[t%2][x+1] + (B[1][x]*A[t%2][x] + B[2][x]*A[t%2][x-1]))
#define XSLOPE  1
#elif point == 5
#define  kernel(A)  A[(t+1)%2][x] = LK * (A[t%2][x-2] + A[t%2][x-1] + A[t%2][x] + A[t%2][x+1] + A[t%2][x+2]);
#define XSLOPE  2
#endif

typedef union avd{
	vec v;
	double d[4];
} avd_t;


static inline vec boundary(vec a){
	avd_t vik,viw; 
	vik.v = a;
	vik.d[0] = 0.0;
	return vik.v;
}

static inline vec headnum(vec a, double b){
	avd_t vik; 
	vik.v = a;
	vik.d[3] = vik.d[2];
	vik.d[2] = vik.d[1];
	vik.d[1] = vik.d[0];
	vik.d[0] = b;
	return vik.v;
}

static inline vec head1(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[3] = vik.d[2];
	vik.d[2] = vik.d[1];
	vik.d[1] = vik.d[0];
	vik.d[0] = viw.d[3];
	return vik.v;
}

static inline vec head2(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[3] = vik.d[2];
	vik.d[2] = vik.d[1];
	vik.d[1] = vik.d[0];
	vik.d[0] = viw.d[2];
	return vik.v;
}

static inline vec head3(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[3] = vik.d[2];
	vik.d[2] = vik.d[1];
	vik.d[1] = vik.d[0];
	vik.d[0] = viw.d[1];
	return vik.v;
}


static inline vec tail1(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[0] = vik.d[1];
	vik.d[1] = vik.d[2];
	vik.d[2] = vik.d[3];
	vik.d[3] = viw.d[0];
	return vik.v;
}

static inline vec tail2(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[0] = vik.d[1];
	vik.d[1] = vik.d[2];
	vik.d[2] = vik.d[3];
	vik.d[3] = viw.d[1];
	return vik.v;
}

static inline vec tail3(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[0] = vik.d[1];
	vik.d[1] = vik.d[2];
	vik.d[2] = vik.d[3];
	vik.d[3] = viw.d[2];
	return vik.v;
}




void ompp(double** A, int N, int T, int Bx, int tb);
void ompp_apop(double** A, double** R, int N, int T, int Bx, int tb);
void check(double** A, double** B, int N, int T);
void regis_mov(double** A, int N, int T);
void redun_load(double** A, int N, int T);
void halfpipe(double** A, double** B, int N, int T, int Bx, int tb);
void one_step(double** A, double** B, int N, int T, int Bx, int tb);
void two_steps(double** A, int N, int T, int Bx, int tb);
void cross_1d2s3p(double** A, double** R, int N, int T, int Bx, int tb);
void cross_1d2s5p(double** A, double** R, int N, int T, int Bx, int tb);
void cross_1d3s3p(double** A, double** R, int N, int T, int Bx, int tb);
void cross_1d4s3p(double** A, double** R, int N, int T, int Bx, int tb);
void fold_1d4s5p(double** A, double** R, int N, int T, int Bx, int tb);
void cross_1d4s5p(double** A, double** R, int N, int T, int Bx, int tb);
void our_1d1sapop(double** A, double** R, int N, int T, int Bx, int tb);
void our_1d1s3p(double** A, double** R, int N, int T, int Bx, int tb);
void our_1d1s5p(double** A, double** R, int N, int T, int Bx, int tb);
void our_1d2s3p(double** A, double** R, int N, int T, int Bx, int tb);
void our_1d2s3ph(double** A, double** R, int N, int T, int Bx, int tb);
void our_1d2s5p(double** A, double** R, int N, int T, int Bx, int tb);
void our_1d3s3p(double** A, double** R, int N, int T, int Bx, int tb);
void four_tiles(double** A, double** R, int N, int T, int Bx, int tb);
void four_steps(double** A, double** B, int N, int T, int Bx, int tb);
void dlt(double** A, double** B, int N, int T);
void dlt_unaligned(double** A, int N, int T);
void dlt_aligned(double** A, int N, int T);
void dlt_aligned_unroll(double** A, int N, int T);
void dlt_tblock2(double** A, int N, int T);

