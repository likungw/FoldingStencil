#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))
#define myabs(x,y)  ((x) > (y)? ((x)-(y)) : ((y)-(x)))
#define ceild(n,d)	(((n)-1)/(d) + 1)// ceil(((double)(n))/((double)(d)))
#define myceil(x,y)  (int)ceil(((double)x)/((double)y)) // if x and y are integers, myceil(x,y) = (x-1)/y + 1
#define myfloor(x,y)  (int)floor(((double)x)/((double)y)) // if x and y are integers, myceil(x,y) = (x-1)/y + 1

#define NUM_THREAD 1

typedef __m256d vec;

#define len 2
#define veclen 4
#define veclen2 8
#define veclen3 12
#define veclen4 16
#define veclen5 20
#define veclen6 24

#define GW 1
#define LK 1//0.14285714
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
#define vload_al(a,b) {a=_mm256_load_pd((&b));}
#define vload1_al(a,b) {a=_mm256_load_pd((&b)+veclen);}
#define vload2_al(a,b) {a=_mm256_load_pd((&b)+veclen2);}
#define vload3_al(a,b) {a=_mm256_load_pd((&b)+veclen3);}
#define vload4_al(a,b) {a=_mm256_load_pd((&b)+veclen4);}

#define setload(a,b,c,d,e) {vload(a,e);vload1(b,e);vload2(c,e);vload3(d,e);}
#define setload_al(a,b,c,d,e) {vload_al(a,e);vload1_al(b,e);vload2_al(c,e);vload3_al(d,e);}
#define setload5(a,b,c,d,e,f) {vload(a,f);vload1(b,f);vload2(c,f);vload3(d,f);vload4(e,f);}
#define setload5_al(a,b,c,d,e,f) {vload_al(a,f);vload1_al(b,f);vload2_al(c,f);vload3_al(d,f);vload4_al(e,f);}
#define setload6(a,b,c,d,e,f,g) {vload(a,g);vload1(b,g);vload2(c,g);vload3(d,g);vload4(e,g);vload5(f,g);}
#define setload7(a,b,c,d,e,f,g,h) {vload(a,h);vload1(b,h);vload2(c,h);vload3(d,h);vload4(e,h);vload5(f,h);vload6(g,h);}

#define vloadset(a,b,c,d,e) {a=_mm256_set_pd(b,c,d,e);}
#define vallset(a,b) {a = _mm256_set1_pd(b);}


#define vstore(a,b) {_mm256_storeu_pd((&a),b);}
#define vstore1(a,b) {_mm256_storeu_pd((&a)+4,b);}
#define vstore2(a,b) {_mm256_storeu_pd((&a)+8,b);}
#define vstore3(a,b) {_mm256_storeu_pd((&a)+12,b);}
#define vstore_al(a,b) {_mm256_store_pd((&a),b);}
#define vstore1_al(a,b) {_mm256_store_pd((&a)+4,b);}
#define vstore2_al(a,b) {_mm256_store_pd((&a)+8,b);}
#define vstore3_al(a,b) {_mm256_store_pd((&a)+12,b);}
#define setstore(a,b,c,d,e) {vstore(e,a);vstore1(e,b);vstore2(e,c);vstore3(e,d);}
#define setstore_al(a,b,c,d,e) {vstore_al(e,a);vstore1_al(e,b);vstore2_al(e,c);vstore3_al(e,d);}


#define kwpair(a) _mm256_permute4x64_pd(a,0b01001110) //	a0 a1 a2 a3 --> a2 a3 a0 a1
#define gwpair(a,b) _mm256_blend_pd(a,b,0b1100) //	--> a0 a1 c0 c1
#define kwpermute(a,b) _mm256_permute2f128_pd(a,b,0b00010011)
#define gwpermute(a,b) _mm256_permute2f128_pd(a,b,0b00000010)
#define trans(a,b,c,d)	{vec l=kwpermute(a,b); vec k=kwpermute(c,d); vec g=gwpermute(a,b); vec w=gwpermute(c,d);\
							a=_mm256_unpackhi_pd(k,l); b=_mm256_unpacklo_pd(k,l);\
						    d=_mm256_unpacklo_pd(w,g); c=_mm256_unpackhi_pd(w,g);}

#define cross_comp_s1_h(v0,v1,v2,v3,v4,v5){v0=_mm256_add_pd(_mm256_add_pd(v0,v1),v2);\
										v1=_mm256_add_pd(_mm256_add_pd(v1,v2),v3);\
										v2=_mm256_add_pd(_mm256_add_pd(v2,v3),v4);\
										v3=_mm256_add_pd(_mm256_add_pd(v3,v4),v5);}

#define up_down_comp_s1(v0,v1,v2)		{v1=_mm256_add_pd(_mm256_add_pd(v0,v1),v2);}						

#define cross_comp_v6(v0,v1,v2,v3,v4,v5){v0=_mm256_add_pd(v0, v2);\
										 v1=_mm256_add_pd(v1, v3);\
										 v2=_mm256_add_pd(v2, v4);\
										 v3=_mm256_add_pd(v3, v5);}

#define cobine_comp_hv4(v0,v1,v2,v3,v4,v5,v6,v7,kk)	{v0=_mm256_mul_pd(_mm256_add_pd(v0,v4), kk);\
													 v1=_mm256_mul_pd(_mm256_add_pd(v1,v5), kk);\
													 v2=_mm256_mul_pd(_mm256_add_pd(v2,v6), kk);\
													 v3=_mm256_mul_pd(_mm256_add_pd(v3,v7), kk);}	

#define cross_comp_s2_nb2_v8(v0,v1,v2,v3,v4,v5,v6,v7)  {v4=_mm256_add_pd(v0,v4);\
														v5=_mm256_add_pd(v1,v5);\
														v6=_mm256_add_pd(v2,v6);\
														v7=_mm256_add_pd(v3,v7);}	

#define cross_comp_s2_nb0_vall(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13)\
													   {v0=_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v0,v8), _mm256_fmadd_pd(v2,v2,v10)),v4);\
														v1=_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v1,v9), _mm256_fmadd_pd(v3,v3,v11)),v5);\
														v2=_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v2,v10),_mm256_fmadd_pd(v4,v4,v12)),v6);\
														v3=_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v3,v11),_mm256_fmadd_pd(v5,v5,v13)),v7);}	

#define cross_comp_s2_nb1_v6(v0,v1,v2,v3,v4,v5)  	   {v0=_mm256_add_pd(_mm256_add_pd(v0,v1),v2);\
														v1=_mm256_add_pd(_mm256_add_pd(v1,v2),v3);\
														v2=_mm256_add_pd(_mm256_add_pd(v2,v3),v4);\
														v3=_mm256_add_pd(_mm256_add_pd(v3,v4),v5);}	
													
#define cobine_comp_s2_hv4(v0,v1,v2,v3,v4,v5,v6,v7)	{v4=_mm256_mul_pd(v0,v4);\
													 v5=_mm256_mul_pd(v1,v5);\
													 v6=_mm256_mul_pd(v2,v6);\
													 v7=_mm256_mul_pd(v3,v7);}		


#define cross_comp_s2p9_ma1(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11) {v0=_mm256_fmadd_pd(v6,v6,v4);\
																	v1=_mm256_fmadd_pd(v7,v7,v5);\
																	v2=_mm256_fmadd_pd(v8,v8,v6);\
																	v3=_mm256_fmadd_pd(v9,v9,v7);}													 										 
#define compute(up, v, down, ww, kk) {v=_mm256_add_pd(v,down);v=_mm256_add_pd(up,v);v=_mm256_mul_pd(v,kk);}
#define computed(up, v, down, ww, kk) {up=_mm256_mul_pd(_mm256_add_pd(up,_mm256_add_pd(v,down)),kk);}
#define setcompute(v0,v1,v2,v3,v4,v5,ww,kk) 	{v0=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v1,v0),v2),kk);\
												v1=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v2,v1),v3),kk);\
												v2=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v3,v2),v4),kk);\
												v3=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(v4,v3),v5),kk);}

#define setcompute2(v0,v1,v2,v3,v4,v5,ww,kk) 	{v0=_mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(v1,ww,v0),v2),kk);\
												v1=_mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(v2,ww,v1),v3),kk);\
												v2=_mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(v3,ww,v2),v4),kk);\
												v3=_mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(v4,ww,v3),v5),kk);}

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

// added
#define computed_7p(a, b, c, d, e, f, g, ww, kk) {\
					a=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a,b),c),d),e),f),g),kk);}
#define compute_7p_res(res, a, b, c, d, e, f, g, ww, kk) {\
					res=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a,b),c),d),e),f),g),kk);}
#define computed_27p(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a1, ww, kk) {a=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd\
(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd\
(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a,b),c),d),e),f),g),h),i),j),k),l),m),n),o),p),q),r),s),t),u),v),w),x),y),z),a1),kk);}

#define shift_left(a) _mm256_permute4x64_pd(a,0b00111001) //	a0 a1 a2 a3 --> a1 a2 a3 a0
#define shift_right(a) _mm256_permute4x64_pd(a,0b10010011) //	a0 a1 a2 a3 --> a3 a0 a1 a2
#define blend(a,b,c) _mm256_blend_pd(a, b, c)

#define shuffle_head(a,b,c) {c = _mm256_blend_pd(a, b, 8); c = _mm256_permute4x64_pd(c,147);}
#define shuffle_headd(a,b) {b = _mm256_blend_pd(a, b, 8); b = _mm256_permute4x64_pd(b,147);}
#define shuffle_tail(a,b,c) {c = _mm256_blend_pd(a, b, 1); c = _mm256_permute4x64_pd(c,57);}


#if !defined(point)
#define point 27
#endif

// modified
#if point == 7
#define kernel(A) A[(t+1)%2][x][y][z] = LK * ((((((A[t%2][x][y][z] + \
										A[t%2][x][y][z-1]) + A[t%2][x][y][z+1]) + \
										A[t%2][x][y+1][z]) + A[t%2][x][y-1][z] )+ \
										A[t%2][x+1][y ][z] )+ A[t%2][x-1][y][z]);
#define XSLOPE 1
#define YSLOPE 1
#define ZSLOPE 1
#elif point == 27
#define kernel(A) A[(t+1)%2][x][y][z] = LK * (A[(t)%2][x][y][z] + \
												A[(t)%2][x][y][z-1] + A[(t)%2][x][y-1][z] + \
												A[(t)%2][x][y+1][z] + A[(t)%2][x][y][z+1] + \
												A[(t)%2][x-1][y][z] + A[(t)%2][x+1][y][z] + \
												A[(t)%2][x-1][y][z-1] + A[(t)%2][x-1][y-1][z] + \
												A[(t)%2][x-1][y+1][z] + A[(t)%2][x-1][y][z+1] + \
												A[(t)%2][x][y-1][z-1] + A[(t)%2][x][y+1][z-1] + \
												A[(t)%2][x][y-1][z+1] + A[(t)%2][x][y+1][z+1] + \
												A[(t)%2][x+1][y][z-1] + A[(t)%2][x+1][y-1][z] + \
												A[(t)%2][x+1][y+1][z] + A[(t)%2][x+1][y][z+1] + \
												A[(t)%2][x-1][y-1][z-1] + A[(t)%2][x-1][y+1][z-1] + \
												A[(t)%2][x-1][y-1][z+1] + A[(t)%2][x-1][y+1][z+1] + \
												A[(t)%2][x+1][y-1][z-1] + A[(t)%2][x+1][y+1][z-1] + \
												A[(t)%2][x+1][y-1][z+1] + A[(t)%2][x+1][y+1][z+1]);
#define XSLOPE 1
#define YSLOPE 1
#define ZSLOPE 1
#endif

#define three_kernel(A,Y,Z) A[(t+1)%2][x][y][z] = LK * ((((((A[t%2][x][y][z] + \
										A[t%2][x][y][z-1]) + A[t%2][x][y][z+1]) + \
										A[t%2][x][y+1][z]) + A[t%2][x][y-1][z] )+ \
										A[t%2][x+1][y ][z] )+ A[t%2][x-1][y][z])+ \
										LK * ((((((Y[t%2][x][y][z] + \
										Y[t%2][x][y][z-1]) + Y[t%2][x][y][z+1]) + \
										Y[t%2][x][y+1][z]) + Y[t%2][x][y-1][z] )+ \
										Y[t%2][x+1][y ][z] )+ Y[t%2][x-1][y][z])+ \
										LK * ((((((Y[t%2][x][y][z] + \
										Z[t%2][x][y][z-1]) + Z[t%2][x][y][z+1]) + \
										Z[t%2][x][y+1][z]) + Z[t%2][x][y-1][z] )+ \
										Z[t%2][x+1][y ][z] )+ Z[t%2][x-1][y][z]);

typedef union avd{
	vec v;
	double d[4];
} avd_t;

// 0 a1 a2 a3
static inline vec boundary(vec a){
	avd_t vik,viw; 
	vik.v = a;
	vik.d[0] = 0.0;
	return vik.v;
}

// b a0 a1 a2
static inline vec headnum(vec a, double b){
	avd_t vik; 
	vik.v = a;
	vik.d[3] = vik.d[2];
	vik.d[2] = vik.d[1];
	vik.d[1] = vik.d[0];
	vik.d[0] = b;
	return vik.v;
}

// b3 a0 a1 a2
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

// b2 a0 a1 a2
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

// b1 a0 a1 a2
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

// a1 a2 a3 b0
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

// a1 a2 a3 b1
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

// a1 a2 a3 b2
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


void ompp_3d(double**** A,double**** Y,double**** Z, int NX, int NY, int NZ, int T, int Bx, int By, int tb);
void check_3d(double**** A, double**** B, int NX, int NY, int NZ, int T);
void allpipe_3d(double**** A, int NX, int NY, int NZ, int T);
void one_tile_3d(double**** A, int NX, int NY, int NZ, int T, int Bx, int By, int tb);
void one_tile_aligned_3d(double**** A, int NX, int NY, int NZ, int T, int Bx, int By, int tb);
//void two_tile_aligned_3d(double**** A, int NX, int NY, int NZ, int T, int Bx, int By, int tb);
void one_tile_cross_3d(double**** A, int NX, int NY, int NZ, int T, int Bx, int By, int tb);
void two_tile_cross_3d(double**** A, int NX, int NY, int NZ, int T, int Bx, int By, int tb);
void one_tile_3d27p(double**** A, int NX, int NY, int NZ, int T, int Bx, int By, int tb);
void one_tile_plane_cross_3d(double ****A, int NX, int NY, int NZ, int T, int Bx, int By, int tb);