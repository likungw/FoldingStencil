#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))
#define myabs(x,y)  ((x) > (y)? ((x)-(y)) : ((y)-(x))) 
#define myceil(x,y)  (int)ceil(((double)x)/((double)y)) // if x and y are integers, myceil(x,y) = (x-1)/y + 1
#define myfloor(x,y)  (int)floor(((double)x)/((double)y)) // if x and y are integers, myceil(x,y) = (x-1)/y + 1

typedef __m512d vec;
#define len 4
#define veclen 8
#define veclen2 16
#define veclen3 24
#define veclen4 32
#define veclen5 40
#define veclen6 48
#define veclen7 56
#define veclen8 64
#define veclen9 72
#define veclen10 80

//#define GW 2.1
//#define LK 0.244
#define GW 1
#define LK 0.33
#define TOLERANCE 0.0001
int check_flag; 

#define DI(v) ((double*)(&(v)))
#define D0(v) (DI(v))[0]
#define D1(v) (DI(v))[1]
#define D2(v) (DI(v))[2]
#define D3(v) (DI(v))[3]

#define vload(a,b) {a=_mm512_loadu_pd((&b));}
#define vload1(a,b) {a=_mm512_loadu_pd((&b)+veclen);}
#define vload2(a,b) {a=_mm512_loadu_pd((&b)+veclen2);}
#define vload3(a,b) {a=_mm512_loadu_pd((&b)+veclen3);}
#define vload4(a,b) {a=_mm512_loadu_pd((&b)+veclen4);}
#define vload5(a,b) {a=_mm512_loadu_pd((&b)+veclen5);}
#define vload6(a,b) {a=_mm512_loadu_pd((&b)+veclen6);}
#define vload7(a,b) {a=_mm512_loadu_pd((&b)+veclen7);}
#define vload8(a,b) {a=_mm512_loadu_pd((&b)+veclen8);}
#define vload9(a,b) {a=_mm512_loadu_pd((&b)+veclen9);}
#define vload10(a,b) {a=_mm512_loadu_pd((&b)+veclen10);}

#define vloadset(a,b,c,d,e,f,g,h,i) {a=_mm512_set_pd(b,c,d,e,f,g,h,i);}
#define vallset(a,b) {a = _mm512_set1_pd(b);}
#define setloadw(a,b,c,d,e,f,g,h,i,j) {vload(a,j);vload1(b,j);vload2(c,j);vload3(d,j);vload4(e,j);vload5(f,j);vload6(g,j);vload7(h,j);vload8(i,j);}
#define setload(a,b,c,d,e,f,g,h,i) {vload(a,i);vload1(b,i);vload2(c,i);vload3(d,i);vload4(e,i);vload5(f,i);vload6(g,i);vload7(h,i);}
//#define setload10(a,b,c,d,e,f,g) {vload(a,g);vload1(b,g);vload2(c,g);vload3(d,g);vload4(e,g);vload5(f,g);}
//#define setload11(a,b,c,d,e,f,g,h) {vload(a,h);vload1(b,h);vload2(c,h);vload3(d,h);vload4(e,h);vload5(f,h);vload6(g,h);}

#define vstore(a,b) {_mm512_storeu_pd((&a),b);}
#define vstore1(a,b) {_mm512_storeu_pd((&a)+veclen,b);}
#define vstore2(a,b) {_mm512_storeu_pd((&a)+veclen2,b);}
#define vstore3(a,b) {_mm512_storeu_pd((&a)+veclen3,b);} 
#define vstore4(a,b) {_mm512_storeu_pd((&a)+veclen4,b);} 
#define vstore5(a,b) {_mm512_storeu_pd((&a)+veclen5,b);} 
#define vstore6(a,b) {_mm512_storeu_pd((&a)+veclen6,b);} 
#define vstore7(a,b) {_mm512_storeu_pd((&a)+veclen7,b);} 
#define setstore(a,b,c,d,e,f,g,h,i) {vstore(i,a);vstore1(i,b);vstore2(i,c);vstore3(i,d);vstore4(i,e);vstore5(i,f);vstore6(i,g);vstore7(i,h);}
 
#define kwpair(kwp,a) _mm512_permutexvar_pd(kwp,a) //	a0 a1 a2 a3 --> a2 a3 a0 a1
#define gwpair(gwp,a,b) _mm512_mask_blend_pd(gwp,a,b) //	--> a0 a1 c0 c1
#define trans(a,b,c,d,e,f,g,h){__m512i kwp = _mm512_set_epi64(3,2,1,0,7,6,5,4);\
							vec a1=kwpair(kwp,a); vec b1=kwpair(kwp,b); vec c1=kwpair(kwp,c); vec d1=kwpair(kwp,d); vec e1=kwpair(kwp,e); vec f1=kwpair(kwp,f); vec g1=kwpair(kwp,g); vec h1=kwpair(kwp,h);\
							__mmask8 gwp = _cvtu32_mask8(15);\
							a1=gwpair(gwp,a1,c); c1=gwpair(gwp,a,c1); b1=gwpair(gwp,b1,d); d1=gwpair(gwp,b,d1); e1=gwpair(gwp,e1,g); g1=gwpair(gwp,e,g1); f1=gwpair(gwp,f1,h); h1=gwpair(gwp,f,h1);\
						    kwp = _mm512_set_epi64(3,2,7,6,1,0,5,4);\
							a=kwpair(kwp,a1); b=kwpair(kwp,b1); c=kwpair(kwp,c1); d=kwpair(kwp,d1); e=kwpair(kwp,e1); f=kwpair(kwp,f1); g=kwpair(kwp,g1); h=kwpair(kwp,h1);\
							kwp = _mm512_set_epi64(3,2,1,0,7,6,5,4);\
							a1=kwpair(kwp,a); b1=kwpair(kwp,b); c1=kwpair(kwp,c); d1=kwpair(kwp,d); e1=kwpair(kwp,e); f1=kwpair(kwp,f); g1=kwpair(kwp,g); h1=kwpair(kwp,h);\
							a=gwpair(gwp,e1,a); c=gwpair(gwp,g1,c); b=gwpair(gwp,f1,b); d=gwpair(gwp,h1,d); e=gwpair(gwp,e,a1); g=gwpair(gwp,g,c1); f=gwpair(gwp,f,b1); h=gwpair(gwp,h,d1);\
							a1 = a; b1 = b; c1 = c; d1 = d; e1 = e; f1 = f; g1 = g; h1 = h;\
							a = _mm512_unpacklo_pd(a1,b1); b = _mm512_unpackhi_pd(a1,b1);\
							e = _mm512_unpacklo_pd(c1,d1); f = _mm512_unpackhi_pd(c1,d1);\
							c = _mm512_unpacklo_pd(e1,f1); d = _mm512_unpackhi_pd(e1,f1);\
							g = _mm512_unpacklo_pd(g1,h1); h = _mm512_unpackhi_pd(g1,h1);}

#define compute(up, v, down, ww, kk) {v=_mm512_add_pd(v,down);v=_mm512_add_pd(up,v);v=_mm512_mul_pd(v,kk);}
#define computed(up, v, down, ww, kk) {up=_mm512_mul_pd(_mm512_add_pd(up,_mm512_add_pd(v,down)),kk);}
#define setcompute(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,ww,kk) 	{v0=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v1,v0),v2),kk);\
												    		v1=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v2,v1),v3),kk);\
												    		v2=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v3,v2),v4),kk);\
												    		v3=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v4,v3),v5),kk);\
															v4=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v5,v4),v6),kk);\
															v5=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v6,v5),v7),kk);\
															v6=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v7,v6),v8),kk);\
															v7=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v8,v7),v9),kk);}
															
#define setcompute_5p(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,ww,kk) 	{v0=_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v1,v0),v2),v3),v4),kk);\
												    		v1=_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v1,v2),v3),v4),v5),kk);\
												    		v2=_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v6,v5),v2),v3),v4),kk);\
												    		v3=_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v7,v6),v5),v3),v4),kk);\
															v4=_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v7,v6),v5),v8),v4),kk);\
															v5=_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v7,v6),v5),v8),v9),kk);\
															v6=_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v7,v6),v10),v8),v9),kk);\
															v7=_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v7,v11),v10),v8),v9),kk);}														
												
#define setcompute8(v0,v1,v2,v3,v4,v5,v6,v7,ww,kk) 	{v0=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v1,v0),v2),kk);\
													 v1=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v2,v1),v3),kk);\
													 v2=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v3,v2),v4),kk);\
													 v3=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v4,v3),v5),kk);\
													 v4=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v5,v4),v6),kk);\
													 v5=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v6,v5),v7),kk);}													

#define redun_compute(v1_up, v1, v1_down, v2_up, v2, v2_down, v3_up, v3, v3_down, v4_up, v4, v4_down, ww, kk) {\
					v1=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v1,v1_down),v1_up),kk);\
					v2=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v2,v2_down),v2_up),kk);\
					v3=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v3,v3_down),v3_up),kk);\
					v4=_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v4,v4_down),v4_up),kk);}

#define regis_compute(v1_up, v1, v1_down, ww, kk) {\
	v1=_mm256_fmadd_pd(v1,ww,v1_down);v1=_mm512_add_pd(v1_up,v1);v1=_mm512_mul_pd(v1,kk);}

#define shift_left(a) _mm256_permute4x64_pd(a,0b00111001) //	a0 a1 a2 a3 --> a1 a2 a3 a0
#define shift_right(a) _mm256_permute4x64_pd(a,0b10010011) //	a0 a1 a2 a3 --> a3 a0 a1 a2
#define blend(a,b,c) _mm256_blend_pd(a, b, c)

#define shuffle_head(a,b,c) {__mmask8 sh = _cvtu32_mask8(128);\
							c = _mm512_mask_blend_pd(sh, a, b);\
							__m512i kwp = _mm512_set_epi64(6,5,4,3,2,1,0,7);\
							c = kwpair(kwp,c);}
#define shuffle_headd(a,b) {__mmask8 sh = _cvtu32_mask8(128);\
							b = _mm512_mask_blend_pd(sh, a, b);\
							__m512i kwp = _mm512_set_epi64(6,5,4,3,2,1,0,7);\
							b = kwpair(kwp,b);}

#define shuffle_head2(a,b,c) {__m512i kwp = _mm512_set_epi64(6,7,5,4,3,2,1,0);\
							b = kwpair(kwp,b);\
							__mmask8 sh = _cvtu32_mask8(128);\
							c = _mm512_mask_blend_pd(sh, a, b);\
							kwp = _mm512_set_epi64(6,5,4,3,2,1,0,7);\
							c = kwpair(kwp,c);}

#define shuffle_head3(a,b,c) {__m512i kwp = _mm512_set_epi64(5,6,7,4,3,2,1,0);\
							b = kwpair(kwp,b);\
							__mmask8 sh = _cvtu32_mask8(128);\
							c = _mm512_mask_blend_pd(sh, a, b);\
							kwp = _mm512_set_epi64(6,5,4,3,2,1,0,7);\
							c = kwpair(kwp,c);}

#define shuffle_head4(a,b) {__m512i kwp = _mm512_set_epi64(6,5,4,3,2,1,0,7);\
							a = kwpair(kwp,a);\
							__mmask8 sh = _cvtu32_mask8(1);\
							a = _mm512_mask_blend_pd(sh, a, b);}

#define shuffle_tail(a,b,c) {__mmask8 st = _cvtu32_mask8(1);\
							c = _mm512_mask_blend_pd(st, a, b);\
							__m512i kwp = _mm512_set_epi64(0,7,6,5,4,3,2,1);\
							c = kwpair(kwp,c);}

#define shuffle_tail2(a,b,c) {__m512i kwp = _mm512_set_epi64(6,7,5,4,3,2,1,0);\
							b = kwpair(kwp,b);\
							__mmask8 st = _cvtu32_mask8(1);\
							c = _mm512_mask_blend_pd(st, a, b);\
							kwp = _mm512_set_epi64(0,7,6,5,4,3,2,1);\
							c = kwpair(kwp,c);}

#define shuffle_tail3(a,b) {__m512i kwp = _mm512_set_epi64(0,7,6,5,4,3,2,1);\
							a = kwpair(kwp,a);\
							__mmask8 st = _cvtu32_mask8(128);\
							a = _mm512_mask_blend_pd(st, a, b);}

#define shuffle_headnum(a,b,c) {c=_mm512_set_pd(b,b,b,b,b,b,b,b);__mmask8 st = _cvtu32_mask8(1);c = _mm512_mask_blend_pd(st, a, c);}
#define shuffle_boundary(a) {vec b=_mm512_set_pd(0,0,0,0,0,0,0,0);__mmask8 st = _cvtu32_mask8(1);a = _mm512_mask_blend_pd(st, a, b);}

#define point 5
#if !defined(point)
#define point 3
#endif


#if point == 3
#define  kernel(A) A[(t+1)%2][x] = LK * ((A[t%2][x+1] + A[t%2][x]) + A[t%2][x-1])
#define XSLOPE  1
#elif point == 5
#define  kernel(A)  A[(t+1)%2][x] = LK * (A[t%2][x-2] + A[t%2][x-1] + A[t%2][x] + A[t%2][x+1] + A[t%2][x+2]);
#define XSLOPE  2
#endif

typedef union avd{
	vec v;
	double d[8];
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
	vik.d[7] = vik.d[6];
	vik.d[6] = vik.d[5];
	vik.d[5] = vik.d[4];
	vik.d[4] = vik.d[3];
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
	vik.d[7] = vik.d[6];
	vik.d[6] = vik.d[5];
	vik.d[5] = vik.d[4];
	vik.d[4] = vik.d[3];
	vik.d[3] = vik.d[2];
	vik.d[2] = vik.d[1];
	vik.d[1] = vik.d[0];
	vik.d[0] = viw.d[7];
	return vik.v;
}

static inline vec head2(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[7] = vik.d[6];
	vik.d[6] = vik.d[5];
	vik.d[5] = vik.d[4];
	vik.d[4] = vik.d[3];
	vik.d[3] = vik.d[2];
	vik.d[2] = vik.d[1];
	vik.d[1] = vik.d[0];
	vik.d[0] = viw.d[6];
	return vik.v;
}

static inline vec head3(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[7] = vik.d[6];
	vik.d[6] = vik.d[5];
	vik.d[5] = vik.d[4];
	vik.d[4] = vik.d[3];
	vik.d[3] = vik.d[2];
	vik.d[2] = vik.d[1];
	vik.d[1] = vik.d[0];
	vik.d[0] = viw.d[5];
	return vik.v;
}


static inline vec tail1(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[0] = vik.d[1];
	vik.d[1] = vik.d[2];
	vik.d[2] = vik.d[3];
	vik.d[3] = vik.d[4];
	vik.d[4] = vik.d[5];
	vik.d[5] = vik.d[6];
	vik.d[6] = vik.d[7];
	vik.d[7] = viw.d[0];	
	return vik.v;
}

static inline vec tail2(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[0] = vik.d[1];
	vik.d[1] = vik.d[2];
	vik.d[2] = vik.d[3];
	vik.d[3] = vik.d[4];
	vik.d[4] = vik.d[5];
	vik.d[5] = vik.d[6];
	vik.d[6] = vik.d[7];
	vik.d[7] = viw.d[1];
	return vik.v;
}

static inline vec tail3(vec a, vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	vik.d[0] = vik.d[1];
	vik.d[1] = vik.d[2];
	vik.d[2] = vik.d[3];
	vik.d[3] = viw.d[4];
	vik.d[4] = vik.d[5];
	vik.d[5] = vik.d[6];
	vik.d[6] = vik.d[7];
	vik.d[7] = viw.d[2];
	return vik.v;
}




void ompp(double** A, int N, int T, int Bx, int tb);
void check(double** A, double** B, int N, int T);
void regis_mov(double** A, int N, int T);
void redun_load(double** A, int N, int T);
void halfpipe(double** A, double** B, int N, int T, int Bx, int tb);
void one_step(double** A, double** B, int N, int T, int Bx, int tb);
void two_steps(double** A, int N, int T, int Bx, int tb);
void one_tile(double** A, double** R, int N, int T, int Bx, int tb);
void two_tiles(double** A, double** R, int N, int T, int Bx, int tb);
void one_tile_5p(double** A, double** R, int N, int T, int Bx, int tb);
void two_tiles_5p(double** A, double** R, int N, int T, int Bx, int tb);
void four_steps(double** A, double** B, int N, int T, int Bx, int tb);
void dlt(double** A, double** B, int N, int T);
