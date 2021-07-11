#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))
#define myabs(x,y)  ((x) > (y)? ((x)-(y)) : ((y)-(x)))
#define ceild(n,d)	(((n)-1)/(d) + 1)// ceil(((double)(n))/((double)(d)))
#define myceil(x,y)  (int)ceil(((double)x)/((double)y)) // if x and y are integers, myceil(x,y) = (x-1)/y + 1
#define myfloor(x,y)  (int)floor(((double)x)/((double)y)) // if x and y are integers, myceil(x,y) = (x-1)/y + 1

//#define NUM_THREAD 1

typedef __m256d vec;

#define len 2
#define veclen 4
#define veclen2 8
#define veclen3 12
#define veclen4 16
#define veclen5 20
#define veclen6 24

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

#define sload(a,b) {a=_mm256_loadu_pd((&b));}
#define sload1(a,b,c) {a=_mm256_loadu_pd((&b)+c);}
#define sload2(a,b,c) {a=_mm256_loadu_pd((&b)+2*c);}
#define sload3(a,b,c) {a=_mm256_loadu_pd((&b)+3*c);}

#define setload(a,b,c,d,e) {vload(a,e);vload1(b,e);vload2(c,e);vload3(d,e);}
#define setload5(a,b,c,d,e,f) {vload(a,f);vload1(b,f);vload2(c,f);vload3(d,f);vload4(e,f);}
#define setload6(a,b,c,d,e,f,g) {vload(a,g);vload1(b,g);vload2(c,g);vload3(d,g);vload4(e,g);vload5(f,g);}
#define setload7(a,b,c,d,e,f,g,h) {vload(a,h);vload1(b,h);vload2(c,h);vload3(d,h);vload4(e,h);vload5(f,h);vload6(g,h);}

#define squareload(a,b,c,d,e,f) {sload(a,e);sload1(b,e,f);sload2(c,e,f);sload3(d,e,f);}


#define vloadset(a,b,c,d,e) {a=_mm256_set_pd(b,c,d,e);}
#define vallset(a,b) {a = _mm256_set1_pd(b);}


#define vstore(a,b) {_mm256_storeu_pd((&a),b);}
#define vstore1(a,b) {_mm256_storeu_pd((&a)+4,b);}
#define vstore2(a,b) {_mm256_storeu_pd((&a)+8,b);}
#define vstore3(a,b) {_mm256_storeu_pd((&a)+12,b);} 
#define setstore(a,b,c,d,e) {vstore(e,a);vstore1(e,b);vstore2(e,c);vstore3(e,d);}

#define sstore(a,b) {_mm256_storeu_pd((&a),b);}
#define sstore1(a,b,f) {_mm256_storeu_pd((&a)+f,b);}
#define sstore2(a,b,f) {_mm256_storeu_pd((&a)+2*f,b);}
#define sstore3(a,b,f) {_mm256_storeu_pd((&a)+3*f,b);} 
#define squarestore(a,b,c,d,e,f) {sstore(e,a);sstore1(e,b,f);sstore2(e,c,f);sstore3(e,d,f);}

#define leftshuffle(a,b,c,d,e) {a = _mm256_blend_pd(b,c,d); a = _mm256_permute4x64_pd(a,e);} 
#define kwpair(a) _mm256_permute4x64_pd(a,0b01001110) //	a0 a1 a2 a3 --> a2 a3 a0 a1
#define gwpair(a,b) _mm256_blend_pd(a,b,0b1100) //	--> a0 a1 c0 c1
#define kwpermute(a,b) _mm256_permute2f128_pd(a,b,0b00010011)
#define gwpermute(a,b) _mm256_permute2f128_pd(a,b,0b00000010)
#define trans(a,b,c,d)	{vec l=kwpermute(a,c); vec k=kwpermute(c,d); vec g=gwpermute(a,b); vec w=gwpermute(c,d);\
							a=_mm256_unpackhi_pd(k,l); b=_mm256_unpacklo_pd(k,l);\
						    d=_mm256_unpacklo_pd(w,g); c=_mm256_unpackhi_pd(w,g);}
#define permute128_lo(a,b) _mm256_permute2f128_pd(a,b, 0x20) //	
#define permute128_hi(a,b) _mm256_permute2f128_pd(a,b, 0x31) //	
#define trans2d(a,b,c,d) {vec i= permute128_lo(a,c); vec j= permute128_hi(a,c); a = permute128_lo(b,d); c = permute128_hi(b,d);\
                          d = _mm256_unpackhi_pd(j,c); c = _mm256_unpacklo_pd(j,c);  \
                          b = _mm256_unpackhi_pd(i,a); a = _mm256_unpacklo_pd(i,a); }

#define cross_comp_h6(v0,v1,v2,v3,v4,v5){v0=_mm256_add_pd(_mm256_add_pd(v0,v1),v2);\
										v1=_mm256_add_pd(_mm256_add_pd(v1,v2),v3);\
										v2=_mm256_add_pd(_mm256_add_pd(v2,v3),v4);\
										v3=_mm256_add_pd(_mm256_add_pd(v3,v4),v5);}								

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

#define cross_comp_p9vertical(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11){v0=_mm256_add_pd(v8, _mm256_fmadd_pd(_mm256_add_pd(v5,v7), _mm256_set1_pd(2.0),_mm256_fmadd_pd(v6,_mm256_set1_pd(3.0),v4)));\
																	 v1=_mm256_add_pd(v9, _mm256_fmadd_pd(_mm256_add_pd(v6,v8), _mm256_set1_pd(2.0),_mm256_fmadd_pd(v7,_mm256_set1_pd(3.0),v5)));\
																	 v2=_mm256_add_pd(v10,_mm256_fmadd_pd(_mm256_add_pd(v7,v9), _mm256_set1_pd(2.0),_mm256_fmadd_pd(v8,_mm256_set1_pd(3.0),v6)));\
																	 v3=_mm256_add_pd(v11,_mm256_fmadd_pd(_mm256_add_pd(v8,v10),_mm256_set1_pd(2.0),_mm256_fmadd_pd(v9,_mm256_set1_pd(3.0),v7)));}															


#define cross_comp_3thvertical(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11){	v8 =_mm256_add_pd(v4, _mm256_fmadd_pd(_mm256_add_pd(v1,v3), _mm256_set1_pd(2.0),_mm256_fmadd_pd(v2,_mm256_set1_pd(5.0),v0)));\
																		v9 =_mm256_add_pd(v5, _mm256_fmadd_pd(_mm256_add_pd(v2,v4), _mm256_set1_pd(2.0),_mm256_fmadd_pd(v3,_mm256_set1_pd(5.0),v1)));\
																		v10=_mm256_add_pd(v6, _mm256_fmadd_pd(_mm256_add_pd(v3,v5), _mm256_set1_pd(2.0),_mm256_fmadd_pd(v4,_mm256_set1_pd(5.0),v2)));\
																		v11=_mm256_add_pd(v7, _mm256_fmadd_pd(_mm256_add_pd(v4,v6), _mm256_set1_pd(2.0),_mm256_fmadd_pd(v5,_mm256_set1_pd(5.0),v3)));}


#define cross_comp_2thvertical_s1(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9){	v0=_mm256_mul_pd(_mm256_set1_pd(2.0),_mm256_add_pd(_mm256_add_pd(v4,v5),v6));\
																	v1=_mm256_mul_pd(_mm256_set1_pd(2.0),_mm256_add_pd(_mm256_add_pd(v5,v6),v7));\
																	v2=_mm256_mul_pd(_mm256_set1_pd(2.0),_mm256_add_pd(_mm256_add_pd(v6,v7),v8));\
																	v4=_mm256_mul_pd(_mm256_set1_pd(2.0),_mm256_add_pd(_mm256_add_pd(v7,v8),v9));}

#define cross_comp_2thvertical_s2(v0,v1,v2,v3,v4,v5,v6,v7,v8){		v0=_mm256_fmadd_pd(_mm256_set1_pd(2.0),_mm256_add_pd(_mm256_add_pd(v4,v5),v6),v0);\
																	v1=_mm256_fmadd_pd(_mm256_set1_pd(2.0),_mm256_add_pd(_mm256_add_pd(v5,v6),v7),v1);\
																	v2=_mm256_fmadd_pd(_mm256_set1_pd(2.0),_mm256_add_pd(_mm256_add_pd(v6,v7),v8),v2);\
																	v3=_mm256_fmadd_pd(_mm256_set1_pd(2.0),_mm256_add_pd(v7,v8),v3);\
																	v4=v8;}						
										

#define cross_comp_1thvertical(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9){	v0=_mm256_add_pd(_mm256_add_pd(v4,v8),v0);\
																v1=_mm256_add_pd(_mm256_add_pd(v5,v9),v1);\
																v2=_mm256_add_pd(v6,v2);\
																v3=_mm256_add_pd(v7,v3);}															


#define cross_comp_s2p9_ma2(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11) {v0=_mm256_add_pd(v0,_mm256_fmadd_pd(_mm256_add_pd(v5,v7), v6,v8));\
																	v1=_mm256_add_pd(v1,_mm256_fmadd_pd(_mm256_add_pd(v6,v8), v7,v9 ));\
																	v2=_mm256_add_pd(v2,_mm256_fmadd_pd(_mm256_add_pd(v7,v9), v8,v10));\
																	v3=_mm256_add_pd(v3,_mm256_fmadd_pd(_mm256_add_pd(v8,v10),v9,v11));}	

#define cross_comp_s2p9_ma3(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9)             {v6=_mm256_fmadd_pd(v1,v1,_mm256_add_pd(_mm256_fmadd_pd(v0,v0,v0),_mm256_fmadd_pd(v2,v2,v3)) );\
																		v7=_mm256_fmadd_pd(v2,v2,_mm256_add_pd(_mm256_fmadd_pd(v1,v1,v0),_mm256_fmadd_pd(v3,v3,v4)) );\
																		v8=_mm256_fmadd_pd(v3,v3,_mm256_add_pd(_mm256_fmadd_pd(v2,v2,v1),_mm256_fmadd_pd(v4,v4,v5)) );\
																		v9=_mm256_fmadd_pd(v4,v4,_mm256_add_pd(_mm256_fmadd_pd(v3,v3,v2),_mm256_fmadd_pd(v5,v5,v6)) );}					

#define cross_comp_s2p9_m4(v0,v1,v2,v3,kk) {v0=_mm256_mul_pd(v0,kk);\
											v1=_mm256_mul_pd(v1,kk);\
											v2=_mm256_mul_pd(v2,kk);\
											v3=_mm256_mul_pd(v3,kk);}

#define cross_comp_gol2nb(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11) {  v0=_mm256_add_pd(v8,_mm256_fmadd_pd(v7,v7,_mm256_fmadd_pd(v5,v5,_mm256_fmadd_pd(v6,v6,v4))));\
																	v1=_mm256_add_pd(v9,_mm256_fmadd_pd(v8,v8,_mm256_fmadd_pd(v6,v6,_mm256_fmadd_pd(v7,v7,v5))));\
																	v2=_mm256_add_pd(v10,_mm256_fmadd_pd(v9 ,v9 ,_mm256_fmadd_pd(v7,v7,_mm256_fmadd_pd(v8,v8,v6))));\
																	v3=_mm256_add_pd(v11,_mm256_fmadd_pd(v10,v10,_mm256_fmadd_pd(v8,v8,_mm256_fmadd_pd(v9,v9,v7))));}	

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

/// added
#define compute_5p(a, b, c, d, e, ww, kk) {b=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a,b),c),d),e),kk);}
#define compute2_5p(a, b, c, d, e, ww, kk) {a=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a,b),c),d),e),kk);}
#define compute2_up(a, b, c, d, e, f, g, h, i) {e=_mm256_fmadd_pd(a,i,e);f=_mm256_fmadd_pd(b,i,f);g=_mm256_fmadd_pd(c,i,g);h=_mm256_fmadd_pd(d,i,h);}
#define compute2_middle(a, b, c, d, e){a=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a,b),c),d),e);}
#define compute_5p_res(res, a, b, c, d, e, ww, kk) {res=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a,b),c),d),e),kk);}
#define compute_9p(a, b, c, d, e, f, g, h, i, ww, kk) {d=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a,b),c),d),e),f),g),h),i),kk);}
#define compute2_9p(a, b, c, d, e, f, g, h, i, ww, kk) {a=_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a,b),c),d),e),f),g),h),i),kk);}

#define shift_left(a) _mm256_permute4x64_pd(a,0b00111001) //	a0 a1 a2 a3 --> a1 a2 a3 a0
#define shift_right(a) _mm256_permute4x64_pd(a,0b10010011) //	a0 a1 a2 a3 --> a3 a0 a1 a2
#define blend(a,b,c) _mm256_blend_pd(a, b, c)

#define shuffle_head(a,b,c) {c = _mm256_blend_pd(a, b, 8); c = _mm256_permute4x64_pd(c,147);}
#define shuffle_headd(a,b) {b = _mm256_blend_pd(a, b, 8); b = _mm256_permute4x64_pd(b,147);}
#define shuffle_tail(a,b,c) {c = _mm256_blend_pd(a, b, 1); c = _mm256_permute4x64_pd(c,57);}


#if !defined(point)
#define point 9
#endif

// modified
#if point == 5
#define kernel(A) A[(t+1)%2][x][y] = LK * (A[t%2][x][y] + A[t%2][x-1][y] + A[t%2][x+1][y] + A[t%2][x][y-1] + A[t%2][x][y+1])
#define XSLOPE 1
#define YSLOPE 1
#elif point == 9
#define kernel(A) A[(t+1)%2][x][y] = LK * (A[t%2][x][y] + A[t%2][x+1][y] + A[t%2][x-1][y] + A[t%2][x][y+1] + A[t%2][x][y-1] + \
										A[t%2][x+1][y-1] + A[t%2][x-1][y+1] + A[t%2][x-1][y-1] + A[t%2][x+1][y+1]);
#define kernel_9w(A) A[(t+1)%2][x][y] =(12.29 * A[t%2][x][y] + 12.06*A[t%2][x+1][y] + 3.28*A[t%2][x-1][y] + 9.28*A[t%2][x][y+1] + 3.29*A[t%2][x][y-1] + \
										2.07*A[t%2][x+1][y-1] + 12.28*A[t%2][x-1][y+1] + 6.66*A[t%2][x-1][y-1] + 9.99*A[t%2][x+1][y+1]);
#define XSLOPE 1
#define YSLOPE 1
#elif point == 8
#define kernel(A) A[(t+1)%2][x][y] = LK * (A[t%2] + A[t%2][x+1][y] + A[t%2][x-1][y] + A[t%2][x][y+1] + A[t%2][x][y-1] + \
										A[t%2][x+1][y-1] + A[t%2][x-1][y+1] + A[t%2][x-1][y-1] + A[t%2][x+1][y+1]);
#define XSLOPE 1
#define YSLOPE 1
#endif


#define kernel_5p(A) A[(t+1)%2][x][y] = LK * (A[t%2][x][y] + A[t%2][x-1][y] + A[t%2][x+1][y] + A[t%2][x][y-1] + A[t%2][x][y+1])

#define kernel_4p(A) A[(t+1)%2][x][y] = LK * (A[t%2][x-1][y] + A[t%2][x+1][y] + A[t%2][x][y-1] + A[t%2][x][y+1])


#define kernel_9p(A) A[(t+1)%2][x][y] = LK * (A[t%2][x][y] + A[t%2][x+1][y] + A[t%2][x-1][y] + A[t%2][x][y+1] + A[t%2][x][y-1] + \
										A[t%2][x+1][y-1] + A[t%2][x-1][y+1] + A[t%2][x-1][y-1] + A[t%2][x+1][y+1]);


#define kernel_2s9p(A) A[(t+1)%2][x][y] = LK * (A[t%2][x-2][y-2] + 2*A[t%2][x-2][y-1] + 3*A[t%2][x-2][y] + 2*A[t%2][x-2][y+1] + A[t%2][x-2][y+2] + \
											2*A[t%2][x-1][y-2] + 4*A[t%2][x-1][y-1] + 6*A[t%2][x-1][y] + 4*A[t%2][x-1][y+1] + 2*A[t%2][x-1][y+2] + \
											3*A[t%2][x  ][y-2] + 6*A[t%2][x  ][y-1] + 9*A[t%2][x  ][y] + 6*A[t%2][x  ][y+1] + 3*A[t%2][x  ][y+2] + \
											2*A[t%2][x+1][y-2] + 4*A[t%2][x+1][y-1] + 6*A[t%2][x+1][y] + 4*A[t%2][x+1][y+1] + 2*A[t%2][x+1][y+2] + \
											A[t%2][x+2][y-2] + 2*A[t%2][x+2][y-1] + 3*A[t%2][x+2][y] + 2*A[t%2][x+2][y+1] + A[t%2][x+2][y+2]);										

#define kernel_gol(neighbors,A) neighbors = LK * (A[t%2][x][y] + A[t%2][x+1][y] + A[t%2][x-1][y] + A[t%2][x][y+1] + A[t%2][x][y-1] + \
										A[t%2][x+1][y-1] + A[t%2][x-1][y+1] + A[t%2][x-1][y-1] + A[t%2][x+1][y+1]);

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


static inline vec vb2s23(vec a, const vec b){
	avd_t vik,viw; 
	vik.v = a;
	viw.v = b;
	for (int i = 0; i < veclen; i++)
    {
		if((vik.d[i] == 1 && ((viw.d[i] < 2) || (viw.d[i] > 3)))) {
			vik.d[i] = 0;
			continue; 
		}

		if((vik.d[i] == 1 && (viw.d[i] == 2 || viw.d[i] == 3))) {
			vik.d[i] = 1;
			continue; 
		}

		if((vik.d[i] == 0 && viw.d[i] == 3)) {
			vik.d[i] =  1;
			continue; 
		}
    }
	return vik.v;
}

void check_2d(double*** A, double*** B, int NX, int NY, int T);
void allpipe_2d(double*** A, double*** B, int NX, int NY, int T, int Bx, int By, int tb);
void ompp_2d(double*** A, int NX, int NY, int T, int Bx, int By, int tb);

void ompp_2d9w(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void sc_2d1s5p(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void sc_2d1s4p(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void sc_2d1sgol(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void two_steps_2d(double*** A, double*** B, int NX, int NY, int T, int Bx, int By, int tb);
void our_2d1s5p(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void two_tile_2d(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void tile_2d2s(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void our_2d1s9p(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void cross_2d1s5p(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void cross_2d1s9p(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void cross_2d1sgol(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void cross_2d2s5p(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void cross_2d2s9p(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void cross_2d2sgol(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void cross_2d2s9w(double*** A, int NX, int NY, int T, int Bx, int By, int tb);
void test();