
#CC = gcc
CC = icc
#AVX = -mavx2
#AVX = -xAVX2
#AVX512 =  -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl 
AVX512 = -xCORE-AVX512
#OMP = -fopenmp
OMP = -qopenmp
#FMA = -mfma

#CHECK = -DCHECK
#DEBUG = -Ddebug

#icc
# CFLAGS = -O3 $(CHECK) $(DEBUG) -xHost -ansi-alias -ipo -fp-model precise -parallel -qopenmp
#MAC
#CC = /usr/local/opt/llvm/bin/clang
#CFLAGS   = -O3  -march=native -mtune=native -ftree-vectorize    -mavx2 -mfma -I/usr/local/opt/llvm/include -fopenmp
#LDFLAGS = -L/usr/local/opt/llvm/lib
ifeq ($(CC),gcc)
	CFLAGS   = -O3 $(CHECK) -march=native -mtune=native -ftree-vectorize $(OMP) $(AVX) $(FMA) $(AVX512)
else
	CFLAGS = -O3 $(CHECK) $(DEBUG) -xHost -ansi-alias -ipo -fp-model precise -parallel $(OMP) $(AVX) $(AVX512)
endif


all: exe_2d 
exe_2d: exe_2d9p 

.PHONY: clean 

exe_2d9p: *.c
	$(CC) $(CFLAGS) -Dpoint=9 $^ -o $@ $(LDFLAGS)

clean:
	rm -f   exe_* 

