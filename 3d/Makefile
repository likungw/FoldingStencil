
CC = gcc
AVX = -mavx2
FMA = -mfma
#OMP = -fopenmp

#CHECK = -DCHECK
#DEBUG = -Ddebug

#icc
# CFLAGS = -O3 $(CHECK) $(DEBUG) -xHost -ansi-alias -ipo -fp-model precise -parallel -qopenmp
#MAC
#CC = /usr/local/opt/llvm/bin/clang
#CFLAGS   = -O3  -march=native -mtune=native -ftree-vectorize    -mavx2 -mfma -I/usr/local/opt/llvm/include -fopenmp
#LDFLAGS = -L/usr/local/opt/llvm/lib
ifeq ($(CC),gcc)
	CFLAGS   = -O3 $(CHECK) -march=native -mtune=native -ftree-vectorize $(OMP) $(AVX) $(FMA) 
else
	CFLAGS = -O3 $(CHECK) $(DEBUG) -xHost -ansi-alias -ipo -fp-model precise -parallel -qopenmp
endif


all: exe_3d 
exe_3d: exe_3d27p

.PHONY: clean 

exe_3d27p: *.c
	$(CC) $(CFLAGS) -Dpoint=7 $^ -o $@ $(LDFLAGS)

clean:
	rm -f   exe_* 

