#
# Makefile options for C++ compilation with PGI and OpenACC
CC = pgc++
CFLAGS = -O3 -fast -acc -Minfo=accel
LDFLAGS = -O3 -fast -acc
TARGET = -ta=tesla,cc60 
# Compilation and build options for FFTW code
include fftw.make
# Compilation and build options for CUDA code
include cuda.make
