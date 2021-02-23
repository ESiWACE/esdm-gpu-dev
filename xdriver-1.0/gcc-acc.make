#
# Makefile options for C++ compilation with gcc and OpenACC
CC = g++
CFLAGS = -O3 -fopenacc
LDFLAGS = $(CFLAGS)
TARGET =
# Compilation and build options for FFTW code
include fftw.make
# Compilation and build options for CUDA code
include cuda.make
