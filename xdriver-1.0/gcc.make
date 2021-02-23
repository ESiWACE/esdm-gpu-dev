#
# Makefile options for C++ compilation with gcc
CC = g++
CFLAGS = -O3
LDFLAGS = $(CFLAGS)
TARGET =
# Compilation and build options for FFTW code
include fftw.make
