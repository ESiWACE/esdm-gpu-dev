#
# Makefile options for C++ compilation with PGI
CC = pgc++
CFLAGS = -O0 -g
LDFLAGS = $(CFLAGS)
TARGET =
# Compilation and build options for FFTW code
include fftw.make
