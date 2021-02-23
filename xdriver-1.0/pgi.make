#
# Makefile options for C++ compilation with PGI
CC = pgc++
CFLAGS = -O3 -fast
LDFLAGS = $(CFLAGS)
TARGET =
# Compilation and build options for FFTW code
include fftw.make
