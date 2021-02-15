CXX = g++
CXXFLAGS = -O3
CXXACCFLAGS = -fopenacc
CDEBUG = -O0 -g

GCC10CC = g++-10
GCC10FLAGS = -O3
GCC10ACCFLAGS = -fopenacc

PGCC = pgc++
PGFLAGS = -O3 -fast
PGACCFLAGS = -acc -ta=tesla,cc60 -Minfo=accel
PGOBJS = cufft.o
PGLIBS = -lcuda -lcudart -lcufft
PGLIBDIR = -L/usr/local/cuda/lib64

SRCS = xdriver.cc test_fft.cc test_varrayMinMaxMV.cc test_vert_interp_lev.cc \
       array.cc fft.cc interpolation.cc \
       dev_array.cc dev_interpolation.cc
OBJS = xdriver.o test_fft.o test_varrayMinMaxMV.o test_vert_interp_lev.o \
       array.o fft.o interpolation.o \
       dev_array.o dev_interpolation.o
INCS = compare.h
LIBS = -lfftw3 -lfftw3f
LIBDIR = -L/usr/local/lib

notarget:
	make gcc

cufft.o:	cufft.cu
	nvcc -c cufft.cu

gcc:	$(SRCS) $(INCS)
	$(CXX) $(CXXFLAGS) -o xdriver-gcc $(SRCS) $(LIBDIR) $(LIBS)

gcc-acc:	$(SRCS) $(INCS)
	$(CXX) $(CXXFLAGS) $(CXXACCFLAGS) -o xdriver-gcc-acc $(SRCS) $(LIBDIR) $(LIBS)

gcc10-acc:	$(SRCS) $(INCS)
	$(GCC10CC) $(GCC10FLAGS) $(GCC10ACCFLAGS) -o xdriver-gcc10-acc $(SRCS) $(LIBDIR) $(LIBS)

pgi:	$(SRCS) $(INCS)
	$(PGCC) $(PGFLAGS) -o xdriver-pgi $(SRCS) $(LIBDIR) $(LIBS)

pgi-acc:	$(SRCS) $(INCS) $(PGOBJS)
	$(PGCC) $(PGFLAGS) $(PGACCFLAGS) -o xdriver-pgi-acc $(SRCS) $(PGOBJS) $(PGLIBDIR) $(PGLIBS) $(LIBDIR) $(LIBS)

gcc-debug:	$(SRCS) $(INCS)
	$(CXX) $(CDEBUG) -o xdriver-gcc-debug $(SRCS) $(LIBDIR) $(LIBS)

pgi-debug:	$(SRCS) $(INCS)
	$(PGCC) $(CDEBUG) -o xdriver-pgi-debug $(SRCS) $(LIBDIR) $(LIBS)

pgi-acc-debug:	$(SRCS) $(INCS) $(PGOBJS)
	$(PGCC) $(CDEBUG) $(PGACCFLAGS) -o xdriver-pgi-acc-debug $(SRCS) $(PGOBJS) $(PGLIBDIR) $(PGLIBS) $(LIBDIR) $(LIBS)

clean:
	rm -f *.o xdriver xdriver-gcc* xdriver-pgi*
