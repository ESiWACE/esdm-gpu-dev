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

SRCS = xdriver.cc test_varrayMinMaxMV.cc test_vert_interp_lev.cc \
       array.cc interpolation.cc \
       dev_array.cc dev_interpolation.cc
INCS = compare.h

gcc:	$(SRCS) $(INCS)
	$(CXX) $(CXXFLAGS) -o xdriver-gcc $(SRCS)

gcc-acc:	$(SRCS) $(INCS)
	$(CXX) $(CXXFLAGS) $(CXXACCFLAGS) -o xdriver-gcc-acc $(SRCS)

gcc10-acc:	$(SRCS) $(INCS)
	$(GCC10CC) $(GCC10FLAGS) $(GCC10ACCFLAGS) -o xdriver-gcc10-acc $(SRCS)

pgi:	$(SRCS) $(INCS)
	$(PGCC) $(PGFLAGS) -o xdriver-pgi $(SRCS)

pgi-acc:	$(SRCS) $(INCS)
	$(PGCC) $(PGFLAGS) $(PGACCFLAGS) -o xdriver-pgi-acc $(SRCS)

gcc-debug:	$(SRCS) $(INCS)
	$(CXX) $(CDEBUG) -o xdriver-gcc-debug $(SRCS)

pgi-debug:	$(SRCS) $(INCS)
	$(PGCC) $(CDEBUG) -o xdriver-pgi-debug $(SRCS)

clean:
	rm -f xdriver xdriver-gcc xdriver-pgi
