
CXX = g++
CXXFLAGS = -O3
CXXACCFLAGS = -fopenacc
CDEBUG = -O0 -g
PGCC = pgc++
PGFLAGS = -O3 -Minfo
#FILES = main.cc interpolation.cc array.cc
FILES = main.cc dev_interpolation.cc

PGACCFLAGS = -acc -ta=tesla,cc60 -Minfo=accel
gcc:
	$(CXX) $(CXXFLAGS) -o interpolation-gcc $(FILES)

gcc-acc:
	$(CXX) $(CXXFLAGS) $(CXXACCFLAGS) -o interpolation-gcc-acc $(FILES)

pgi:
	$(PGCC) $(PGFLAGS) -o interpolation-pgi $(FILES)

pgi-acc:
	$(PGCC) $(PGFLAGS) $(PGACCFLAGS) -o interpolation-pgi-acc $(FILES)

gcc-debug:
	$(CXX) $(CDEBUG) -o interpolation-gcc $(FILES)

pgi-debug:
	$(PGCC) $(CDEBUG) -o interpolation-pgi $(FILES)

clean:
	rm -f xdriver xdriver-gcc xdriver-pgi
