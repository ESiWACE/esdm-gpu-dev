all:
	c++ -O3 -c xdriver.cc
	pgc++ -O3 -acc -ta=tesla,cc60 -Minfo=accel -c kernels.cc
	pgc++ -O3 -acc -o xdriver xdriver.o kernels.o

clean:
	rm -f *.o xdriver
