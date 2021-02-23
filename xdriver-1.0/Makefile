TARGET = none

CC=
CFLAGS=
LDFLAGS=
LIBDIR=
FFTLIBS=
CUCC=
CUFLAGS=

OBJS = xdriver.o test_fft1d.o test_fft2d.o test_varrayMinMaxMV.o test_vert_interp_lev.o \
       array.o fft.o interpolation.o \
       dev_array.o dev_interpolation.o
INCS = compare.h functions.h xd_timer.h

include $(TARGET).make

gcc:
	make build TARGET=$@

gcc-acc:
	make build TARGET=$@

gcc-acc-nocuda:
	make build TARGET=$@

gcc10-acc:
	make build TARGET=$@

pgi:
	make build TARGET=$@

pgi-acc:
	make build TARGET=$@

gcc-debug:
	make build TARGET=$@

pgi-debug:
	make build TARGET=$@

pgi-acc-debug:
	make build TARGET=$@

%.o: %.cc
	$(CC) -c $(CFLAGS) -o $@ $<

%.o: %.cu
	$(CUCC) -c $(CUFLAGS) -o $@ $<

build:	$(OBJS)
	$(CC) $(LDFLAGS) -o xdriver-$(TARGET) $(OBJS) $(LIBDIR) $(LIBS)

clean:
	rm -f *.o $(OBJS) xdriver-*

# Dependencies
array.o:			compare.h
array_noisnan.o:		compare.h
dev_array.o:			dev_compare.h dev_array.h
dev_interpolation.o:		dev_interpolation.h
interpolation.o:		compare.h
test_fft1d.o:			xd_timer.h functions.h
test_fft2d.o:			xd_timer.h functions.h
test_interp_main.o:		dev_interpolation.h
test_varrayMinMaxMV.o:		xd_timer.h functions.h compare.h
test_varrayMinMaxMV_noisnan.o:	xd_timer.h functions.h compare.h
test_vert_interp_lev.o:		xd_timer.h functions.h compare.h dev_interpolation.h
xdriver.cc:			compare.h functions.h

