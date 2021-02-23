# Compilation and build options for CUDA code
CUCC = nvcc
CUFLAGS = -O3
CFLAGS += -DUSE_CUDA
OBJS += cufft.o
LIBS += -lcuda -lcudart -lcufft
LIBDIR += -L/usr/local/cuda/lib64
