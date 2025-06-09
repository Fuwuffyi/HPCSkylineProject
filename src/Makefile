CC := gcc
NVCC := nvcc

CFLAGS := -std=c99 -Wall -Wpedantic -O2 -D_XOPEN_SOURCE=600
NVCFLAGS := -Wno-deprecated-gpu-targets
LDLIBS := -lm
NVLDLIBS := -lm

C_SOURCES := $(wildcard *.c)
CU_SOURCES := $(wildcard *.cu)

SERIAL_OBJS := $(patsubst %.c,serial-%.o,$(C_SOURCES))
OMP_OBJS := $(patsubst %.c,omp-%.o,$(C_SOURCES))
CUDA_OBJS := $(patsubst %.cu,cuda-%.o,$(CU_SOURCES))

ifeq ($(OS),Windows_NT)
	RM := del /Q
else
	RM := rm -f
endif

.PHONY: all clean serial omp cuda

all: serial omp cuda

clean:
	-$(RM) -f serial-*.o omp-*.o cuda-*.o serial omp cuda serial.exe omp.exe cuda.exe *~

serial: CFLAGS += -fopenmp -DTHREADS=1
serial: $(SERIAL_OBJS)
	$(CC) $(CFLAGS) $(SERIAL_OBJS) -o $@ $(LDLIBS)

omp: CFLAGS += -fopenmp
omp: $(OMP_OBJS)
	$(CC) $(CFLAGS) $(OMP_OBJS) -o $@ $(LDLIBS)

cuda: $(CUDA_OBJS)
	$(NVCC) $(NVCFLAGS) $(CUDA_OBJS) -o $@ $(NVLDLIBS)

serial-%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

omp-%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

cuda-%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c $< -o $@
