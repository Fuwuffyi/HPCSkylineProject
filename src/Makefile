## Cognome e nome:  	Palazzini Luca
## Codice matricola: 0001070910

## Targets
##
## make: 			Builds all targets.
## make serial: 	Builds the code using single threaded OpenMP
## make omp:		Builds the code using OpenMP
## make cuda:		Builds the code using Cuda

CC := gcc
NVCC := nvcc

CFLAGS := -std=c99 -Wall -Wpedantic -O2
NVCFLAGS := -Wno-deprecated-gpu-targets -O2

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
	-$(RM) -f serial-*.o omp-*.o cuda-*.o cuda-skyline.exp cuda-skyline.lib serial-skyline omp-skyline cuda-skyline serial-skyline.exe omp-skyline.exe cuda-skyline.exe *~

serial: CFLAGS += -fopenmp -DSINGLE_THREAD=1
serial: $(SERIAL_OBJS)
	$(CC) $(CFLAGS) $(SERIAL_OBJS) -o serial-skyline

omp: CFLAGS += -fopenmp
omp: $(OMP_OBJS)
	$(CC) $(CFLAGS) $(OMP_OBJS) -o omp-skyline

cuda: $(CUDA_OBJS)
	$(NVCC) $(NVCFLAGS) $(CUDA_OBJS) -o cuda-skyline

serial-%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

omp-%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

cuda-%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c $< -o $@
