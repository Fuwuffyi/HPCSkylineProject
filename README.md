# Multithreded Skyline Operator
This repository contains an university project for High Performance Computing. It contains both an OpenMP and a Cuda implementation for the skyline operator.
## Compilation
To compile the program you can run the following commands:
- `make clean` removes all executables and object files.
- `make serial` for the serial version of the program.
- `make omp` for the OpenMP version of the program, multithreaded on the CPU.
- `make cuda` for the CUDA version of the program, using CUDA on the GPU.
## Requirements
- *gcc* compiler: to compile the serial and OpenMP versions of the code.
- *OpenMP*: the library used to multithreaded the GPU code
- *nvcc* compiler: to compile the CUDA code.
- *cuda*: the nvidia library used to compile the code for CUDA.
