#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "hpc.h"

/**
 * Point data structure.
 */
typedef struct {
   float *P;       /* pointer to flat array of coordinates (size N * D) */
   unsigned int N; /* number of points */
   unsigned int D; /* dimension of all points */
} points_t;

/**
 * Read dimension, number of points, and coordinates from stdin.
 */
void read_input(points_t *points) {
   char buf[1024];
   unsigned int N, D;
   float *P;
   // Read the dimension
   if (1 != scanf("%u", &D)) {
      fprintf(stderr, "FATAL: can not read the dimension\n");
      exit(EXIT_FAILURE);
   }
   assert(D >= 2);
   // Skip rest of line
   if (NULL == fgets(buf, sizeof(buf), stdin)) {
      fprintf(stderr, "FATAL: can not read the first line\n");
      exit(EXIT_FAILURE);
   }
   // Read point count
   if (1 != scanf("%u", &N)) {
      fprintf(stderr, "FATAL: can not read the number of points\n");
      exit(EXIT_FAILURE);
   }
   // Allocate point array
   P = (float *)malloc(D * N * sizeof(*P));
   assert(P);
   // Read all points
   for (unsigned int i = 0; i < N; ++i) {
      for (unsigned int k = 0; k < D; ++k) {
         if (1 != scanf("%f", &(P[i * D + k]))) {
            fprintf(stderr, "FATAL: failed to get coordinate %u of point %u\n", k, i);
            exit(EXIT_FAILURE);
         }
      }
   }
   points->P = P;
   points->N = N;
   points->D = D;
}

/**
 * Frees points memory.
 */
void free_points(points_t *points) {
   free(points->P);
   points->P = NULL;
   points->N = points->D = 0;
}

/**
 * Check if point p dominates point q in all dimensions.
 * Returns 1 if p >= q in every coordinate and p > q in at least one.
 * Returns 0 otherwise.
 */
__device__ char dominates(const float *p, const float *q, const unsigned int D) {
   char strictly_greater = 0;
   for (unsigned int k = 0; k < D; ++k) {
      if (p[k] < q[k]) return 0;
      strictly_greater |= (p[k] > q[k]);
   }
   return strictly_greater;
}

/**
 * Compute the skyline of a set of points.
 * Uses an array of skyline_flags (size N) marking whether each point remains in the skyline (1 = in, 0 = out)
 * Returns the number of skyline points.
 */
__global__ void skyline(const float *points_data, char *skyline_flags, const unsigned int N, const unsigned int D) {
   const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= N) return;
   const float *pi = points_data + i * D;
   for (unsigned int j = 0; j < N; ++j) {
      if (!skyline_flags[j]) continue;
      const float *pj = points_data + j * D;
      if (dominates(pi, pj, D)) {
         skyline_flags[j] = 0;
      }
   }
}

/**
 * Output the skyline to stdout in the expected format.
 * First prints D, then r (number of skyline points), then each skyline point.
 */
void print_skyline(const points_t *points, const char *skyline_flags, const unsigned int r) {
   const unsigned int D = points->D;
   const unsigned int N = points->N;
   const float *P = points->P;
   // Print dimension and skyline size
   printf("%u\n", D);
   printf("%u\n", r);
   // Print each skyline point's coordinates
   for (unsigned int i = 0; i < N; ++i) {
      if (!skyline_flags[i]) continue;
      for (unsigned int k = 0; k < D; ++k) {
         printf("%f ", P[i * D + k]);
      }
      printf("\n");
   }
}

int main(int argc, char *argv[]) {
   points_t points;
   if (argc != 1) {
      fprintf(stderr, "Usage: %s < input_file > output_file\n", argv[0]);
      return EXIT_FAILURE;
   }
   // Read in data
   read_input(&points);
   const unsigned int N = points.N;
   const unsigned int D = points.D;
   const size_t point_bytes = (size_t)N * D * sizeof(float);
   const size_t flag_bytes = (size_t)N * sizeof(char);
   // Allocate local flags
   char *h_skyline_flags = (char *)malloc(points.N * sizeof(*h_skyline_flags));
   assert(h_skyline_flags);
   // Allocate data on the gpu
   float *d_P;
   char *d_skyline_flags;
   cudaSafeCall(cudaMalloc(&d_P, point_bytes));
   cudaSafeCall(cudaMalloc(&d_skyline_flags, flag_bytes));
   cudaSafeCall(cudaMemcpy(d_P, points.P, point_bytes, cudaMemcpyHostToDevice));
   // Calculate blocks
   const unsigned int threads_per_block = 256;
   const unsigned int blocks = (N + threads_per_block - 1) / threads_per_block;
   // Run the skyline algorithm
   const double tstart = hpc_gettime();
   cudaSafeCall(cudaMemset(d_skyline_flags, 1, flag_bytes));
   skyline<<<blocks, threads_per_block>>>(d_P, d_skyline_flags, N, D);
   cudaSafeCall(cudaDeviceSynchronize());
   // Copy data from host
   cudaSafeCall(cudaMemcpy(h_skyline_flags, d_skyline_flags, flag_bytes, cudaMemcpyDeviceToHost));
   // TODO: Implement this within the kernel?
   unsigned int r = 0;
   for (unsigned int i = 0; i < N; ++i) {
      if (h_skyline_flags[i] == 1) {
         ++r;
      }
   }
   const double elapsed = hpc_gettime() - tstart;
   // Print results
   print_skyline(&points, h_skyline_flags, r);
   fprintf(stderr, "\n\t%u points\n", points.N);
   fprintf(stderr, "\t%u dimensions\n", points.D);
   fprintf(stderr, "\t%u points in skyline\n\n", r);
   fprintf(stderr, "Execution time (s) %f\n", elapsed);
   // Free and exit
   free_points(&points);
   cudaSafeCall(cudaFree(d_P));
   free(h_skyline_flags);
   cudaSafeCall(cudaFree(d_skyline_flags));
   return EXIT_SUCCESS;
}
