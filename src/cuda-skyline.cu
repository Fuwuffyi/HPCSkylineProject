#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "hpc.h"

typedef struct {
   float *P; /* coordinates P[i][j] of point i               */
   int N;    /* Number of points (rows of matrix P)          */
   int D;    /* Number of dimensions (columns of matrix P)   */
} points_t;

/**
 * Read input from stdin. Input format is:
 *
 * d [other ignored stuff]
 * N
 * p0,0 p0,1 ... p0,d-1
 * p1,0 p1,1 ... p1,d-1
 * ...
 * pn-1,0 pn-1,1 ... pn-1,d-1
 *
 */
void read_input(points_t *points) {
   char buf[1024];
   int N, D;
   float *P;

   if (1 != scanf("%d", &D)) {
      fprintf(stderr, "FATAL: can not read the dimension\n");
      exit(EXIT_FAILURE);
   }
   assert(D >= 2);
   if (NULL == fgets(buf, sizeof(buf), stdin)) { /* ignore rest of the line */
      fprintf(stderr, "FATAL: can not read the first line\n");
      exit(EXIT_FAILURE);
   }
   if (1 != scanf("%d", &N)) {
      fprintf(stderr, "FATAL: can not read the number of points\n");
      exit(EXIT_FAILURE);
   }
   P = (float *)malloc(D * N * sizeof(*P));
   assert(P);
   for (int i = 0; i < N; ++i) {
      for (int k = 0; k < D; ++k) {
         if (1 != scanf("%f", &(P[i * D + k]))) {
            fprintf(stderr, "FATAL: failed to get coordinate %d of point %d\n", k, i);
            exit(EXIT_FAILURE);
         }
      }
   }
   points->P = P;
   points->N = N;
   points->D = D;
}

void free_points(points_t *points) {
   free(points->P);
   points->P = NULL;
   points->N = points->D = -1;
}

/* Returns 1 iff |p| dominates |q| */
__device__ char dominates(const float *p, const float *q, const int D) {
   char strictly_greater = 0;
   for (int k = 0; k < D; ++k) {
      if (p[k] < q[k]) return 0;
      strictly_greater |= (p[k] > q[k]);
   }
   return strictly_greater;
}

/**
 * Compute the skyline of `points`. At the end, `s[i] == 1` iff point
 * `i` belongs to the skyline. The function returns the number `r` of
 * points that belongs to the skyline. The caller is responsible for
 * allocating the array `s` of length at least `points->N`.
 */
__global__ void skyline(const float *points_data, char *flags, const int N, const int D) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= N) return;
   if (!flags[i]) return;
   const float *pi = points_data + i * D;
   for (int j = 0; j < N; ++j) {
      if (!flags[j]) continue;
      const float *pj = points_data + j * D;
      if (dominates(pi, pj, D)) {
         flags[j] = 0;
      }
   }
}

/**
 * Print the coordinates of points belonging to the skyline `s` to
 * standard ouptut. `s[i] == 1` iff point `i` belongs to the skyline.
 * The output format is the same as the input format, so that this
 * program can process its own output.
 */
void print_skyline(const points_t *points, const char *s, int r) {
   const int D = points->D;
   const int N = points->N;
   const float *P = points->P;

   printf("%d\n", D);
   printf("%d\n", r);
   for (int i = 0; i < N; ++i) {
      if (s[i]) {
         for (int k = 0; k < D; ++k) {
            printf("%f ", P[i * D + k]);
         }
         printf("\n");
      }
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
   const int N = points.N;
   const int D = points.D;
   const size_t point_bytes = (size_t)N * D * sizeof(float);
   const size_t flag_bytes = (size_t)N * sizeof(char);
   // Allocate local flags
   char *h_flags = (char *)malloc(points.N * sizeof(*h_flags));
   assert(h_flags);
   // Allocate data on the gpu
   float *d_P;
   char *d_flags;
   cudaSafeCall(cudaMalloc(&d_P, point_bytes));
   cudaSafeCall(cudaMalloc(&d_flags, flag_bytes));
   cudaSafeCall(cudaMemcpy(d_P, points.P, point_bytes, cudaMemcpyHostToDevice));
   // Calculate blocks
   const int threads_per_block = 256;
   const int total = N * N;
   const int blocks = (total + threads_per_block - 1) / threads_per_block;
   // Run the skyline algorithm
   const double tstart = hpc_gettime();
   // TODO: Implement this within the kernel?
   for (int i = 0; i < N; ++i) {
      h_flags[i] = 1;
   }
   cudaSafeCall(cudaMemcpy(d_flags, h_flags, flag_bytes, cudaMemcpyHostToDevice));
   skyline<<<blocks, threads_per_block>>>(d_P, d_flags, N, D);
   cudaSafeCall(cudaDeviceSynchronize());
   // Copy data from host
   cudaSafeCall(cudaMemcpy(h_flags, d_flags, flag_bytes, cudaMemcpyDeviceToHost));
   // TODO: Implement this within the kernel?
   int r = 0;
   for (int i = 0; i < N; ++i) {
      if (h_flags[i] == 1) {
         ++r;
      }
   }
   const double elapsed = hpc_gettime() - tstart;
   // Print results
   print_skyline(&points, h_flags, r);
   fprintf(stderr, "\n\t%d points\n", points.N);
   fprintf(stderr, "\t%d dimensions\n", points.D);
   fprintf(stderr, "\t%d points in skyline\n\n", r);
   fprintf(stderr, "Execution time (s) %f\n", elapsed);
   // Free and exit
   free_points(&points);
   free(h_flags);
   return EXIT_SUCCESS;
}
