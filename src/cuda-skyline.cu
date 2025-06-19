/**
 * Cognome e nome:   Palazzini Luca
 * Codice matricola: 0001070910
 */

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "hpc.h"

#define GLOBAL_BLKDIM 32
#define SHARED_BLKDIM 512
#define REDUCTION_BLKDIM 512

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
__device__ inline char dominates(const float *p, const float *q, const unsigned int D) {
   char strictly_greater = 0;
   for (unsigned int k = 0; k < D; ++k) {
      if (p[k] < q[k]) return 0;
      strictly_greater |= (p[k] > q[k]);
   }
   return strictly_greater;
}

/**
 * Compute the skyline of a set of points, using shared memory to locally store points and reduce global memory overhead.
 * Uses an array of skyline_flags (size N) marking whether each point remains in the skyline (1 = in, 0 = out)
 */
__global__ void shared_skyline(const float *points_data, char *skyline_flags, const unsigned int N, const unsigned int D) {
#define SHMEM_FLOATS 12288
   __shared__ float shmem[SHMEM_FLOATS];
   const unsigned int tile_size = SHMEM_FLOATS / D;
   const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   const float *pi = points_data + i * D;
   char in_skyline = 1;
   // Loop over tiles
   for (unsigned int tile_start = 0; tile_start < N; tile_start += tile_size) {
      // Load tile into shared memory
      const unsigned int curr_tile_size = min(tile_size, N - tile_start);
      for (unsigned int idx = threadIdx.x; idx < curr_tile_size; idx += blockDim.x) {
         const unsigned int j = tile_start + idx;
         if (j < N) {
            for (unsigned int k = 0; k < D; ++k) {
               shmem[idx * D + k] = points_data[j * D + k];
            }
         }
      }
      __syncthreads();
      // Compare with points in the tile
      for (unsigned int tj = 0; tj < curr_tile_size && i < N; ++tj) {
         const unsigned int global_j = tile_start + tj;
         if (!skyline_flags[global_j]) continue;
         // Check if the other point dominates the current one
         if (dominates(shmem + tj * D, pi, D)) {
            in_skyline = 0;
            break;
         }
      }
      __syncthreads();
   }
   if (i < N) skyline_flags[i] = in_skyline;
}

/**
 * Compute the skyline of a set of points, using exclusively global memory.
 * Uses an array of skyline_flags (size N) marking whether each point remains in the skyline (1 = in, 0 = out)
 */
__global__ void global_skyline(const float *points_data, char *skyline_flags, const unsigned int N, const unsigned int D) {
   const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= N) return;
   const float *pi = points_data + i * D;
   char in_skyline = 1;
   // Loop over all other tiles in global memory
   for (unsigned int j = 0; j < N; ++j) {
      if (!skyline_flags[j]) continue;
      const float *pj = points_data + j * D;
      // Check if the other point dominates the current one
      if (dominates(pj, pi, D)) {
         in_skyline = 0;
         break;
      }
   }
   skyline_flags[i] = in_skyline;
}

/**
 * Calculates the amount of points on the skyline using a reduction kernel.
 */
__global__ void r_reduction(char *skyline_flags, unsigned int *r, const unsigned int N) {
   // Setup shared memory
   __shared__ unsigned int shmem[REDUCTION_BLKDIM];
   // Thread and global indices
   const unsigned int tid = threadIdx.x;
   const unsigned int gindex = blockIdx.x * blockDim.x + threadIdx.x;
   // Setup initial shared memory
   shmem[tid] = (gindex < N) ? skyline_flags[gindex] : 0;
   __syncthreads();
   // Tree reduction
   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
         shmem[tid] += shmem[tid + s];
      }
      __syncthreads();
   }
   // Atomic add block result to global counter
   if (tid == 0) {
      atomicAdd(r, shmem[0]);
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
   const unsigned int point_bytes = N * D * sizeof(float);
   const unsigned int flag_bytes = N * sizeof(char);
   // Allocate local flags
   char *h_skyline_flags = (char *)malloc(points.N * sizeof(*h_skyline_flags));
   assert(h_skyline_flags);
   unsigned int h_r = 0;
   float *d_P;
   char *d_skyline_flags;
   unsigned int *d_r;
   // Calculate blocks
   const unsigned int shared_blocks = (N + SHARED_BLKDIM - 1) / SHARED_BLKDIM;
   const unsigned int global_blocks = (N + GLOBAL_BLKDIM - 1) / GLOBAL_BLKDIM;
   const unsigned int reduction_blocks = (N + REDUCTION_BLKDIM - 1) / REDUCTION_BLKDIM;
   // Allocate data on the gpu
   cudaSafeCall(cudaMalloc(&d_P, point_bytes));
   cudaSafeCall(cudaMalloc(&d_skyline_flags, flag_bytes));
   cudaSafeCall(cudaMalloc(&d_r, sizeof(unsigned int)));
   cudaSafeCall(cudaMemcpy(d_r, &h_r, sizeof(unsigned int), cudaMemcpyHostToDevice));
   cudaSafeCall(cudaMemcpy(d_P, points.P, point_bytes, cudaMemcpyHostToDevice));
   cudaSafeCall(cudaMemset(d_skyline_flags, 1, flag_bytes));
   // Run the skyline algorithm
   const double tstart = hpc_gettime();
   if (D >= 1536) {
      global_skyline<<<global_blocks, GLOBAL_BLKDIM>>>(d_P, d_skyline_flags, N, D);
   } else {
      shared_skyline<<<shared_blocks, SHARED_BLKDIM>>>(d_P, d_skyline_flags, N, D);
   }
   // Reduction to get r count
   r_reduction<<<reduction_blocks, REDUCTION_BLKDIM>>>(d_skyline_flags, d_r, N);
   cudaSafeCall(cudaDeviceSynchronize());
   const double elapsed = hpc_gettime() - tstart;
   // Copy data from host
   cudaSafeCall(cudaMemcpy(h_skyline_flags, d_skyline_flags, flag_bytes, cudaMemcpyDeviceToHost));
   cudaSafeCall(cudaMemcpy(&h_r, d_r, sizeof(unsigned int), cudaMemcpyDeviceToHost));
   // Print results
   print_skyline(&points, h_skyline_flags, h_r);
   fprintf(stderr, "\n\t%u points\n", points.N);
   fprintf(stderr, "\t%u dimensions\n", points.D);
   fprintf(stderr, "\t%u points in skyline\n\n", h_r);
   fprintf(stderr, "Execution time (s) %f\n", elapsed);
   // Free and exit
   cudaSafeCall(cudaFree(d_r));
   free_points(&points);
   cudaSafeCall(cudaFree(d_P));
   free(h_skyline_flags);
   cudaSafeCall(cudaFree(d_skyline_flags));
   return EXIT_SUCCESS;
}
