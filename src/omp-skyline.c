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
char dominates(const float *p, const float *q, const unsigned int D) {
   char strictly_greater = 0;
   for (unsigned int k = 0; k < D; ++k) {
      if (p[k] < q[k]) return 0;
      strictly_greater |= (p[k] > q[k]);
   }
   return strictly_greater;
}

/**
 * Compute the skyline of a set of points.
 * Uses an array of flags (size N) marking whether each point remains in the skyline (1 = in, 0 = out)
 * Returns the number of skyline points.
 */
unsigned int skyline(const points_t *points, char *skyline_flags) {
   const unsigned int D = points->D;
   const unsigned int N = points->N;
   const float *P = points->P;
   unsigned int r = N;
   // Initialize all flags to be in skyline
#pragma omp parallel default(none) shared(skyline_flags, D, N, P) reduction(- : r)
   {
#pragma omp for schedule(static)
      for (unsigned int i = 0; i < N; ++i) {
         skyline_flags[i] = 1;
      }
      // For each point
      for (unsigned int i = 0; i < N; ++i) {
         if (!skyline_flags[i]) continue;
         // Compare against all others (in parallel)
#pragma omp for schedule(guided, 256)
         for (unsigned int j = 0; j < N; ++j) {
            // If point i dominates point j, then point j is removed from the skyline
            if (skyline_flags[j] && dominates(&(P[i * D]), &(P[j * D]), D)) {
               skyline_flags[j] = 0;
               --r;
            }
         }
      }
   }
   return r;
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
#ifdef SINGLE_THREAD
   // Set threads to one if single threaded mode
   omp_set_num_threads(1);
#endif
   if (argc != 1) {
      fprintf(stderr, "Usage: %s < input_file > output_file\n", argv[0]);
      return EXIT_FAILURE;
   }
   // Read the input from stdin
   read_input(&points);
   // Setup the flag array
   char *skyline_flags = (char *)malloc(points.N * sizeof(*skyline_flags));
   assert(skyline_flags);
   // Run and time the algorithm
   const double tstart = hpc_gettime();
   const unsigned int r = skyline(&points, skyline_flags);
   const double elapsed = hpc_gettime() - tstart;
   // Print the results
   print_skyline(&points, skyline_flags, r);
   fprintf(stderr, "\n\t%u points\n", points.N);
   fprintf(stderr, "\t%u dimensions\n", points.D);
   fprintf(stderr, "\t%u points in skyline\n\n", r);
   fprintf(stderr, "Execution time (s) %f\n", elapsed);
   free_points(&points);
   free(skyline_flags);
   return EXIT_SUCCESS;
}
