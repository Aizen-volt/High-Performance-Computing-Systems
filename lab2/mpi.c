#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define TAG_WORK   0
#define TAG_RESULT 1
#define TAG_DONE   2
#define CHUNK_SIZE 1000000  /* ~1MB sieve */

/*
 * Result layout sent from slave to master (7 longs):
 *  [0] count          - twin prime pairs fully resolved inside chunk
 *  [1] start          - starting index of chunk
 *  [2] len            - chunk length
 *  [3] p_first        - is_prime(first value in chunk)
 *  [4] p_second       - is_prime(second value in chunk)
 *  [5] p_second_last  - is_prime(second-to-last value)
 *  [6] p_last         - is_prime(last value)
 */
#define RES_FIELDS 7

static unsigned long *small_primes_up_to(unsigned long limit, long *count) {
  char *mark = (char *)calloc(limit + 1, 1);   /* 0 = prime */
  mark[0] = mark[1] = 1;
  for (unsigned long i = 2; i * i <= limit; i++)
    if (!mark[i])
      for (unsigned long j = i * i; j <= limit; j += i)
        mark[j] = 1;

  /* Count and collect */
  long n = 0;
  for (unsigned long i = 2; i <= limit; i++)
    if (!mark[i]) n++;

  unsigned long *primes = (unsigned long *)malloc(n * sizeof(unsigned long));
  long idx = 0;
  for (unsigned long i = 2; i <= limit; i++)
    if (!mark[i]) primes[idx++] = i;

  free(mark);
  *count = n;
  return primes;
}

static void process_chunk(long start, long len, long result[RES_FIELDS]) {
  unsigned long lo = (unsigned long)(start + 1);
  unsigned long hi = (unsigned long)(start + len);

  unsigned long sqrt_hi = (unsigned long)sqrt((double)hi) + 1;
  long nprimes;
  unsigned long *sprimes = small_primes_up_to(sqrt_hi, &nprimes);

  char *seg = (char *)calloc(len, 1);

  if (lo <= 1 && 1 - lo < (unsigned long)len)
    seg[1 - lo] = 1;

  for (long p = 0; p < nprimes; p++) {
    unsigned long pr = sprimes[p];

    unsigned long first;
    if (pr * pr >= lo)
      first = pr * pr;
    else
      first = lo + ((pr - lo % pr) % pr);
    if (first == pr) first += pr;

    for (unsigned long j = first; j <= hi; j += pr)
      seg[j - lo] = 1;
  }

  free(sprimes);

  long count = 0;
  for (long i = 0; i + 2 < len; i++)
    if (!seg[i] && !seg[i + 2])
      count++;

  /* Step 4: boundary primality */
  result[0] = count;
  result[1] = start;
  result[2] = len;
  result[3] = !seg[0];                             /* first value  */
  result[4] = (len > 1) ? !seg[1] : 0;             /* second       */
  result[5] = (len > 1) ? (int)!seg[len - 2] : 0;  /* second-last  */
  result[6] = !seg[len - 1];                        /* last         */

  free(seg);
}

static int cmp_by_start(const void *a, const void *b) {
  long sa = ((const long *)a)[1];
  long sb = ((const long *)b)[1];
  return (sa > sb) - (sa < sb);
}

int main(int argc, char **argv) {

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  long inputArgument = ins__args.arg;   /* N: array is {1, 2, ..., N} */

  struct timeval ins__tstart, ins__tstop;
  int myrank, nproc;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (!myrank)
    gettimeofday(&ins__tstart, NULL);

  /* ================================================================ */

  long N = inputArgument;
  long total_twin_primes = 0;

  if (myrank == 0) {
    /* ======================== MASTER ======================== */
    int num_slaves = nproc - 1;

    if (num_slaves == 0) {
      long chunk = CHUNK_SIZE;
      if (chunk > N) chunk = N;

      long prev_res[RES_FIELDS] = {0};
      int have_prev = 0;

      for (long pos = 0; pos < N; pos += chunk) {
        long len = (pos + chunk <= N) ? chunk : (N - pos);
        long res[RES_FIELDS];
        process_chunk(pos, len, res);
        total_twin_primes += res[0];

        if (have_prev) {
          if (prev_res[5] && res[3]) total_twin_primes++;
          if (prev_res[6] && res[4]) total_twin_primes++;
        }
        memcpy(prev_res, res, RES_FIELDS * sizeof(long));
        have_prev = 1;
      }
    } else {

      long chunk = CHUNK_SIZE;
      if (chunk > N) chunk = N;
      if (chunk < 4) chunk = 4;

      long next = 0;
      int  active_slaves = 0;

      int  max_chunks = (N / chunk) + num_slaves + 1;
      long *all_results = (long *)malloc(max_chunks * RES_FIELDS * sizeof(long));
      int  n_results = 0;

      /* --- initial non-blocking distribution --- */
      for (int s = 1; s <= num_slaves && next < N; s++) {
        long len = (next + chunk <= N) ? chunk : (N - next);
        long range[2] = { next, len };
        MPI_Request req;
        MPI_Isend(range, 2, MPI_LONG, s, TAG_WORK, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        next += len;
        active_slaves++;
      }

      /* --- dynamic scheduling with non-blocking recv --- */
      long recv_buf[RES_FIELDS];
      MPI_Request recv_req;
      MPI_Irecv(recv_buf, RES_FIELDS, MPI_LONG, MPI_ANY_SOURCE,
                 TAG_RESULT, MPI_COMM_WORLD, &recv_req);

      while (active_slaves > 0) {
        MPI_Status status;
        MPI_Wait(&recv_req, &status);
        int source = status.MPI_SOURCE;

        memcpy(&all_results[n_results * RES_FIELDS], recv_buf,
               RES_FIELDS * sizeof(long));
        n_results++;

        if (next < N) {
          long len = (next + chunk <= N) ? chunk : (N - next);
          long range[2] = { next, len };
          MPI_Request req;
          MPI_Isend(range, 2, MPI_LONG, source, TAG_WORK, MPI_COMM_WORLD, &req);
          MPI_Wait(&req, MPI_STATUS_IGNORE);
          next += len;
        } else {
          long dummy = 0;
          MPI_Request req;
          MPI_Isend(&dummy, 1, MPI_LONG, source, TAG_DONE, MPI_COMM_WORLD, &req);
          MPI_Request_free(&req);
          active_slaves--;
        }

        if (active_slaves > 0)
          MPI_Irecv(recv_buf, RES_FIELDS, MPI_LONG, MPI_ANY_SOURCE,
                     TAG_RESULT, MPI_COMM_WORLD, &recv_req);
      }

      for (int i = 0; i < n_results; i++)
        total_twin_primes += all_results[i * RES_FIELDS + 0];

      qsort(all_results, n_results, RES_FIELDS * sizeof(long), cmp_by_start);

      for (int i = 0; i < n_results - 1; i++) {
        long *A = &all_results[i * RES_FIELDS];       /* earlier chunk */
        long *B = &all_results[(i + 1) * RES_FIELDS]; /* next chunk    */

        /* Pair (A_second_last, A_second_last + 2 = B_first):
         *   A's 2nd-to-last is prime -> A[5]
         *   B's 1st is prime         -> B[3] */
        if (A[5] && B[3])
          total_twin_primes++;

        /* Pair (A_last, A_last + 2 = B_second):
         *   A's last is prime   -> A[6]
         *   B's 2nd is prime    -> B[4] */
        if (A[6] && B[4])
          total_twin_primes++;
      }

      free(all_results);
    }

    printf("Number of twin prime pairs found: %ld\n", total_twin_primes);

  } else {
    /* ======================== SLAVE ======================== */

    while (1) {
      MPI_Status status;
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (status.MPI_TAG == TAG_DONE) {
        long dummy;
        MPI_Recv(&dummy, 1, MPI_LONG, 0, TAG_DONE, MPI_COMM_WORLD, &status);
        break;
      }

      long range[2];
      MPI_Recv(range, 2, MPI_LONG, 0, TAG_WORK, MPI_COMM_WORLD, &status);

      long result[RES_FIELDS];
      process_chunk(range[0], range[1], result);

      MPI_Request req;
      MPI_Isend(result, RES_FIELDS, MPI_LONG, 0, TAG_RESULT, MPI_COMM_WORLD, &req);
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
  }

  /* ================================================================ */

  if (!myrank) {
    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
  }

  MPI_Finalize();
}
