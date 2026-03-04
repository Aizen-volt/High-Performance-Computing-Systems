#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

static void parseArguments(int argc, char **argv, long long *first, long long *last) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") != 0 && strcmp(argv[i], "-v") != 0 &&
            strcmp(argv[i], "--extra-verbose") != 0 && strcmp(argv[i], "-vv") != 0) {
            if (*first == -1) {
                *first = atoll(argv[i]);
            } else if (*last == -1) {
                *last = atoll(argv[i]);
            }
        }
    }
}

static void generateBasePrimes(long long limit, long long **base_primes, long long *base_count) {
    char *base_sieve = calloc(limit + 1, 1);
    
    for (long long p = 2; p * p <= limit; p++) {
        if (!base_sieve[p]) {
            for (long long i = p * p; i <= limit; i += p) {
                base_sieve[i] = 1;
            }
        }
    }

    *base_count = 0;
    for (long long p = 2; p <= limit; p++) {
        if (!base_sieve[p]) {
            (*base_count)++;
        }
    }

    *base_primes = malloc(*base_count * sizeof(long long));
    long long idx = 0;
    for (long long p = 2; p <= limit; p++) {
        if (!base_sieve[p]) {
            (*base_primes)[idx++] = p;
        }
    }
    free(base_sieve);
}

static void calculateLocalRange(long long first, long long last, int myrank, int proccount, 
                                long long *local_first, long long *local_last) {
    long long range_size = last - first + 1;
    long long block_size = range_size / proccount;
    long long remainder = range_size % proccount;

    *local_first = first + myrank * block_size + (myrank < remainder ? myrank : remainder);
    *local_last = *local_first + block_size - 1 + (myrank < remainder ? 1 : 0);
}

static void runSegmentedSieve(long long local_first, long long local_last, 
                              long long *base_primes, long long base_count, 
                              long long *local_primes, long long *local_twins, 
                              long long *local_first_prime, long long *local_last_prime) {
    long long SEGMENT_SIZE = 262144;
    char *sieve = malloc(SEGMENT_SIZE);
    long long prev_prime = -1;

    *local_primes = 0;
    *local_twins = 0;
    *local_first_prime = -1;
    *local_last_prime = -1;

    for (long long low = local_first; low <= local_last; low += SEGMENT_SIZE) {
        long long high = low + SEGMENT_SIZE - 1;
        if (high > local_last) {
            high = local_last;
        }
        long long current_seg_size = high - low + 1;

        memset(sieve, 0, current_seg_size);

        for (long long i = 0; i < base_count; i++) {
            long long p = base_primes[i];
            long long start_idx = (low + p - 1) / p * p;
            if (start_idx < p * p) {
                start_idx = p * p;
            }

            long long start_offset = start_idx - low;
            for (long long j = start_offset; j < current_seg_size; j += p) {
                sieve[j] = 1;
            }
        }

        if (low == 0) {
            sieve[0] = 1;
            if (current_seg_size > 1) sieve[1] = 1;
        } else if (low == 1) {
            sieve[0] = 1;
        }

        for (long long i = 0; i < current_seg_size; i++) {
            if (sieve[i] == 0) {
                long long current_prime = low + i;
                (*local_primes)++;
                
                if (*local_first_prime == -1) {
                    *local_first_prime = current_prime;
                }
                
                if (prev_prime != -1 && current_prime - prev_prime == 2) {
                    (*local_twins)++;
                }
                
                prev_prime = current_prime;
                *local_last_prime = current_prime;
            }
        }
    }
    free(sieve);
}

static void communicateBoundaries(int myrank, int proccount, 
                                  long long local_first_prime, long long local_last_prime, 
                                  long long *local_twins) {
    long long next_first = -1;
    
    if (myrank < proccount - 1) {
        MPI_Recv(&next_first, 1, MPI_LONG_LONG, myrank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (myrank > 0) {
        MPI_Send(&local_first_prime, 1, MPI_LONG_LONG, myrank - 1, 0, MPI_COMM_WORLD);
    }

    if (myrank < proccount - 1 && next_first != -1 && local_last_prime != -1) {
        if (next_first - local_last_prime == 2) {
            (*local_twins)++;
        }
    }
}

static void printResults(long long first, long long last, int proccount, 
                         long long total_primes, long long total_twins) {
    printf("Range            : [%lld, %lld]\n", first, last);
    printf("Processes        : %d\n", proccount);
    printf("Primes found     : %lld\n", total_primes);
    printf("Twin prime pairs : %lld\n", total_twins);
}

int main(int argc, char **argv) {
    long long first = -1, last = -1;
    int myrank, proccount;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccount);

    parseArguments(argc, argv, &first, &last);

    if (first == -1 || last == -1 || first > last) {
        if (myrank == 0) {
            printf("Usage: mpirun -np N ./primes <first> <last>\n");
        }
        MPI_Finalize();
        return -1;
    }

    long long limit = (long long)sqrt((double)last);
    long long *base_primes = NULL;
    long long base_count = 0;
    
    generateBasePrimes(limit, &base_primes, &base_count);

    long long local_first, local_last;
    calculateLocalRange(first, last, myrank, proccount, &local_first, &local_last);

    long long local_primes = 0, local_twins = 0;
    long long local_first_prime = -1, local_last_prime = -1;

    if (local_first <= last) {
        runSegmentedSieve(local_first, local_last, base_primes, base_count, 
                          &local_primes, &local_twins, &local_first_prime, &local_last_prime);
    }
    
    free(base_primes);

    communicateBoundaries(myrank, proccount, local_first_prime, local_last_prime, &local_twins);

    long long total_primes = 0, total_twins = 0;
    MPI_Reduce(&local_primes, &total_primes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_twins,  &total_twins,  1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        printResults(first, last, proccount, total_primes, total_twins);
    }

    MPI_Finalize();
    return 0;
}
