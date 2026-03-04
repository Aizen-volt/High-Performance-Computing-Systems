#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

typedef struct {
    long long *data;
    int count;
    int capacity;
} DynamicArray;

static int cmp_ll(const void *a, const void *b) {
    long long arg1 = *(const long long*)a;
    long long arg2 = *(const long long*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

static int isPrime(long long n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    long long limit = (long long)sqrt((double)n);
    for (long long i = 3; i <= limit; i += 2)
        if (n % i == 0) return 0;
    return 1;
}

static void initArray(DynamicArray *arr, int initial_capacity) {
    arr->capacity = initial_capacity;
    arr->count = 0;
    arr->data = malloc(arr->capacity * sizeof(long long));
}

static void pushToArray(DynamicArray *arr, long long value) {
    if (arr->count >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = realloc(arr->data, arr->capacity * sizeof(long long));
    }
    arr->data[arr->count++] = value;
}

static void freeArray(DynamicArray *arr) {
    free(arr->data);
    arr->data = NULL;
    arr->count = 0;
    arr->capacity = 0;
}

static long long* gatherArrays(DynamicArray *local_arr, int myrank, int proccount, int *total_count) {
    int *recvcounts = NULL;
    int *displs = NULL;
    long long *all_data = NULL;

    if (myrank == 0) {
        recvcounts = malloc(proccount * sizeof(int));
        displs = malloc(proccount * sizeof(int));
    }

    MPI_Gather(&local_arr->count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        displs[0] = 0;
        *total_count = recvcounts[0];
        
        for (int i = 1; i < proccount; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
            *total_count += recvcounts[i];
        }
        
        all_data = malloc(*total_count * sizeof(long long));
    }

    MPI_Gatherv(local_arr->data, local_arr->count, MPI_LONG_LONG, 
                all_data, recvcounts, displs, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        free(recvcounts);
        free(displs);
    }

    return all_data;
}

static void printResults(long long first, long long last, int proccount, int total_primes, int total_twins, 
                         long long *all_primes, long long *all_twins, int verbose) {
    printf("Range            : [%lld, %lld]\n", first, last);
    printf("Processes        : %d\n", proccount);
    printf("Primes found     : %d\n", total_primes);
    printf("Twin prime pairs : %d\n", total_twins);

    if (verbose) {
        qsort(all_primes, total_primes, sizeof(long long), cmp_ll);
        qsort(all_twins, total_twins, sizeof(long long), cmp_ll);

        printf("\n--- ALL PRIMES ---\n");
        for (int i = 0; i < total_primes; i++) {
            printf("%lld ", all_primes[i]);
        }
        printf("\n\n--- ALL TWIN PAIRS ---\n");
        for (int i = 0; i < total_twins; i++) {
            printf("(%lld, %lld) ", all_twins[i], all_twins[i] + 2);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    long long first = -1, last = -1;
    int verbose = 0, extra_verbose = 0;
    int myrank, proccount;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccount);

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v")) {
            verbose = 1;
        } else if (!strcmp(argv[i], "--extra-verbose") || !strcmp(argv[i], "-vv")) {
            verbose = 1;
            extra_verbose = 1;
        } else if (first == -1) {
            first = atoll(argv[i]);
        } else if (last == -1) {
            last = atoll(argv[i]);
        }
    }

    if (first == -1 || last == -1 || first > last) {
        if (myrank == 0) {
            printf("Usage: mpirun -np N ./primes <first> <last> [-v|--verbose] [-vv|--extra-verbose]\n");
        }
        MPI_Finalize();
        return -1;
    }

    DynamicArray local_primes, local_twins;
    initArray(&local_primes, 1000);
    initArray(&local_twins, 1000);

    long long start = first;
    if (start <= 2) {
        if (myrank == 0 && first <= 2 && last >= 2) {
            pushToArray(&local_primes, 2);
        }
        start = 3;
    } else if (start % 2 == 0) {
        start++;
    }

    for (long long n = start + (2 * myrank); n <= last; n += (2 * proccount)) {
        if (isPrime(n)) {
            pushToArray(&local_primes, n);
            
            if (extra_verbose) {
                printf("[rank %d] prime: %lld\n", myrank, n);
            }

            if (n + 2 <= last && isPrime(n + 2)) {
                pushToArray(&local_twins, n);
                
                if (extra_verbose) {
                    printf("[rank %d] twin pair: (%lld, %lld)\n", myrank, n, n + 2);
                }
            }
        }
    }

    int total_primes = 0, total_twins = 0;
    long long *all_primes = gatherArrays(&local_primes, myrank, proccount, &total_primes);
    long long *all_twins = gatherArrays(&local_twins, myrank, proccount, &total_twins);

    if (myrank == 0) {
        printResults(first, last, proccount, total_primes, total_twins, all_primes, all_twins, verbose);
        free(all_primes);
        free(all_twins);
    }

    freeArray(&local_primes);
    freeArray(&local_twins);

    MPI_Finalize();
    return 0;
}
