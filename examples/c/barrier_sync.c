/*
 * barrier_sync.c - PARSEC-inspired barrier contention example
 *
 * This example demonstrates a key insight from the Coz paper: sometimes
 * optimizing code would actually SLOW DOWN the program due to increased
 * contention. The Coz paper showed this with PARSEC's fluidanimate benchmark,
 * where a custom barrier implementation was the bottleneck.
 *
 * This example implements a similar pattern:
 * - Worker threads do parallel computation
 * - Workers synchronize at barriers between phases
 * - The barrier uses a spin-wait pattern that creates contention
 *
 * Causal profiling should show NEGATIVE impact for the spin-wait code,
 * indicating that "speeding it up" would increase contention and hurt throughput.
 *
 * BUILD:
 *   gcc -O2 -g -pthread barrier_sync.c -o barrier_sync
 *
 * RUN (without profiling):
 *   ./barrier_sync
 *
 * RUN (with profiling):
 *   LD_PRELOAD=/path/to/libandweorc.so ANDWEORC_ENABLED=1 ./barrier_sync
 *
 * EXPECTED RESULTS:
 *   Negative causal impact for the spin-wait barrier code, showing that
 *   optimizing it would increase contention and slow down the program.
 *   Replacing the custom barrier with pthread_barrier gives significant speedup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <stdatomic.h>
#include <unistd.h>
#include <string.h>

/* Include andweorc header for progress points */
#include "../../andweorc/include/andweorc.h"

/*
 * Weak symbol stubs for andweorc functions.
 * These allow the program to link without the profiler library.
 * When run with LD_PRELOAD, the real implementations override these.
 */
__attribute__((weak))
void andweorc_progress(const char* name) {
    (void)name; /* No-op when profiler not loaded */
}

__attribute__((weak))
void andweorc_progress_named(const char* file, int line) {
    (void)file;
    (void)line;
}

/* Configuration */
#define NUM_THREADS 8
#define NUM_PHASES 100
#define WORK_PER_PHASE 10000

/* Custom barrier using spin-wait (inefficient, causes contention) */
typedef struct {
    atomic_int count;
    atomic_int generation;
    int num_threads;
    pthread_mutex_t mutex;
} spin_barrier_t;

/* Statistics */
static atomic_uint_fast64_t g_total_work = 0;
static atomic_uint_fast64_t g_barrier_waits = 0;

/* Initialize spin barrier */
static void spin_barrier_init(spin_barrier_t *barrier, int num_threads) {
    atomic_store(&barrier->count, 0);
    atomic_store(&barrier->generation, 0);
    barrier->num_threads = num_threads;
    pthread_mutex_init(&barrier->mutex, NULL);
}

/* Destroy spin barrier */
static void spin_barrier_destroy(spin_barrier_t *barrier) {
    pthread_mutex_destroy(&barrier->mutex);
}

/*
 * Wait at barrier using spin-wait pattern (CONTENTION POINT)
 *
 * This implementation intentionally uses an inefficient spin-wait pattern
 * similar to the one in PARSEC's parsec_barrier.cpp. The Coz paper showed
 * that speeding up this code would INCREASE contention and slow things down.
 */
static void spin_barrier_wait(spin_barrier_t *barrier) {
    int gen = atomic_load(&barrier->generation);

    /* Try to acquire mutex to increment count */
    pthread_mutex_lock(&barrier->mutex);

    int arrived = atomic_fetch_add(&barrier->count, 1) + 1;

    if (arrived == barrier->num_threads) {
        /* Last thread to arrive - reset and advance generation */
        atomic_store(&barrier->count, 0);
        atomic_fetch_add(&barrier->generation, 1);
        pthread_mutex_unlock(&barrier->mutex);
    } else {
        pthread_mutex_unlock(&barrier->mutex);

        /*
         * SPIN-WAIT: This is the contention point!
         *
         * This spin-wait code is inefficient and causes cache line bouncing.
         * Causal profiling should show that "speeding up" this code would
         * actually slow down the program because threads would contend more
         * aggressively on the shared atomic variable.
         */
        while (atomic_load(&barrier->generation) == gen) {
            /* Spin - intentionally inefficient */
            atomic_fetch_add(&g_barrier_waits, 1);

            /* Small pause to simulate the original PARSEC pattern */
            for (volatile int i = 0; i < 10; i++) {
                /* Empty spin */
            }
        }
    }
}

/* Simulate parallel computation work */
static uint64_t do_work(int thread_id, int phase, int amount) {
    uint64_t result = thread_id * phase;
    for (int i = 0; i < amount; i++) {
        result = result * 31 + i;
        result ^= (result >> 7);
    }
    atomic_fetch_add(&g_total_work, amount);
    return result;
}

/* Global barrier */
static spin_barrier_t g_barrier;

/* Thread arguments */
typedef struct {
    int thread_id;
    uint64_t result;
} thread_arg_t;

/* Worker thread function */
static void *worker_thread(void *arg) {
    thread_arg_t *targ = (thread_arg_t *)arg;
    int thread_id = targ->thread_id;
    uint64_t local_result = 0;

    printf("Worker %d started\n", thread_id);

    for (int phase = 0; phase < NUM_PHASES; phase++) {
        /* Phase 1: Parallel computation */
        local_result += do_work(thread_id, phase, WORK_PER_PHASE);

        /* Phase 2: Synchronize at barrier (BOTTLENECK) */
        spin_barrier_wait(&g_barrier);

        /* Mark progress after each phase */
        ANDWEORC_PROGRESS_NAMED("phase_complete");
    }

    targ->result = local_result;
    printf("Worker %d completed (result=%lu)\n", thread_id, local_result);
    return NULL;
}

/* Run benchmark with custom spin barrier */
static double run_with_spin_barrier(void) {
    pthread_t threads[NUM_THREADS];
    thread_arg_t thread_args[NUM_THREADS];
    struct timespec start, end;

    spin_barrier_init(&g_barrier, NUM_THREADS);
    atomic_store(&g_total_work, 0);
    atomic_store(&g_barrier_waits, 0);

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].result = 0;
        pthread_create(&threads[i], NULL, worker_thread, &thread_args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    spin_barrier_destroy(&g_barrier);

    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

/* pthread_barrier version for comparison */
static pthread_barrier_t g_pthread_barrier;

static void *worker_thread_pthread_barrier(void *arg) {
    thread_arg_t *targ = (thread_arg_t *)arg;
    int thread_id = targ->thread_id;
    uint64_t local_result = 0;

    for (int phase = 0; phase < NUM_PHASES; phase++) {
        /* Phase 1: Parallel computation */
        local_result += do_work(thread_id, phase, WORK_PER_PHASE);

        /* Phase 2: Synchronize with efficient pthread_barrier */
        pthread_barrier_wait(&g_pthread_barrier);
    }

    targ->result = local_result;
    return NULL;
}

static double run_with_pthread_barrier(void) {
    pthread_t threads[NUM_THREADS];
    thread_arg_t thread_args[NUM_THREADS];
    struct timespec start, end;

    pthread_barrier_init(&g_pthread_barrier, NULL, NUM_THREADS);
    atomic_store(&g_total_work, 0);

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].result = 0;
        pthread_create(&threads[i], NULL, worker_thread_pthread_barrier, &thread_args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    pthread_barrier_destroy(&g_pthread_barrier);

    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(void) {
    printf("=== Barrier Synchronization Example ===\n");
    printf("Inspired by Coz paper's PARSEC fluidanimate analysis\n\n");
    printf("Configuration:\n");
    printf("  Threads: %d\n", NUM_THREADS);
    printf("  Phases: %d\n", NUM_PHASES);
    printf("  Work per phase: %d operations\n\n", WORK_PER_PHASE);

    /* Run with custom spin barrier (inefficient) */
    printf("Running with custom spin barrier...\n");
    double spin_time = run_with_spin_barrier();
    uint64_t barrier_waits = atomic_load(&g_barrier_waits);
    uint64_t work_done = atomic_load(&g_total_work);

    printf("\n=== Spin Barrier Results ===\n");
    printf("Elapsed time: %.3f seconds\n", spin_time);
    printf("Total work: %lu operations\n", work_done);
    printf("Barrier spin iterations: %lu\n", barrier_waits);
    printf("Phases per second: %.1f\n", (NUM_PHASES * NUM_THREADS) / spin_time);

    /* Run with pthread_barrier (efficient) */
    printf("\nRunning with pthread_barrier (efficient)...\n");
    double pthread_time = run_with_pthread_barrier();
    work_done = atomic_load(&g_total_work);

    printf("\n=== pthread_barrier Results ===\n");
    printf("Elapsed time: %.3f seconds\n", pthread_time);
    printf("Total work: %lu operations\n", work_done);
    printf("Phases per second: %.1f\n", (NUM_PHASES * NUM_THREADS) / pthread_time);

    /* Compare */
    double speedup = spin_time / pthread_time;
    printf("\n=== Comparison ===\n");
    printf("pthread_barrier is %.1fx faster than spin barrier\n", speedup);
    printf("Improvement: %.1f%%\n", (speedup - 1.0) * 100);

    printf("\n=== Profiling Analysis ===\n");
    printf("When profiled with causal profiling, you should see:\n");
    printf("\n");
    printf("1. NEGATIVE causal impact for spin_barrier_wait spin loop:\n");
    printf("   - The spin-wait code shows that 'optimizing' it would slow things down\n");
    printf("   - This is because faster spinning = more contention on atomic variable\n");
    printf("   - The downward-sloping causal profile reveals this paradox\n");
    printf("\n");
    printf("2. This matches the Coz paper's PARSEC findings:\n");
    printf("   - fluidanimate: 37.5%% speedup by replacing custom barrier\n");
    printf("   - streamcluster: 68.4%% speedup with same fix\n");
    printf("\n");
    printf("3. The fix is NOT to optimize the spin-wait code, but to REPLACE it\n");
    printf("   with an efficient barrier implementation (pthread_barrier).\n");

    return 0;
}
