/*
 * signal_trigger.c - Demonstrates SIGUSR1-based experiment triggering
 *
 * This example shows how to profile a program via signal-based triggering.
 * Workers run continuously, and experiments are triggered by SIGUSR1.
 *
 * BUILD:
 *   gcc -O2 -g -pthread -ldl signal_trigger.c -o signal_trigger
 *
 * RUN:
 *   LD_PRELOAD=../../target/release/libandweorc.so ANDWEORC_ENABLED=1 \
 *     ANDWEORC_EXPERIMENT_TARGET="work_done" ./signal_trigger
 *
 * Then in another terminal:
 *   kill -USR1 $(pgrep signal_trigger)
 *
 * Or run with auto-trigger (sends SIGUSR1 to self after warmup):
 *   LD_PRELOAD=../../target/release/libandweorc.so ANDWEORC_ENABLED=1 \
 *     ANDWEORC_EXPERIMENT_TARGET="work_done" ./signal_trigger --auto
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <stdatomic.h>
#include <unistd.h>
#include <signal.h>
#include <dlfcn.h>

/*
 * Function pointer for andweorc_progress.
 */
typedef void (*andweorc_progress_fn)(const char*);
static andweorc_progress_fn real_andweorc_progress = NULL;
static int andweorc_initialized = 0;

static void init_andweorc(void) {
    if (andweorc_initialized) return;
    andweorc_initialized = 1;

    real_andweorc_progress = (andweorc_progress_fn)dlsym(RTLD_DEFAULT, "andweorc_progress");
    if (real_andweorc_progress) {
        printf("[signal_trigger] Found andweorc_progress via dlsym\n");
    }
}

static void andweorc_progress_wrapper(const char* name) {
    if (!andweorc_initialized) init_andweorc();
    if (real_andweorc_progress) {
        real_andweorc_progress(name);
    }
}

#define ANDWEORC_PROGRESS(name) andweorc_progress_wrapper(name)

/* Configuration */
#define NUM_THREADS 4
#define WORK_ITERATIONS 50000  /* Work per unit */

/* Global state */
static atomic_int g_running = 1;
static atomic_uint_fast64_t g_total_work = 0;

/*
 * Simulated work function - hash computation.
 * This is the "bottleneck" we're profiling.
 */
static uint64_t do_work(uint64_t seed) {
    uint64_t hash = seed;
    for (int i = 0; i < WORK_ITERATIONS; i++) {
        hash ^= (hash << 13);
        hash ^= (hash >> 7);
        hash ^= (hash << 17);
    }
    return hash;
}

/*
 * Worker thread - runs continuously doing work.
 */
static void *worker_thread(void *arg) {
    int thread_id = *(int *)arg;
    uint64_t rng_state = (uint64_t)thread_id * 12345 + 67890;
    uint64_t local_work = 0;

    printf("Worker %d started\n", thread_id);

    while (atomic_load(&g_running)) {
        /* Do some work */
        rng_state = do_work(rng_state);
        local_work++;

        /* Mark progress - this is where experiments get triggered */
        ANDWEORC_PROGRESS("work_done");
    }

    atomic_fetch_add(&g_total_work, local_work);
    printf("Worker %d completed %lu work units\n", thread_id, local_work);
    return NULL;
}

int main(int argc, char **argv) {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    int auto_trigger = 0;

    /* Check for --auto flag */
    if (argc > 1 && strcmp(argv[1], "--auto") == 0) {
        auto_trigger = 1;
    }

    printf("=== Signal-Triggered Causal Profiling Demo ===\n\n");
    printf("This example demonstrates SIGUSR1-based experiment triggering.\n");
    printf("Workers run continuously until experiments complete.\n\n");
    printf("Configuration:\n");
    printf("  Threads: %d\n", NUM_THREADS);
    printf("  Work iterations: %d per unit\n", WORK_ITERATIONS);
    printf("  Auto-trigger: %s\n\n", auto_trigger ? "YES" : "NO");

    if (!auto_trigger) {
        printf("To trigger experiments, run in another terminal:\n");
        printf("  kill -USR1 %d\n\n", getpid());
    }

    /* Start worker threads */
    printf("Starting %d worker threads...\n", NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, worker_thread, &thread_ids[i]);
    }

    if (auto_trigger) {
        /* Wait for warmup then send signal to self */
        printf("\nWaiting 3 seconds for warmup...\n");
        sleep(3);

        printf("Sending SIGUSR1 to self (pid %d)...\n", getpid());
        kill(getpid(), SIGUSR1);

        /* Wait for experiments to run (they run at next progress point) */
        printf("Waiting for experiments to complete...\n");
        sleep(30);  /* Give experiments time to run */

        /* Signal workers to stop */
        printf("\nSignaling workers to stop...\n");
        atomic_store(&g_running, 0);
    } else {
        /* Manual mode - wait for user to send signal */
        printf("\nWorkers running. Send SIGUSR1 to trigger experiments.\n");
        printf("Press Ctrl+C to stop without running experiments.\n\n");

        /* Wait indefinitely - signal handler will trigger experiments */
        while (atomic_load(&g_running)) {
            sleep(1);
        }
    }

    /* Wait for threads */
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    uint64_t total = atomic_load(&g_total_work);
    printf("\n=== Results ===\n");
    printf("Total work units completed: %lu\n", total);

    return 0;
}
