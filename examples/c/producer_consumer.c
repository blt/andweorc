/*
 * producer_consumer.c - Lock contention in a bounded queue
 *
 * This example demonstrates how causal profiling identifies lock contention
 * in a classic producer-consumer pattern with a shared bounded queue.
 *
 * The queue uses a single mutex for both producers and consumers, creating
 * contention. Causal profiling should reveal that the lock is the bottleneck
 * and guide optimization (e.g., lock-free queue, separate producer/consumer locks).
 *
 * BUILD:
 *   gcc -O2 -g -pthread producer_consumer.c -o producer_consumer
 *
 * RUN (without profiling):
 *   ./producer_consumer
 *
 * RUN (with profiling):
 *   LD_PRELOAD=/path/to/libandweorc.so ANDWEORC_ENABLED=1 ./producer_consumer
 *
 * EXPECTED RESULTS:
 *   High causal impact for pthread_mutex_lock calls, indicating the queue
 *   lock is limiting throughput.
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
#define QUEUE_SIZE 64           /* Small queue to increase contention */
#define NUM_PRODUCERS 4
#define NUM_CONSUMERS 4
#define ITEMS_PER_PRODUCER 50000

/* Work item */
typedef struct {
    uint64_t id;
    uint64_t data;
    int producer_id;
} work_item_t;

/* Bounded queue with lock */
typedef struct {
    work_item_t items[QUEUE_SIZE];
    int head;
    int tail;
    int count;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} bounded_queue_t;

/* Global queue */
static bounded_queue_t g_queue;

/* Statistics */
static atomic_uint_fast64_t g_items_produced = 0;
static atomic_uint_fast64_t g_items_consumed = 0;
static atomic_int g_producers_done = 0;

/* Initialize queue */
static void queue_init(bounded_queue_t *q) {
    q->head = 0;
    q->tail = 0;
    q->count = 0;
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

/* Destroy queue */
static void queue_destroy(bounded_queue_t *q) {
    pthread_mutex_destroy(&q->lock);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

/* Enqueue an item (blocks if full) - CONTENTION POINT */
static void queue_enqueue(bounded_queue_t *q, work_item_t *item) {
    /* This lock is a major contention point */
    pthread_mutex_lock(&q->lock);

    /* Wait while queue is full */
    while (q->count == QUEUE_SIZE) {
        pthread_cond_wait(&q->not_full, &q->lock);
    }

    /* Add item */
    q->items[q->tail] = *item;
    q->tail = (q->tail + 1) % QUEUE_SIZE;
    q->count++;

    /* Signal consumers */
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
}

/* Dequeue an item (blocks if empty) - CONTENTION POINT */
static int queue_dequeue(bounded_queue_t *q, work_item_t *item_out) {
    /* This lock is a major contention point */
    pthread_mutex_lock(&q->lock);

    /* Wait while queue is empty, but check if producers are done */
    while (q->count == 0) {
        if (atomic_load(&g_producers_done) == NUM_PRODUCERS) {
            pthread_mutex_unlock(&q->lock);
            return 0; /* No more items */
        }
        /* Use timed wait to periodically check if producers are done */
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 10000000; /* 10ms timeout */
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&q->not_empty, &q->lock, &ts);
    }

    /* Remove item */
    *item_out = q->items[q->head];
    q->head = (q->head + 1) % QUEUE_SIZE;
    q->count--;

    /* Signal producers */
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->lock);
    return 1;
}

/* Simulate work on an item */
static uint64_t process_item(work_item_t *item) {
    /* Light processing to keep focus on queue contention */
    uint64_t result = item->data;
    for (int i = 0; i < 100; i++) {
        result = result * 31 + i;
    }
    return result;
}

/* Producer thread */
static void *producer_thread(void *arg) {
    int producer_id = *(int *)arg;
    uint64_t rng_state = (uint64_t)producer_id * 12345 + time(NULL);

    printf("Producer %d started\n", producer_id);

    for (int i = 0; i < ITEMS_PER_PRODUCER; i++) {
        work_item_t item;
        item.id = (uint64_t)producer_id * ITEMS_PER_PRODUCER + i;
        item.producer_id = producer_id;

        /* Generate some data */
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        item.data = rng_state;

        queue_enqueue(&g_queue, &item);
        atomic_fetch_add(&g_items_produced, 1);

        /* Mark progress for throughput measurement */
        ANDWEORC_PROGRESS_NAMED("item_produced");
    }

    printf("Producer %d finished (%d items)\n", producer_id, ITEMS_PER_PRODUCER);
    atomic_fetch_add(&g_producers_done, 1);

    /* Wake up consumers that might be waiting */
    pthread_mutex_lock(&g_queue.lock);
    pthread_cond_broadcast(&g_queue.not_empty);
    pthread_mutex_unlock(&g_queue.lock);

    return NULL;
}

/* Consumer thread */
static void *consumer_thread(void *arg) {
    int consumer_id = *(int *)arg;
    uint64_t items_processed = 0;
    uint64_t checksum = 0;

    printf("Consumer %d started\n", consumer_id);

    work_item_t item;
    while (queue_dequeue(&g_queue, &item)) {
        checksum += process_item(&item);
        items_processed++;
        atomic_fetch_add(&g_items_consumed, 1);

        /* Mark progress for throughput measurement */
        ANDWEORC_PROGRESS_NAMED("item_consumed");
    }

    printf("Consumer %d finished (%lu items, checksum=%lu)\n",
           consumer_id, items_processed, checksum);

    return NULL;
}

int main(void) {
    pthread_t producers[NUM_PRODUCERS];
    pthread_t consumers[NUM_CONSUMERS];
    int producer_ids[NUM_PRODUCERS];
    int consumer_ids[NUM_CONSUMERS];
    struct timespec start, end;

    printf("=== Producer-Consumer Contention Example ===\n\n");
    printf("Configuration:\n");
    printf("  Queue size: %d (small for high contention)\n", QUEUE_SIZE);
    printf("  Producers: %d\n", NUM_PRODUCERS);
    printf("  Consumers: %d\n", NUM_CONSUMERS);
    printf("  Items per producer: %d\n", ITEMS_PER_PRODUCER);
    printf("  Total items: %d\n\n", NUM_PRODUCERS * ITEMS_PER_PRODUCER);

    /* Initialize queue */
    queue_init(&g_queue);

    /* Start timing */
    clock_gettime(CLOCK_MONOTONIC, &start);

    /* Start consumers first (they'll block until items arrive) */
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        consumer_ids[i] = i;
        pthread_create(&consumers[i], NULL, consumer_thread, &consumer_ids[i]);
    }

    /* Start producers */
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        producer_ids[i] = i;
        pthread_create(&producers[i], NULL, producer_thread, &producer_ids[i]);
    }

    /* Wait for producers */
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        pthread_join(producers[i], NULL);
    }

    /* Wait for consumers */
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        pthread_join(consumers[i], NULL);
    }

    /* End timing */
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    uint64_t produced = atomic_load(&g_items_produced);
    uint64_t consumed = atomic_load(&g_items_consumed);

    printf("\n=== Results ===\n");
    printf("Items produced: %lu\n", produced);
    printf("Items consumed: %lu\n", consumed);
    printf("Elapsed time: %.2f seconds\n", elapsed);
    printf("Producer throughput: %.0f items/sec\n", produced / elapsed);
    printf("Consumer throughput: %.0f items/sec\n", consumed / elapsed);

    printf("\n=== Profiling Analysis ===\n");
    printf("If run with causal profiling enabled, you should see:\n");
    printf("  - High causal impact for pthread_mutex_lock in queue operations\n");
    printf("  - Both queue_enqueue and queue_dequeue share the same lock\n");
    printf("  - The single lock serializes all access, creating contention\n");
    printf("\nPotential optimizations (guided by profiling):\n");
    printf("  - Use separate locks for head and tail (two-lock queue)\n");
    printf("  - Use a lock-free queue implementation\n");
    printf("  - Increase queue size to reduce blocking\n");

    /* Cleanup */
    queue_destroy(&g_queue);

    return 0;
}
