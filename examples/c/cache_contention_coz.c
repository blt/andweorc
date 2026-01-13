/*
 * cache_contention_coz.c - Memcached-inspired cache contention example (coz version)
 *
 * This is a coz-compatible version of cache_contention.c for comparison testing
 * between andweorc and coz causal profilers.
 *
 * BUILD:
 *   gcc -O2 -g -pthread -ldl cache_contention_coz.c -o cache_contention_coz
 *
 * RUN (with coz):
 *   coz run --- ./cache_contention_coz
 *   coz plot profile.coz
 *
 * EXPECTED RESULTS:
 *   The profiler should identify the bucket lock acquisition (pthread_mutex_lock
 *   in cache_get/cache_set) as having high causal impact on throughput.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <stdatomic.h>
#include <unistd.h>

/* Include coz header for progress points */
#include <coz.h>

/* Configuration */
#define NUM_BUCKETS 16          /* Small number to increase contention */
#define NUM_THREADS 8           /* Worker threads */
#define RUN_DURATION_SEC 30     /* Fixed run duration for coz */
#define MAX_KEY_LEN 32
#define MAX_VAL_LEN 128

/* Cache entry */
typedef struct cache_entry {
    char key[MAX_KEY_LEN];
    char value[MAX_VAL_LEN];
    struct cache_entry *next;
} cache_entry_t;

/* Cache bucket with lock */
typedef struct {
    pthread_mutex_t lock;
    cache_entry_t *head;
    uint64_t hits;
    uint64_t misses;
} cache_bucket_t;

/* The cache */
typedef struct {
    cache_bucket_t buckets[NUM_BUCKETS];
} cache_t;

/* Global cache */
static cache_t g_cache;

/* Statistics */
static atomic_uint_fast64_t g_total_ops = 0;
static atomic_int g_running = 1;

/* Simple hash function */
static uint32_t hash_key(const char *key) {
    uint32_t hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

/* Initialize the cache */
static void cache_init(cache_t *cache) {
    for (int i = 0; i < NUM_BUCKETS; i++) {
        pthread_mutex_init(&cache->buckets[i].lock, NULL);
        cache->buckets[i].head = NULL;
        cache->buckets[i].hits = 0;
        cache->buckets[i].misses = 0;
    }
}

/* Destroy the cache */
static void cache_destroy(cache_t *cache) {
    for (int i = 0; i < NUM_BUCKETS; i++) {
        pthread_mutex_destroy(&cache->buckets[i].lock);
        cache_entry_t *entry = cache->buckets[i].head;
        while (entry) {
            cache_entry_t *next = entry->next;
            free(entry);
            entry = next;
        }
    }
}

/* Get a value from cache - CONTENTION POINT */
static int cache_get(cache_t *cache, const char *key, char *value_out) {
    uint32_t bucket_idx = hash_key(key) % NUM_BUCKETS;
    cache_bucket_t *bucket = &cache->buckets[bucket_idx];

    /* This lock is the contention point - profiler should identify this */
    pthread_mutex_lock(&bucket->lock);

    /* Traverse the bucket chain */
    cache_entry_t *entry = bucket->head;
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            strcpy(value_out, entry->value);
            bucket->hits++;
            pthread_mutex_unlock(&bucket->lock);
            return 1; /* Found */
        }
        entry = entry->next;
    }

    bucket->misses++;
    pthread_mutex_unlock(&bucket->lock);
    return 0; /* Not found */
}

/* Set a value in cache - CONTENTION POINT */
static void cache_set(cache_t *cache, const char *key, const char *value) {
    uint32_t bucket_idx = hash_key(key) % NUM_BUCKETS;
    cache_bucket_t *bucket = &cache->buckets[bucket_idx];

    /* This lock is the contention point - profiler should identify this */
    pthread_mutex_lock(&bucket->lock);

    /* Check if key exists */
    cache_entry_t *entry = bucket->head;
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            strcpy(entry->value, value);
            pthread_mutex_unlock(&bucket->lock);
            return;
        }
        entry = entry->next;
    }

    /* Add new entry at head */
    cache_entry_t *new_entry = malloc(sizeof(cache_entry_t));
    if (new_entry) {
        strncpy(new_entry->key, key, MAX_KEY_LEN - 1);
        new_entry->key[MAX_KEY_LEN - 1] = '\0';
        strncpy(new_entry->value, value, MAX_VAL_LEN - 1);
        new_entry->value[MAX_VAL_LEN - 1] = '\0';
        new_entry->next = bucket->head;
        bucket->head = new_entry;
    }

    pthread_mutex_unlock(&bucket->lock);
}

/* Simple LCG random number generator (thread-local state) */
static uint32_t rand_next(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(*state >> 33);
}

/* Worker thread function */
static void *worker_thread(void *arg) {
    int thread_id = *(int *)arg;
    uint64_t rng_state = (uint64_t)thread_id * 12345 + time(NULL);
    char key[MAX_KEY_LEN];
    char value[MAX_VAL_LEN];
    uint64_t local_ops = 0;

    printf("Worker %d started\n", thread_id);

    while (atomic_load(&g_running)) {
        /* Generate a key (small key space to increase contention) */
        int key_id = rand_next(&rng_state) % 1000;
        snprintf(key, sizeof(key), "key:%d", key_id);

        /* 80% reads, 20% writes (typical cache workload) */
        if (rand_next(&rng_state) % 100 < 80) {
            /* GET operation */
            cache_get(&g_cache, key, value);
        } else {
            /* SET operation */
            snprintf(value, sizeof(value), "value-%d-from-thread-%d",
                     rand_next(&rng_state), thread_id);
            cache_set(&g_cache, key, value);
        }

        local_ops++;

        /* Mark progress for coz causal profiling */
        COZ_PROGRESS_NAMED("cache_op_done");
    }

    atomic_fetch_add(&g_total_ops, local_ops);
    printf("Worker %d completed %lu operations\n", thread_id, local_ops);
    return NULL;
}

/* Print cache statistics */
static void print_stats(cache_t *cache) {
    uint64_t total_hits = 0;
    uint64_t total_misses = 0;

    printf("\nBucket statistics:\n");
    for (int i = 0; i < NUM_BUCKETS; i++) {
        uint64_t hits = cache->buckets[i].hits;
        uint64_t misses = cache->buckets[i].misses;
        total_hits += hits;
        total_misses += misses;

        /* Count entries in bucket */
        int count = 0;
        cache_entry_t *entry = cache->buckets[i].head;
        while (entry) {
            count++;
            entry = entry->next;
        }
        printf("  Bucket %2d: %3d entries, %8lu hits, %8lu misses\n",
               i, count, hits, misses);
    }

    printf("\nTotal: %lu hits, %lu misses (%.1f%% hit rate)\n",
           total_hits, total_misses,
           total_hits * 100.0 / (total_hits + total_misses + 1));
}

int main(void) {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    struct timespec start, end;

    printf("=== Cache Contention Example (coz version) ===\n");
    printf("Inspired by Coz paper's Memcached analysis\n\n");
    printf("Configuration:\n");
    printf("  Buckets: %d (intentionally small for contention)\n", NUM_BUCKETS);
    printf("  Threads: %d\n", NUM_THREADS);
    printf("  Key space: 1000 keys\n");
    printf("  Run duration: %d seconds\n\n", RUN_DURATION_SEC);

    /* Initialize cache */
    cache_init(&g_cache);

    /* Pre-populate cache with some entries */
    printf("Pre-populating cache...\n");
    for (int i = 0; i < 500; i++) {
        char key[MAX_KEY_LEN];
        char value[MAX_VAL_LEN];
        snprintf(key, sizeof(key), "key:%d", i);
        snprintf(value, sizeof(value), "initial-value-%d", i);
        cache_set(&g_cache, key, value);
    }

    /* Start timing */
    clock_gettime(CLOCK_MONOTONIC, &start);

    /* Start worker threads */
    printf("Starting %d worker threads...\n", NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, worker_thread, &thread_ids[i]);
    }

    /* Run for fixed duration (coz style) */
    printf("Running for %d seconds...\n\n", RUN_DURATION_SEC);
    sleep(RUN_DURATION_SEC);

    /* Signal threads to stop */
    atomic_store(&g_running, 0);

    /* Wait for threads */
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    /* End timing */
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    uint64_t total_ops = atomic_load(&g_total_ops);

    printf("\n=== Results ===\n");
    printf("Total operations: %lu\n", total_ops);
    printf("Elapsed time: %.2f seconds\n", elapsed);
    printf("Throughput: %.0f ops/sec\n", total_ops / elapsed);

    print_stats(&g_cache);

    printf("\n=== Profiling Analysis ===\n");
    printf("If run with coz (coz run --- ./cache_contention_coz), you should see:\n");
    printf("  - High causal impact for pthread_mutex_lock in cache_get/cache_set\n");
    printf("  - Use 'coz plot profile.coz' to visualize results\n");

    /* Cleanup */
    cache_destroy(&g_cache);

    return 0;
}
