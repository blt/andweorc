/*
 * dedup_hash.c - Deduplication hash table benchmark (coz paper inspired)
 *
 * This example demonstrates how causal profiling identifies hash function
 * bottlenecks in a deduplication workload. From the coz paper, fixing hash
 * distribution in dedup yielded an 8% improvement.
 *
 * The bottleneck is hash bucket traversal when there's poor distribution.
 * A bad hash function creates hot buckets that require long chain traversals.
 *
 * BUILD:
 *   gcc -O2 -g -pthread -ldl dedup_hash.c -o dedup_hash
 *
 * RUN (without profiling):
 *   ./dedup_hash
 *
 * RUN (with profiling):
 *   LD_PRELOAD=/path/to/libandweorc.so ANDWEORC_ENABLED=1 ./dedup_hash
 *
 * EXPECTED RESULTS:
 *   The profiler should identify the hash bucket traversal (in lookup/insert)
 *   as having high causal impact on throughput.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <stdatomic.h>
#include <unistd.h>
#include <dlfcn.h>

/*
 * Function pointers for andweorc functions.
 */
typedef void (*andweorc_progress_fn)(const char*);
typedef void (*andweorc_run_experiments_fn)(const char*);

static andweorc_progress_fn real_andweorc_progress = NULL;
static andweorc_run_experiments_fn real_andweorc_run_experiments = NULL;
static int andweorc_initialized = 0;

static void init_andweorc(void) {
    if (andweorc_initialized) return;
    andweorc_initialized = 1;

    real_andweorc_progress = (andweorc_progress_fn)dlsym(RTLD_DEFAULT, "andweorc_progress");
    real_andweorc_run_experiments = (andweorc_run_experiments_fn)dlsym(RTLD_DEFAULT, "andweorc_run_experiments");

    if (real_andweorc_progress) {
        printf("[dedup_hash] Found andweorc_progress via dlsym\n");
    }
    if (real_andweorc_run_experiments) {
        printf("[dedup_hash] Found andweorc_run_experiments via dlsym\n");
    }
}

static void andweorc_progress_wrapper(const char* name) {
    if (!andweorc_initialized) init_andweorc();
    if (real_andweorc_progress) {
        real_andweorc_progress(name);
    }
}

static void andweorc_run_experiments_wrapper(const char* name) {
    if (!andweorc_initialized) init_andweorc();
    if (real_andweorc_run_experiments) {
        real_andweorc_run_experiments(name);
    } else {
        printf("[dedup_hash] Profiler not loaded, sleeping 5s...\n");
        sleep(5);
    }
}

#define ANDWEORC_PROGRESS(name) andweorc_progress_wrapper(name)

/* Configuration */
#define NUM_BUCKETS 256           /* Small bucket count for contention */
#define NUM_THREADS 8             /* Worker threads */
#define CHUNK_SIZE 64             /* Simulated chunk size for hashing */
#define NUM_UNIQUE_CHUNKS 50000   /* Number of unique chunks to generate */

/* Entry in the hash table (chained) */
typedef struct hash_entry {
    uint8_t chunk_hash[32];       /* SHA-256 would go here; we use simplified hash */
    uint64_t chunk_id;            /* ID of the deduplicated chunk */
    struct hash_entry *next;      /* Chain pointer */
} hash_entry_t;

/* Hash table with per-bucket locks */
typedef struct {
    pthread_mutex_t lock;
    hash_entry_t *head;
    uint64_t lookups;
    uint64_t inserts;
    uint64_t chain_traversals;    /* Count of chain node visits */
} hash_bucket_t;

typedef struct {
    hash_bucket_t buckets[NUM_BUCKETS];
    atomic_uint_fast64_t next_chunk_id;
} dedup_table_t;

/* Global dedup table */
static dedup_table_t g_dedup;

/* Statistics */
static atomic_uint_fast64_t g_total_ops = 0;
static atomic_uint_fast64_t g_duplicates_found = 0;
static atomic_int g_running = 1;

/* Pre-generated chunk data for testing */
static uint8_t g_chunks[NUM_UNIQUE_CHUNKS][CHUNK_SIZE];

/*
 * BAD hash function - intentionally creates hot buckets.
 * This is the bottleneck that causal profiling should identify.
 */
static uint32_t bad_hash(const uint8_t *data, size_t len) {
    /* Simple sum with poor distribution */
    uint32_t hash = 0;
    for (size_t i = 0; i < len; i++) {
        hash += data[i];
    }
    return hash;
}

/*
 * GOOD hash function - better distribution (for comparison).
 */
static uint32_t good_hash(const uint8_t *data, size_t len) {
    /* FNV-1a hash */
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash;
}

/* Which hash to use (can toggle for comparison) */
#define USE_BAD_HASH 1

static uint32_t compute_hash(const uint8_t *data, size_t len) {
#if USE_BAD_HASH
    return bad_hash(data, len);
#else
    return good_hash(data, len);
#endif
}

/* Create simplified chunk fingerprint */
static void compute_fingerprint(const uint8_t *chunk, size_t len, uint8_t *fingerprint) {
    /* Simplified: just use first 32 bytes or pad with zeros */
    memset(fingerprint, 0, 32);
    size_t copy_len = len < 32 ? len : 32;
    memcpy(fingerprint, chunk, copy_len);

    /* Mix in a hash to differentiate similar chunks */
    uint32_t h = compute_hash(chunk, len);
    fingerprint[0] ^= (h >> 24) & 0xFF;
    fingerprint[1] ^= (h >> 16) & 0xFF;
    fingerprint[2] ^= (h >> 8) & 0xFF;
    fingerprint[3] ^= h & 0xFF;
}

/* Initialize the dedup table */
static void dedup_init(dedup_table_t *table) {
    for (int i = 0; i < NUM_BUCKETS; i++) {
        pthread_mutex_init(&table->buckets[i].lock, NULL);
        table->buckets[i].head = NULL;
        table->buckets[i].lookups = 0;
        table->buckets[i].inserts = 0;
        table->buckets[i].chain_traversals = 0;
    }
    atomic_store(&table->next_chunk_id, 0);
}

/* Destroy the dedup table */
static void dedup_destroy(dedup_table_t *table) {
    for (int i = 0; i < NUM_BUCKETS; i++) {
        pthread_mutex_destroy(&table->buckets[i].lock);
        hash_entry_t *entry = table->buckets[i].head;
        while (entry) {
            hash_entry_t *next = entry->next;
            free(entry);
            entry = next;
        }
    }
}

/*
 * Lookup or insert a chunk in the dedup table.
 * Returns: chunk_id (existing if duplicate, new if unique)
 * Sets *is_duplicate to 1 if chunk was already in table.
 *
 * THE BOTTLENECK: Chain traversal in this function is where
 * the profiler should identify high causal impact.
 */
static uint64_t dedup_lookup_insert(dedup_table_t *table, const uint8_t *chunk,
                                     size_t len, int *is_duplicate) {
    uint8_t fingerprint[32];
    compute_fingerprint(chunk, len, fingerprint);

    uint32_t hash = compute_hash(chunk, len);
    uint32_t bucket_idx = hash % NUM_BUCKETS;
    hash_bucket_t *bucket = &table->buckets[bucket_idx];

    pthread_mutex_lock(&bucket->lock);
    bucket->lookups++;

    /* Traverse chain looking for match - THIS IS THE BOTTLENECK */
    hash_entry_t *entry = bucket->head;
    while (entry) {
        bucket->chain_traversals++;  /* Count work done */

        if (memcmp(entry->chunk_hash, fingerprint, 32) == 0) {
            /* Found duplicate */
            uint64_t id = entry->chunk_id;
            pthread_mutex_unlock(&bucket->lock);
            *is_duplicate = 1;
            return id;
        }
        entry = entry->next;
    }

    /* Not found - insert new entry */
    hash_entry_t *new_entry = malloc(sizeof(hash_entry_t));
    if (!new_entry) {
        pthread_mutex_unlock(&bucket->lock);
        *is_duplicate = 0;
        return 0;
    }

    memcpy(new_entry->chunk_hash, fingerprint, 32);
    new_entry->chunk_id = atomic_fetch_add(&table->next_chunk_id, 1);
    new_entry->next = bucket->head;
    bucket->head = new_entry;
    bucket->inserts++;

    uint64_t id = new_entry->chunk_id;
    pthread_mutex_unlock(&bucket->lock);
    *is_duplicate = 0;
    return id;
}

/* Simple LCG random number generator */
static uint32_t rand_next(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(*state >> 33);
}

/* Worker thread function */
static void *worker_thread(void *arg) {
    int thread_id = *(int *)arg;
    uint64_t rng_state = (uint64_t)thread_id * 12345 + time(NULL);
    uint64_t local_ops = 0;
    uint64_t local_duplicates = 0;

    printf("Worker %d started\n", thread_id);

    while (atomic_load(&g_running)) {
        /* Pick a random chunk (with some locality to create duplicates) */
        int chunk_idx;
        if (rand_next(&rng_state) % 100 < 70) {
            /* 70% chance: pick from a hot set (first 1000 chunks) */
            chunk_idx = rand_next(&rng_state) % 1000;
        } else {
            /* 30% chance: pick from full range */
            chunk_idx = rand_next(&rng_state) % NUM_UNIQUE_CHUNKS;
        }

        int is_duplicate = 0;
        uint64_t chunk_id = dedup_lookup_insert(&g_dedup,
                                                 g_chunks[chunk_idx],
                                                 CHUNK_SIZE,
                                                 &is_duplicate);
        (void)chunk_id;

        if (is_duplicate) {
            local_duplicates++;
        }

        local_ops++;
        ANDWEORC_PROGRESS("dedup_done");
    }

    atomic_fetch_add(&g_total_ops, local_ops);
    atomic_fetch_add(&g_duplicates_found, local_duplicates);
    printf("Worker %d completed %lu operations (%lu duplicates)\n",
           thread_id, local_ops, local_duplicates);
    return NULL;
}

/* Generate random chunk data */
static void generate_chunks(void) {
    printf("Generating %d unique chunks...\n", NUM_UNIQUE_CHUNKS);
    uint64_t rng = 0x12345678;

    for (int i = 0; i < NUM_UNIQUE_CHUNKS; i++) {
        for (int j = 0; j < CHUNK_SIZE; j++) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            g_chunks[i][j] = (uint8_t)(rng >> 56);
        }
    }
}

/* Print statistics */
static void print_stats(dedup_table_t *table) {
    uint64_t total_lookups = 0;
    uint64_t total_inserts = 0;
    uint64_t total_traversals = 0;
    int max_chain = 0;
    int bucket_counts[10] = {0};  /* Chain length histogram */

    printf("\nBucket statistics:\n");
    for (int i = 0; i < NUM_BUCKETS; i++) {
        total_lookups += table->buckets[i].lookups;
        total_inserts += table->buckets[i].inserts;
        total_traversals += table->buckets[i].chain_traversals;

        /* Count chain length */
        int chain_len = 0;
        hash_entry_t *entry = table->buckets[i].head;
        while (entry) {
            chain_len++;
            entry = entry->next;
        }

        if (chain_len > max_chain) max_chain = chain_len;

        int hist_idx = chain_len < 9 ? chain_len : 9;
        bucket_counts[hist_idx]++;
    }

    printf("  Total lookups: %lu\n", total_lookups);
    printf("  Total inserts: %lu\n", total_inserts);
    printf("  Total chain traversals: %lu\n", total_traversals);
    printf("  Avg traversals per lookup: %.2f\n",
           (double)total_traversals / (total_lookups > 0 ? total_lookups : 1));
    printf("  Max chain length: %d\n", max_chain);

    printf("\nChain length distribution:\n");
    for (int i = 0; i < 10; i++) {
        if (bucket_counts[i] > 0) {
            printf("  Length %s%d: %d buckets\n",
                   i == 9 ? ">=" : "", i, bucket_counts[i]);
        }
    }
}

int main(void) {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    struct timespec start, end;

    printf("=== Dedup Hash Table Benchmark ===\n");
    printf("Inspired by Coz paper's dedup analysis (8%% improvement)\n\n");
    printf("Configuration:\n");
    printf("  Buckets: %d\n", NUM_BUCKETS);
    printf("  Threads: %d\n", NUM_THREADS);
    printf("  Unique chunks: %d\n", NUM_UNIQUE_CHUNKS);
    printf("  Hash function: %s (intentionally creates hot buckets)\n\n",
           USE_BAD_HASH ? "BAD" : "GOOD");

    /* Generate chunk data */
    generate_chunks();

    /* Initialize dedup table */
    dedup_init(&g_dedup);

    /* Start timing */
    clock_gettime(CLOCK_MONOTONIC, &start);

    /* Start worker threads */
    printf("Starting %d worker threads...\n", NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, worker_thread, &thread_ids[i]);
    }

    /* Run causal profiling experiments */
    printf("Running causal profiling experiments...\n");
    printf("Workers will continue running until experiments complete.\n\n");
    andweorc_run_experiments_wrapper("dedup_done");

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
    uint64_t duplicates = atomic_load(&g_duplicates_found);

    printf("\n=== Results ===\n");
    printf("Total operations: %lu\n", total_ops);
    printf("Duplicates found: %lu (%.1f%%)\n", duplicates,
           duplicates * 100.0 / (total_ops > 0 ? total_ops : 1));
    printf("Elapsed time: %.2f seconds\n", elapsed);
    printf("Throughput: %.0f ops/sec\n", total_ops / elapsed);

    print_stats(&g_dedup);

    printf("\n=== Profiling Analysis ===\n");
    printf("If run with causal profiling enabled, you should see:\n");
    printf("  - High causal impact for hash bucket chain traversal\n");
    printf("  - The bad hash function creates uneven distribution\n");
    printf("  - Some buckets have very long chains (hot spots)\n");
    printf("\nPotential optimizations (guided by profiling):\n");
    printf("  - Use a better hash function (FNV-1a, MurmurHash, etc.)\n");
    printf("  - Increase bucket count\n");
    printf("  - Use open addressing instead of chaining\n");

    /* Cleanup */
    dedup_destroy(&g_dedup);

    return 0;
}
