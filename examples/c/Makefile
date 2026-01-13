# Makefile for andweorc C examples
#
# These examples demonstrate causal profiling with C programs using LD_PRELOAD.
#
# BUILD:
#   make all         - Build all examples
#   make clean       - Remove built binaries
#
# RUN (without profiling):
#   ./cache_contention
#   ./producer_consumer
#   ./barrier_sync
#
# RUN (with profiling):
#   make run-cache ANDWEORC_LIB=/path/to/libandweorc.so
#   make run-producer ANDWEORC_LIB=/path/to/libandweorc.so
#   make run-barrier ANDWEORC_LIB=/path/to/libandweorc.so

CC = gcc
CFLAGS = -O2 -g -Wall -Wextra -pthread
LDFLAGS = -pthread

# Path to andweorc shared library (set when running with profiling)
ANDWEORC_LIB ?= ../../target/release/libandweorc.so

# Include path for andweorc.h
INCLUDES = -I../../andweorc/include

# Targets
TARGETS = cache_contention producer_consumer barrier_sync dedup_hash

.PHONY: all clean run-cache run-producer run-barrier run-dedup run-all

all: $(TARGETS)

cache_contention: cache_contention.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

producer_consumer: producer_consumer.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

barrier_sync: barrier_sync.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

dedup_hash: dedup_hash.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)

# Run targets with profiling enabled
run-cache: cache_contention
	@echo "=== Running cache_contention with causal profiling ==="
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./cache_contention

run-producer: producer_consumer
	@echo "=== Running producer_consumer with causal profiling ==="
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./producer_consumer

run-barrier: barrier_sync
	@echo "=== Running barrier_sync with causal profiling ==="
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./barrier_sync

run-dedup: dedup_hash
	@echo "=== Running dedup_hash with causal profiling ==="
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./dedup_hash

# Run all examples sequentially
run-all: $(TARGETS)
	@echo "=== Running all examples with causal profiling ==="
	@echo ""
	@echo "--- Cache Contention (Memcached-inspired) ---"
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./cache_contention
	@echo ""
	@echo "--- Producer-Consumer Queue ---"
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./producer_consumer
	@echo ""
	@echo "--- Barrier Synchronization (PARSEC-inspired) ---"
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./barrier_sync
	@echo ""
	@echo "--- Dedup Hash Table (coz paper inspired) ---"
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./dedup_hash

# Run without profiling (for comparison)
run-baseline: $(TARGETS)
	@echo "=== Running all examples WITHOUT profiling (baseline) ==="
	./cache_contention
	@echo ""
	./producer_consumer
	@echo ""
	./barrier_sync
	@echo ""
	./dedup_hash
