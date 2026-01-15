# Makefile for andweorc C examples
#
# These examples demonstrate CAUSAL PROFILING with C programs.
# Each example calls andweorc_run_experiments() to perform virtual speedup
# injection and impact calculation - the core of causal profiling.
#
# BUILD:
#   make all         - Build all examples
#   make clean       - Remove built binaries
#
# RUN (with causal profiling):
#   make run-cache ANDWEORC_LIB=/path/to/libandweorc.so
#   make run-dedup ANDWEORC_LIB=/path/to/libandweorc.so

CC = gcc
CFLAGS = -O2 -g -Wall -Wextra -pthread
LDFLAGS = -pthread -ldl

# Path to andweorc shared library (set when running with profiling)
ANDWEORC_LIB ?= ../../target/release/libandweorc.so

# Include path for andweorc.h
INCLUDES = -I../../andweorc/include

# Targets - only examples that do FULL CAUSAL PROFILING
TARGETS = cache_contention dedup_hash signal_trigger

.PHONY: all clean run-cache run-dedup run-signal run-all

all: $(TARGETS)

cache_contention: cache_contention.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

dedup_hash: dedup_hash.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

signal_trigger: signal_trigger.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)

# Run targets with causal profiling enabled
run-cache: cache_contention
	@echo "=== Running cache_contention with causal profiling ==="
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./cache_contention

run-dedup: dedup_hash
	@echo "=== Running dedup_hash with causal profiling ==="
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./dedup_hash

run-signal: signal_trigger
	@echo "=== Running signal_trigger with causal profiling (auto-trigger) ==="
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ANDWEORC_EXPERIMENT_TARGET="work_done" ./signal_trigger --auto

# Run all examples sequentially
run-all: $(TARGETS)
	@echo "=== Running all examples with causal profiling ==="
	@echo ""
	@echo "--- Cache Contention (Memcached-inspired) ---"
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./cache_contention
	@echo ""
	@echo "--- Dedup Hash Table (coz paper inspired) ---"
	LD_PRELOAD=$(ANDWEORC_LIB) ANDWEORC_ENABLED=1 ./dedup_hash
