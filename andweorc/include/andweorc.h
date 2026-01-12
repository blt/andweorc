/*
 * andweorc.h - C API for the Andweorc Causal Profiler
 *
 * This header provides C-compatible functions for marking progress points
 * in your code. While the profiler automatically intercepts synchronization
 * points via LD_PRELOAD, these functions allow more precise throughput
 * measurement.
 *
 * USAGE:
 *
 * 1. Build your program normally
 * 2. Run with LD_PRELOAD:
 *    LD_PRELOAD=/path/to/libandweorc.so ANDWEORC_ENABLED=1 ./your_program
 *
 * EXAMPLE:
 *
 *   #include "andweorc.h"
 *
 *   void process_request(void) {
 *       // ... do work ...
 *       andweorc_progress("request_done");
 *   }
 *
 * THREAD SAFETY:
 *
 * All functions in this header are thread-safe and can be called from any
 * thread without additional synchronization.
 *
 * PERFORMANCE:
 *
 * When profiling is not active (ANDWEORC_ENABLED is not set or set to 0),
 * all functions return immediately with minimal overhead (a single atomic load).
 */

#ifndef ANDWEORC_H
#define ANDWEORC_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Records a visit to a named progress point.
 *
 * This function marks that one unit of useful work has completed. The profiler
 * uses these markers to measure throughput and correlate it with virtual speedups.
 *
 * @param name A null-terminated string identifying the progress point.
 *             Should be a descriptive name like "requests_processed".
 *             The name is interned, so the same pointer doesn't need to be
 *             passed each time.
 *
 * Example:
 *   void handle_request(void) {
 *       // ... process request ...
 *       andweorc_progress("request_done");
 *   }
 */
void andweorc_progress(const char* name);

/**
 * Records a visit to a progress point identified by file and line number.
 *
 * This is a convenience function that creates a progress point name from
 * the source location. Best used with the ANDWEORC_PROGRESS() macro.
 *
 * @param file A null-terminated string containing the source file name.
 * @param line The line number in the source file.
 *
 * Example:
 *   void process(void) {
 *       andweorc_progress_named(__FILE__, __LINE__);
 *   }
 */
void andweorc_progress_named(const char* file, int line);

/**
 * Begins a latency measurement section.
 *
 * Call this at the start of an operation you want to measure, then call
 * andweorc_end() with the same name when the operation completes.
 *
 * @param name A null-terminated string identifying the operation.
 *
 * NOTE: Begin/end pairs must be called from the same thread.
 *       Nested pairs with different names are allowed.
 *
 * Example:
 *   void database_query(void) {
 *       andweorc_begin("db_query");
 *       // ... perform query ...
 *       andweorc_end("db_query");
 *   }
 */
void andweorc_begin(const char* name);

/**
 * Ends a latency measurement section.
 *
 * @param name A null-terminated string identifying the operation.
 *             Must match the name passed to andweorc_begin().
 */
void andweorc_end(const char* name);

/*
 * Convenience Macros
 */

/**
 * Records a progress point at the current source location.
 *
 * This macro uses __FILE__ and __LINE__ to create a unique progress point
 * name based on the source location.
 *
 * Example:
 *   void process_item(void) {
 *       // ... do work ...
 *       ANDWEORC_PROGRESS();
 *   }
 */
#define ANDWEORC_PROGRESS() andweorc_progress_named(__FILE__, __LINE__)

/**
 * Records a named progress point.
 *
 * @param name A string literal identifying the progress point.
 *
 * Example:
 *   ANDWEORC_PROGRESS_NAMED("items_processed");
 */
#define ANDWEORC_PROGRESS_NAMED(name) andweorc_progress(name)

/**
 * Measures the latency of a code block.
 *
 * This macro starts timing at the point of invocation. You must call
 * ANDWEORC_END() with the same name to complete the measurement.
 *
 * @param name A string literal identifying the operation.
 *
 * Example:
 *   void complex_operation(void) {
 *       ANDWEORC_BEGIN("complex_op");
 *       // ... do work ...
 *       ANDWEORC_END("complex_op");
 *   }
 */
#define ANDWEORC_BEGIN(name) andweorc_begin(name)

/**
 * Ends a latency measurement started with ANDWEORC_BEGIN().
 *
 * @param name A string literal matching the ANDWEORC_BEGIN() call.
 */
#define ANDWEORC_END(name) andweorc_end(name)

#ifdef __cplusplus
}
#endif

#endif /* ANDWEORC_H */
