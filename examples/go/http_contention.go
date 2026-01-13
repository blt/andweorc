/*
Package main demonstrates causal profiling on a Go HTTP server.

This example shows how LD_PRELOAD-based causal profiling works with Go programs.
The server has shared state protected by a mutex, creating contention under load.

BUILD:
  cd examples/go
  go build -o http_contention http_contention.go

RUN (without profiling):
  ./http_contention

RUN (with profiling):
  LD_PRELOAD=/path/to/libandweorc.so ANDWEORC_ENABLED=1 ./http_contention

LOAD TESTING (in another terminal):
  # Using curl in a loop
  for i in $(seq 1 1000); do curl -s http://localhost:8080/counter > /dev/null; done

  # Or using hey (https://github.com/rakyll/hey)
  hey -n 10000 -c 50 http://localhost:8080/counter

EXPECTED RESULTS:
  The profiler should identify mutex contention in the shared counter operations
  as limiting throughput. Go's sync.Mutex uses pthread_mutex under the hood,
  which is intercepted by the LD_PRELOAD mechanism.

NOTES:
  - Go uses its own goroutine scheduler, but sync primitives map to pthreads
  - CGO_ENABLED=1 is required for LD_PRELOAD interception to work
  - Progress points could be added via CGO if finer control is needed
*/
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// SharedState holds counters protected by a mutex (contention point)
type SharedState struct {
	mu       sync.Mutex
	counters map[string]int64
	reads    int64
	writes   int64
}

// Global shared state
var state = &SharedState{
	counters: make(map[string]int64),
}

// Stats for monitoring
var (
	totalRequests  int64
	totalLatencyNs int64
)

// IncrementCounter atomically increments a named counter
// This is the CONTENTION POINT - mutex protects shared map
func (s *SharedState) IncrementCounter(name string) int64 {
	s.mu.Lock() // CONTENTION: This lock is the bottleneck
	defer s.mu.Unlock()

	s.counters[name]++
	s.writes++

	// Simulate some work while holding the lock
	// (makes contention more visible)
	sum := int64(0)
	for i := 0; i < 100; i++ {
		sum += int64(i)
	}
	_ = sum

	return s.counters[name]
}

// GetCounter reads a counter value
func (s *SharedState) GetCounter(name string) int64 {
	s.mu.Lock() // CONTENTION: Same lock for reads
	defer s.mu.Unlock()

	s.reads++
	return s.counters[name]
}

// GetAllCounters returns a copy of all counters
func (s *SharedState) GetAllCounters() map[string]int64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	result := make(map[string]int64, len(s.counters))
	for k, v := range s.counters {
		result[k] = v
	}
	return result
}

// GetStats returns read/write statistics
func (s *SharedState) GetStats() (reads, writes int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reads, s.writes
}

// Handler for incrementing the counter
func counterHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	// Increment the "requests" counter (contention point)
	value := state.IncrementCounter("requests")

	// Track stats
	atomic.AddInt64(&totalRequests, 1)
	atomic.AddInt64(&totalLatencyNs, time.Since(start).Nanoseconds())

	// Respond with the new value
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int64{"value": value})
}

// Handler for reading counter stats
func statsHandler(w http.ResponseWriter, r *http.Request) {
	reads, writes := state.GetStats()
	reqs := atomic.LoadInt64(&totalRequests)
	latency := atomic.LoadInt64(&totalLatencyNs)

	avgLatencyUs := float64(0)
	if reqs > 0 {
		avgLatencyUs = float64(latency) / float64(reqs) / 1000.0
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"reads":           reads,
		"writes":          writes,
		"total_requests":  reqs,
		"avg_latency_us":  avgLatencyUs,
		"counters":        state.GetAllCounters(),
	})
}

// Handler for health check
func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func main() {
	fmt.Println("=== Go HTTP Server Contention Example ===")
	fmt.Println()
	fmt.Println("This server demonstrates mutex contention in a Go HTTP handler.")
	fmt.Println("The shared counter is protected by sync.Mutex, which maps to")
	fmt.Println("pthread_mutex_lock under the hood (intercepted by LD_PRELOAD).")
	fmt.Println()
	fmt.Println("Endpoints:")
	fmt.Println("  GET /counter  - Increment shared counter (contention point)")
	fmt.Println("  GET /stats    - View statistics")
	fmt.Println("  GET /health   - Health check")
	fmt.Println()

	// Set up routes
	http.HandleFunc("/counter", counterHandler)
	http.HandleFunc("/stats", statsHandler)
	http.HandleFunc("/health", healthHandler)

	// Start server in goroutine
	addr := ":8080"
	server := &http.Server{Addr: addr}

	go func() {
		fmt.Printf("Starting server on %s\n", addr)
		fmt.Println()
		fmt.Println("Load test with:")
		fmt.Println("  curl http://localhost:8080/counter")
		fmt.Println("  hey -n 10000 -c 50 http://localhost:8080/counter")
		fmt.Println()
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	<-sigChan
	fmt.Println("\nShutting down...")

	// Print final stats
	reads, writes := state.GetStats()
	reqs := atomic.LoadInt64(&totalRequests)
	latency := atomic.LoadInt64(&totalLatencyNs)

	fmt.Println("\n=== Final Statistics ===")
	fmt.Printf("Total requests: %d\n", reqs)
	fmt.Printf("Counter reads:  %d\n", reads)
	fmt.Printf("Counter writes: %d\n", writes)
	if reqs > 0 {
		fmt.Printf("Avg latency:    %.2f us\n", float64(latency)/float64(reqs)/1000.0)
	}
	fmt.Println()
	fmt.Println("=== Profiling Analysis ===")
	fmt.Println("If run with causal profiling enabled, you should see:")
	fmt.Println("  - High causal impact for sync.Mutex operations")
	fmt.Println("  - The shared state lock is the bottleneck under load")
	fmt.Println()
	fmt.Println("Potential optimizations (guided by profiling):")
	fmt.Println("  - Use sync.RWMutex for read-heavy workloads")
	fmt.Println("  - Shard the counter map to reduce contention")
	fmt.Println("  - Use atomic operations for simple counters")
	fmt.Println("  - Consider lock-free data structures")

	server.Close()
}
