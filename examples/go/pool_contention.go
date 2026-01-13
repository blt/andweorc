/*
Package main demonstrates causal profiling on connection pool contention.

This example shows a common pattern in database clients and network services
where multiple goroutines compete for a limited pool of connections.

The bottleneck patterns are:
1. Pool lock contention when acquiring/releasing connections
2. Blocking when all connections are in use
3. Work done while holding a connection (simulated query time)

BUILD:
  cd examples/go
  go build -o pool_contention pool_contention.go

RUN (without profiling):
  ./pool_contention

RUN (with profiling):
  LD_PRELOAD=/path/to/libandweorc.so ANDWEORC_ENABLED=1 ./pool_contention

EXPECTED RESULTS:
  The profiler should identify:
  - sync.Mutex operations as high-impact when pool is saturated
  - Simulated query work as high-impact when pool is large enough

NOTES:
  - Go's sync.Mutex uses pthread_mutex under the hood
  - sync.Cond.Wait uses pthread_cond_wait (both intercepted by LD_PRELOAD)
  - CGO_ENABLED=1 is required for LD_PRELOAD interception
*/
package main

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Configuration
const (
	PoolSize         = 4          // Small pool to maximize contention
	NumWorkers       = 16         // More workers than connections
	RunDuration      = 30 * time.Second
	SimulatedQueryNs = 100_000    // 100us simulated query time
)

// Connection represents a pooled resource (database connection, socket, etc.)
type Connection struct {
	id       int
	useCount int64
}

// ConnectionPool manages a fixed set of connections with FIFO wait queue
type ConnectionPool struct {
	mu          sync.Mutex
	cond        *sync.Cond
	connections []*Connection
	available   []*Connection
	inUse       int
	waiters     int64

	// Stats
	acquires     int64
	releases     int64
	waits        int64
	totalWaitNs  int64
}

// NewConnectionPool creates a pool with the given capacity
func NewConnectionPool(size int) *ConnectionPool {
	pool := &ConnectionPool{
		connections: make([]*Connection, size),
		available:   make([]*Connection, 0, size),
	}
	pool.cond = sync.NewCond(&pool.mu)

	// Initialize connections
	for i := 0; i < size; i++ {
		conn := &Connection{id: i}
		pool.connections[i] = conn
		pool.available = append(pool.available, conn)
	}

	return pool
}

// Acquire gets a connection from the pool, blocking if none available.
// THIS IS THE CONTENTION POINT - multiple goroutines compete for limited connections
func (p *ConnectionPool) Acquire() *Connection {
	p.mu.Lock() // CONTENTION: Lock protects pool state
	defer p.mu.Unlock()

	waitStart := time.Now()
	waited := false

	// Wait for an available connection
	for len(p.available) == 0 {
		p.waiters++
		waited = true
		p.cond.Wait() // CONTENTION: Blocks on condition variable
		p.waiters--
	}

	if waited {
		p.waits++
		p.totalWaitNs += time.Since(waitStart).Nanoseconds()
	}

	// Get connection from available pool
	n := len(p.available)
	conn := p.available[n-1]
	p.available = p.available[:n-1]
	p.inUse++
	p.acquires++

	return conn
}

// Release returns a connection to the pool
func (p *ConnectionPool) Release(conn *Connection) {
	p.mu.Lock() // CONTENTION: Lock for release
	defer p.mu.Unlock()

	conn.useCount++
	p.available = append(p.available, conn)
	p.inUse--
	p.releases++

	// Signal one waiting goroutine
	if p.waiters > 0 {
		p.cond.Signal()
	}
}

// Stats returns pool statistics
func (p *ConnectionPool) Stats() (acquires, releases, waits int64, avgWaitUs float64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	acquires = p.acquires
	releases = p.releases
	waits = p.waits

	if waits > 0 {
		avgWaitUs = float64(p.totalWaitNs) / float64(waits) / 1000.0
	}

	return
}

// simulateQuery does work while holding a connection
// This represents actual database query or network operation time
func simulateQuery(conn *Connection) int64 {
	// Simulate query work
	var result int64 = 0
	iterations := SimulatedQueryNs / 10 // Rough approximation

	for i := int64(0); i < iterations; i++ {
		result ^= i
		result = result*0x5851F42D4C957F2D + 1
	}

	return result
}

// Worker function - repeatedly acquires connection, does work, releases
func worker(id int, pool *ConnectionPool, ops *int64, running *int32) {
	for atomic.LoadInt32(running) == 1 {
		// Acquire connection (may block)
		conn := pool.Acquire()

		// Simulate query work while holding connection
		_ = simulateQuery(conn)

		// Release connection
		pool.Release(conn)

		// Count operation
		atomic.AddInt64(ops, 1)
	}
}

func main() {
	fmt.Println("=== Connection Pool Contention Example ===")
	fmt.Println()
	fmt.Println("This example demonstrates contention in a connection pool,")
	fmt.Println("a common pattern in database clients and network services.")
	fmt.Println()
	fmt.Printf("Configuration:\n")
	fmt.Printf("  Pool size:    %d connections\n", PoolSize)
	fmt.Printf("  Workers:      %d goroutines (%.0fx oversubscription)\n",
		NumWorkers, float64(NumWorkers)/float64(PoolSize))
	fmt.Printf("  Query time:   %d ns (simulated)\n", SimulatedQueryNs)
	fmt.Printf("  Duration:     %v\n", RunDuration)
	fmt.Println()
	fmt.Println("Expected profiling results:")
	fmt.Println("  - High impact for sync.Mutex.Lock in Acquire/Release")
	fmt.Println("  - High impact for sync.Cond.Wait when pool is saturated")
	fmt.Println()

	// Create pool
	pool := NewConnectionPool(PoolSize)

	// Start workers
	fmt.Printf("Starting %d workers competing for %d connections...\n\n", NumWorkers, PoolSize)

	var totalOps int64
	var running int32 = 1
	var wg sync.WaitGroup

	start := time.Now()

	for i := 0; i < NumWorkers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			worker(id, pool, &totalOps, &running)
		}(i)
	}

	// Run for duration
	time.Sleep(RunDuration)
	atomic.StoreInt32(&running, 0)
	wg.Wait()

	elapsed := time.Since(start)

	// Get stats
	acquires, releases, waits, avgWaitUs := pool.Stats()
	ops := atomic.LoadInt64(&totalOps)

	fmt.Println("=== Results ===")
	fmt.Printf("Total operations: %d\n", ops)
	fmt.Printf("Elapsed time:     %.2f seconds\n", elapsed.Seconds())
	fmt.Printf("Throughput:       %.0f ops/sec\n", float64(ops)/elapsed.Seconds())
	fmt.Println()
	fmt.Printf("Pool statistics:\n")
	fmt.Printf("  Acquires:       %d\n", acquires)
	fmt.Printf("  Releases:       %d\n", releases)
	fmt.Printf("  Blocked waits:  %d\n", waits)
	fmt.Printf("  Avg wait time:  %.2f us\n", avgWaitUs)
	fmt.Printf("  Wait ratio:     %.1f%%\n", float64(waits)*100.0/float64(acquires))
	fmt.Println()

	// Print connection usage distribution
	fmt.Println("Connection usage:")
	for _, conn := range pool.connections {
		fmt.Printf("  Connection %d: %d uses\n", conn.id, conn.useCount)
	}
	fmt.Println()

	fmt.Println("=== Profiling Analysis ===")
	fmt.Println("If run with causal profiling enabled, you should see:")
	fmt.Println("  - sync.Mutex: HIGH impact (pool lock is bottleneck)")
	fmt.Println("  - sync.Cond.Wait: Impact depends on saturation level")
	fmt.Println()
	fmt.Println("Potential optimizations (guided by profiling):")
	fmt.Println("  - Increase pool size to reduce contention")
	fmt.Println("  - Use connection pooling libraries with better strategies")
	fmt.Println("  - Implement connection reuse within goroutines")
	fmt.Println("  - Consider lock-free pool implementations")
}
