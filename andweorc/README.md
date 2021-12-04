Needs:

 * a 'profiler thread' that selects lines to virtually speed up (profiler.cpp => profiler::profiler_thread)
   * 'selects lines' implies it knows what lines appear in practice
   * 'speed up' implies it knows how much time a line took to execute unmolested
 * per-thread event sampling, both the callstack and program counter (profiler.cpp => profiler::begin_sampling)
   * why do we want the program counter?
     * oh, is this how we figure out which line we're on when we sample?! yes!

So, per-thread samplers chunk which IP they're encountering on a timer.
 * timer sets up a signal handler, target is `profiler::samples_ready`
   * https://stackoverflow.com/questions/24643311/unix-pthreads-and-signals-per-thread-signal-handlers
   * random thread is chosen to process its samples; this explains why delay/global-delay mechanism exists

Anyway. Post-experiment the sampled line
is counted RE how many times it was hit and the progress points are also
interrogated.
