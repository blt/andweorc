# andweorc - a Rust causal profiler

NOTE This is a seriously work-in-progress kind of thing. I regularly rebase the
commit history. Please be aware of this if you fork. I'll cut that out once this
is miminally usable.

Inspired by [criterion][criterion] and [coz][coz].

'andweorc' an Old English for 'cause': [Wiktionary][wiktionary].

Coz profiler promises to determine the slow spots in unmofied program, assuming
some conditions are met. In practice the most interesting runs with coz are done
over modified programs. Criterion runs as a sub-program in a project and does
careful timing a benchmarking function. Coz needs to fork to do its LD_PRELOAD
trick, criterion does not. Coz overrides the following functions:

* `exit`
* `_exit`
* `_Exit`
* `fork`
* `sigaction`
* `signal`
* `kill`
* `sigprocmask`
* `sigwait`
* `sigwaitinfo`
* `sigtimedwait`
* `pthread_create`
* `pthread_exit`
* `pthread_join`
* `pthread_tryjoin_np`
* `pthread_timedjoin_np`
* `pthread_sigmask`
* `pthread_kill`
* `pthread_sigqueue`
* `pthread_mutex_lock`
* `pthread_mutex_trylock`
* `pthread_mutex_unlock`
* `pthread_cond_wait`
* `pthread_cond_timedwait`
* `pthread_cond_signal`
* `pthread_cond_broadcast`
* `pthread_barrier_wait`
* `pthread_rwlock_rdlock`
* `pthread_rwlock_tryrdlock`
* `pthread_rwlock_timedrdlock`
* `pthread_rwlock_wrlock`
* `pthread_rwlock_trywrlock`
* `pthread_rwlock_timedwrlock`
* `pthread_rwlock_unlock`

https://users.rust-lang.org/t/override-c-standard-functions-with-ld-preload-without-using-crates/40893/2
https://stackoverflow.com/questions/426230/what-is-the-ld-preload-trick
https://stackoverflow.com/questions/48270676/mac-assembly-segfault-with-libc-exit
https://stackoverflow.com/questions/65099252/how-to-use-dlsym-in-rust?noredirect=1&lq=1
https://internals.rust-lang.org/t/ld-preload-to-intercept-readlink-of-rustc-compiler-hangs/13000

NOTE:

To make this simpler we DO NOT support forking.

[criterion]: https://github.com/bheisler/criterion.rs
[coz]: https://github.com/plasma-umass/coz
[Wiktionary]: https://en.wiktionary.org/wiki/andweorc#Old_English
