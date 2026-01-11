use core::mem;
use libc::{
    c_int,
    c_void,
    pthread_attr_t,
    // pthread_cond_t, pthread_mutex_t, pthread_rwlock_t,
    pthread_t,
    // sigset_t,
    // sigval,
    // timespec,
    RTLD_NEXT,
};
use nix::sys::pthread::pthread_self;
use spin::Once;
// use std::ffi::{CStr, CString};

// //
// // pthread_tryjoin_np
// //
// static PTHREAD_TRYJOIN_NP: Lazy<extern "C" fn(pthread_t, *mut *mut c_void) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_tryjoin_np").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(pthread_t, *mut *mut c_void) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_tryjoin_np(thread: pthread_t, retval: *mut *mut c_void) -> c_int {
//     PTHREAD_TRYJOIN_NP(thread, retval)
// }

// //
// // pthread_timedjoin_np
// //
// static PTHREAD_TIMEDJOIN_NP: Lazy<
//     extern "C" fn(pthread_t, *mut *mut c_void, *const timespec) -> c_int,
// > = Lazy::new(|| unsafe {
//     let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_timedjoin_np").as_ptr());
//     mem::transmute::<
//         *mut c_void,
//         extern "C" fn(pthread_t, *mut *mut c_void, *const timespec) -> c_int,
//     >(ptr)
// });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_timedjoin_np(
//     thread: pthread_t,
//     retval: *mut *mut c_void,
//     abstime: *const timespec,
// ) -> c_int {
//     PTHREAD_TIMEDJOIN_NP(thread, retval, abstime)
// }

// //
// // pthread_sigmask
// //
// static PTHREAD_SIGMASK: Lazy<extern "C" fn(c_int, *const sigset_t, *mut sigset_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_sigmask").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(c_int, *const sigset_t, *mut sigset_t) -> c_int>(
//             ptr,
//         )
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_sigmask(
//     how: c_int,
//     set: *const sigset_t,
//     oldset: *mut sigset_t,
// ) -> c_int {
//     PTHREAD_SIGMASK(how, set, oldset)
// }

// //
// // pthread_kill
// //
// static PTHREAD_KILL: Lazy<extern "C" fn(pthread_t, c_int) -> c_int> = Lazy::new(|| unsafe {
//     let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_kill").as_ptr());
//     mem::transmute::<*mut c_void, extern "C" fn(pthread_t, c_int) -> c_int>(ptr)
// });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_kill(thread: pthread_t, sig: c_int) -> c_int {
//     PTHREAD_KILL(thread, sig)
// }

// //
// // pthread_sigqueue
// //
// static PTHREAD_SIGQUEUE: Lazy<extern "C" fn(pthread_t, c_int, sigval) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_sigqueue").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(pthread_t, c_int, sigval) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_sigqueue(thread: pthread_t, sig: c_int, value: sigval) -> c_int {
//     PTHREAD_SIGQUEUE(thread, sig, value)
// }

// //
// // pthread_mutex_lock
// //
// static PTHREAD_MUTEX_LOCK: Lazy<extern "C" fn(*mut pthread_mutex_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_mutex_lock").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_mutex_t) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_mutex_lock(mutex: *mut pthread_mutex_t) -> c_int {
//     PTHREAD_MUTEX_LOCK(mutex)
// }

// //
// // pthread_mutex_trylock
// //
// static PTHREAD_MUTEX_TRYLOCK: Lazy<extern "C" fn(*mut pthread_mutex_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_mutex_trylock").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_mutex_t) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_mutex_trylock(mutex: *mut pthread_mutex_t) -> c_int {
//     PTHREAD_MUTEX_TRYLOCK(mutex)
// }

// //
// // pthread_mutex_unlock
// //
// static PTHREAD_MUTEX_UNLOCK: Lazy<extern "C" fn(*mut pthread_mutex_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_mutex_unlock").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_mutex_t) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_mutex_unlock(mutex: *mut pthread_mutex_t) -> c_int {
//     PTHREAD_MUTEX_UNLOCK(mutex)
// }

// //
// // pthread_cond_timedwait
// //
// static PTHREAD_COND_TIMEDWAIT: Lazy<
//     extern "C" fn(*mut pthread_cond_t, *mut pthread_mutex_t, *const timespec) -> c_int,
// > = Lazy::new(|| unsafe {
//     let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_cond_timedwait").as_ptr());
//     mem::transmute::<
//         *mut c_void,
//         extern "C" fn(*mut pthread_cond_t, *mut pthread_mutex_t, *const timespec) -> c_int,
//     >(ptr)
// });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_cond_timedwait(
//     cond: *mut pthread_cond_t,
//     mutex: *mut pthread_mutex_t,
//     abstime: *const timespec,
// ) -> c_int {
//     PTHREAD_COND_TIMEDWAIT(cond, mutex, abstime)
// }

// //
// // pthread_cond_wait
// //
// static PTHREAD_COND_WAIT: Lazy<extern "C" fn(*mut pthread_cond_t, *mut pthread_mutex_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_cond_wait").as_ptr());
//         mem::transmute::<
//             *mut c_void,
//             extern "C" fn(*mut pthread_cond_t, *mut pthread_mutex_t) -> c_int,
//         >(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_cond_wait(
//     cond: *mut pthread_cond_t,
//     mutex: *mut pthread_mutex_t,
// ) -> c_int {
//     PTHREAD_COND_WAIT(cond, mutex)
// }

// //
// // pthread_cond_signal
// //
// static PTHREAD_COND_SIGNAL: Lazy<extern "C" fn(*mut pthread_cond_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_cond_signal").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_cond_t) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_cond_signal(cond: *mut pthread_cond_t) -> c_int {
//     PTHREAD_COND_SIGNAL(cond)
// }

// //
// // pthread_cond_broadcast
// //
// static PTHREAD_COND_BROADCAST: Lazy<extern "C" fn(*mut pthread_cond_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_cond_broadcast").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_cond_t) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_cond_broadcast(cond: *mut pthread_cond_t) -> c_int {
//     PTHREAD_COND_BROADCAST(cond)
// }

// //
// // pthread_rwlock_rdlock
// //
// static PTHREAD_RWLOCK_RDLOCK: Lazy<extern "C" fn(*mut pthread_rwlock_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_rwlock_rdlock").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_rwlock_t) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_rwlock_rdlock(rwlock: *mut pthread_rwlock_t) -> c_int {
//     PTHREAD_RWLOCK_RDLOCK(rwlock)
// }

// //
// // pthread_rwlock_tryrdlock
// //
// static PTHREAD_RWLOCK_TRYRDLOCK: Lazy<extern "C" fn(*mut pthread_rwlock_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_rwlock_tryrdlock").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_rwlock_t) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_rwlock_tryrdlock(rwlock: *mut pthread_rwlock_t) -> c_int {
//     PTHREAD_RWLOCK_TRYRDLOCK(rwlock)
// }

// //
// // pthread_rwlock_timedrdlock
// //
// static PTHREAD_RWLOCK_TIMEDRDLOCK: Lazy<
//     extern "C" fn(*mut pthread_rwlock_t, *const timespec) -> c_int,
// > = Lazy::new(|| unsafe {
//     let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_rwlock_timedrdlock").as_ptr());
//     mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_rwlock_t, *const timespec) -> c_int>(
//         ptr,
//     )
// });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_rwlock_timedrdlock(
//     rwlock: *mut pthread_rwlock_t,
//     abstime: *const timespec,
// ) -> c_int {
//     PTHREAD_RWLOCK_TIMEDRDLOCK(rwlock, abstime)
// }

// //
// // pthread_rwlock_wrlock
// //
// static PTHREAD_RWLOCK_WRLOCK: Lazy<extern "C" fn(*mut pthread_rwlock_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_rwlock_wrlock").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_rwlock_t) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_rwlock_wrlock(rwlock: *mut pthread_rwlock_t) -> c_int {
//     PTHREAD_RWLOCK_WRLOCK(rwlock)
// }

// //
// // pthread_rwlock_trywrlock
// //
// static PTHREAD_RWLOCK_TRYWRLOCK: Lazy<extern "C" fn(*mut pthread_rwlock_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_rwlock_trywrlock").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_rwlock_t) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_rwlock_trywrlock(rwlock: *mut pthread_rwlock_t) -> c_int {
//     PTHREAD_RWLOCK_TRYWRLOCK(rwlock)
// }

// //
// // pthread_rwlock_timedwrlock
// //
// static PTHREAD_RWLOCK_TIMEDWRLOCK: Lazy<
//     extern "C" fn(*mut pthread_rwlock_t, *const timespec) -> c_int,
// > = Lazy::new(|| unsafe {
//     let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_rwlock_timedwrlock").as_ptr());
//     mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_rwlock_t, *const timespec) -> c_int>(
//         ptr,
//     )
// });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_rwlock_timedwrlock(
//     rwlock: *mut pthread_rwlock_t,
//     abstime: *const timespec,
// ) -> c_int {
//     PTHREAD_RWLOCK_TIMEDWRLOCK(rwlock, abstime)
// }

// //
// // pthread_rwlock_unlock
// //
// static PTHREAD_RWLOCK_UNLOCK: Lazy<extern "C" fn(*mut pthread_rwlock_t) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!("pthread_rwlock_unlock").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(*mut pthread_rwlock_t) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_rwlock_unlock(rwlock: *mut pthread_rwlock_t) -> c_int {
//     PTHREAD_RWLOCK_UNLOCK(rwlock)
// }

//
// pthread_exit
//
static PTHREAD_EXIT: Once<extern "C" fn(*const c_void) -> !> = Once::new();
/// Intercepts `pthread_exit` for profiler bookkeeping.
///
/// # Safety
///
/// This function is called from C code via the cdylib. The caller must ensure
/// that `retval` is valid or null.
///
/// # Panics
///
/// Panics if dlsym cannot find the real `pthread_exit` function. This should
/// only happen if the module is used incorrectly (not via LD_PRELOAD).
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI, not Rust module visibility
pub unsafe extern "C" fn pthread_exit(retval: *const c_void) -> ! {
    // crate::experiment::EXPERIMENT.deregister_thread(pthread_self());

    PTHREAD_EXIT.call_once(|| {
        let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"pthread_exit\0".as_ptr().cast::<i8>());
        // dlsym returns NULL if RTLD_NEXT lookup fails (e.g., not loaded via LD_PRELOAD)
        assert!(!ptr.is_null(), "dlsym failed to find pthread_exit - this module requires LD_PRELOAD");
        mem::transmute::<*mut c_void, extern "C" fn(*const c_void) -> !>(ptr)
    })(retval)
}

// //
// // pthread_join
// //
// static PTHREAD_JOIN: Lazy<extern "C" fn(pthread_t, *const *const c_void) -> c_int> =
//     Lazy::new(|| unsafe {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, cstr!(b"pthread_join").as_ptr());
//         mem::transmute::<*mut c_void, extern "C" fn(pthread_t, *const *const c_void) -> c_int>(ptr)
//     });
// #[no_mangle]
// pub unsafe extern "C" fn pthread_join(thread: pthread_t, retval: *const *const c_void) -> c_int {
//     PTHREAD_JOIN(thread, retval)
// }

//
// pthread_create
//
static PTHREAD_CREATE: Once<
    extern "C" fn(
        thread: *const pthread_t,
        attr: *const pthread_attr_t,
        start_routine: unsafe extern "C" fn(*const c_void),
        arg: *const c_void,
    ) -> c_int,
> = Once::new();
/// Intercepts `pthread_create` to register new threads with the profiler.
///
/// # Safety
///
/// This function is called from C code via the cdylib. All pointer parameters
/// must be valid according to `pthread_create`'s contract.
///
/// # Panics
///
/// Panics if dlsym cannot find the real `pthread_create` function. This should
/// only happen if the module is used incorrectly (not via LD_PRELOAD).
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI, not Rust module visibility
pub unsafe extern "C" fn pthread_create(
    thread: *const pthread_t,
    attr: *const pthread_attr_t,
    start_routine: unsafe extern "C" fn(*const c_void),
    arg: *const c_void,
) -> c_int {
    crate::experiment::get_instance().register_thread(pthread_self());
    PTHREAD_CREATE.call_once(|| {
        let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"pthread_create\0".as_ptr().cast::<i8>());
        // dlsym returns NULL if RTLD_NEXT lookup fails (e.g., not loaded via LD_PRELOAD)
        assert!(!ptr.is_null(), "dlsym failed to find pthread_create - this module requires LD_PRELOAD");
        mem::transmute::<
            *mut c_void,
            extern "C" fn(
                *const pthread_t,
                *const pthread_attr_t,
                unsafe extern "C" fn(*const c_void),
                *const c_void,
            ) -> c_int,
        >(ptr)
    })(thread, attr, start_routine, arg)
}
