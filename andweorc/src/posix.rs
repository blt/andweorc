// use core::mem;
// use libc::{c_int, c_void, pid_t, sighandler_t, siginfo_t, sigset_t, RTLD_NEXT};
// pub use pthread::*;
// use spin::once::Once;

// mod pthread;

// //
// // signal
// //
// static SIGNAL: Once<extern "C" fn(c_int, sighandler_t) -> sighandler_t> = Once::new();
// #[no_mangle]
// pub unsafe extern "C" fn signal(signum: c_int, handler: sighandler_t) -> sighandler_t {
//     SIGNAL.call_once(|| {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"signal\0".as_ptr() as *const i8);
//         mem::transmute::<*mut c_void, extern "C" fn(c_int, sighandler_t) -> sighandler_t>(ptr)
//     })(signum, handler)
// }

// //
// // sigaction
// //
// static SIGACTION: Once<
//     extern "C" fn(c_int, *const libc::sigaction, *mut libc::sigaction) -> c_int,
// > = Once::new();
// #[no_mangle]
// pub unsafe extern "C" fn sigaction(
//     sig: c_int,
//     act: *const libc::sigaction,
//     oact: *mut libc::sigaction,
// ) -> c_int {
//     SIGACTION.call_once(|| {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"sigaction\0".as_ptr() as *const i8);
//         mem::transmute::<
//             *mut c_void,
//             extern "C" fn(c_int, *const libc::sigaction, *mut libc::sigaction) -> c_int,
//         >(ptr)
//     })(sig, act, oact)
// }

// //
// // sigprocmask
// //
// static SIGPROCMASK: Once<extern "C" fn(c_int, *const sigset_t, *mut sigset_t) -> c_int> =
//     Once::new();
// #[no_mangle]
// pub unsafe extern "C" fn sigprocmask(
//     how: c_int,
//     set: *const sigset_t,
//     oset: *mut sigset_t,
// ) -> c_int {
//     SIGPROCMASK.call_once(|| {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"sigprocmask".as_ptr() as *const i8);
//         mem::transmute::<*mut c_void, extern "C" fn(c_int, *const sigset_t, *mut sigset_t) -> c_int>(
//             ptr,
//         )
//     })(how, set, oset)
// }

// //
// // sigwait
// //
// static SIGWAIT: Once<extern "C" fn(*const sigset_t, *mut c_int) -> c_int> = Once::new();
// #[no_mangle]
// pub unsafe extern "C" fn sigwait(set: *const sigset_t, info: *mut c_int) -> c_int {
//     SIGWAIT.call_once(|| {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"sigwait".as_ptr() as *const i8);
//         mem::transmute::<*mut c_void, extern "C" fn(*const sigset_t, *mut c_int) -> c_int>(ptr)
//     })(set, info)
// }

// //
// // sigwaitinfo
// //
// static SIGWAITINFO: Once<extern "C" fn(*const sigset_t, *mut siginfo_t) -> c_int> = Once::new();
// #[no_mangle]
// pub unsafe extern "C" fn sigwaitinfo(set: *const sigset_t, info: *mut siginfo_t) -> c_int {
//     SIGWAITINFO.call_once(|| {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"sigwaitinfo".as_ptr() as *const i8);
//         mem::transmute::<*mut c_void, extern "C" fn(*const sigset_t, *mut siginfo_t) -> c_int>(ptr)
//     })(set, info)
// }

// //
// // sigtimedwait
// //
// static SIGTIMEDWAIT: Once<extern "C" fn(*const sigset_t, *mut siginfo_t) -> c_int> = Once::new();
// #[no_mangle]
// pub unsafe extern "C" fn sigtimedwait(set: *const sigset_t, info: *mut siginfo_t) -> c_int {
//     SIGTIMEDWAIT.call_once(|| {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"sigtimedwait".as_ptr() as *const i8);
//         mem::transmute::<*mut c_void, extern "C" fn(*const sigset_t, *mut siginfo_t) -> c_int>(ptr)
//     })(set, info)
// }

// //
// // kill
// //
// static KILL: Once<extern "C" fn(pid: pid_t, sig: c_int) -> c_int> = Once::new();
// #[no_mangle]
// pub unsafe extern "C" fn kill(pid: pid_t, sig: c_int) -> c_int {
//     KILL.call_once(|| {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"kill".as_ptr() as *const i8);
//         mem::transmute::<*mut c_void, extern "C" fn(pid_t, c_int) -> c_int>(ptr)
//     })(pid, sig)
// }

// //
// // exit
// //
// static EXIT: Once<extern "C" fn(c_int) -> !> = Once::new();
// #[no_mangle]
// pub unsafe extern "C" fn exit(status: c_int) -> ! {
//     EXIT.call_once(|| {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"exit".as_ptr() as *const i8);
//         mem::transmute::<*mut c_void, extern "C" fn(c_int) -> !>(ptr)
//     })(status)
// }

// //
// // _exit
// //
// static _EXIT: Once<extern "C" fn(c_int) -> !> = Once::new();
// #[no_mangle]
// pub unsafe extern "C" fn _exit(status: c_int) -> ! {
//     _EXIT.call_once(|| {
//         let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"_exit".as_ptr() as *const i8);
//         mem::transmute::<*mut c_void, extern "C" fn(c_int) -> !>(ptr)
//     })(status)
// }

// //
// // fork
// //
// static FORK: Lazy<extern "C" fn(c_void) -> ()> = Lazy::new(|| unsafe {
//     let ptr: *mut c_void = libc::dlsym(RTLD_NEXT, b"fork".as_ptr() as *const i8);
//     mem::transmute::<*mut c_void, extern "C" fn(c_void) -> ()>(ptr)
// });
// #[no_mangle]
// pub unsafe extern "C" fn fork(inp: c_void) -> () {
//     FORK(inp)
// }
