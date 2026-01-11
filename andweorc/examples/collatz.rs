//! Example: Multi-threaded Collatz sequence computation with progress tracking.
//!
//! This example spawns multiple threads that compute Collatz sequences and
//! demonstrates how to use progress points to mark work completion.
//!
//! The Collatz conjecture states that repeatedly applying the following rules
//! to any positive integer will eventually reach 1:
//! - If n is even: n → n/2
//! - If n is odd: n → 3n + 1

// Examples are demonstration code - allow more relaxed rules
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::print_stdout)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::manual_is_multiple_of)]

use std::fs::File;
use std::io::Write;
use std::num::NonZeroU64;
use std::sync::mpsc::sync_channel;
use std::thread;

use andweorc::progress;

/// Computes the number of steps to reach 1 in the Collatz sequence.
fn collatz(n: NonZeroU64) -> u64 {
    let mut cur = n.get();
    let mut iters = 0;
    while cur != 1 {
        iters += 1;
        if cur % 2 == 0 {
            cur /= 2;
        } else {
            cur = (3 * cur) + 1;
        }
    }
    iters
}

/// Entry point for the collatz example.
fn main() -> std::io::Result<()> {
    let mut fp = File::create("/tmp/collatz.txt").unwrap();
    let (snd, rcv) = sync_channel(8);

    let outer = 100;
    let inner = 10_000;
    let total = outer * inner;

    for id in 0..outer {
        let thr_snd = snd.clone();
        let builder = thread::Builder::new();
        let _ = builder
            .spawn(move || {
                println!("[{id}] spawned new thread");
                let snd = thr_snd;
                for i in 1..inner {
                    let ctz: u64 = collatz(NonZeroU64::new(i).unwrap());
                    snd.send(ctz).expect("failed to send collatz number");
                }
                progress!("collatz_done");
                println!("[{id}] thread is done");
            })
            .unwrap();
    }
    drop(snd);

    let mut i = 0;
    while let Ok(ctz) = rcv.recv() {
        writeln!(fp, "{i} {ctz}").unwrap();
        i += 1;
        fp.flush().unwrap();
        if i == total {
            break;
        }
    }

    Ok(())
}
