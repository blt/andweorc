use std::fs::File;
use std::io::Write;
use std::num::NonZeroU64;
use std::sync::mpsc::sync_channel;
use std::thread;

use andweorc::progress;

fn collatz(n: NonZeroU64) -> u64 {
    let mut cur = n.get();
    let mut iters = 0;
    while cur != 1 {
        iters += 1;
        if cur % 2 == 0 {
            cur = cur / 2;
        } else {
            cur = (3 * cur) + 1;
        }
    }
    iters
}

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
                println!("[{}] spawned new thread", id);
                let snd = thr_snd;
                for i in 1..inner {
                    let ctz: u64 = collatz(NonZeroU64::new(i).unwrap());
                    snd.send(ctz).expect("failed to send collatz number");
                }
                progress!("collatz_done");
                println!("[{}] thread is done", id);
            })
            .unwrap();
    }
    drop(snd);

    let mut i = 0;
    while let Ok(ctz) = rcv.recv() {
        writeln!(fp, "{} {}", i, ctz).unwrap();
        i += 1;
        fp.flush().unwrap();
        if i == total {
            break;
        }
    }

    Ok(())
}
