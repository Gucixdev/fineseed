//! An ordered parallel pipeline: jobs come out of an iterator, `work` runs on all cores, and `sink` sees the results
//! in the order the jobs were made. At most `window` jobs are in flight (made but not yet through `sink`), so memory
//! stays bounded however slow the sink is. This replaces "take a batch, spawn a thread per block, wait for the
//! slowest, write" which left every core idle while the batch was read and while it was written.
//!
//! The first error, from the iterator, from `work` or from `sink`, ends the run; results before it are still
//! delivered in order.
use crate::Res;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Condvar, Mutex};

struct Feed<I> {
    jobs: I,
    next: usize,
    exhausted: bool,
}

/// Threads to use: one per core, at most 32, at least 1.
pub fn threads() -> usize {
    std::thread::available_parallelism().map_or(2, |n| n.get()).clamp(1, 32)
}

pub fn run<J, T, I, W, S>(threads: usize, window: usize, jobs: I, work: W, mut sink: S) -> Res<()>
where
    J: Send,
    T: Send,
    I: Iterator<Item = Res<J>> + Send,
    W: Fn(usize, J) -> Res<T> + Sync,
    S: FnMut(usize, T) -> Res<()>,
{
    let threads = threads.max(1);
    let window = window.max(threads);
    let feed = Mutex::new(Feed {
        jobs,
        next: 0,
        exhausted: false,
    });
    let room = Condvar::new();
    let consumed = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    let (tx, rx) = mpsc::channel::<(usize, Res<T>)>();
    let mut result: Res<()> = Ok(());
    std::thread::scope(|s| {
        for _ in 0..threads {
            let tx = tx.clone();
            let (feed, room, consumed, abort, work) = (&feed, &room, &consumed, &abort, &work);
            s.spawn(move || {
                loop {
                    let (idx, job) = {
                        let Ok(mut g) = feed.lock() else { return };
                        loop {
                            if abort.load(Ordering::Relaxed) || g.exhausted {
                                return;
                            }
                            if g.next - consumed.load(Ordering::Acquire) < window {
                                break;
                            }
                            g = match room.wait(g) {
                                Ok(g) => g,
                                Err(_) => return,
                            };
                        }
                        let idx = g.next;
                        match g.jobs.next() {
                            None => {
                                g.exhausted = true;
                                room.notify_all();
                                return;
                            }
                            Some(Err(e)) => {
                                g.exhausted = true;
                                g.next += 1;
                                room.notify_all();
                                let _ = tx.send((idx, Err(e)));
                                return;
                            }
                            Some(Ok(j)) => {
                                g.next += 1;
                                (idx, j)
                            }
                        }
                    };
                    let r = work(idx, job);
                    let failed = r.is_err();
                    if tx.send((idx, r)).is_err() || failed {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut pending: BTreeMap<usize, Res<T>> = BTreeMap::new();
        let mut want = 0usize;
        'recv: while let Ok((idx, r)) = rx.recv() {
            pending.insert(idx, r);
            while let Some(r) = pending.remove(&want) {
                let step = r.and_then(|t| sink(want, t));
                if let Err(e) = step {
                    result = Err(e);
                    abort.store(true, Ordering::Relaxed);
                    drop(feed.lock());
                    room.notify_all();
                    break 'recv;
                }
                want += 1;
                consumed.store(want, Ordering::Release);
                // taking the lock first means no worker can miss the wake-up between its check and its wait
                drop(feed.lock());
                room.notify_all();
            }
        }
        abort.store(true, Ordering::Relaxed);
        drop(feed.lock());
        room.notify_all();
        drop(rx);
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_arrive_in_order() {
        let mut got = Vec::new();
        run(
            4,
            8,
            (0..1000usize).map(Ok),
            |_, j| {
                if j % 7 == 0 {
                    std::thread::yield_now();
                }
                Ok(j * 2)
            },
            |i, t| {
                assert_eq!(t, i * 2);
                got.push(t);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(got.len(), 1000);
    }

    #[test]
    fn first_error_stops_the_run() {
        let mut seen = 0;
        let r = run(
            3,
            6,
            (0..100usize).map(Ok),
            |_, j| if j == 10 { Err("boom".to_string()) } else { Ok(j) },
            |_, _| {
                seen += 1;
                Ok(())
            },
        );
        assert_eq!(r, Err("boom".to_string()));
        assert_eq!(seen, 10);
    }
}
