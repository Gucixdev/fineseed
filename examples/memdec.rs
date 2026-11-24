//! Compress and decompress a whole file through `api`, bytes in and bytes out, best of five: `cargo run --release
//! --example memdec FILE`. Decompress checks the BLAKE3 of the result; nothing is written to disk. Pin the cores with taskset.
use fineseed::api::{self, Options};
use std::time::Instant;
fn main() {
    let path = std::env::args().nth(1).expect("FILE");
    let data = std::fs::read(&path).expect("read");
    let opt = Options::default();
    let best = |f: &mut dyn FnMut()| (0..5).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::MAX, f64::min);
    let mut coded = Vec::new();
    let tc = best(&mut || coded = api::compress(&data, None, &opt).expect("compress"));
    let mut back = Vec::new();
    let td = best(&mut || back = api::decompress(&coded, None, &opt).expect("decompress"));
    assert!(back == data, "roundtrip");
    let mb = data.len() as f64 / 1e6;
    println!("fineseed api: {} -> {} ({:.1} %) pack {:.0} MB/s unpack {:.0} MB/s (in memory, blake3 checked)", data.len(), coded.len(), 100.0 * coded.len() as f64 / data.len() as f64, mb / tc, mb / td);
}
