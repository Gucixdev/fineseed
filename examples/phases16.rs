//! Time of each phase of unpacking F16 data, one thread: `cargo run --release --example phases16 FILE [MiB]`.
//! The three planes (exponent, sign and top mantissa bits, low mantissa byte) are coded and decoded one by one.
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mib: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let all = std::fs::read(&a[1]).expect("read");
    let start = 1 << 16;
    let data = &all[start..(start + (mib << 20)).min(all.len())];
    let mb = data.len() as f64 / 1e6;
    let s = Instant::now();
    let planes = fineseed::split::split(fineseed::split::F16, data);
    println!("split F16               {:7.1} MB/s", mb / s.elapsed().as_secs_f64());
    let mut coded = Vec::new();
    for (i, p) in planes.iter().enumerate() {
        let t = fineseed::huffman::table(p);
        let c = fineseed::huffman::encode_chunked(p, &t).expect("non-empty");
        println!("plane {i}: {} -> {} ({:.1} %)", p.len(), c.len(), 100.0 * c.len() as f64 / p.len() as f64);
        coded.push(c);
    }
    let mut dec = Vec::new();
    for (i, (c, p)) in coded.iter().zip(&planes).enumerate() {
        let best = (0..5)
            .map(|_| {
                let s = Instant::now();
                let d = fineseed::huffman::decode_chunked(c, p.len()).expect("decodes");
                assert!(d == *p);
                s.elapsed().as_secs_f64()
            })
            .fold(f64::MAX, f64::min);
        println!("decode plane {i}         {:7.1} MB/s of the plane", p.len() as f64 / 1e6 / best);
        dec.push(best);
    }
    println!("decode planes 0 and 1   {:7.1} MB/s of the data (plane 2 stays raw)", mb / (dec[0] + dec[1]));
    let mut out = vec![0u8; data.len()];
    let s = Instant::now();
    fineseed::split::merge(fineseed::split::F16, &planes, &mut out).expect("merge");
    println!("merge F16               {:7.1} MB/s", mb / s.elapsed().as_secs_f64());
    assert!(out == data);
    let s = Instant::now();
    let h = blake3::hash(data);
    println!("blake3                  {:7.1} MB/s", mb / s.elapsed().as_secs_f64());
    let _ = h;
}
