//! Huffman speed per plane, one thread: `cargo run --release --example huff FILE [MiB]`.
use fineseed::huffman as h;
use std::time::Instant;
fn best<F: FnMut()>(mut f: F) -> f64 {
    (0..5).map(|_| { let s = Instant::now(); f(); s.elapsed().as_secs_f64() }).fold(f64::MAX, f64::min)
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mib: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let all = std::fs::read(&a[1]).expect("read");
    let data = &all[1 << 20..(1 << 20) + (mib << 20)];
    for (i, p) in fineseed::planes::split(data, 2).iter().enumerate() {
        let mb = p.len() as f64 / 1e6;
        let t = h::table(p);
        let one = h::encode_with(p, &t).unwrap();
        let chk = h::encode_chunked(p, &t).unwrap();
        println!("plane {i}: {} -> single {} chunked {}", p.len(), one.len(), chk.len());
        println!("  encode single  {:7.0} MB/s", mb / best(|| { std::hint::black_box(h::encode_with(p, &t)); }));
        println!("  encode chunked {:7.0} MB/s", mb / best(|| { std::hint::black_box(h::encode_chunked(p, &t)); }));
        println!("  decode single  {:7.0} MB/s", mb / best(|| { std::hint::black_box(h::decode(&one, p.len()).unwrap()); }));
        println!("  decode chunked {:7.0} MB/s", mb / best(|| { std::hint::black_box(h::decode_chunked(&chk, p.len()).unwrap()); }));
        assert!(h::decode_chunked(&chk, p.len()).unwrap() == *p);
    }
}
