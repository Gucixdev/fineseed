//! Time of each phase of coding one BF16 block, one thread: `cargo run --release --example phases FILE [MiB]`.
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mib: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let all = std::fs::read(&a[1]).expect("read");
    let data = &all[1 << 20..(1 << 20) + (mib << 20)];
    let mb = data.len() as f64 / 1e6;
    let t = |name: &str, s: Instant| println!("{name:<22} {:7.1} MB/s", mb / s.elapsed().as_secs_f64());
    let s = Instant::now();
    let planes = fineseed::planes::split(data, 2);
    t("planes::split", s);
    let s = Instant::now();
    let planes2 = fineseed::split::split(fineseed::split::BF16, data);
    t("split::split BF16", s);
    let _ = planes2;
    let s = Instant::now();
    let tabs: Vec<_> = planes.iter().map(|p| fineseed::huffman::table(p)).collect();
    t("huffman::table", s);
    let s = Instant::now();
    let coded: Vec<_> = planes.iter().zip(&tabs).map(|(p, tb)| fineseed::huffman::encode_chunked(p, tb).unwrap()).collect();
    t("huffman::encode_chunked", s);
    let s = Instant::now();
    for (c, p) in coded.iter().zip(&planes) {
        let d = fineseed::huffman::decode_chunked(c, p.len()).unwrap();
        assert!(d == *p);
    }
    t("huffman::decode", s);
    let s = Instant::now();
    let mut out = vec![0u8; data.len()];
    fineseed::planes::merge(&planes, &mut out);
    t("planes::merge", s);
    assert!(out == data);
    let s = Instant::now();
    let h = blake3::hash(data);
    t("blake3", s);
    let _ = h;
    let s = Instant::now();
    let z = zstd::bulk::compress(&planes[0], 3).unwrap();
    println!("zstd3 on plane0 {:.1} MB/s ({} -> {})", planes[0].len() as f64 / 1e6 / s.elapsed().as_secs_f64(), planes[0].len(), z.len());
    let s = Instant::now();
    let z = zstd::bulk::compress(&planes[1], 3).unwrap();
    println!("zstd3 on plane1 {:.1} MB/s ({} -> {})", planes[1].len() as f64 / 1e6 / s.elapsed().as_secs_f64(), planes[1].len(), z.len());
}
