//! Chunked Huffman decode on the GPU against the CPU: `cargo run --release --features gpu --example gpu_huff FILE [MiB]`.
use fineseed::huffman as h;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mib: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let all = std::fs::read(&a[1]).expect("read");
    let data = &all[1 << 20..(1 << 20) + (mib << 20)];
    let Some(gpu) = fineseed::gpu::Gpu::new() else {
        println!("no GPU");
        return;
    };
    println!("GPU: {}", gpu.name);
    for (i, p) in fineseed::planes::split(data, 2).iter().enumerate() {
        let mb = p.len() as f64 / 1e6;
        let chunked = h::encode_chunked(p, &h::table(p)).unwrap();
        let s = Instant::now();
        let cpu = h::decode_chunked(&chunked, p.len()).unwrap();
        let tc = s.elapsed().as_secs_f64();
        assert!(cpu == *p);
        println!("plane {i}: {} -> {} bytes; cpu 1 thread {:.0} MB/s", p.len(), chunked.len(), mb / tc);
        let time = |reps| {
            let s = Instant::now();
            gpu.decode_reps(&chunked, p.len(), reps).expect("gpu");
            s.elapsed().as_secs_f64()
        };
        time(1);
        let (t1, t21) = (time(1), time(21));
        let kernel = (t21 - t1) / 20.0;
        println!("  kernel alone {:.1} ms = {:.0} MB/s; the rest (upload, download, setup) {:.1} ms", kernel * 1e3, mb / kernel, (t1 - kernel) * 1e3);
        for round in 0..2 {
            let s = Instant::now();
            let g = gpu.decode_chunked(&chunked, p.len()).expect("gpu decode");
            let t = s.elapsed().as_secs_f64();
            assert!(g == *p, "gpu output differs");
            println!("  gpu round {round}: {:.0} MB/s incl. upload and download ({:.0} ms)", mb / t, t * 1e3);
        }
    }
}
