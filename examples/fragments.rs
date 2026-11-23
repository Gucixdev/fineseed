//! How `api::encode_auto` does on pieces of a real file cut like a store cuts them: cut at random places, 64..256 KiB.
//! `cargo run --release --example fragments FILE [COUNT]` prints sizes against zstd level 3 and the time of each.
use fineseed::api::{self, Options};
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("FILE");
    let count: usize = args.next().and_then(|c| c.parse().ok()).unwrap_or(200);
    let mut f = std::fs::File::open(&path).expect("open");
    let len = f.metadata().expect("stat").len();
    let opt = Options::default();
    let (mut raw, mut zs, mut fs) = (0usize, 0usize, 0usize);
    let (mut tz, mut tf, mut td) = (0f64, 0f64, 0f64);
    let mut x = 88172645463325252u64;
    for _ in 0..count {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let size = 65536 + (x >> 40) as usize % (192 * 1024);
        let at = (x >> 3) % (len - size as u64);
        let mut buf = vec![0u8; size];
        f.seek(SeekFrom::Start(at)).expect("seek");
        f.read_exact(&mut buf).expect("read");
        let t = Instant::now();
        let z = zstd::bulk::compress(&buf, 3).expect("zstd");
        tz += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let c = api::encode_auto(&buf, None, &opt).expect("encode");
        tf += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let back = api::decode_auto(&c, None).expect("decode");
        td += t.elapsed().as_secs_f64();
        assert_eq!(back, buf, "round trip at {at}");
        raw += size;
        zs += z.len().min(size);
        fs += c.len().min(size);
    }
    let mb = raw as f64 / 1e6;
    println!(
        "{count} pieces, {mb:.1} MB: zstd3 {:.1}%  fineseed {:.1}%",
        zs as f64 * 100.0 / raw as f64,
        fs as f64 * 100.0 / raw as f64
    );
    println!(
        "MB/s: zstd3 {:.0}  fineseed encode {:.0}  fineseed decode {:.0}",
        mb / tz,
        mb / tf,
        mb / td
    );
}
