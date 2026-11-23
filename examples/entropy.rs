//! entropy FILE: bits per bf16 value for several ways of modelling the weights of the biggest tensor, to see what is worth building.
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

fn h(counts: &[u64]) -> f64 {
    let n: u64 = counts.iter().sum();
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n as f64;
            -p * p.log2()
        })
        .sum()
}

/// conditional entropy H(sym | ctx) in bits, sym < 256
fn hc(pairs: impl Iterator<Item = (u32, u8)>) -> (f64, usize) {
    let mut m: HashMap<u32, [u64; 256]> = HashMap::new();
    let mut n = 0u64;
    for (c, s) in pairs {
        m.entry(c).or_insert([0; 256])[s as usize] += 1;
        n += 1;
    }
    let tot: f64 = m
        .values()
        .map(|c| h(c) * c.iter().sum::<u64>() as f64)
        .sum();
    (tot / n as f64, m.len())
}

fn main() {
    let path = std::env::args().nth(1).expect("file");
    let mut f = File::open(&path).unwrap();
    let len = f.metadata().unwrap().len();
    let lay = fineseed::safetensors::parse(&mut f, len).expect("safetensors");
    for t in lay
        .tensors
        .iter()
        .filter(|t| t.dtype == "BF16" && t.end - t.begin > (40 << 20))
        .take(3)
    {
        let n = ((t.end - t.begin) / 2).min(16 << 20) as usize;
        let mut buf = vec![0u8; n * 2];
        f.seek(SeekFrom::Start(t.begin)).unwrap();
        f.read_exact(&mut buf).unwrap();
        let row = *t.shape.last().unwrap_or(&1) as usize;
        let hi: Vec<u8> = buf.chunks(2).map(|v| v[1]).collect();
        let lo: Vec<u8> = buf.chunks(2).map(|v| v[0]).collect();
        let e: Vec<u8> = buf.chunks(2).map(|v| (v[1] << 1) | (v[0] >> 7)).collect();
        let sm: Vec<u8> = buf
            .chunks(2)
            .map(|v| (v[1] & 0x80) | (v[0] & 0x7f))
            .collect();
        let c0 = |x: &[u8]| {
            let mut c = [0u64; 256];
            x.iter().for_each(|&b| c[b as usize] += 1);
            h(&c)
        };
        println!("{} {:?} {} values, row {row}", t.name, t.shape, n);
        println!(
            "  planes:  high byte {:.3}  low byte {:.3}  -> {:.3} bits/value",
            c0(&hi),
            c0(&lo),
            c0(&hi) + c0(&lo)
        );
        println!(
            "  fields:  exponent {:.3}  sign+mantissa {:.3}  -> {:.3}",
            c0(&e),
            c0(&sm),
            c0(&e) + c0(&sm)
        );
        let prev = (1..n).map(|i| (e[i - 1] as u32, e[i]));
        let (hp, k) = hc(prev);
        let above = (row..n).map(|i| (e[i - row] as u32, e[i]));
        let (ha, _) = hc(above);
        let both = (row + 1..n).map(|i| ((e[i - 1] as u32) << 8 | e[i - row] as u32, e[i]));
        let (hb, kb) = hc(both);
        let lastq = |x: u8| (x as u32) / 2;
        let two = (2..n).map(|i| (lastq(e[i - 1]) << 8 | lastq(e[i - 2]), e[i]));
        let (h2, _) = hc(two);
        println!(
            "  exponent given: prev {hp:.3} ({k} ctx)  above {ha:.3}  prev+above {hb:.3} ({kb} ctx, overfits)  prev2 coarse {h2:.3}"
        );
        let sgn_given_e = hc((0..n).map(|i| (e[i] as u32, sm[i] >> 7)));
        let m3 = hc((0..n).map(|i| (e[i] as u32, (sm[i] & 0x7f) >> 4)));
        println!(
            "  sign given exponent {:.3} (was 1)   top 3 mantissa bits given exponent {:.3} (was 3)",
            sgn_given_e.0, m3.0
        );
        // whole low byte but with the exponent's low bit moved out: which is what the field split gains
        println!(
            "  best simple model: exponent|prev {hp:.3} + sign+mant {:.3} = {:.3} bits/value ({:.1}% of 16)",
            c0(&sm),
            hp + c0(&sm),
            100.0 * (hp + c0(&sm)) / 16.0
        );
    }
}
