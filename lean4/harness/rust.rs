// The Rust side of the differential test: the real `split.rs` and `planes.rs` of fineseed, compiled as they are,
// driven by a seeded stream of operations. HARNESS SEED OPS prints one line per operation, `OP ARGS => RESULT`,
// bytes as hex (`-` for none). The Lean oracle in ../Oracle.lean replays the lines on the model; the outputs must
// be identical.
use std::env;

#[path = "../../src/split.rs"]
#[allow(dead_code)]
mod split;

#[path = "../../src/planes.rs"]
#[allow(dead_code)]
mod planes;

fn hex(b: &[u8]) -> String {
    if b.is_empty() {
        return "-".to_string();
    }
    b.iter().map(|x| format!("{:02x}", x)).collect::<Vec<_>>().join("")
}

fn streams(ss: &[Vec<u8>]) -> String {
    ss.iter().map(|s| hex(s)).collect::<Vec<_>>().join(" ")
}

fn fields(t: u32) -> String {
    let q = split::quant(t).unwrap();
    q.fields
        .iter()
        .map(|f| match *f {
            split::Field::Planes(o, l) => format!("p{}.{}", o, l),
            split::Field::Bytes(o, l) => format!("b{}.{}", o, l),
            split::Field::Nibs(o, l) => format!("n{}.{}", o, l),
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let n: u32 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(100);
    let mut x: u32 = a.get(1).and_then(|v| v.parse::<u32>().ok()).unwrap_or(0).wrapping_mul(2654435761).wrapping_add(1);
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    };
    let known: Vec<u32> = (0..64).filter(|t| split::quant(*t).is_some()).collect();
    let flat = [1u8, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12];
    for _ in 0..n {
        match rnd() % 5 {
            0 => {
                let kind = flat[rnd() as usize % flat.len()];
                let count = (rnd() % 24) as usize;
                let skew = rnd() % 3 == 0;
                let data: Vec<u8> = (0..count * split::unit(kind)).map(|_| if skew { (rnd() % 3) as u8 * 0x7f } else { rnd() as u8 }).collect();
                let ss = split::split(kind, &data);
                println!("split {} {} => {}", kind, hex(&data), streams(&ss));
                let mut back = vec![0u8; data.len()];
                let r = split::merge(kind, &ss, &mut back);
                let tail: Vec<String> = ss.iter().map(|s| hex(s)).collect();
                println!("merge {} {} {} => {}", kind, count, tail.join(" "), if r.is_ok() { hex(&back) } else { "err".to_string() });
            }
            1 | 2 => {
                let t = known[rnd() as usize % known.len()];
                let q = split::quant(t).unwrap();
                let blocks = (rnd() % 4) as usize;
                let data: Vec<u8> = (0..blocks * q.bytes).map(|_| rnd() as u8).collect();
                let kind = split::QUANT + t as u8;
                let mut ss = split::split(kind, &data);
                println!("qsplit {} {} {} => {}", q.bytes, fields(t), hex(&data), streams(&ss));
                // now and then damage one byte of one stream: a symbol above 15 in a 4-bit stream must be refused
                if rnd() % 2 == 0 && !ss.is_empty() {
                    let k = rnd() as usize % ss.len();
                    if !ss[k].is_empty() {
                        let at = rnd() as usize % ss[k].len();
                        ss[k][at] = 16 + (rnd() % 200) as u8;
                    }
                }
                let mut back = vec![0u8; data.len()];
                let r = split::merge(kind, &ss, &mut back);
                println!("qmerge {} {} {} {} => {}", q.bytes, fields(t), blocks, ss.iter().map(|s| hex(s)).collect::<Vec<_>>().join(" "), if r.is_ok() { hex(&back) } else { "err".to_string() });
            }
            3 => {
                let len = (rnd() % 40) as usize;
                let a: Vec<u8> = (0..len).map(|_| rnd() as u8).collect();
                let b: Vec<u8> = (0..len).map(|_| rnd() as u8).collect();
                let mut c = a.clone();
                planes::xor_into(&mut c, &b);
                println!("xor {} {} => {}", hex(&a), hex(&b), hex(&c));
                // xoring the reference in again gives the block back
                planes::xor_into(&mut c, &b);
                assert_eq!(c, a);
            }
            _ => {
                let begin = (rnd() % 50) as u64;
                let end = begin + (rnd() % 400) as u64;
                let unit = [1u64, 2, 4, 8, 18, 144, 210][rnd() as usize % 7];
                let block = (rnd() % 300) as u64 + 1;
                let l = split::block_lens(begin, end, unit, block);
                println!("blocks {} {} {} {} => {}", begin, end, unit, block, if l.is_empty() { "-".to_string() } else { l.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ") });
            }
        }
    }
}
