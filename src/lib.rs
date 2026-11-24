//! fineseed: a lossless packer for model files.
//!
//! The file is read as SafeTensors, GGUF or ONNX (`format`) and cut into blocks of at most `BLOCK` bytes that
//! each lie inside one tensor. Each block is taken apart by what its bytes are (`split`: the exponent of a float
//! apart from its mantissa, the scales of a quantized block apart from its 4-bit values, byte planes for
//! integers) and every part is coded as the smallest of raw, zstd and an own Huffman coder (`huffman`). With a
//! base model a block is also tried as `block ^ base block`, which is mostly zeros where a fine-tune left a
//! weight alone. The packed file holds the blake3 of the original: `unpack` refuses to finish unless the bytes
//! come back exact. Many files in one archive, with reference blocks from any earlier file: `archive`.
//!
//! Layout, little endian: magic `FSEED\x02\0\0`, original length `u64`, blake3 of the original (32), blake3 of
//! the base file (32, zeros without a base), block count `u32`, then the blocks in file order. A block is
//! `kind u8` (`split`), `flags u8` (bit 0: xor with the base), `offset u64`, `len u32`, `base offset u64` if bit 0,
//! then one part per stream of the kind: `method u8` (0 raw, 1 zstd, 2 huffman), `coded length u32`, bytes.
pub mod api;
pub mod archive;
pub mod format;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod gguf;
pub mod huffman;
#[cfg(feature = "mount")]
pub mod mount;
pub mod onnx;
pub mod parity;
pub mod pipe;
pub mod planes;
pub mod safetensors;
pub mod split;
pub mod zip;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const BLOCK: u64 = 4 << 20;
const MAGIC: &[u8; 8] = b"FSEED\x02\0\0";
const HEAD_LEN: u64 = 8 + 8 + 32 + 32 + 4;

pub type Res<T> = Result<T, String>;

fn io<T>(r: std::io::Result<T>, what: &str) -> Res<T> {
    r.map_err(|e| format!("{what}: {e}"))
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub original: u64,
    pub packed: u64,
    pub blocks: u32,
    pub xor_blocks: u32,
    pub raw_parts: u32,
    pub zstd_parts: u32,
    pub huff_parts: u32,
    pub seconds: f64,
}

struct Plan {
    off: u64,
    len: u64,
    kind: u8,
    /// The earlier bytes this block may be coded against: `(member, offset)`; the member is 0 in a single file.
    base: Option<(u32, u64)>,
}

struct Part {
    method: u8,
    bytes: Vec<u8>,
}

struct Encoded {
    kind: u8,
    xor: bool,
    off: u64,
    len: u32,
    base_member: u32,
    base_off: u64,
    parts: Vec<Part>,
}

impl Encoded {
    fn size(&self) -> usize {
        self.parts.iter().map(|p| p.bytes.len()).sum()
    }
}

fn workers() -> usize {
    workers_for(0)
}

/// `0` = one per core (at most 32), else the number asked for.
fn workers_for(threads: usize) -> usize {
    if threads > 0 {
        return threads.min(64);
    }
    std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .clamp(1, 32)
}

/// `len` bytes at `off` without moving any cursor, so several threads can read one file.
fn pread(file: &File, off: u64, len: u64) -> Res<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let mut buf = vec![0u8; len as usize];
    io(file.read_exact_at(&mut buf, off), "read")?;
    Ok(buf)
}

fn read_at<R: Read + Seek>(file: &mut R, off: u64, len: u64) -> Res<Vec<u8>> {
    let mut buf = vec![0u8; len as usize];
    io(file.seek(SeekFrom::Start(off)), "seek")?;
    io(file.read_exact(&mut buf), "read")?;
    Ok(buf)
}

/// `--fast`: numeric planes get the own Huffman coder only (what ZipNN does), without the zstd attempt. The format does
/// not change, every part says which coder made it; only the choice is narrower, so it can cost a little ratio.
pub static FAST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// zstd at 130 MB/s on an exponent plane costs more than everything else together, and on weights it rarely beats the
/// order-0 coder. So look at a sample first: the full run is made only when the sample already codes smaller than the
/// order-0 estimate says the whole plane will. Level 4 and up (the user asked for the best) always runs in full.
fn zstd_may_win(plane: &[u8], est: usize, level: i32) -> bool {
    const SAMPLE: usize = 128 << 10;
    if level > 3 || plane.len() <= 4 * SAMPLE {
        return true;
    }
    let mid = plane.len() / 2 - SAMPLE / 2;
    let Ok(z) = zstd::bulk::compress(&plane[mid..mid + SAMPLE], level) else {
        return true;
    };
    (z.len() as u128) * (plane.len() as u128) <= (est as u128) * (SAMPLE as u128)
}

fn code_plane(plane: &[u8], level: i32, plain: bool) -> Res<Part> {
    let table = huffman::table(plane);
    let est = table.size;
    // an order-0 coder that gains almost nothing means noise, and zstd will not find more in noise; plain blocks
    // (text, metadata) keep the zstd attempt because matches, not symbol counts, are what it finds there
    let mut best = Part {
        method: 0,
        bytes: Vec::new(),
    };
    let mut best_len = plane.len();
    let fast = FAST.load(std::sync::atomic::Ordering::Relaxed);
    if plain || (!fast && est < plane.len() - plane.len() / 50 && zstd_may_win(plane, est, level)) {
        let coded = io(zstd::bulk::compress(plane, level), "zstd")?;
        if coded.len() < best_len {
            best_len = coded.len();
            best = Part {
                method: 1,
                bytes: coded,
            };
        }
    }
    if est < best_len
        && let Some(h) = huffman::encode_chunked(plane, &table).filter(|h| h.len() < best_len)
    {
        best = Part {
            method: 3,
            bytes: h,
        };
    }
    if best.method == 0 {
        best.bytes = plane.to_vec();
    }
    Ok(best)
}

/// What the planes of `data` would cost under an order-0 coder, without coding anything.
fn estimate(data: &[u8], kind: u8) -> usize {
    split::split(kind, data).iter().map(|p| huffman::table(p).size.min(p.len())).sum()
}

fn code_block(data: &[u8], kind: u8, level: i32) -> Res<Vec<Part>> {
    let streams = split::split(kind, data);
    streams
        .iter()
        .map(|p| code_plane(p, level, kind == split::PLAIN))
        .collect()
}

fn encode(plan: &Plan, data: &[u8], base: Option<&[u8]>, level: i32) -> Res<Encoded> {
    let mut best = Encoded {
        kind: plan.kind,
        xor: false,
        off: plan.off,
        len: data.len() as u32,
        base_member: 0,
        base_off: 0,
        parts: code_block(data, plan.kind, level)?,
    };
    if let (Some(base), Some((base_member, base_off))) = (base, plan.base) {
        let mut delta = data.to_vec();
        planes::xor_into(&mut delta, base);
        // coding the delta in full costs as much as coding the block again, so look at the order-0 sizes of its planes
        // first: a delta that does not even come close to what the block cost is noise (a base that is not a base)
        if plan.kind != split::PLAIN && estimate(&delta, plan.kind) >= best.size() - best.size() / 50 {
            return Ok(best);
        }
        let parts = code_block(&delta, plan.kind, level)?;
        let candidate = Encoded {
            xor: true,
            base_member,
            base_off,
            parts,
            ..best_header(&best)
        };
        if candidate.size() < best.size() {
            best = candidate;
        }
    }
    Ok(best)
}

fn best_header(e: &Encoded) -> Encoded {
    Encoded {
        kind: e.kind,
        xor: e.xor,
        off: e.off,
        len: e.len,
        base_member: e.base_member,
        base_off: e.base_off,
        parts: Vec::new(),
    }
}

/// Cut `[begin, end)` into blocks of at most `BLOCK` bytes, each a whole number of units of `kind` and with the
/// base offset of the same bytes.
fn cut(plans: &mut Vec<Plan>, begin: u64, end: u64, kind: u8, base_begin: Option<(u32, u64)>) {
    let mut at = begin;
    for len in split::block_lens(begin, end, split::unit(kind) as u64, BLOCK) {
        let base = base_begin.map(|(m, b)| (m, b + (at - begin)));
        plans.push(Plan {
            off: at,
            len,
            kind,
            base,
        });
        at += len;
    }
}

fn make_plans<R: Read + Seek, B: Read + Seek>(
    input: &mut R,
    len: u64,
    base: Option<(&mut B, u64)>,
) -> Res<Vec<Plan>> {
    let mut plans = Vec::new();
    let Some(segs) = format::layout(input, len) else {
        cut(&mut plans, 0, len, split::PLAIN, None);
        return Ok(plans);
    };
    let mut base_index: HashMap<String, (u8, u64, u64)> = HashMap::new();
    if let Some((b, blen)) = base {
        let bsegs = format::layout(b, blen).ok_or("the base is not a model file this reads")?;
        for t in bsegs {
            base_index.insert(t.name, (t.kind, t.begin, t.end));
        }
    }
    let mut pos = 0u64;
    for t in &segs {
        cut(&mut plans, pos, t.begin, split::PLAIN, None);
        let same = base_index
            .get(&t.name)
            .filter(|(k, b, e)| *k == t.kind && e - b == t.end - t.begin)
            .map(|(_, b, _)| (0, *b));
        cut(&mut plans, t.begin, t.end, t.kind, same);
        pos = t.end;
    }
    cut(&mut plans, pos, len, split::PLAIN, None);
    Ok(plans)
}

fn write_block<W: Write>(out: &mut W, e: &Encoded) -> Res<()> {
    let mut head = vec![e.kind, u8::from(e.xor)];
    head.extend(e.off.to_le_bytes());
    head.extend(e.len.to_le_bytes());
    if e.xor {
        head.extend(e.base_off.to_le_bytes());
    }
    io(out.write_all(&head), "write")?;
    for p in &e.parts {
        io(out.write_all(&[p.method]), "write")?;
        io(
            out.write_all(&(p.bytes.len() as u32).to_le_bytes()),
            "write",
        )?;
        io(out.write_all(&p.bytes), "write")?;
    }
    Ok(())
}

pub fn file_hash(path: &Path) -> Res<[u8; 32]> {
    let mut f = io(File::open(path), "open")?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = io(f.read(&mut buf), "read")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(*hasher.finalize().as_bytes())
}

pub fn pack(input: &Path, base: Option<&Path>, output: &Path, level: i32) -> Res<Stats> {
    let started = std::time::Instant::now();
    let mut f = io(File::open(input), "open input")?;
    let len = io(f.metadata(), "stat input")?.len();
    let mut base_file = match base {
        Some(p) => Some(io(File::open(p), "open base")?),
        None => None,
    };
    let base_hash = match base {
        Some(p) => file_hash(p)?,
        None => [0u8; 32],
    };
    let base_len = match base_file.as_ref() {
        Some(b) => io(b.metadata(), "stat base")?.len(),
        None => 0,
    };
    let plans = make_plans(&mut f, len, base_file.as_mut().map(|b| (b, base_len)))?;
    let mut out = BufWriter::new(io(File::create(output), "create output")?);
    let mut head = Vec::with_capacity(HEAD_LEN as usize);
    head.extend(MAGIC);
    head.extend(len.to_le_bytes());
    head.extend([0u8; 32]);
    head.extend(base_hash);
    head.extend((plans.len() as u32).to_le_bytes());
    io(out.write_all(&head), "write")?;

    let mut stats = Stats {
        original: len,
        blocks: plans.len() as u32,
        ..Stats::default()
    };
    let mut hasher = blake3::Hasher::new();
    for batch in plans.chunks(workers()) {
        let mut data = Vec::new();
        let mut bases = Vec::new();
        for plan in batch {
            let d = read_at(&mut f, plan.off, plan.len)?;
            hasher.update(&d);
            let b = match (plan.base.map(|(_, o)| o), base_file.as_mut()) {
                (Some(off), Some(bf)) => Some(read_at(bf, off, plan.len)?),
                _ => None,
            };
            data.push(d);
            bases.push(b);
        }
        let encoded: Vec<Res<Encoded>> = std::thread::scope(|s| {
            let handles: Vec<_> = batch
                .iter()
                .zip(&data)
                .zip(&bases)
                .map(|((plan, d), b)| s.spawn(move || encode(plan, d, b.as_deref(), level)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("a worker panicked".into())))
                .collect()
        });
        for e in encoded {
            let e = e?;
            stats.xor_blocks += u32::from(e.xor);
            for p in &e.parts {
                match p.method {
                    0 => stats.raw_parts += 1,
                    1 => stats.zstd_parts += 1,
                    _ => stats.huff_parts += 1,
                }
            }
            write_block(&mut out, &e)?;
        }
    }
    io(out.flush(), "flush")?;
    let mut file = io(out.into_inner().map_err(|e| e.into_error()), "flush")?;
    io(file.seek(SeekFrom::Start(16)), "seek")?;
    io(file.write_all(hasher.finalize().as_bytes()), "write hash")?;
    stats.packed = io(file.metadata(), "stat output")?.len();
    stats.seconds = started.elapsed().as_secs_f64();
    Ok(stats)
}

struct Header {
    len: u64,
    hash: [u8; 32],
    base_hash: [u8; 32],
    blocks: u32,
}

fn read_header<R: Read>(r: &mut R) -> Res<Header> {
    let mut b = [0u8; HEAD_LEN as usize];
    io(r.read_exact(&mut b), "read header")?;
    if &b[..8] != MAGIC {
        return Err("not a fineseed file".into());
    }
    let n = |i: usize, k: usize| &b[i..i + k];
    Ok(Header {
        len: u64::from_le_bytes(n(8, 8).try_into().unwrap_or([0; 8])),
        hash: n(16, 32).try_into().unwrap_or([0; 32]),
        base_hash: n(48, 32).try_into().unwrap_or([0; 32]),
        blocks: u32::from_le_bytes(n(80, 4).try_into().unwrap_or([0; 4])),
    })
}

struct Raw {
    kind: u8,
    xor: bool,
    off: u64,
    len: usize,
    base_off: u64,
    parts: Vec<(u8, Vec<u8>)>,
}

fn read_u<const N: usize, R: Read>(r: &mut R) -> Res<[u8; N]> {
    let mut b = [0u8; N];
    io(r.read_exact(&mut b), "read block")?;
    Ok(b)
}

fn read_block<R: Read>(r: &mut R) -> Res<Raw> {
    let [kind, flags] = read_u::<2, _>(r)?;
    let off = u64::from_le_bytes(read_u(r)?);
    let len = u32::from_le_bytes(read_u(r)?) as usize;
    let xor = flags & 1 == 1;
    let base_off = if xor {
        u64::from_le_bytes(read_u(r)?)
    } else {
        0
    };
    let unit = split::unit(kind);
    if unit == 0 || len > BLOCK as usize || !len.is_multiple_of(unit) || flags > 1 {
        return Err("a damaged block header".into());
    }
    let mut parts = Vec::new();
    for _ in 0..split::streams(kind) {
        let [method] = read_u::<1, _>(r)?;
        let clen = u32::from_le_bytes(read_u(r)?) as usize;
        if clen > 2 * len + 1024 {
            return Err("a damaged part length".into());
        }
        let mut bytes = vec![0u8; clen];
        io(r.read_exact(&mut bytes), "read part")?;
        parts.push((method, bytes));
    }
    Ok(Raw {
        kind,
        xor,
        off,
        len,
        base_off,
        parts,
    })
}

fn decode(raw: &Raw, base: Option<&[u8]>) -> Res<Vec<u8>> {
    let lens = split::lens(raw.kind, raw.len / split::unit(raw.kind));
    let mut streams = Vec::with_capacity(raw.parts.len());
    for ((method, bytes), want) in raw.parts.iter().zip(&lens) {
        let s: std::borrow::Cow<[u8]> = match method {
            0 => std::borrow::Cow::Borrowed(bytes.as_slice()),
            1 => io(zstd::bulk::decompress(bytes, *want), "unzstd")?.into(),
            2 => huffman::decode(bytes, *want)?.into(),
            3 => huffman::decode_chunked(bytes, *want)?.into(),
            _ => return Err("an unknown part method".into()),
        };
        if s.len() != *want {
            return Err("a part of the wrong length".into());
        }
        streams.push(s);
    }
    let mut out = vec![0u8; raw.len];
    split::merge(raw.kind, &streams, &mut out)?;
    if raw.xor {
        planes::xor_into(&mut out, base.ok_or("the block needs the base file")?);
    }
    Ok(out)
}

pub fn unpack(input: &Path, base: Option<&Path>, output: &Path) -> Res<Stats> {
    let started = std::time::Instant::now();
    let mut f = std::io::BufReader::with_capacity(1 << 20, io(File::open(input), "open input")?);
    let h = read_header(&mut f)?;
    let mut base_file = match base {
        Some(p) => {
            if file_hash(p)? != h.base_hash {
                return Err("the base is not the file this was packed against".into());
            }
            Some(io(File::open(p), "open base")?)
        }
        None => None,
    };
    let mut out = BufWriter::with_capacity(1 << 20, io(File::create(output), "create output")?);
    let mut hasher = blake3::Hasher::new();
    let mut written = 0u64;
    let mut left = h.blocks;
    while left > 0 {
        let n = (left as usize).min(workers());
        let mut raws = Vec::new();
        for _ in 0..n {
            raws.push(read_block(&mut f)?);
        }
        let mut bases = Vec::new();
        for r in &raws {
            bases.push(match (r.xor, base_file.as_mut()) {
                (true, Some(bf)) => Some(read_at(bf, r.base_off, r.len as u64)?),
                (true, None) => return Err("the file was packed against a base: pass it".into()),
                _ => None,
            });
        }
        let decoded: Vec<Res<Vec<u8>>> = std::thread::scope(|s| {
            let handles: Vec<_> = raws
                .iter()
                .zip(&bases)
                .map(|(r, b)| s.spawn(move || decode(r, b.as_deref())))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("a worker panicked".into())))
                .collect()
        });
        for (r, d) in raws.iter().zip(decoded) {
            if r.off != written {
                return Err("blocks out of order".into());
            }
            let d = d?;
            hasher.update(&d);
            io(out.write_all(&d), "write")?;
            written += d.len() as u64;
        }
        left -= n as u32;
    }
    io(out.flush(), "flush")?;
    if written != h.len || hasher.finalize().as_bytes() != &h.hash {
        return Err("the unpacked bytes differ from the original (blake3 mismatch)".into());
    }
    Ok(Stats {
        original: h.len,
        packed: io(std::fs::metadata(input), "stat")?.len(),
        blocks: h.blocks,
        seconds: started.elapsed().as_secs_f64(),
        ..Stats::default()
    })
}

pub fn info(input: &Path) -> Res<Stats> {
    let mut f = std::io::BufReader::new(io(File::open(input), "open")?);
    let h = read_header(&mut f)?;
    let mut s = Stats {
        original: h.len,
        packed: io(std::fs::metadata(input), "stat")?.len(),
        blocks: h.blocks,
        ..Stats::default()
    };
    for _ in 0..h.blocks {
        let r = read_block(&mut f)?;
        s.xor_blocks += u32::from(r.xor);
        for (m, _) in &r.parts {
            match m {
                0 => s.raw_parts += 1,
                1 => s.zstd_parts += 1,
                _ => s.huff_parts += 1,
            }
        }
    }
    Ok(s)
}
