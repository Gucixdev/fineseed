//! How the bytes of one block are taken apart into streams that a coder can treat each on its own, and put
//! back together. `kind` is one byte in the packed file:
//!
//! * `1..=8`: byte planes of `kind`-byte values (1 = the bytes as they are);
//! * `10` BF16, `12` F32: the exponent as a stream of its own, then the rest of the float cut into
//!   bytes (a float's exponent repeats, its mantissa is noise: kept apart each costs what it is worth);
//! * `11` F16: the high byte (sign, exponent, top two mantissa bits) and the low byte, two streams. Cutting the
//!   sign and mantissa bits off the exponent costs 1 % more (measured on SD 1.5), the bits tell about each other;
//! * `64 + t`: a quantized GGUF block of ggml type `t`, cut by field: scales apart from the quantized values,
//!   4-bit values one per symbol so that an order-0 coder sees their real, uneven distribution.
//!
//! `split` then `merge` gives the input back for every kind and every input that is a whole number of units;
//! that is what the tests and the Lean model in `lean4/` check.

pub const PLAIN: u8 = 1;
pub const BF16: u8 = 10;
pub const F16: u8 = 11;
pub const F32: u8 = 12;
pub const QUANT: u8 = 64;

/// A part of a quantized block, `(offset, length)` in bytes inside the block.
#[derive(Clone, Copy, Debug)]
pub enum Field {
    /// Each byte position is a stream of its own (a small float: its bytes differ in what they hold).
    Planes(usize, usize),
    /// The bytes of the field of all blocks, one stream.
    Bytes(usize, usize),
    /// Each byte is two 4-bit symbols (low, then high), one stream.
    Nibs(usize, usize),
}

use self::Field::{Bytes, Nibs, Planes};

pub struct Quant {
    pub elems: usize,
    pub bytes: usize,
    pub fields: &'static [Field],
}

/// The layout of the ggml block types; `None` for a type this does not know (it is then stored as plain bytes).
pub fn quant(t: u32) -> Option<Quant> {
    let q = |elems, bytes, fields| {
        Some(Quant {
            elems,
            bytes,
            fields,
        })
    };
    match t {
        2 => q(32, 18, &[Planes(0, 2), Nibs(2, 16)]), // Q4_0
        3 => q(32, 20, &[Planes(0, 4), Nibs(4, 16)]), // Q4_1
        6 => q(32, 22, &[Planes(0, 2), Bytes(2, 4), Nibs(6, 16)]), // Q5_0
        7 => q(32, 24, &[Planes(0, 4), Bytes(4, 4), Nibs(8, 16)]), // Q5_1
        8 => q(32, 34, &[Planes(0, 2), Bytes(2, 32)]), // Q8_0
        10 => q(256, 84, &[Bytes(0, 16), Bytes(16, 64), Planes(80, 4)]), // Q2_K
        11 => q(
            256,
            110,
            &[Bytes(0, 32), Bytes(32, 64), Bytes(96, 12), Planes(108, 2)],
        ), // Q3_K
        12 => q(256, 144, &[Planes(0, 4), Bytes(4, 12), Nibs(16, 128)]), // Q4_K
        13 => q(
            256,
            176,
            &[Planes(0, 4), Bytes(4, 12), Bytes(16, 32), Nibs(48, 128)],
        ), // Q5_K
        14 => q(
            256,
            210,
            &[Nibs(0, 128), Bytes(128, 64), Bytes(192, 16), Planes(208, 2)],
        ), // Q6_K
        16 => q(256, 66, &[Planes(0, 2), Bytes(2, 64)]), // IQ2_XXS
        17 => q(256, 74, &[Planes(0, 2), Bytes(2, 64), Bytes(66, 8)]), // IQ2_XS
        18 => q(256, 98, &[Planes(0, 2), Bytes(2, 96)]), // IQ3_XXS
        19 => q(256, 50, &[Planes(0, 2), Bytes(2, 32), Bytes(34, 16)]), // IQ1_S
        20 => q(32, 18, &[Planes(0, 2), Nibs(2, 16)]), // IQ4_NL
        21 => q(
            256,
            110,
            &[
                Planes(0, 2),
                Bytes(2, 64),
                Bytes(66, 8),
                Bytes(74, 32),
                Bytes(106, 4),
            ],
        ), // IQ3_S
        22 => q(
            256,
            82,
            &[Planes(0, 2), Bytes(2, 64), Bytes(66, 8), Bytes(74, 8)],
        ), // IQ2_S
        23 => q(
            256,
            136,
            &[Planes(0, 2), Bytes(2, 2), Bytes(4, 4), Nibs(8, 128)],
        ), // IQ4_XS
        29 => q(256, 56, &[Bytes(0, 32), Bytes(32, 16), Bytes(48, 8)]), // IQ1_M
        34 => q(256, 54, &[Bytes(0, 52), Planes(52, 2)]), // TQ1_0
        35 => q(256, 66, &[Bytes(0, 64), Planes(64, 2)]), // TQ2_0
        _ => None,
    }
}

/// Bytes of one unit: a value for planes and floats, a block for quantized types. 0 for an unknown kind.
pub fn unit(kind: u8) -> usize {
    match kind {
        1..=8 => kind as usize,
        BF16 | F16 => 2,
        F32 => 4,
        k if k >= QUANT => quant(u32::from(k - QUANT)).map_or(0, |q| q.bytes),
        _ => 0,
    }
}

fn field_streams(f: &Field) -> usize {
    match f {
        Planes(_, len) => *len,
        _ => 1,
    }
}

/// The number of streams a block of `kind` is cut into; 0 for an unknown kind.
pub fn streams(kind: u8) -> usize {
    match kind {
        1..=8 => kind as usize,
        BF16 | F16 => 2,
        F32 => 4,
        k if k >= QUANT => {
            quant(u32::from(k - QUANT)).map_or(0, |q| q.fields.iter().map(field_streams).sum())
        }
        _ => 0,
    }
}

/// The length of each stream for `n` units.
pub fn lens(kind: u8, n: usize) -> Vec<usize> {
    match kind {
        1..=8 => vec![n; kind as usize],
        BF16 | F16 => vec![n; 2],
        F32 => vec![n; 4],
        k if k >= QUANT => {
            let mut v = Vec::new();
            for f in quant(u32::from(k - QUANT)).map_or(&[][..], |q| q.fields) {
                match f {
                    Planes(_, len) => v.extend(std::iter::repeat_n(n, *len)),
                    Bytes(_, len) => v.push(n * len),
                    Nibs(_, len) => v.push(n * len * 2),
                }
            }
            v
        }
        _ => Vec::new(),
    }
}

/// The lengths of the blocks `[begin, end)` is cut into when a block holds at most `block` bytes and a whole number of
/// `unit`s: every block but the last is full, so a range that is a whole number of units gives only such blocks.
pub fn block_lens(begin: u64, end: u64, unit: u64, block: u64) -> Vec<u64> {
    let step = (block / unit.max(1)).max(1) * unit.max(1);
    let mut out = Vec::new();
    let mut at = begin;
    while at < end {
        let len = (end - at).min(step);
        out.push(len);
        at += len;
    }
    out
}

/// `data` must be a whole number of units of `kind`.
pub fn split(kind: u8, data: &[u8]) -> Vec<Vec<u8>> {
    let u = unit(kind);
    debug_assert!(u > 0 && data.len().is_multiple_of(u));
    let n = data.len() / u;
    let mut out: Vec<Vec<u8>> = lens(kind, n).into_iter().map(Vec::with_capacity).collect();
    match kind {
        1 => out[0].extend_from_slice(data),
        2..=8 => {
            for o in out.iter_mut() {
                o.resize(n, 0);
            }
            for (i, v) in data.chunks_exact(u).enumerate() {
                for (k, b) in v.iter().enumerate() {
                    out[k][i] = *b;
                }
            }
        }
        BF16 => {
            let [a, b] = out.get_disjoint_mut([0, 1]).unwrap_or_else(|_| unreachable!());
            a.resize(n, 0);
            b.resize(n, 0);
            for ((v, e), s) in data.as_chunks::<2>().0.iter().zip(a.iter_mut()).zip(b.iter_mut()) {
                *e = v[1] << 1 | v[0] >> 7;
                *s = v[1] & 0x80 | v[0] & 0x7f;
            }
        }
        F16 => {
            let [a, b] = out.get_disjoint_mut([0, 1]).unwrap_or_else(|_| unreachable!());
            a.resize(n, 0);
            b.resize(n, 0);
            for ((v, h), l) in data.as_chunks::<2>().0.iter().zip(a.iter_mut()).zip(b.iter_mut()) {
                *h = v[1];
                *l = v[0];
            }
        }
        F32 => {
            let [a, b, c, d] = out.get_disjoint_mut([0, 1, 2, 3]).unwrap_or_else(|_| unreachable!());
            a.resize(n, 0);
            b.resize(n, 0);
            c.resize(n, 0);
            d.resize(n, 0);
            for ((((v, e), s), m1), m0) in data
                .as_chunks::<4>()
                .0
                .iter()
                .zip(a.iter_mut())
                .zip(b.iter_mut())
                .zip(c.iter_mut())
                .zip(d.iter_mut())
            {
                let x = u32::from_le_bytes(*v);
                *e = (x >> 23) as u8;
                *s = (x >> 31 << 7 | x >> 16 & 0x7f) as u8;
                *m1 = (x >> 8) as u8;
                *m0 = x as u8;
            }
        }
        k => {
            let Some(q) = quant(u32::from(k - QUANT)) else {
                return out;
            };
            let mut at = 0;
            for f in q.fields {
                match *f {
                    Planes(off, len) => {
                        for b in data.chunks_exact(q.bytes) {
                            for j in 0..len {
                                out[at + j].push(b[off + j]);
                            }
                        }
                        at += len;
                    }
                    Bytes(off, len) => {
                        for b in data.chunks_exact(q.bytes) {
                            out[at].extend_from_slice(&b[off..off + len]);
                        }
                        at += 1;
                    }
                    Nibs(off, len) => {
                        for b in data.chunks_exact(q.bytes) {
                            for x in &b[off..off + len] {
                                out[at].push(x & 15);
                                out[at].push(x >> 4);
                            }
                        }
                        at += 1;
                    }
                }
            }
        }
    }
    out
}

/// The inverse of `split`: `streams` as `split` made them, `out` the whole block. `Err` when the streams do not fit.
pub fn merge<S: AsRef<[u8]>>(kind: u8, streams_in: &[S], out: &mut [u8]) -> Result<(), String> {
    let st: Vec<&[u8]> = streams_in.iter().map(|s| s.as_ref()).collect();
    let u = unit(kind);
    if u == 0 || !out.len().is_multiple_of(u) {
        return Err("a block of an unknown kind or an odd length".into());
    }
    let n = out.len() / u;
    let want = lens(kind, n);
    if st.len() != want.len() || st.iter().zip(&want).any(|(s, l)| s.len() != *l) {
        return Err("streams of the wrong number or length".into());
    }
    match kind {
        1 => out.copy_from_slice(&st[0]),
        2..=8 => {
            for (i, v) in out.chunks_exact_mut(u).enumerate() {
                for (k, b) in v.iter_mut().enumerate() {
                    *b = st[k][i];
                }
            }
        }
        BF16 => {
            for ((v, &e), &s) in out.as_chunks_mut::<2>().0.iter_mut().zip(st[0]).zip(st[1]) {
                v[1] = s & 0x80 | e >> 1;
                v[0] = (e & 1) << 7 | s & 0x7f;
            }
        }
        F16 => {
            for ((v, &h), &l) in out.as_chunks_mut::<2>().0.iter_mut().zip(st[0]).zip(st[1]) {
                *v = [l, h];
            }
        }
        F32 => {
            for ((((v, &e), &s), &m1), &m0) in out
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(st[0])
                .zip(st[1])
                .zip(st[2])
                .zip(st[3])
            {
                let (e, s, m1, m0) = (u32::from(e), u32::from(s), u32::from(m1), u32::from(m0));
                *v = (s >> 7 << 31 | e << 23 | (s & 0x7f) << 16 | m1 << 8 | m0).to_le_bytes();
            }
        }
        k => {
            let q = quant(u32::from(k - QUANT)).ok_or("an unknown block type")?;
            let mut at = 0;
            for f in q.fields {
                match *f {
                    Planes(off, len) => {
                        for (i, b) in out.chunks_exact_mut(q.bytes).enumerate() {
                            for j in 0..len {
                                b[off + j] = st[at + j][i];
                            }
                        }
                        at += len;
                    }
                    Bytes(off, len) => {
                        for (i, b) in out.chunks_exact_mut(q.bytes).enumerate() {
                            b[off..off + len]
                                .copy_from_slice(&st[at][i * len..(i + 1) * len]);
                        }
                        at += 1;
                    }
                    Nibs(off, len) => {
                        for (i, b) in out.chunks_exact_mut(q.bytes).enumerate() {
                            for j in 0..len {
                                let (lo, hi) = (
                                    st[at][(i * len + j) * 2],
                                    st[at][(i * len + j) * 2 + 1],
                                );
                                if lo > 15 || hi > 15 {
                                    return Err("a nibble stream with a symbol above 15".into());
                                }
                                b[off + j] = lo | hi << 4;
                            }
                        }
                        at += 1;
                    }
                }
            }
        }
    }
    Ok(())
}

/// A block type known to this module covers every byte of its block exactly once (checked by the tests).
pub fn covers(q: &Quant) -> bool {
    let mut seen = vec![0u8; q.bytes];
    for f in q.fields {
        let (off, len) = match *f {
            Planes(o, l) | Bytes(o, l) | Nibs(o, l) => (o, l),
        };
        if off + len > q.bytes {
            return false;
        }
        seen[off..off + len].iter_mut().for_each(|s| *s += 1);
    }
    seen.iter().all(|&s| s == 1)
}

/// Bytes of a block the choice made by [`guess`] looks at.
pub const GUESS_SAMPLE: usize = 16 << 10;

/// What a run of bytes of unknown type most likely is: `(kind, shift)`, the kind of the values that start `shift` bytes
/// in. Plain bytes and the float kinds at every alignment are scored by the smaller of the order-0 size and a fast zstd of each stream the kind cuts
/// a sample into; a fast zstd stands in for the coder; plain wins every tie. Used for pieces of files whose layout is not read (a
/// TensorFlow data file, a TFLite buffer, a PyTorch zip entry) and for the pieces a content-defined chunker cuts.
pub fn guess(data: &[u8]) -> (u8, usize) {
    let mut best: Option<(usize, u8, usize)> = None;
    let candidates = std::iter::once((PLAIN, 0usize)).chain(
        [BF16, F16, F32]
            .into_iter()
            .flat_map(|k| (0..unit(k)).map(move |s| (k, s))),
    );
    for (kind, shift) in candidates {
        let u = unit(kind);
        let n = data.len().saturating_sub(shift) / u * u;
        if n == 0 {
            continue;
        }
        let part = &data[shift..shift + n.min(GUESS_SAMPLE / u * u).max(u)];
        // what the coder would keep of each stream: the smaller of its order-0 size and a fast zstd, so structure a
        // match finder sees (text, code, tables) is not mistaken for noise that planes would suit
        let size: usize = split(kind, part)
            .iter()
            .map(|s| {
                let z = zstd::bulk::compress(s, 1).map_or(s.len(), |c| c.len());
                crate::huffman::estimate(s).min(z).min(s.len())
            })
            .sum();
        let score = size * 1024 / part.len();
        if best.is_none_or(|(s, _, _)| score < s) {
            best = Some((score, kind, shift));
        }
    }
    best.map_or((PLAIN, 0), |(_, k, s)| (k, s))
}
