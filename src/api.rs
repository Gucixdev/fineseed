//! The surface other programs use: bytes in, bytes out. No files, no global state, no output of its own.
//!
//! Two levels:
//!
//! * a whole buffer, [`compress`] / [`decompress`]: any file, the model formats (SafeTensors, GGUF, ONNX) are
//!   read and coded tensor by tensor, anything else is coded as plain blocks. The result is the same byte stream the
//!   `fineseed pack` command writes, carries the blake3 of the original and is checked on the way back.
//! * one block, [`encode_block`] / [`decode_block`]: the bytes of one tensor (or a piece of one) of a type the
//!   caller already knows, with an optional reference block of the same length (`block ^ reference` is coded where
//!   that is smaller, which is where a fine-tune of a model pays off). It is what a store that keeps its own
//!   chunks and hashes wants; the caller keeps the hash and the reference.
//!
//! Both are blocking and CPU bound: from async code call them from a blocking pool. A call uses up to
//! [`Options::threads`] threads and none outlive it. Input from outside is never trusted: a damaged or hostile
//! stream gives an [`Error`], never a panic, and the output of [`decompress`] can be capped.
use super::{
    Encoded, HEAD_LEN, MAGIC, Plan, Raw, Stats, decode, encode, make_plans, read_block,
    read_header, split, workers_for, write_block,
};
use std::io::Cursor;

pub use crate::format::Seg;

/// A block of bytes no longer than this is accepted by [`encode_block`] and [`decode_block`].
pub const MAX_BLOCK: usize = 64 << 20;

/// What went wrong. The text is for people; match on the variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The coded bytes are damaged, cut short, or decode to something other than what was stored.
    Corrupt(String),
    /// The data needs a reference that was not given, or the reference given is not the one used when coding.
    Reference(String),
    /// An argument the library cannot take: a block too long, not a whole number of values, a limit exceeded.
    Limit(String),
    /// Anything else (a worker failed, a coder refused).
    Failed(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Corrupt(m) => write!(f, "corrupt fineseed data: {m}"),
            Error::Reference(m) => write!(f, "fineseed reference: {m}"),
            Error::Limit(m) => write!(f, "fineseed limit: {m}"),
            Error::Failed(m) => write!(f, "fineseed: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// How to code. `Default` is level 3, all cores, no output cap.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// zstd level for the parts that zstd codes best (1..=22).
    pub level: i32,
    /// Threads one call may use; 0 = the number of cores.
    pub threads: usize,
    /// [`decompress`] refuses a stream that says it holds more than this many bytes.
    pub max_output: u64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            level: 3,
            threads: 0,
            max_output: u64::MAX,
        }
    }
}

/// How a block of bytes is cut into streams before coding: what the bytes are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Kind(u8);

impl Kind {
    /// Bytes with no structure known.
    pub const PLAIN: Kind = Kind(split::PLAIN);
    pub const BF16: Kind = Kind(split::BF16);
    pub const F16: Kind = Kind(split::F16);
    pub const F32: Kind = Kind(split::F32);

    /// Little endian integers of `width` bytes (2, 4 or 8; 1 is [`Kind::PLAIN`]).
    pub fn integers(width: usize) -> Option<Kind> {
        (1..=8).contains(&width).then_some(Kind(width as u8))
    }

    /// A quantized block type of ggml, by its `ggml_type` number in a GGUF file; `None` for a type not known here
    /// (store those as [`Kind::PLAIN`]).
    pub fn ggml(ggml_type: u32) -> Option<Kind> {
        split::quant(ggml_type).map(|_| Kind(split::QUANT + ggml_type as u8))
    }

    /// A SafeTensors `dtype` name.
    pub fn safetensors(dtype: &str) -> Kind {
        match dtype {
            "BF16" => Kind::BF16,
            "F16" => Kind::F16,
            "F32" => Kind::F32,
            d => Kind::integers(crate::safetensors::width(d)).unwrap_or(Kind::PLAIN),
        }
    }

    /// Bytes of one value (or quantized block): a block coded as this kind is a whole number of them.
    pub fn unit(self) -> usize {
        split::unit(self.0)
    }

    /// The code a block of this kind carries in the coded stream.
    pub fn code(self) -> u8 {
        self.0
    }
}

/// Where the tensors of a SafeTensors, GGUF or ONNX file lie and what each holds; `None` for anything else,
/// or for a file that does not read cleanly as one of them.
pub fn layout(bytes: &[u8]) -> Option<Vec<Seg>> {
    crate::format::layout(&mut Cursor::new(bytes), bytes.len() as u64)
}

/// The kind of a [`Seg`] of [`layout`].
pub fn kind_of(seg: &Seg) -> Kind {
    Kind(seg.kind)
}

fn failed(e: String) -> Error {
    Error::Failed(e)
}

fn corrupt(e: String) -> Error {
    Error::Corrupt(e)
}

/// One block: `kind u8`, `flags u8` (bit 0: xor with the reference), `len u32` little endian, then one part per
/// stream of the kind: `method u8` (0 raw, 1 zstd, 2 huffman in one stream, 3 huffman in chunks), `coded length u32`, bytes.
///
/// `data` is a whole number of values of `kind` and at most [`MAX_BLOCK`]. With a `reference` of the same length
/// the block is also tried as `data ^ reference` and the smaller form kept; [`decode_block`] then needs the same
/// reference.
pub fn encode_block(
    kind: Kind,
    data: &[u8],
    reference: Option<&[u8]>,
    opt: &Options,
) -> Result<Vec<u8>, Error> {
    let unit = kind.unit();
    if unit == 0 || data.len() > MAX_BLOCK || !data.len().is_multiple_of(unit) {
        return Err(Error::Limit(format!(
            "a block of {} bytes is not a whole number of values of this kind, or too long",
            data.len()
        )));
    }
    if reference.is_some_and(|r| r.len() != data.len()) {
        return Err(Error::Reference(
            "the reference has not the length of the block".into(),
        ));
    }
    let plan = Plan {
        off: 0,
        len: data.len() as u64,
        kind: kind.0,
        base: reference.map(|_| (0, 0)),
    };
    let e = encode(&plan, data, reference, opt.level).map_err(failed)?;
    let mut out = vec![e.kind, u8::from(e.xor)];
    out.extend((data.len() as u32).to_le_bytes());
    for p in &e.parts {
        out.push(p.method);
        out.extend((p.bytes.len() as u32).to_le_bytes());
        out.extend(&p.bytes);
    }
    Ok(out)
}

fn take<'a>(r: &mut &'a [u8], n: usize) -> Result<&'a [u8], Error> {
    if r.len() < n {
        return Err(Error::Corrupt("cut short".into()));
    }
    let (a, b) = r.split_at(n);
    *r = b;
    Ok(a)
}

/// The inverse of [`encode_block`]. The reference must be the one given when coding (the caller keeps its hash).
pub fn decode_block(coded: &[u8], reference: Option<&[u8]>) -> Result<Vec<u8>, Error> {
    let mut r = coded;
    let head = take(&mut r, 6)?;
    let (kind, flags) = (head[0], head[1]);
    let len = u32::from_le_bytes([head[2], head[3], head[4], head[5]]) as usize;
    let unit = split::unit(kind);
    if unit == 0 || flags > 1 || len > MAX_BLOCK || !len.is_multiple_of(unit) {
        return Err(Error::Corrupt("a damaged block header".into()));
    }
    let mut parts = Vec::new();
    for _ in 0..split::streams(kind) {
        let m = take(&mut r, 5)?;
        let clen = u32::from_le_bytes([m[1], m[2], m[3], m[4]]) as usize;
        if clen > 2 * len + 1024 {
            return Err(Error::Corrupt("a damaged part length".into()));
        }
        parts.push((m[0], take(&mut r, clen)?.to_vec()));
    }
    if !r.is_empty() {
        return Err(Error::Corrupt("bytes after the block".into()));
    }
    let xor = flags == 1;
    let reference = match (xor, reference) {
        (true, None) => {
            return Err(Error::Reference(
                "the block was coded against a reference: pass it".into(),
            ));
        }
        (true, Some(b)) if b.len() != len => {
            return Err(Error::Reference(
                "the reference has not the length of the block".into(),
            ));
        }
        (_, b) => b.filter(|_| xor),
    };
    decode(
        &Raw {
            kind,
            xor,
            off: 0,
            len,
            base_off: 0,
            parts,
        },
        reference,
    )
    .map_err(corrupt)
}

/// A whole buffer. With a `reference` (an earlier version of the same model) tensors of the same name, type and
/// size are also tried as `tensor ^ reference tensor`; [`decompress`] then needs the same reference bytes.
pub fn compress(data: &[u8], reference: Option<&[u8]>, opt: &Options) -> Result<Vec<u8>, Error> {
    let len = data.len() as u64;
    let mut input = Cursor::new(data);
    let mut base = reference.map(Cursor::new);
    let plans = make_plans(
        &mut input,
        len,
        base.as_mut()
            .map(|b| (b, reference.map_or(0, |r| r.len() as u64))),
    )
    .map_err(Error::Reference)?;
    let mut out = Vec::with_capacity(HEAD_LEN as usize + data.len() / 2);
    out.extend(MAGIC);
    out.extend(len.to_le_bytes());
    out.extend(blake3::hash(data).as_bytes());
    out.extend(reference.map_or([0u8; 32], |r| *blake3::hash(r).as_bytes()));
    out.extend((plans.len() as u32).to_le_bytes());
    for batch in plans.chunks(workers_for(opt.threads)) {
        let encoded: Vec<Result<Encoded, String>> = std::thread::scope(|s| {
            let handles: Vec<_> = batch
                .iter()
                .map(|plan| {
                    let d = &data[plan.off as usize..(plan.off + plan.len) as usize];
                    let b = match (plan.base, reference) {
                        (Some((_, off)), Some(r)) => r.get(off as usize..(off + plan.len) as usize),
                        _ => None,
                    };
                    s.spawn(move || encode(plan, d, b, opt.level))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("a worker panicked".into())))
                .collect()
        });
        for e in encoded {
            write_block(&mut out, &e.map_err(failed)?).map_err(failed)?;
        }
    }
    Ok(out)
}

/// The length of the original, from the header of a [`compress`] stream.
pub fn original_len(coded: &[u8]) -> Result<u64, Error> {
    let mut r = coded;
    Ok(read_header(&mut r).map_err(corrupt)?.len)
}

/// The inverse of [`compress`]. The blake3 of the original is checked; the result is exactly what went in or an error.
pub fn decompress(coded: &[u8], reference: Option<&[u8]>, opt: &Options) -> Result<Vec<u8>, Error> {
    let mut r = coded;
    let h = read_header(&mut r).map_err(corrupt)?;
    if h.len > opt.max_output {
        return Err(Error::Limit(format!(
            "the stream holds {} bytes, more than the {} allowed",
            h.len, opt.max_output
        )));
    }
    match reference {
        Some(b) if *blake3::hash(b).as_bytes() != h.base_hash => {
            return Err(Error::Reference(
                "the reference is not the one this was coded against".into(),
            ));
        }
        None if h.base_hash != [0u8; 32] => {
            return Err(Error::Reference(
                "coded against a reference: pass it".into(),
            ));
        }
        _ => {}
    }
    // never trust the length in the header for an allocation: the output grows with the blocks actually read
    let mut out: Vec<u8> = Vec::new();
    let mut left = h.blocks;
    while left > 0 {
        let n = (left as usize).min(workers_for(opt.threads));
        let mut raws = Vec::with_capacity(n);
        for _ in 0..n {
            raws.push(read_block(&mut r).map_err(corrupt)?);
        }
        let mut bases: Vec<Option<&[u8]>> = Vec::with_capacity(n);
        for raw in &raws {
            bases.push(if raw.xor {
                let b = reference
                    .ok_or_else(|| Error::Reference("a block needs the reference".into()))?;
                let end = raw
                    .base_off
                    .checked_add(raw.len as u64)
                    .filter(|e| *e <= b.len() as u64);
                Some(
                    &b[raw.base_off as usize
                        ..end.ok_or_else(|| {
                            Error::Corrupt("a reference range past its end".into())
                        })? as usize],
                )
            } else {
                None
            });
        }
        let decoded: Vec<Result<Vec<u8>, String>> = std::thread::scope(|s| {
            let handles: Vec<_> = raws
                .iter()
                .zip(&bases)
                .map(|(raw, b)| s.spawn(move || decode(raw, *b)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("a worker panicked".into())))
                .collect()
        });
        for (raw, d) in raws.iter().zip(decoded) {
            if raw.off != out.len() as u64 {
                return Err(Error::Corrupt("blocks out of order".into()));
            }
            out.extend(d.map_err(corrupt)?);
            if out.len() as u64 > h.len {
                return Err(Error::Corrupt("more bytes than the header says".into()));
            }
        }
        left -= n as u32;
    }
    if out.len() as u64 != h.len || blake3::hash(&out).as_bytes() != &h.hash {
        return Err(Error::Corrupt(
            "the bytes differ from the original (blake3 mismatch)".into(),
        ));
    }
    Ok(out)
}

/// What a coded buffer holds, without decoding it.
pub fn inspect(coded: &[u8]) -> Result<Stats, Error> {
    let mut r = coded;
    let h = read_header(&mut r).map_err(corrupt)?;
    let mut s = Stats {
        original: h.len,
        packed: coded.len() as u64,
        blocks: h.blocks,
        ..Stats::default()
    };
    for _ in 0..h.blocks {
        let raw = read_block(&mut r).map_err(corrupt)?;
        s.xor_blocks += u32::from(raw.xor);
        for (m, _) in &raw.parts {
            match m {
                0 => s.raw_parts += 1,
                1 => s.zstd_parts += 1,
                _ => s.huff_parts += 1,
            }
        }
    }
    Ok(s)
}

/// A piece of a file whose type is not known, as a chunker of a store hands out: it may start anywhere inside a tensor.
/// Tries plain bytes and the float kinds at every alignment on a sample, then codes the whole with the smallest.
/// The result is `shift u8, tail u8`, the `shift` bytes in front and the `tail` bytes behind kept raw (so the rest is a
/// whole number of values), then one [`encode_block`] block. [`decode_auto`] needs only that.
///
/// `data` is at most [`MAX_BLOCK`]. The choice costs a few order-0 counts over a 16 KiB sample, then one coding of the whole;
/// a store that knows the type picks the kind itself and calls [`encode_block`].
pub fn encode_auto(data: &[u8], reference: Option<&[u8]>, opt: &Options) -> Result<Vec<u8>, Error> {
    if reference.is_some_and(|r| r.len() != data.len()) {
        return Err(Error::Reference(
            "the reference has not the length of the block".into(),
        ));
    }
    let (code, shift) = split::guess(data);
    let kind = Kind(code);
    let unit = kind.unit();
    let n = (data.len().saturating_sub(shift) / unit) * unit;
    let tail = data.len() - shift - n;
    let body = &data[shift..shift + n];
    let reference = reference.map(|r| &r[shift..shift + n]);
    let mut out = vec![shift as u8, tail as u8];
    out.extend(&data[..shift]);
    out.extend(&data[shift + n..]);
    out.extend(encode_block(kind, body, reference, opt)?);
    Ok(out)
}

/// The inverse of [`encode_auto`].
pub fn decode_auto(coded: &[u8], reference: Option<&[u8]>) -> Result<Vec<u8>, Error> {
    let mut r = coded;
    let head = take(&mut r, 2)?;
    let (shift, tail) = (head[0] as usize, head[1] as usize);
    let front = take(&mut r, shift)?;
    let back = take(&mut r, tail)?;
    let reference = match reference {
        Some(b) if b.len() >= shift + tail => Some(&b[shift..b.len() - tail]),
        Some(_) => {
            return Err(Error::Reference(
                "the reference is shorter than the block".into(),
            ));
        }
        None => None,
    };
    let body = decode_block(r, reference)?;
    let mut out = Vec::with_capacity(shift + body.len() + tail);
    out.extend(front);
    out.extend(body);
    out.extend(back);
    Ok(out)
}

/// The length [`decode_auto`] will return for `coded`, read from its header without decoding, so a caller can refuse a
/// stream that claims more than it may hold before anything is allocated.
pub fn auto_len(coded: &[u8]) -> Result<usize, Error> {
    let mut r = coded;
    let head = take(&mut r, 2)?;
    let (shift, tail) = (head[0] as usize, head[1] as usize);
    take(&mut r, shift + tail)?;
    let b = take(&mut r, 6)?;
    Ok(shift + tail + u32::from_le_bytes([b[2], b[3], b[4], b[5]]) as usize)
}
