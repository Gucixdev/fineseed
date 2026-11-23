//! GGUF (v2, v3) reader: where each tensor's bytes are and what type they hold. The header is read with bounds on
//! every length, so a damaged or hostile file gives `None` and is then packed as plain bytes.

use crate::format::Seg;
use crate::split;
use std::io::{BufReader, Read, Seek, SeekFrom};

const HEADER_MAX: u64 = 512 << 20;
const STRING_MAX: u64 = 1 << 24;
const MAX_DIMS: u32 = 8;

struct Cur<'a, R> {
    r: BufReader<&'a mut R>,
    pos: u64,
}

impl<R: Read + Seek> Cur<'_, R> {
    fn bytes<const N: usize>(&mut self) -> Option<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b).ok()?;
        self.pos += N as u64;
        if self.pos > HEADER_MAX { None } else { Some(b) }
    }
    fn u32(&mut self) -> Option<u32> {
        self.bytes().map(u32::from_le_bytes)
    }
    fn u64(&mut self) -> Option<u64> {
        self.bytes().map(u64::from_le_bytes)
    }
    fn skip(&mut self, n: u64) -> Option<()> {
        self.pos = self.pos.checked_add(n).filter(|p| *p <= HEADER_MAX)?;
        self.r.seek_relative(i64::try_from(n).ok()?).ok()
    }
    fn string(&mut self) -> Option<String> {
        let n = self.u64()?;
        if n > STRING_MAX {
            return None;
        }
        let mut v = vec![0u8; n as usize];
        self.r.read_exact(&mut v).ok()?;
        self.pos += n;
        String::from_utf8(v).ok()
    }
}

/// Skip a metadata value of type `t`; return it when it is an unsigned integer (for `general.alignment`).
fn value<R: Read + Seek>(c: &mut Cur<'_, R>, t: u32, depth: u32) -> Option<Option<u64>> {
    let fixed = |c: &mut Cur<'_, R>, n: u64| c.skip(n).map(|()| None);
    match t {
        0 | 1 | 7 => fixed(c, 1),
        2 | 3 => fixed(c, 2),
        4..=6 => {
            let v = u64::from(c.u32()?);
            Some((t == 4).then_some(v))
        }
        10..=12 => fixed(c, 8),
        8 => c.string().map(|_| None),
        9 if depth == 0 => {
            let (et, n) = (c.u32()?, c.u64()?);
            let width = match et {
                0 | 1 | 7 => Some(1),
                2 | 3 => Some(2),
                4..=6 => Some(4),
                10..=12 => Some(8),
                _ => None,
            };
            match width {
                Some(w) => c.skip(n.checked_mul(w)?)?,
                None => {
                    if n > HEADER_MAX {
                        return None;
                    }
                    for _ in 0..n {
                        value(c, et, depth + 1)?;
                    }
                }
            }
            Some(None)
        }
        _ => None,
    }
}

/// The scalar width of an unquantized ggml type, and the kind its bytes are cut by.
fn plain_type(t: u32) -> Option<(u64, u8)> {
    Some(match t {
        0 => (4, split::F32),
        1 => (2, split::F16),
        30 => (2, split::BF16),
        24 => (1, split::PLAIN),
        25 => (2, 2),
        26 => (4, 4),
        27 | 28 => (8, 8),
        _ => return None,
    })
}

pub fn parse<R: Read + Seek>(file: &mut R, len: u64) -> Option<Vec<Seg>> {
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut c = Cur {
        r: BufReader::with_capacity(1 << 16, file),
        pos: 0,
    };
    if &c.bytes::<4>()? != b"GGUF" {
        return None;
    }
    let version = c.u32()?;
    if version != 2 && version != 3 {
        return None;
    }
    let (n_tensors, n_kv) = (c.u64()?, c.u64()?);
    if n_tensors > 1 << 20 || n_kv > 1 << 20 {
        return None;
    }
    let mut align = 32u64;
    for _ in 0..n_kv {
        let key = c.string()?;
        let t = c.u32()?;
        let v = value(&mut c, t, 0)?;
        if key == "general.alignment" {
            align = v.filter(|a| *a > 0 && a.is_power_of_two())?;
        }
    }
    let mut infos = Vec::new();
    for _ in 0..n_tensors {
        let name = c.string()?;
        let nd = c.u32()?;
        if nd > MAX_DIMS {
            return None;
        }
        let mut elems = 1u64;
        for _ in 0..nd {
            elems = elems.checked_mul(c.u64()?)?;
        }
        let t = c.u32()?;
        let off = c.u64()?;
        infos.push((name, elems, t, off));
    }
    let data_start = c.pos.div_ceil(align).checked_mul(align)?;
    let mut segs = Vec::new();
    for (name, elems, t, off) in infos {
        let (kind, size) = if let Some((w, kind)) = plain_type(t) {
            (kind, elems.checked_mul(w)?)
        } else {
            // a type this does not know has no known size either, so the layout cannot be trusted
            let q = split::quant(t)?;
            if !elems.is_multiple_of(q.elems as u64) {
                return None;
            }
            (
                split::QUANT + t as u8,
                elems / q.elems as u64 * q.bytes as u64,
            )
        };
        let begin = data_start.checked_add(off)?;
        let end = begin.checked_add(size)?;
        if end > len {
            return None;
        }
        // a whole number of units, or the kind falls back to plain bytes
        let kind = if size % split::unit(kind).max(1) as u64 == 0 {
            kind
        } else {
            split::PLAIN
        };
        segs.push(Seg {
            name,
            kind,
            begin,
            end,
        });
    }
    segs.sort_by_key(|s| (s.begin, s.end));
    if segs.windows(2).any(|p| p[1].begin < p[0].end) {
        return None;
    }
    Some(segs)
}
