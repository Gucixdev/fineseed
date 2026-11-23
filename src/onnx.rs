//! ONNX (a protobuf file): finds the `raw_data` of every initializer of the main graph, the bytes that hold the
//! weights, and what type they are. Only the path ModelProto.graph (7) -> GraphProto.initializer (5) ->
//! TensorProto.{name 8, data_type 2, raw_data 9, data_location 14} is read; everything else is skipped without
//! being read. The whole file must parse as protobuf or the answer is `None` (plain bytes).

use crate::format::Seg;
use crate::split;
use std::io::{BufReader, Read, Seek, SeekFrom};

struct Pb<'a, R> {
    r: BufReader<&'a mut R>,
    pos: u64,
}

impl<R: Read + Seek> Pb<'_, R> {
    fn varint(&mut self) -> Option<u64> {
        let (mut v, mut shift) = (0u64, 0u32);
        loop {
            let mut b = [0u8; 1];
            self.r.read_exact(&mut b).ok()?;
            self.pos += 1;
            if shift >= 64 {
                return None;
            }
            v |= u64::from(b[0] & 0x7f) << shift;
            if b[0] & 0x80 == 0 {
                return Some(v);
            }
            shift += 7;
        }
    }
    fn skip(&mut self, n: u64, end: u64) -> Option<()> {
        let to = self.pos.checked_add(n).filter(|t| *t <= end)?;
        self.r.seek_relative(i64::try_from(n).ok()?).ok()?;
        self.pos = to;
        Some(())
    }
    fn text(&mut self, n: u64, end: u64) -> Option<String> {
        if n > 1 << 16 || self.pos + n > end {
            return None;
        }
        let mut v = vec![0u8; n as usize];
        self.r.read_exact(&mut v).ok()?;
        self.pos += n;
        Some(String::from_utf8_lossy(&v).into_owned())
    }
    /// The next field header inside a message ending at `end`: `(field number, wire type)`.
    fn tag(&mut self) -> Option<(u64, u64)> {
        let t = self.varint()?;
        Some((t >> 3, t & 7))
    }
    /// Skip the value of a field of wire type `wire`, length-delimited ones included.
    fn skip_value(&mut self, wire: u64, end: u64) -> Option<()> {
        match wire {
            0 => self.varint().map(|_| ()),
            1 => self.skip(8, end),
            5 => self.skip(4, end),
            2 => {
                let n = self.varint()?;
                self.skip(n, end)
            }
            _ => None,
        }
    }
}

fn kind_of(data_type: u64) -> (u8, u64) {
    match data_type {
        1 => (split::F32, 4),
        10 => (split::F16, 2),
        16 => (split::BF16, 2),
        7 | 11 | 13 => (8, 8),
        6 | 12 => (4, 4),
        4 | 5 => (2, 2),
        _ => (split::PLAIN, 1),
    }
}

fn tensor<R: Read + Seek>(p: &mut Pb<'_, R>, end: u64) -> Option<Option<Seg>> {
    let (mut name, mut dtype, mut raw, mut external) = (String::new(), 0u64, None, false);
    while p.pos < end {
        let (f, w) = p.tag()?;
        match (f, w) {
            (8, 2) => {
                let n = p.varint()?;
                name = p.text(n, end)?;
            }
            (2, 0) => dtype = p.varint()?,
            (14, 0) => external = p.varint()? == 1,
            (9, 2) => {
                let n = p.varint()?;
                let begin = p.pos;
                p.skip(n, end)?;
                raw = Some((begin, begin + n));
            }
            _ => p.skip_value(w, end)?,
        }
    }
    if p.pos != end {
        return None;
    }
    Some(match (raw, external) {
        (Some((begin, e)), false) if e > begin => {
            let (mut kind, unit) = kind_of(dtype);
            if (e - begin) % unit != 0 {
                kind = split::PLAIN;
            }
            Some(Seg {
                name,
                kind,
                begin,
                end: e,
            })
        }
        _ => None,
    })
}

fn graph<R: Read + Seek>(p: &mut Pb<'_, R>, end: u64, out: &mut Vec<Seg>) -> Option<()> {
    while p.pos < end {
        let (f, w) = p.tag()?;
        if (f, w) == (5, 2) {
            let n = p.varint()?;
            let stop = p.pos.checked_add(n).filter(|s| *s <= end)?;
            if let Some(seg) = tensor(p, stop)? {
                out.push(seg);
            }
        } else {
            p.skip_value(w, end)?;
        }
    }
    (p.pos == end).then_some(())
}

pub fn parse<R: Read + Seek>(file: &mut R, len: u64) -> Option<Vec<Seg>> {
    // a ModelProto nearly always starts with `ir_version` (field 1, varint): cheap test before reading anything
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut first = [0u8; 1];
    file.read_exact(&mut first).ok()?;
    if first[0] != 0x08 || len < 16 {
        return None;
    }
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut p = Pb {
        r: BufReader::with_capacity(1 << 16, file),
        pos: 0,
    };
    let mut out = Vec::new();
    let mut seen_graph = false;
    while p.pos < len {
        let (f, w) = p.tag()?;
        if (f, w) == (7, 2) && !seen_graph {
            let n = p.varint()?;
            let stop = p.pos.checked_add(n).filter(|s| *s <= len)?;
            graph(&mut p, stop, &mut out)?;
            seen_graph = true;
        } else {
            p.skip_value(w, len)?;
        }
    }
    if p.pos != len || !seen_graph || out.is_empty() {
        return None;
    }
    out.sort_by_key(|s| (s.begin, s.end));
    if out.windows(2).any(|q| q[1].begin < q[0].end) {
        return None;
    }
    Some(out)
}
