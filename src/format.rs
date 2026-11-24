//! What kind of file this is and where its tensors lie: SafeTensors, GGUF and ONNX are read, anything else (or a
//! file that does not read cleanly as one of them) is `None` and is packed as plain bytes.

use crate::{gguf, onnx, safetensors, split, zip};
use std::io::{Read, Seek, SeekFrom};

/// A run of bytes of one tensor and how to cut it into streams (`split::*`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seg {
    pub name: String,
    pub kind: u8,
    pub begin: u64,
    pub end: u64,
}

/// A byte range of a file read as a file of its own, so a model inside a zip is parsed as if it stood alone.
struct Window<'a, R> {
    inner: &'a mut R,
    begin: u64,
    len: u64,
    pos: u64,
}

impl<R: Read + Seek> Read for Window<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = buf.len().min(self.len.saturating_sub(self.pos) as usize);
        self.inner.seek(SeekFrom::Start(self.begin + self.pos))?;
        let k = self.inner.read(&mut buf[..n])?;
        self.pos += k as u64;
        Ok(k)
    }
}

impl<R: Read + Seek> Seek for Window<'_, R> {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let p = match to {
            SeekFrom::Start(p) => Some(p),
            SeekFrom::End(d) => self.len.checked_add_signed(d),
            SeekFrom::Current(d) => self.pos.checked_add_signed(d),
        };
        self.pos = p.ok_or_else(|| std::io::Error::other("seek before the start"))?;
        Ok(self.pos)
    }
}

/// A zip (DDUF, a PyTorch checkpoint, a model that ships as a zip): every stored entry is a segment, the ones that
/// are themselves a model file with the tensors of it, the others as plain runs that are cut and typed by a sample
/// from the start of the entry, so a block never straddles two storages of different types.
fn zip_layout<R: Read + Seek>(file: &mut R, len: u64) -> Option<Vec<Seg>> {
    let entries = zip::stored(file, len)?;
    let mut out = Vec::new();
    for (name, start, size) in entries {
        let inner = {
            let mut w = Window {
                inner: &mut *file,
                begin: start,
                len: size,
                pos: 0,
            };
            layout_plain(&mut w, size)
        };
        match inner {
            Some(segs) => out.extend(segs.into_iter().map(|s| Seg {
                name: format!("{name}/{}", s.name),
                begin: s.begin + start,
                end: s.end + start,
                ..s
            })),
            None if size > 0 => out.push(Seg {
                name,
                kind: split::PLAIN,
                begin: start,
                end: start + size,
            }),
            None => {}
        }
    }
    Some(out)
}

/// Segments sorted by `begin`, none overlapping; each a whole number of units of its kind.
pub fn layout<R: Read + Seek>(file: &mut R, len: u64) -> Option<Vec<Seg>> {
    layout_plain(file, len).or_else(|| zip_layout(file, len))
}

/// `layout` without the zip reader: SafeTensors, GGUF, ONNX.
fn layout_plain<R: Read + Seek>(file: &mut R, len: u64) -> Option<Vec<Seg>> {
    if let Some(l) = safetensors::parse(file, len) {
        return Some(
            l.tensors
                .into_iter()
                .map(|t| {
                    let kind = match t.dtype.as_str() {
                        "BF16" => split::BF16,
                        "F16" => split::F16,
                        "F32" => split::F32,
                        d => safetensors::width(d) as u8,
                    };
                    let kind = if (t.end - t.begin) % split::unit(kind) as u64 == 0 {
                        kind
                    } else {
                        split::PLAIN
                    };
                    Seg {
                        name: t.name,
                        kind,
                        begin: t.begin,
                        end: t.end,
                    }
                })
                .collect(),
        );
    }
    gguf::parse(file, len).or_else(|| onnx::parse(file, len))
}
