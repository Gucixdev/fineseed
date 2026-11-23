//! What kind of file this is and where its tensors lie: SafeTensors, GGUF and ONNX are read, anything else (or a
//! file that does not read cleanly as one of them) is `None` and is packed as plain bytes.

use crate::{gguf, onnx, safetensors, split};
use std::io::{Read, Seek};

/// A run of bytes of one tensor and how to cut it into streams (`split::*`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seg {
    pub name: String,
    pub kind: u8,
    pub begin: u64,
    pub end: u64,
}

/// Segments sorted by `begin`, none overlapping; each a whole number of units of its kind.
pub fn layout<R: Read + Seek>(file: &mut R, len: u64) -> Option<Vec<Seg>> {
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
