//! The layout of a SafeTensors file: a `u64` header length, a JSON header, then the tensor bytes.
use std::io::{Read, Seek, SeekFrom};

pub struct Tensor {
    pub name: String,
    pub dtype: String,
    /// Absolute offsets in the file.
    pub begin: u64,
    pub end: u64,
    /// Dimensions as the header gives them (empty when it gives none); the last one is the length of a row.
    pub shape: Vec<u64>,
}

pub struct Layout {
    pub data_start: u64,
    /// Sorted by `begin`, no two overlap.
    pub tensors: Vec<Tensor>,
}

const HEADER_MAX: u64 = 256 << 20;

/// The width in bytes of one value of `dtype`: how many byte planes the tensor splits into.
pub fn width(dtype: &str) -> usize {
    match dtype {
        "F64" | "I64" | "U64" | "C64" => 8,
        "F32" | "I32" | "U32" => 4,
        "BF16" | "F16" | "I16" | "U16" => 2,
        _ => 1,
    }
}

/// `None` when the file is not a SafeTensors file this reader can lay out exactly (a header that does
/// not parse, a range outside the file, overlapping ranges): the caller then packs it as plain bytes.
pub fn parse<R: Read + Seek>(file: &mut R, len: u64) -> Option<Layout> {
    if len < 8 {
        return None;
    }
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut first = [0u8; 8];
    file.read_exact(&mut first).ok()?;
    let n = u64::from_le_bytes(first);
    if n > HEADER_MAX || 8 + n > len {
        return None;
    }
    let mut header = vec![0u8; n as usize];
    file.read_exact(&mut header).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&header).ok()?;
    let data_start = 8 + n;
    let mut tensors = Vec::new();
    for (name, value) in json.as_object()? {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value.get("dtype")?.as_str()?.to_string();
        let range = value.get("data_offsets")?.as_array()?;
        let (b, e) = (range.first()?.as_u64()?, range.get(1)?.as_u64()?);
        let (begin, end) = (data_start.checked_add(b)?, data_start.checked_add(e)?);
        if e < b || end > len {
            return None;
        }
        let shape = value
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|d| d.as_u64()).collect())
            .unwrap_or_default();
        tensors.push(Tensor {
            name: name.clone(),
            dtype,
            begin,
            end,
            shape,
        });
    }
    tensors.sort_by_key(|t| (t.begin, t.end));
    if tensors.windows(2).any(|p| p[1].begin < p[0].end) {
        return None;
    }
    Some(Layout {
        data_start,
        tensors,
    })
}
