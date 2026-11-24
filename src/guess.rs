//! Which float kind a run of bytes of unknown type most likely is. Kept out of `split.rs`, which has to stay free of
//! other modules and crates so that `lean4/harness/rust.rs` can compile it alone.
use crate::split::{BF16, F16, F32, PLAIN, split, unit};

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
