//! `layout FILE`: what the reader finds in a model file - tensors, kinds, and whether they tile the data area.
use std::collections::BTreeMap;
fn main() {
    let path = std::env::args().nth(1).expect("layout FILE");
    let mut f = std::fs::File::open(&path).expect("open");
    let len = f.metadata().expect("stat").len();
    let Some(segs) = fineseed::format::layout(&mut f, len) else {
        println!("not read as safetensors, gguf or onnx: packed as plain bytes");
        return;
    };
    let mut by: BTreeMap<u8, (usize, u64)> = BTreeMap::new();
    let (mut gaps, mut pos) = (0u64, segs.first().map_or(0, |s| s.begin));
    for s in &segs {
        gaps += s.begin - pos;
        pos = s.end;
        let e = by.entry(s.kind).or_default();
        e.0 += 1;
        e.1 += s.end - s.begin;
    }
    let covered: u64 = by.values().map(|v| v.1).sum();
    println!(
        "{} tensors, {covered} of {len} bytes, {gaps} bytes of padding between them, header {} bytes",
        segs.len(),
        segs[0].begin
    );
    for (k, (n, b)) in by {
        println!("  kind {k:>3}: {n:>5} tensors {b:>14} bytes");
    }
}
