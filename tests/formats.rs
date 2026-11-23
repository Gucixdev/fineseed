use fineseed::{archive, format, pack, split, unpack};
use std::fs;
use std::path::PathBuf;

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn noise(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

fn all_kinds() -> Vec<u8> {
    let mut k: Vec<u8> = (1..=8).collect();
    k.extend([split::BF16, split::F16, split::F32]);
    for t in 0..64u32 {
        if split::quant(t).is_some() {
            k.push(split::QUANT + t as u8);
        }
    }
    k
}

#[test]
fn split_then_merge_gives_the_input_for_every_kind() {
    for kind in all_kinds() {
        let u = split::unit(kind);
        for n in [0usize, 1, 3, 70] {
            let data = noise(kind as u64 * 31 + n as u64, n * u);
            let parts = split::split(kind, &data);
            assert_eq!(parts.len(), split::streams(kind), "kind {kind}");
            assert_eq!(
                parts.iter().map(Vec::len).collect::<Vec<_>>(),
                split::lens(kind, n),
                "kind {kind}"
            );
            let mut back = vec![0u8; data.len()];
            split::merge(kind, &parts, &mut back).unwrap();
            assert_eq!(back, data, "kind {kind} n {n}");
        }
    }
}

#[test]
fn every_block_descriptor_covers_its_bytes_exactly_once() {
    for t in 0..64u32 {
        if let Some(q) = split::quant(t) {
            assert!(split::covers(&q), "ggml type {t}");
        }
    }
}

#[test]
fn merge_refuses_wrong_streams_and_nibbles_above_fifteen() {
    let kind = split::QUANT + 2;
    let data = noise(1, split::unit(kind) * 4);
    let mut parts = split::split(kind, &data);
    let mut out = vec![0u8; data.len()];
    assert!(split::merge(kind, &parts[..1], &mut out).is_err());
    let last = parts.len() - 1;
    parts[last][3] = 16;
    assert!(split::merge(kind, &parts, &mut out).is_err());
    assert!(split::merge(0, &[], &mut []).is_err());
}

fn kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
    out.extend((key.len() as u64).to_le_bytes());
    out.extend(key.as_bytes());
    out.extend(8u32.to_le_bytes());
    out.extend((value.len() as u64).to_le_bytes());
    out.extend(value.as_bytes());
}

/// A GGUF v3 with the given tensors `(name, ggml type, elements, bytes)`, data aligned to 32.
fn gguf_file(tensors: &[(&str, u32, u64, Vec<u8>)]) -> Vec<u8> {
    let mut f = b"GGUF".to_vec();
    f.extend(3u32.to_le_bytes());
    f.extend((tensors.len() as u64).to_le_bytes());
    f.extend(1u64.to_le_bytes());
    kv_string(&mut f, "general.name", "test");
    let mut off = 0u64;
    for (name, t, elems, bytes) in tensors {
        f.extend((name.len() as u64).to_le_bytes());
        f.extend(name.as_bytes());
        f.extend(1u32.to_le_bytes());
        f.extend(elems.to_le_bytes());
        f.extend(t.to_le_bytes());
        f.extend(off.to_le_bytes());
        off += (bytes.len() as u64).next_multiple_of(32);
    }
    while !f.len().is_multiple_of(32) {
        f.push(0);
    }
    for (_, _, _, bytes) in tensors {
        f.extend(bytes);
        while !f.len().is_multiple_of(32) {
            f.push(0);
        }
    }
    f
}

fn layout_of(bytes: &[u8], name: &str) -> Option<Vec<format::Seg>> {
    let p = dir(name).join("f");
    fs::write(&p, bytes).unwrap();
    let mut f = fs::File::open(&p).unwrap();
    format::layout(&mut f, bytes.len() as u64)
}

#[test]
fn a_gguf_file_is_read_tensor_by_tensor() {
    let q4k = split::QUANT + 12;
    let file = gguf_file(&[
        ("a.weight", 12, 512, noise(1, 288)),
        ("b.weight", 1, 100, noise(2, 200)),
        ("c.weight", 30, 64, noise(3, 128)),
        ("d.weight", 0, 8, noise(4, 32)),
    ]);
    let segs = layout_of(&file, "gguf_layout").unwrap();
    assert_eq!(segs.len(), 4);
    assert_eq!(segs[0].kind, q4k);
    assert_eq!(segs[0].end - segs[0].begin, 288);
    assert_eq!(
        (segs[1].kind, segs[2].kind, segs[3].kind),
        (split::F16, split::BF16, split::F32)
    );
    assert!(segs.windows(2).all(|p| p[0].end <= p[1].begin));
    assert_eq!(
        &file[segs[0].begin as usize..segs[0].end as usize].len(),
        &288
    );
}

#[test]
fn a_gguf_that_lies_is_read_as_plain_bytes() {
    // an element count that does not fill whole blocks, a tensor past the end, an unknown type, garbage after the magic
    assert!(
        layout_of(
            &gguf_file(&[("x", 12, 100, noise(1, 144))]),
            "gguf_bad_elems"
        )
        .is_none()
    );
    let mut cut = gguf_file(&[("x", 12, 512, noise(1, 288))]);
    cut.truncate(cut.len() - 100);
    assert!(layout_of(&cut, "gguf_cut").is_none());
    assert!(layout_of(&gguf_file(&[("x", 9999, 32, noise(1, 64))]), "gguf_unknown").is_none());
    let mut junk = b"GGUF".to_vec();
    junk.extend(noise(5, 300));
    assert!(layout_of(&junk, "gguf_junk").is_none());
    let mut huge = b"GGUF".to_vec();
    huge.extend(3u32.to_le_bytes());
    huge.extend(u64::MAX.to_le_bytes());
    huge.extend(u64::MAX.to_le_bytes());
    assert!(layout_of(&huge, "gguf_huge").is_none());
}

fn varint(mut v: u64) -> Vec<u8> {
    let mut o = Vec::new();
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            o.push(b);
            return o;
        }
        o.push(b | 0x80);
    }
}

fn field(num: u64, wire: u64, body: &[u8]) -> Vec<u8> {
    let mut o = varint(num << 3 | wire);
    if wire == 2 {
        o.extend(varint(body.len() as u64));
    }
    o.extend(body);
    o
}

fn tensor_proto(name: &str, dtype: u64, raw: &[u8], external: bool) -> Vec<u8> {
    let mut t = field(8, 2, name.as_bytes());
    t.extend(field(2, 0, &varint(dtype)));
    if external {
        t.extend(field(14, 0, &varint(1)));
    } else {
        t.extend(field(9, 2, raw));
    }
    t
}

fn onnx_file(tensors: &[(&str, u64, Vec<u8>, bool)]) -> Vec<u8> {
    let mut graph = field(2, 2, b"main");
    for (n, d, raw, ext) in tensors {
        graph.extend(field(5, 2, &tensor_proto(n, *d, raw, *ext)));
    }
    let mut m = field(1, 0, &varint(8));
    m.extend(field(2, 2, b"test"));
    m.extend(field(7, 2, &graph));
    m
}

#[test]
fn an_onnx_file_is_read_and_external_data_is_left_alone() {
    let file = onnx_file(&[
        ("w1", 1, noise(1, 4096), false),
        ("w2", 16, noise(2, 2048), false),
        ("ext", 1, Vec::new(), true),
        ("i", 7, noise(3, 800), false),
    ]);
    let segs = layout_of(&file, "onnx_layout").unwrap();
    assert_eq!(segs.len(), 3);
    assert_eq!(
        (segs[0].kind, segs[1].kind, segs[2].kind),
        (split::F32, split::BF16, 8)
    );
    assert_eq!(
        &file[segs[0].begin as usize..segs[0].end as usize].len(),
        &4096
    );
}

#[test]
fn bytes_that_only_look_like_onnx_are_plain() {
    let mut f = onnx_file(&[("w1", 1, noise(1, 4096), false)]);
    f.truncate(f.len() - 10);
    assert!(layout_of(&f, "onnx_cut").is_none());
    let mut g = vec![0x08, 0x08];
    g.extend(noise(9, 500));
    assert!(layout_of(&g, "onnx_noise").is_none());
}

fn roundtrip_file(name: &str, bytes: &[u8]) -> (u64, u64) {
    let d = dir(name);
    let (i, s, o) = (d.join("in"), d.join("in.seed"), d.join("out"));
    fs::write(&i, bytes).unwrap();
    pack(&i, None, &s, 3).unwrap();
    unpack(&s, None, &o).unwrap();
    assert_eq!(fs::read(&o).unwrap(), bytes);
    (bytes.len() as u64, fs::metadata(&s).unwrap().len())
}

#[test]
fn gguf_and_onnx_files_come_back_byte_for_byte() {
    let mut q = Vec::new();
    for t in [
        2u32, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 34, 35,
    ] {
        let qq = split::quant(t).unwrap();
        q.push((t, qq.elems as u64 * 5, noise(t as u64, qq.bytes * 5)));
    }
    let names: Vec<String> = (0..q.len()).map(|i| format!("t{i}")).collect();
    let tensors: Vec<(&str, u32, u64, Vec<u8>)> = q
        .iter()
        .zip(&names)
        .map(|((t, e, b), n)| (n.as_str(), *t, *e, b.clone()))
        .collect();
    roundtrip_file("rt_gguf_all_quants", &gguf_file(&tensors));
    let o = onnx_file(&[
        ("a", 1, noise(1, 8000), false),
        ("b", 10, noise(2, 4000), false),
        ("c", 5, noise(3, 99), false),
    ]);
    roundtrip_file("rt_onnx", &o);
}

#[test]
fn quantized_nibbles_that_are_not_uniform_cost_less() {
    // a Q4_0 tensor whose 4-bit values lean to a few symbols: the field split must find it
    let blocks = 4000;
    let mut data = Vec::new();
    let r = noise(7, blocks * 16);
    for b in 0..blocks {
        data.extend([0x3c, 0x00]);
        data.extend((0..16).map(|j| {
            let x = r[b * 16 + j];
            (7 + (x % 3)) | (7 + ((x >> 4) % 3)) << 4
        }));
    }
    let (orig, packed) = roundtrip_file(
        "rt_nibbles",
        &gguf_file(&[("w", 2, blocks as u64 * 32, data)]),
    );
    assert!(packed * 100 < orig * 40, "{packed} of {orig}");
}

#[test]
fn similar_and_equal_tensors_of_other_models_are_found() {
    let d = dir("cross_model");
    let base = noise(11, 600_000);
    let mut tuned = base.clone();
    for i in (0..tuned.len()).step_by(7) {
        tuned[i] ^= 1;
    }
    let ordered = |a: &[u8], b: &[u8]| {
        let mut f = Vec::new();
        for (n, v) in [("blk.0.w", a), ("blk.1.w", b)] {
            f.push((n.to_string(), v.to_vec()));
        }
        f
    };
    // model A has two layers; model B has the same first layer, a lightly changed second layer under another name
    let mk = |layers: Vec<(String, Vec<u8>)>| {
        let t: Vec<(&str, u32, u64, Vec<u8>)> = layers
            .iter()
            .map(|(n, v)| (n.as_str(), 8u32, v.len() as u64 / 34 * 32, v.clone()))
            .collect();
        gguf_file(&t)
    };
    let a_layers = ordered(&base[..34 * 8000], &base[34 * 8000..34 * 16000]);
    let mut b_layers = ordered(&base[..34 * 8000], &tuned[34 * 8000..34 * 16000]);
    b_layers[1].0 = "renamed.w".into();
    let (pa, pb) = (d.join("a.gguf"), d.join("b.gguf"));
    fs::write(&pa, mk(a_layers)).unwrap();
    fs::write(&pb, mk(b_layers)).unwrap();
    let out = d.join("x.fseed");
    let rep = archive::store(
        &archive::collect(&[pa.clone(), pb.clone()]).unwrap(),
        &out,
        3,
    )
    .unwrap();
    assert!(rep.dup_blocks > 0, "the equal layer is not stored twice");
    assert!(
        rep.xor_blocks > 0,
        "the changed layer under another name is coded against the first model's"
    );
    let r = d.join("r");
    archive::restore(&out, &r).unwrap();
    assert_eq!(fs::read(r.join("a.gguf")).unwrap(), fs::read(&pa).unwrap());
    assert_eq!(fs::read(r.join("b.gguf")).unwrap(), fs::read(&pb).unwrap());
}
