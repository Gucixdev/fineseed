use fineseed::{info, pack, unpack};
use std::fs;
use std::path::PathBuf;

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Deterministic noise, so a test needs no random crate.
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

/// Fake bf16 weights: a repeating high byte, a random low byte, the shape of real weights.
fn bf16(seed: u64, values: usize) -> Vec<u8> {
    let low = noise(seed, values);
    (0..values)
        .flat_map(|i| [low[i], 0x3c + (low[(i * 7) % values] % 3)])
        .collect()
}

fn safetensors(tensors: &[(&str, &str, Vec<u8>)], gap: usize, tail: &[u8]) -> Vec<u8> {
    let mut fields = Vec::new();
    let mut at = 0usize;
    for (name, dtype, bytes) in tensors {
        fields.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{}],"data_offsets":[{at},{}]}}"#,
            bytes.len(),
            at + bytes.len()
        ));
        at += bytes.len() + gap;
    }
    let mut header = format!("{{{}}}", fields.join(",")).into_bytes();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    for (_, _, bytes) in tensors {
        file.extend(bytes);
        file.extend(vec![0xAA; gap]);
    }
    file.extend(tail);
    file
}

fn roundtrip(name: &str, bytes: &[u8], base: Option<&[u8]>) -> (u64, u64) {
    let d = dir(name);
    let (i, s, o, b) = (
        d.join("in"),
        d.join("in.seed"),
        d.join("out"),
        d.join("base"),
    );
    fs::write(&i, bytes).unwrap();
    let base_path = base.map(|x| {
        fs::write(&b, x).unwrap();
        b.as_path()
    });
    let stats = pack(&i, base_path, &s, 3).unwrap();
    assert_eq!(info(&s).unwrap().blocks, stats.blocks);
    unpack(&s, base_path, &o).unwrap();
    assert_eq!(
        fs::read(&o).unwrap(),
        bytes,
        "{name}: the unpacked bytes differ"
    );
    (stats.original, stats.packed)
}

#[test]
fn a_safetensors_file_comes_back_byte_for_byte() {
    let f = safetensors(
        &[
            ("a", "BF16", bf16(1, 40000)),
            ("b", "F32", noise(2, 4096)),
            ("c", "U8", noise(3, 77)),
        ],
        3,
        b"tail",
    );
    let (original, packed) = roundtrip("st", &f, None);
    assert!(
        packed < original,
        "bf16-like weights must shrink: {packed} of {original}"
    );
}

#[test]
fn bytes_that_are_not_safetensors_come_back_byte_for_byte() {
    roundtrip("plain", &noise(5, 100_000), None);
    roundtrip("empty", b"", None);
    roundtrip("short", b"abc", None);
    roundtrip("bad_header", &[255; 64], None);
}

#[test]
fn a_header_with_overlapping_tensors_is_packed_as_plain_bytes() {
    let header = br#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}"#;
    let mut f = (header.len() as u64).to_le_bytes().to_vec();
    f.extend(header);
    f.extend([1u8; 6]);
    roundtrip("overlap", &f, None);
}

#[test]
fn a_tensor_longer_than_a_block_is_cut_and_comes_back() {
    let big = bf16(9, (fineseed::BLOCK as usize) / 2 + 12345);
    roundtrip("big", &safetensors(&[("w", "BF16", big)], 0, b""), None);
}

#[test]
fn a_fine_tune_packs_much_smaller_against_its_base_and_needs_it_to_unpack() {
    let w1 = bf16(11, 60000);
    let w2 = bf16(12, 60000);
    let base = safetensors(
        &[("w1", "BF16", w1.clone()), ("w2", "BF16", w2.clone())],
        0,
        b"",
    );
    let mut w2_tuned = w2.clone();
    for byte in w2_tuned.iter_mut().step_by(97) {
        *byte ^= 1;
    }
    let tuned = safetensors(&[("w1", "BF16", w1), ("w2", "BF16", w2_tuned)], 0, b"");
    let (_, alone) = roundtrip("tuned_alone", &tuned, None);
    let (_, delta) = roundtrip("tuned_delta", &tuned, Some(&base));
    assert!(
        delta * 2 < alone,
        "against its base it must be far smaller: {delta} vs {alone}"
    );

    let d = dir("tuned_needs_base");
    let (i, s) = (d.join("in"), d.join("in.seed"));
    fs::write(&i, &tuned).unwrap();
    fs::write(d.join("base"), &base).unwrap();
    pack(&i, Some(&d.join("base")), &s, 3).unwrap();
    assert!(unpack(&s, None, &d.join("o1")).is_err(), "no base given");
    fs::write(
        d.join("other"),
        safetensors(&[("x", "BF16", bf16(99, 10))], 0, b""),
    )
    .unwrap();
    assert!(
        unpack(&s, Some(&d.join("other")), &d.join("o2")).is_err(),
        "a different base"
    );
}

#[test]
fn a_damaged_seed_file_is_refused() {
    let d = dir("damaged");
    let (i, s) = (d.join("in"), d.join("in.seed"));
    fs::write(&i, safetensors(&[("a", "BF16", bf16(21, 30000))], 0, b"")).unwrap();
    pack(&i, None, &s, 3).unwrap();
    let mut seed = fs::read(&s).unwrap();
    let at = seed.len() / 2;
    seed[at] ^= 0x55;
    fs::write(&s, &seed).unwrap();
    assert!(unpack(&s, None, &d.join("out")).is_err());
    fs::write(&s, b"not a seed file at all").unwrap();
    assert!(unpack(&s, None, &d.join("out")).is_err());
}

mod huffman {
    use fineseed::huffman::{decode, decode_chunked, encode, encode_chunked, estimate, table};

    fn lcg(seed: u64, n: usize) -> Vec<u8> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 33) as u8
            })
            .collect()
    }

    #[test]
    fn roundtrips_skewed_uniform_single_and_tiny_planes() {
        let skewed: Vec<u8> = lcg(1, 100_000).iter().map(|b| 0x3c + (b % 7)).collect();
        let cases: Vec<Vec<u8>> = vec![
            skewed,
            lcg(2, 50_000),
            vec![9; 1000],
            vec![5],
            vec![1, 2],
            (0..=255).collect(),
        ];
        for data in cases {
            let coded = encode(&data).expect("non-empty");
            assert_eq!(
                coded.len(),
                estimate(&data),
                "estimate must equal the real size"
            );
            assert_eq!(decode(&coded, data.len()).expect("decodes"), data);
        }
    }

    #[test]
    fn deep_codes_stay_within_fifteen_bits() {
        // Fibonacci-like counts make a degenerate tree that would need codes longer than 15 bits
        let mut data = Vec::new();
        let (mut a, mut b) = (1usize, 1usize);
        for s in 0..30u8 {
            data.extend(std::iter::repeat_n(s, a));
            let c = a + b;
            a = b;
            b = c;
        }
        let coded = encode(&data).expect("non-empty");
        assert_eq!(decode(&coded, data.len()).expect("decodes"), data);
    }

    #[test]
    fn refuses_damaged_input() {
        let data: Vec<u8> = lcg(3, 4000).iter().map(|b| b % 11).collect();
        let coded = encode(&data).expect("non-empty");
        assert!(
            decode(&coded[..100], data.len()).is_err(),
            "table cut short"
        );
        assert!(
            decode(&coded[..coded.len() - 40], data.len()).is_err(),
            "stream cut short"
        );
        let mut overfull = coded.clone();
        overfull[..128].fill(0x11);
        assert!(
            decode(&overfull, data.len()).is_err(),
            "lengths overfill the code space"
        );
        assert!(
            decode(&coded, data.len() + 5000).is_err(),
            "asking for more than was coded"
        );
    }

    fn chunked(data: &[u8]) -> Vec<u8> {
        encode_chunked(data, &table(data)).expect("non-empty")
    }

    #[test]
    fn chunked_roundtrips_every_size_around_the_chunk_and_the_lane_group() {
        let chunk = fineseed::huffman::CHUNK;
        let skewed = |n| -> Vec<u8> { lcg(7, n).iter().map(|b| 0x3c + (b % 5)).collect() };
        for n in [1, 2, 3, 63, chunk - 1, chunk, chunk + 1, 3 * chunk + 17, 4 * chunk, 4 * chunk + 1, 9 * chunk + 5] {
            for data in [skewed(n), lcg(n as u64, n), vec![7; n]] {
                let coded = chunked(&data);
                assert_eq!(decode_chunked(&coded, n).expect("decodes"), data, "n = {n}");
            }
        }
    }

    #[test]
    fn chunked_roundtrips_codes_of_fifteen_bits() {
        let mut data = Vec::new();
        let (mut a, mut b) = (1usize, 1usize);
        for s in 0..30u8 {
            data.extend(std::iter::repeat_n(s, a));
            let c = a + b;
            a = b;
            b = c;
        }
        let coded = chunked(&data);
        assert_eq!(decode_chunked(&coded, data.len()).expect("decodes"), data);
    }

    #[test]
    fn chunked_refuses_damaged_input_and_never_panics() {
        let n = 5 * fineseed::huffman::CHUNK + 100;
        let data: Vec<u8> = lcg(3, n).iter().map(|b| b % 11).collect();
        let coded = chunked(&data);
        assert!(decode_chunked(&coded[..100], n).is_err(), "table cut short");
        assert!(decode_chunked(&coded[..140], n).is_err(), "chunk lengths cut short");
        assert!(decode_chunked(&coded[..coded.len() - 40], n).is_err(), "last chunk cut short");
        let mut more = coded.clone();
        more.push(0);
        assert!(decode_chunked(&more, n).is_err(), "bytes after the last chunk");
        let mut lie = coded.clone();
        lie[128..132].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_chunked(&lie, n).is_err(), "a chunk longer than the part");
        assert!(decode_chunked(&coded, n + 3 * fineseed::huffman::CHUNK).is_err(), "more than was coded");
        // flipped bytes may give wrong data (the archive's hash catches that) but never a panic
        for at in (128..coded.len()).step_by(997) {
            let mut bad = coded.clone();
            bad[at] ^= 0xa5;
            let _ = decode_chunked(&bad, n);
        }
    }
}

mod archive {
    use fineseed::archive::{collect, list, restore, store};
    use std::path::{Path, PathBuf};

    fn lcg(seed: u64, n: usize) -> Vec<u8> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 33) as u8
            })
            .collect()
    }

    /// A safetensors file of two bf16 tensors of `n` values each.
    fn model(dir: &Path, name: &str, a: &[u8], b: &[u8]) -> PathBuf {
        let header = format!(
            "{{\"a\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[0,{}]}},\"b\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{},{}]}}}}",
            a.len() / 2,
            a.len(),
            b.len() / 2,
            a.len(),
            a.len() + b.len()
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(a);
        bytes.extend(b);
        let p = dir.join(name);
        std::fs::write(&p, bytes).expect("write model");
        p
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fineseed-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("tmp dir");
        d
    }

    #[test]
    fn a_copy_and_a_fine_tune_cost_little_and_everything_comes_back() {
        let src = tmp("src");
        // bf16-like data: a skewed high byte, random low byte
        let weights = |seed| -> Vec<u8> {
            lcg(seed, 400_000)
                .chunks(2)
                .flat_map(|p| [p[0], 0x3c + (p[1] % 5)])
                .collect()
        };
        let (a, b) = (weights(1), weights(2));
        let base = model(&src, "base.safetensors", &a, &b);
        let copy = model(&src, "copy.safetensors", &a, &b);
        // the fine-tune keeps tensor a and nudges the low byte of every 50th value of b
        let mut b2 = b.clone();
        for i in (0..b2.len()).step_by(100) {
            b2[i] ^= 1;
        }
        let tuned = model(&src, "tuned.safetensors", &a, &b2);
        std::fs::write(src.join("notes.txt"), b"not a model").expect("write");

        let files = collect(&[base.clone(), copy, tuned, src.join("notes.txt")]).expect("collect");
        let arc = src.join("all.fseed");
        let rep = store(&files, &arc, 3).expect("store");
        assert_eq!(rep.members, 4);
        assert!(
            rep.dup_blocks >= 2,
            "the copy and the shared tensor are not stored again: {rep:?}"
        );
        let alone = std::fs::metadata(&base).expect("stat").len();
        assert!(
            rep.packed < alone * 14 / 10,
            "four files cost under 1.4 times the first: {} vs {alone}",
            rep.packed
        );

        let names: Vec<_> = list(&arc)
            .expect("list")
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert_eq!(
            names,
            [
                "base.safetensors",
                "copy.safetensors",
                "tuned.safetensors",
                "notes.txt"
            ]
        );

        let out = tmp("out");
        restore(&arc, &out).expect("restore");
        for n in names {
            assert_eq!(
                std::fs::read(out.join(&n)).expect("read restored"),
                std::fs::read(src.join(&n)).expect("read original"),
                "{n}"
            );
        }
    }

    #[test]
    fn a_damaged_or_cut_archive_is_refused() {
        let src = tmp("dam");
        let f = src.join("x.bin");
        std::fs::write(&f, lcg(9, 70_000)).expect("write");
        let arc = src.join("x.fseed");
        store(&collect(&[f]).expect("collect"), &arc, 3).expect("store");
        let mut bytes = std::fs::read(&arc).expect("read");
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xff;
        let bad = src.join("bad.fseed");
        std::fs::write(&bad, &bytes).expect("write");
        assert!(
            restore(&bad, &tmp("dam-out")).is_err(),
            "a flipped byte is caught"
        );
        let cut = src.join("cut.fseed");
        std::fs::write(&cut, &bytes[..bytes.len() - 20]).expect("write");
        assert!(list(&cut).is_err(), "a cut archive is refused");
    }

    #[test]
    fn a_member_name_that_climbs_out_of_the_folder_is_refused() {
        let src = tmp("evil");
        let f = src.join("x.bin");
        std::fs::write(&f, lcg(4, 1000)).expect("write");
        let arc = src.join("x.fseed");
        store(&collect(&[f]).expect("collect"), &arc, 3).expect("store");
        let mut bytes = std::fs::read(&arc).expect("read");
        let at = bytes
            .windows(5)
            .rposition(|w| w == b"x.bin")
            .expect("name in the table");
        bytes[at..at + 5].copy_from_slice(b"../ab");
        std::fs::write(&arc, &bytes).expect("write");
        let out = tmp("evil-out");
        let err = restore(&arc, &out.join("inside")).expect_err("refused");
        assert!(err.contains("refusing"), "{err}");
        assert!(!out.join("ab").exists());
    }
}
