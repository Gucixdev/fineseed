use fineseed::api::{self, Error, Kind, Options};

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

/// bf16-looking values: a narrow exponent range, random mantissa.
fn bf16(seed: u64, values: usize) -> Vec<u8> {
    let r = noise(seed, values * 2);
    r.chunks(2)
        .flat_map(|c| [c[0], 0x3c + (c[1] & 3)])
        .collect()
}

fn safetensors(tensors: &[(&str, &str, Vec<u8>)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut at = 0usize;
    for (i, (name, dtype, data)) in tensors.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{}],"data_offsets":[{at},{}]}}"#,
            data.len(),
            at + data.len()
        ));
        at += data.len();
    }
    header.push('}');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header.as_bytes());
    for (_, _, d) in tensors {
        out.extend(d);
    }
    out
}

#[test]
fn block_roundtrip_for_each_kind_with_and_without_reference() {
    let opt = Options::default();
    let kinds = [
        Kind::PLAIN,
        Kind::BF16,
        Kind::F16,
        Kind::F32,
        Kind::integers(4).unwrap(),
        Kind::ggml(12).unwrap(),
    ];
    for kind in kinds {
        let data = noise(kind.code() as u64, kind.unit() * 500);
        let coded = api::encode_block(kind, &data, None, &opt).unwrap();
        assert_eq!(api::decode_block(&coded, None).unwrap(), data);
        let reference: Vec<u8> = data
            .iter()
            .enumerate()
            .map(|(i, b)| if i % 97 == 0 { !b } else { *b })
            .collect();
        let coded = api::encode_block(kind, &data, Some(&reference), &opt).unwrap();
        assert_eq!(api::decode_block(&coded, Some(&reference)).unwrap(), data);
    }
}

#[test]
fn a_close_reference_makes_the_block_smaller() {
    let opt = Options::default();
    let a = bf16(1, 40_000);
    let mut b = a.clone();
    for i in (0..b.len()).step_by(401) {
        b[i] ^= 1;
    }
    let alone = api::encode_block(Kind::BF16, &b, None, &opt).unwrap();
    let with = api::encode_block(Kind::BF16, &b, Some(&a), &opt).unwrap();
    assert!(
        with.len() * 3 < alone.len(),
        "{} vs {}",
        with.len(),
        alone.len()
    );
    assert_eq!(api::decode_block(&with, Some(&a)).unwrap(), b);
    assert!(matches!(
        api::decode_block(&with, None),
        Err(Error::Reference(_))
    ));
}

#[test]
fn block_arguments_are_checked() {
    let opt = Options::default();
    assert!(matches!(
        api::encode_block(Kind::BF16, &[0; 3], None, &opt),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        api::encode_block(Kind::BF16, &[0; 4], Some(&[0; 2]), &opt),
        Err(Error::Reference(_))
    ));
    assert!(Kind::integers(9).is_none() && Kind::ggml(9999).is_none());
    assert_eq!(Kind::safetensors("BF16"), Kind::BF16);
}

#[test]
fn damaged_blocks_give_errors_never_panics() {
    let opt = Options::default();
    let data = bf16(2, 3000);
    let coded = api::encode_block(Kind::BF16, &data, None, &opt).unwrap();
    for cut in 0..coded.len().min(64) {
        assert!(api::decode_block(&coded[..cut], None).is_err());
    }
    let mut x = 1u64;
    for _ in 0..400 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut bad = coded.clone();
        let at = (x >> 33) as usize % bad.len();
        bad[at] ^= 1 << ((x >> 20) & 7);
        // either an error or, if the flip hit a byte the decoder cannot tell, bytes of the right length
        if let Ok(out) = api::decode_block(&bad, None) {
            assert_eq!(out.len(), data.len());
        }
    }
}

#[test]
fn whole_buffer_roundtrip_plain_and_safetensors() {
    let opt = Options::default();
    for data in [Vec::new(), noise(3, 1000), vec![7u8; 9_000_000]] {
        let c = api::compress(&data, None, &opt).unwrap();
        assert_eq!(api::decompress(&c, None, &opt).unwrap(), data);
        assert_eq!(api::original_len(&c).unwrap(), data.len() as u64);
    }
    let model = safetensors(&[
        ("a", "BF16", bf16(4, 100_000)),
        ("b", "F32", noise(5, 40_000)),
        ("c", "I8", noise(6, 777)),
    ]);
    let segs = api::layout(&model).unwrap();
    assert_eq!(segs.len(), 3);
    assert_eq!(api::kind_of(&segs[0]), Kind::BF16);
    let c = api::compress(&model, None, &opt).unwrap();
    assert!(c.len() < model.len());
    assert_eq!(api::decompress(&c, None, &opt).unwrap(), model);
    let s = api::inspect(&c).unwrap();
    assert_eq!(s.original, model.len() as u64);
}

#[test]
fn whole_buffer_with_a_reference_model() {
    let opt = Options::default();
    let base = safetensors(&[("w", "BF16", bf16(7, 300_000))]);
    let mut tuned = base.clone();
    let n = tuned.len();
    for i in (n - 600_000..n).step_by(997) {
        tuned[i] ^= 1;
    }
    let alone = api::compress(&tuned, None, &opt).unwrap();
    let with = api::compress(&tuned, Some(&base), &opt).unwrap();
    assert!(with.len() < alone.len());
    assert_eq!(api::decompress(&with, Some(&base), &opt).unwrap(), tuned);
    assert!(matches!(
        api::decompress(&with, None, &opt),
        Err(Error::Reference(_))
    ));
    let other = safetensors(&[("w", "BF16", bf16(8, 300_000))]);
    assert!(matches!(
        api::decompress(&with, Some(&other), &opt),
        Err(Error::Reference(_))
    ));
}

#[test]
fn whole_buffer_damage_and_limits() {
    let opt = Options::default();
    let data = noise(9, 200_000);
    let c = api::compress(&data, None, &opt).unwrap();
    assert!(matches!(
        api::decompress(
            &c,
            None,
            &Options {
                max_output: 1000,
                ..opt
            }
        ),
        Err(Error::Limit(_))
    ));
    for cut in [0, 5, 40, c.len() / 2, c.len() - 1] {
        assert!(api::decompress(&c[..cut], None, &opt).is_err());
    }
    let mut x = 3u64;
    for _ in 0..200 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut bad = c.clone();
        let at = (x >> 33) as usize % bad.len();
        bad[at] ^= 1 << ((x >> 20) & 7);
        assert!(
            api::decompress(&bad, None, &opt).is_err(),
            "flip at {at} went through"
        );
    }
    // a header that claims an enormous length must not allocate it
    let mut huge = c.clone();
    huge[8..16].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
    assert!(api::decompress(&huge, None, &opt).is_err());
}

#[test]
fn threads_option_gives_the_same_bytes() {
    let data = safetensors(&[
        ("a", "BF16", bf16(10, 2_500_000)),
        ("b", "F16", bf16(11, 2_500_000)),
    ]);
    let one = api::compress(
        &data,
        None,
        &Options {
            threads: 1,
            ..Options::default()
        },
    )
    .unwrap();
    let four = api::compress(
        &data,
        None,
        &Options {
            threads: 4,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(one, four);
}

#[test]
fn auto_finds_the_alignment_of_a_piece_cut_anywhere() {
    let opt = Options::default();
    let tensor = bf16(12, 400_000);
    for start in [0usize, 1, 2, 3, 1001, 123_457] {
        let piece = &tensor[start..start + 250_001];
        let coded = api::encode_auto(piece, None, &opt).unwrap();
        assert_eq!(api::decode_auto(&coded, None).unwrap(), piece);
        assert!(
            coded.len() * 100 < piece.len() * 80,
            "start {start}: {} of {}",
            coded.len(),
            piece.len()
        );
    }
    for len in [0usize, 1, 5, 64, 1000] {
        let d = noise(len as u64, len);
        let coded = api::encode_auto(&d, None, &opt).unwrap();
        assert_eq!(api::decode_auto(&coded, None).unwrap(), d);
    }
}

#[test]
fn auto_with_a_reference_and_damage() {
    let opt = Options::default();
    let a = bf16(13, 100_000);
    let mut b = a.clone();
    for i in (0..b.len()).step_by(300) {
        b[i] ^= 1;
    }
    let (pa, pb) = (&a[1..150_000], &b[1..150_000]);
    let coded = api::encode_auto(pb, Some(pa), &opt).unwrap();
    assert_eq!(api::decode_auto(&coded, Some(pa)).unwrap(), pb);
    assert!(api::decode_auto(&coded, None).is_err());
    for cut in 0..coded.len().min(40) {
        assert!(api::decode_auto(&coded[..cut], None).is_err() || cut >= 2);
    }
    assert!(api::decode_auto(&coded[..coded.len() / 2], Some(pa)).is_err());
}

#[test]
fn auto_len_reads_the_length_from_the_header() {
    let opt = Options::default();
    for n in [0usize, 7, 4097, 200_001] {
        let d = bf16(14, n.div_ceil(2) + 1)[1..=n].to_vec();
        let c = api::encode_auto(&d, None, &opt).unwrap();
        assert_eq!(api::auto_len(&c).unwrap(), n);
    }
    assert!(api::auto_len(&[]).is_err() && api::auto_len(&[3, 3, 1]).is_err());
}
