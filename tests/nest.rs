//! `.fseed` in `.fseed`, and files of formats that are not read: everything must come back byte for byte.
use fineseed::archive::{collect, list, restore, store};
use std::fs;
use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("nest-{name}"));
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

/// bf16-looking values: a narrow exponent range, random mantissa.
fn bf16(seed: u64, values: usize) -> Vec<u8> {
    noise(seed, values * 2)
        .chunks(2)
        .flat_map(|c| [c[0], 0x3c + (c[1] & 3)])
        .collect()
}

fn safetensors(dir: &Path, name: &str, data: &[u8]) -> PathBuf {
    let header = format!(
        r#"{{"w":{{"dtype":"BF16","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        data.len() / 2,
        data.len()
    );
    let mut b = (header.len() as u64).to_le_bytes().to_vec();
    b.extend(header.as_bytes());
    b.extend(data);
    let p = dir.join(name);
    fs::write(&p, b).unwrap();
    p
}

fn pack(inputs: &[&Path], out: &Path, level: i32) -> fineseed::archive::Report {
    let v: Vec<PathBuf> = inputs.iter().map(|p| p.to_path_buf()).collect();
    store(&collect(&v).unwrap(), out, level).unwrap()
}

fn same(a: &Path, b: &Path) {
    assert_eq!(fs::read(a).unwrap(), fs::read(b).unwrap(), "{}", a.display());
}

#[test]
fn an_archive_inside_an_archive_comes_back_identical_and_shares_blocks() {
    let d = tmp("one");
    let base = bf16(1, 400_000);
    let mut tuned = base.clone();
    for i in (0..tuned.len()).step_by(997) {
        tuned[i] ^= 1;
    }
    let a = safetensors(&d, "a.safetensors", &base);
    let b = safetensors(&d, "b.safetensors", &tuned);
    let inner = d.join("pair.fseed");
    pack(&[&a, &b], &inner, 3);
    // the same two models once more, as a file next to the inner archive: a copy that costs nothing
    let c = safetensors(&d, "c.safetensors", &base);
    let outer = d.join("all.fseed");
    let rep = pack(&[&inner, &c], &outer, 3);
    assert_eq!(rep.nested, 1);
    let inner_len = fs::metadata(&inner).unwrap().len();
    let outer_len = fs::metadata(&outer).unwrap().len();
    assert!(
        outer_len < inner_len + inner_len / 20,
        "{outer_len} vs inner {inner_len}"
    );
    let entries = list(&outer).unwrap();
    assert!(entries.iter().any(|m| m.nest.is_some() && m.depth == 0));
    assert!(entries.iter().any(|m| m.depth == 1));
    let out = d.join("out");
    restore(&outer, &out).unwrap();
    same(&inner, &out.join("pair.fseed"));
    same(&c, &out.join("c.safetensors"));
    assert!(!out.join(".fseed-stage").exists(), "the stage folder is removed");
    // and the inner archive that came back is a real one
    let again = d.join("again");
    restore(&out.join("pair.fseed"), &again).unwrap();
    same(&a, &again.join("a.safetensors"));
    same(&b, &again.join("b.safetensors"));
}

#[test]
fn three_levels_deep() {
    let d = tmp("deep");
    let m = safetensors(&d, "m.safetensors", &bf16(2, 100_000));
    let l1 = d.join("l1.fseed");
    pack(&[&m], &l1, 3);
    let l2 = d.join("l2.fseed");
    pack(&[&l1], &l2, 5);
    let l3 = d.join("l3.fseed");
    let rep = pack(&[&l2], &l3, 1);
    assert_eq!(rep.nested, 2);
    let depths: Vec<u32> = list(&l3).unwrap().iter().map(|m| m.depth).collect();
    assert_eq!(depths, [0, 1, 2]);
    let out = d.join("out");
    restore(&l3, &out).unwrap();
    same(&l2, &out.join("l2.fseed"));
    let again = d.join("again");
    restore(&out.join("l2.fseed"), &again).unwrap();
    same(&l1, &again.join("l1.fseed"));
}

#[test]
fn an_archive_that_would_not_be_built_again_the_same_is_kept_as_bytes() {
    let d = tmp("odd");
    let m = safetensors(&d, "m.safetensors", &bf16(3, 50_000));
    let inner = d.join("i.fseed");
    pack(&[&m], &inner, 3);
    // another version's archive: same magic, a flipped byte in the last block's body is still a valid-looking file
    // only if the table survives, so damage the one thing that does not change the contents: the stored level
    let mut bytes = fs::read(&inner).unwrap();
    let at = bytes.len() - 16;
    let table_at = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
    bytes[table_at + 8] = 22; // the archive says level 22, the blocks were made at 3
    let odd = d.join("odd.fseed");
    fs::write(&odd, &bytes).unwrap();
    let outer = d.join("o.fseed");
    let rep = pack(&[&odd], &outer, 3);
    assert_eq!(rep.nested, 0, "kept as a plain file");
    let out = d.join("out");
    restore(&outer, &out).unwrap();
    same(&odd, &out.join("odd.fseed"));
}

#[test]
fn a_damaged_nested_archive_or_table_is_an_error_not_a_panic() {
    let d = tmp("bad");
    let m = safetensors(&d, "m.safetensors", &bf16(4, 60_000));
    let inner = d.join("i.fseed");
    pack(&[&m], &inner, 3);
    let outer = d.join("o.fseed");
    pack(&[&inner], &outer, 3);
    let good = fs::read(&outer).unwrap();
    let mut x = 7u64;
    for _ in 0..150 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut bad = good.clone();
        let at = (x >> 33) as usize % bad.len();
        bad[at] ^= 1 << ((x >> 20) & 7);
        let f = d.join("bad.fseed");
        fs::write(&f, &bad).unwrap();
        let out = d.join("badout");
        // either an error, or the very same bytes back (the flip hit nothing that is read)
        if restore(&f, &out).is_ok() {
            same(&inner, &out.join("i.fseed"));
        }
        let _ = fs::remove_dir_all(&out);
    }
    for cut in [0, 7, 30, good.len() / 2, good.len() - 1] {
        fs::write(d.join("cut.fseed"), &good[..cut]).unwrap();
        assert!(restore(&d.join("cut.fseed"), &d.join("cutout")).is_err());
    }
}

#[test]
fn a_file_of_a_format_that_is_not_read_still_gets_its_floats_split() {
    let d = tmp("raw");
    // a TensorFlow-data-file-like blob: a few bytes of header, then bf16 values, then a tail; no layout is read
    let mut blob = noise(9, 13);
    blob.extend(bf16(5, 600_000));
    blob.extend(noise(10, 7));
    let f = d.join("model.data-00000-of-00001");
    fs::write(&f, &blob).unwrap();
    let text = d.join("notes.txt");
    let words = "the quick brown fox jumps over the lazy dog ".repeat(30_000);
    fs::write(&text, words.as_bytes()).unwrap();
    let out = d.join("x.fseed");
    pack(&[&f, &text], &out, 3);
    let ratio = fs::metadata(&out).unwrap().len() as f64
        / (blob.len() + words.len()) as f64;
    // bf16 alone is about 66%, the text is nearly nothing
    assert!(ratio < 0.40, "ratio {ratio}");
    let r = d.join("r");
    restore(&out, &r).unwrap();
    same(&f, &r.join("model.data-00000-of-00001"));
    same(&text, &r.join("notes.txt"));
    // the text is not made worse by being looked at as floats: smaller than zstd of it alone would be, give or take
    let only_text = d.join("t.fseed");
    pack(&[&text], &only_text, 3);
    assert!(fs::metadata(&only_text).unwrap().len() < 20_000);
}
