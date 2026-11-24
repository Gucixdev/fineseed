//! Random access (`Archive`), the CLI's cat/extract/stdin, zip containers and parity: everything comes back byte for byte.
use fineseed::archive::{access::Archive, collect, restore, store};
use fineseed::parity;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn tmp(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("access-{name}"));
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

fn bf16(seed: u64, values: usize) -> Vec<u8> {
    noise(seed, values * 2)
        .chunks(2)
        .flat_map(|c| [c[0], 0x3c + (c[1] & 3)])
        .collect()
}

fn safetensors(data: &[u8]) -> Vec<u8> {
    let header = format!(
        r#"{{"w":{{"dtype":"BF16","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        data.len() / 2,
        data.len()
    );
    let mut b = (header.len() as u64).to_le_bytes().to_vec();
    b.extend(header.as_bytes());
    b.extend(data);
    b
}

fn pack(inputs: &[&Path], out: &Path) {
    let v: Vec<PathBuf> = inputs.iter().map(|p| p.to_path_buf()).collect();
    store(&collect(&v).unwrap(), out, 3).unwrap();
}

/// A zip with stored entries, made by hand.
fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let (mut z, mut cd) = (Vec::new(), Vec::new());
    for (name, data) in entries {
        let off = z.len() as u32;
        z.extend(b"PK\x03\x04");
        z.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend((data.len() as u32).to_le_bytes());
        z.extend((data.len() as u32).to_le_bytes());
        z.extend((name.len() as u16).to_le_bytes());
        z.extend([0, 0]);
        z.extend(name.as_bytes());
        z.extend(*data);
        cd.extend(b"PK\x01\x02");
        cd.extend([20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        cd.extend((data.len() as u32).to_le_bytes());
        cd.extend((data.len() as u32).to_le_bytes());
        cd.extend((name.len() as u16).to_le_bytes());
        cd.extend([0u8; 12]);
        cd.extend(off.to_le_bytes());
        cd.extend(name.as_bytes());
    }
    let cd_off = z.len() as u32;
    z.extend(&cd);
    z.extend(b"PK\x05\x06");
    z.extend([0, 0, 0, 0]);
    z.extend((entries.len() as u16).to_le_bytes());
    z.extend((entries.len() as u16).to_le_bytes());
    z.extend((cd.len() as u32).to_le_bytes());
    z.extend(cd_off.to_le_bytes());
    z.extend([0, 0]);
    z
}

#[test]
fn ranges_members_and_extract_match_the_originals() {
    let d = tmp("ranges");
    let base = safetensors(&bf16(1, 300_000));
    let mut tuned = base.clone();
    for i in (1000..tuned.len()).step_by(997) {
        tuned[i] ^= 1;
    }
    let other = noise(5, 200_000);
    let src = d.join("in");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("base.safetensors"), &base).unwrap();
    fs::write(src.join("tuned.safetensors"), &tuned).unwrap();
    fs::write(src.join("notes.bin"), &other).unwrap();
    let ar_path = d.join("a.fseed");
    pack(&[&src], &ar_path);
    let ar = Archive::open(&ar_path, 8 << 20).unwrap();
    for (name, want) in [("base.safetensors", &base), ("tuned.safetensors", &tuned), ("notes.bin", &other)] {
        let i = (0..ar.members().len())
            .find(|&i| ar.name(i).ends_with(name))
            .unwrap_or_else(|| panic!("no {name}"));
        // a whole member
        let mut got = Vec::new();
        ar.write_member(i, &mut got).unwrap();
        assert_eq!(&got, want, "{name}");
        ar.verify_member(i).unwrap();
        // ranges: start, middle, across block edges, the very end, past the end
        for (off, len) in [(0usize, 10usize), (12_345, 70_000), (65_530, 20), (want.len() - 7, 7)] {
            let mut buf = vec![0u8; len];
            let n = ar.read_range(i, off as u64, &mut buf).unwrap();
            assert_eq!(n, len);
            assert_eq!(buf, want[off..off + len], "{name} at {off}");
        }
        let mut buf = [0u8; 8];
        assert_eq!(ar.read_range(i, want.len() as u64 + 5, &mut buf).unwrap(), 0);
    }
    // one entry out, the rest untouched
    let out = d.join("out");
    let i = (0..ar.members().len()).find(|&i| ar.name(i).ends_with("tuned.safetensors")).unwrap();
    assert_eq!(ar.extract(&[ar.name(i).to_string()], &out).unwrap(), 1);
    assert_eq!(fs::read(out.join(ar.name(i))).unwrap(), tuned);
    assert!(!out.join("notes.bin").exists());
}

#[test]
fn a_zip_container_comes_back_identical() {
    let d = tmp("zip");
    let st = safetensors(&bf16(2, 100_000));
    let storage = bf16(3, 80_000);
    let z = zip(&[("m/data.pkl", b"\x80\x02pickle"), ("m/data/0", &storage), ("unet/model.safetensors", &st)]);
    let p = d.join("model.pt");
    fs::write(&p, &z).unwrap();
    let ar = d.join("z.fseed");
    pack(&[&p], &ar);
    let out = d.join("out");
    restore(&ar, &out).unwrap();
    assert_eq!(fs::read(out.join("model.pt")).unwrap(), z);
    // not a zip: a damaged one is plain bytes, still identical
    let mut bad = z.clone();
    let n = bad.len();
    bad[n - 8] ^= 0xff;
    fs::write(&p, &bad).unwrap();
    pack(&[&p], &ar);
    let out2 = d.join("out2");
    restore(&ar, &out2).unwrap();
    assert_eq!(fs::read(out2.join("model.pt")).unwrap(), bad);
}

#[test]
fn an_archive_with_parity_is_read_and_can_be_nested() {
    let d = tmp("parity");
    let p = d.join("m.safetensors");
    fs::write(&p, safetensors(&bf16(4, 200_000))).unwrap();
    let inner = d.join("inner.fseed");
    pack(&[&p], &inner);
    let plain_len = fs::metadata(&inner).unwrap().len();
    parity::protect(&inner, None, 10).unwrap();
    assert!(fs::metadata(&inner).unwrap().len() > plain_len);
    // readers ignore the parity
    let out = d.join("out");
    restore(&inner, &out).unwrap();
    assert_eq!(fs::read(out.join("m.safetensors")).unwrap(), fs::read(&p).unwrap());
    // rot it, repair it, the file is the protected file again
    let good = fs::read(&inner).unwrap();
    let mut b = good.clone();
    b[plain_len as usize / 2] ^= 0x42;
    fs::write(&inner, &b).unwrap();
    assert!(!parity::verify(&inner).unwrap().clean());
    parity::repair(&inner).unwrap();
    assert_eq!(fs::read(&inner).unwrap(), good);
    // inside another archive, with its parity, byte for byte
    let other = d.join("other.bin");
    fs::write(&other, noise(9, 50_000)).unwrap();
    let outer = d.join("outer.fseed");
    pack(&[&inner, &other], &outer);
    let out2 = d.join("out2");
    restore(&outer, &out2).unwrap();
    assert_eq!(fs::read(out2.join("inner.fseed")).unwrap(), good);
}

#[test]
fn the_command_line_streams_through_stdin_cat_and_extract() {
    let d = tmp("cli");
    let exe = env!("CARGO_BIN_EXE_fineseed");
    let data = bf16(7, 300_000);
    let ar = d.join("s.fseed");
    let mut child = Command::new(exe)
        .args(["pack", "-", "--dtype", "bf16", "--name", "kv.bin", "-o"])
        .arg(&ar)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&data).unwrap();
    assert!(child.wait().unwrap().success());
    assert!(fs::metadata(&ar).unwrap().len() < data.len() as u64);
    let cat = Command::new(exe).args(["cat"]).arg(&ar).arg("kv.bin").output().unwrap();
    assert!(cat.status.success());
    assert_eq!(cat.stdout, data);
    let out = d.join("out");
    let ex = Command::new(exe).args(["extract"]).arg(&ar).arg("kv.bin").arg("-o").arg(&out).output().unwrap();
    assert!(ex.status.success());
    assert_eq!(fs::read(out.join("kv.bin")).unwrap(), data);
    // and an archive to standard output
    let piped = Command::new(exe)
        .args(["pack", "-", "--name", "x", "-o", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut piped = piped;
    let mut si = piped.stdin.take().unwrap();
    let data2 = data.clone();
    let t = std::thread::spawn(move || si.write_all(&data2).unwrap());
    let o = piped.wait_with_output().unwrap();
    t.join().unwrap();
    assert!(o.status.success());
    let ar2 = d.join("t.fseed");
    fs::write(&ar2, &o.stdout).unwrap();
    let cat2 = Command::new(exe).args(["cat"]).arg(&ar2).arg("x").output().unwrap();
    assert_eq!(cat2.stdout, data);
}
