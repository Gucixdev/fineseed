//! The stored (uncompressed) entries of a zip file, which is what a PyTorch checkpoint (`.pt`, `.pth`: `data.pkl`
//! plus one raw file per storage) and a DDUF file (safetensors and json, stored) are. Only the central directory is
//! read; every length and offset is checked against the file, and anything odd (a compressed entry, an entry that
//! lies outside the file or over another) is left out, so the caller packs those bytes as plain.
use std::io::{Read, Seek, SeekFrom};

/// More entries than this and the file is not read as a zip.
const MAX_ENTRIES: u64 = 1 << 20;

fn le<const N: usize>(b: &[u8], at: usize) -> Option<[u8; N]> {
    b.get(at..at + N)?.try_into().ok()
}

fn u16_at(b: &[u8], at: usize) -> Option<u64> {
    le::<2>(b, at).map(|x| u64::from(u16::from_le_bytes(x)))
}

fn u32_at(b: &[u8], at: usize) -> Option<u64> {
    le::<4>(b, at).map(|x| u64::from(u32::from_le_bytes(x)))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    le::<8>(b, at).map(u64::from_le_bytes)
}

fn read_at<R: Read + Seek>(f: &mut R, at: u64, n: u64) -> Option<Vec<u8>> {
    let mut v = vec![0u8; usize::try_from(n).ok()?];
    f.seek(SeekFrom::Start(at)).ok()?;
    f.read_exact(&mut v).ok()?;
    Some(v)
}

/// `(name, first byte of the data, length)` of every stored entry, sorted by position, none overlapping.
pub fn stored<R: Read + Seek>(f: &mut R, len: u64) -> Option<Vec<(String, u64, u64)>> {
    if len < 22 {
        return None;
    }
    if read_at(f, 0, 4)? != b"PK\x03\x04" {
        return None;
    }
    // the end record is in the last 64 KiB (a comment may follow it)
    let tail_len = len.min(22 + 65535);
    let tail = read_at(f, len - tail_len, tail_len)?;
    let eocd = (0..=tail.len().checked_sub(22)?)
        .rev()
        .find(|&i| &tail[i..i + 4] == b"PK\x05\x06")?;
    let (mut count, mut cd_size, mut cd_off) = (
        u16_at(&tail, eocd + 10)?,
        u32_at(&tail, eocd + 12)?,
        u32_at(&tail, eocd + 16)?,
    );
    if count == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_off == 0xFFFF_FFFF {
        // zip64: the locator sits right before the end record and points at the zip64 end record
        let loc_at = (len - tail_len) + eocd as u64;
        let loc = read_at(f, loc_at.checked_sub(20)?, 20)?;
        if &loc[..4] != b"PK\x06\x07" {
            return None;
        }
        let rec_at = u64_at(&loc, 8)?;
        let rec = read_at(f, rec_at, 56)?;
        if &rec[..4] != b"PK\x06\x06" {
            return None;
        }
        count = u64_at(&rec, 32)?;
        cd_size = u64_at(&rec, 40)?;
        cd_off = u64_at(&rec, 48)?;
    }
    if count > MAX_ENTRIES || cd_off.checked_add(cd_size)? > len || cd_size > 1 << 30 {
        return None;
    }
    let cd = read_at(f, cd_off, cd_size)?;
    let mut out: Vec<(String, u64, u64)> = Vec::new();
    let mut at = 0usize;
    for _ in 0..count {
        if cd.get(at..at + 4)? != b"PK\x01\x02" {
            return None;
        }
        let method = u16_at(&cd, at + 10)?;
        let (mut comp, mut size, mut local) = (
            u32_at(&cd, at + 20)?,
            u32_at(&cd, at + 24)?,
            u32_at(&cd, at + 42)?,
        );
        let (nl, el, cl) = (
            u16_at(&cd, at + 28)? as usize,
            u16_at(&cd, at + 30)? as usize,
            u16_at(&cd, at + 32)? as usize,
        );
        let name = String::from_utf8_lossy(cd.get(at + 46..at + 46 + nl)?).into_owned();
        let extra = cd.get(at + 46 + nl..at + 46 + nl + el)?;
        // zip64 extra field: the 64-bit values of the fields that were 0xFFFFFFFF, in this order
        let mut e = 0usize;
        while e + 4 <= extra.len() {
            let (id, sz) = (u16_at(extra, e)?, u16_at(extra, e + 2)? as usize);
            if id == 1 {
                let mut p = e + 4;
                for v in [&mut size, &mut comp, &mut local] {
                    if *v == 0xFFFF_FFFF {
                        *v = u64_at(extra, p)?;
                        p += 8;
                    }
                }
            }
            e += 4 + sz;
        }
        at += 46 + nl + el + cl;
        if method != 0 || comp != size || name.ends_with('/') {
            continue;
        }
        // the data starts after the local header, whose name and extra lengths may differ from the central ones
        let lh = read_at(f, local, 30)?;
        if &lh[..4] != b"PK\x03\x04" {
            return None;
        }
        let start = local + 30 + u16_at(&lh, 26)? + u16_at(&lh, 28)?;
        if start.checked_add(size)? > len {
            return None;
        }
        out.push((name, start, size));
    }
    out.sort_by_key(|e| e.1);
    out.dedup_by(|b, a| b.1 < a.1 + a.2);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zip with two stored entries, made by hand so the test needs no zip tool.
    fn tiny() -> Vec<u8> {
        let mut z = Vec::new();
        let mut cd = Vec::new();
        for (name, data) in [("a/data.pkl", &b"hello"[..]), ("a/data/0", &[7u8; 100][..])] {
            let off = z.len() as u32;
            z.extend(b"PK\x03\x04");
            z.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            z.extend((data.len() as u32).to_le_bytes());
            z.extend((data.len() as u32).to_le_bytes());
            z.extend((name.len() as u16).to_le_bytes());
            z.extend([0, 0]);
            z.extend(name.as_bytes());
            z.extend(data);
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
        z.extend([0, 0, 0, 0, 2, 0, 2, 0]);
        z.extend((cd.len() as u32).to_le_bytes());
        z.extend(cd_off.to_le_bytes());
        z.extend([0, 0]);
        z
    }

    #[test]
    fn reads_stored_entries() {
        let z = tiny();
        let e = stored(&mut std::io::Cursor::new(&z), z.len() as u64).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].0, "a/data.pkl");
        assert_eq!(&z[e[0].1 as usize..(e[0].1 + e[0].2) as usize], b"hello");
        assert_eq!(e[1].2, 100);
    }

    #[test]
    fn rejects_garbage() {
        let mut z = tiny();
        let n = z.len();
        z[n - 6] = 0xff;
        z[n - 5] = 0xff;
        assert!(stored(&mut std::io::Cursor::new(&z), z.len() as u64).is_none());
        assert!(stored(&mut std::io::Cursor::new(&[1u8; 40][..]), 40).is_none());
    }
}
