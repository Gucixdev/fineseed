//! Bitrot protection: Reed-Solomon parity behind an archive, in the shape of par2 but inside the one file.
//!
//! The archive bytes are cut into shards of 64 KiB (the last one padded with zeros) and the shards into groups of
//! 128. Every shard gets a CRC32, and every group `max(1, ceil(shards * pct / 100))` recovery shards, so a group
//! survives that many shards damaged, wherever in the group they are and however many bits of each are wrong. Rot
//! comes in small spots, so a few percent of parity covers a lot. The archive proper is not touched: a protected file
//! is the archive, then the recovery shards, then a header (the CRCs and the layout, with a CRC of its own), then a
//! footer `header offset u64` + magic. `archive::read_table` and everything else reads the part before the parity,
//! and an archive written by an older version simply has no footer.
//!
//! Damage to the header (a few kilobytes at the very end) is detected but cannot be repaired: it is the one place
//! that is not protected.
use crate::{Res, io, pipe, pread};
use reed_solomon_simd::{ReedSolomonDecoder, ReedSolomonEncoder};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

const PMAGIC: &[u8; 8] = b"FSEEDP\x01\0";
pub const SHARD: usize = 64 << 10;
const GROUP: u64 = 128;

/// Where the shards are, from the archive length and the parity percentage alone.
#[derive(Clone, Copy, Debug)]
struct Layout {
    len: u64,
    pct: u32,
}

impl Layout {
    fn shards(&self) -> u64 {
        self.len.div_ceil(SHARD as u64)
    }
    fn groups(&self) -> u64 {
        self.shards().div_ceil(GROUP)
    }
    /// Data shards in group `g`.
    fn count(&self, g: u64) -> u64 {
        GROUP.min(self.shards() - g * GROUP)
    }
    /// Recovery shards of group `g`.
    fn recov(&self, g: u64) -> u64 {
        (self.count(g) * u64::from(self.pct)).div_ceil(100).max(1)
    }
    fn recov_total(&self) -> u64 {
        (0..self.groups()).map(|g| self.recov(g)).sum()
    }
    /// Recovery shards before group `g`.
    fn recov_before(&self, g: u64) -> u64 {
        (0..g).map(|h| self.recov(h)).sum()
    }
}

struct Header {
    lay: Layout,
    data_crc: Vec<u32>,
    rec_crc: Vec<u32>,
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at + 4)?.try_into().ok().map(u32::from_le_bytes)
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    b.get(at..at + 8)?.try_into().ok().map(u64::from_le_bytes)
}

/// The header of a protected file, `None` for a file without parity.
fn read_header(f: &File) -> Res<Option<Header>> {
    let total = io(f.metadata(), "stat")?.len();
    if total < 16 {
        return Ok(None);
    }
    let foot = pread(f, total - 16, 16)?;
    if &foot[8..] != PMAGIC {
        return Ok(None);
    }
    let at = u64_at(&foot, 0).ok_or("a damaged parity footer")?;
    if at >= total - 16 || total - 16 - at > 1 << 30 {
        return Err("a damaged parity footer".into());
    }
    let h = pread(f, at, total - 16 - at)?;
    let bad = || "the parity header is damaged (and it is the one part that cannot be repaired)";
    if h.len() < 28 || &h[..8] != PMAGIC {
        return Err(bad().into());
    }
    let (body, crc) = h.split_at(h.len() - 4);
    if crc32fast::hash(body) != u32_at(crc, 0).ok_or(bad())? {
        return Err(bad().into());
    }
    let lay = Layout {
        len: u64_at(&h, 8).ok_or(bad())?,
        pct: u32_at(&h, 16).ok_or(bad())?,
    };
    if u32_at(&h, 20).ok_or(bad())? as usize != SHARD || lay.len == 0 || lay.pct == 0 || lay.pct > 400 {
        return Err(bad().into());
    }
    let (n, r) = (lay.shards() as usize, lay.recov_total() as usize);
    if body.len() != 24 + 4 * (n + r) || lay.len + (r * SHARD) as u64 != at {
        return Err(bad().into());
    }
    let crcs: Vec<u32> = body[24..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Some(Header {
        lay,
        data_crc: crcs[..n].to_vec(),
        rec_crc: crcs[n..].to_vec(),
    }))
}

/// The length of the archive proper: the file without its parity, or all of it when there is none.
pub fn archive_len(f: &File) -> Res<u64> {
    Ok(match read_header(f)? {
        Some(h) => h.lay.len,
        None => io(f.metadata(), "stat")?.len(),
    })
}

/// Shard `i` of the archive bytes, padded with zeros to a whole shard.
fn data_shard(f: &File, lay: &Layout, i: u64) -> Res<Vec<u8>> {
    let off = i * SHARD as u64;
    let n = (lay.len - off).min(SHARD as u64);
    let mut v = pread(f, off, n)?;
    v.resize(SHARD, 0);
    Ok(v)
}

/// Add parity of `pct` percent to the archive in `input`, written to `output` (a copy with the parity behind it) or,
/// for `None`, appended in place. An archive that already has parity gets new parity in place of the old.
pub fn protect(input: &Path, output: Option<&Path>, pct: u32) -> Res<u64> {
    if pct == 0 || pct > 400 {
        return Err("the parity is 1 to 400 percent".into());
    }
    let src = io(OpenOptions::new().read(true).write(output.is_none()).open(input), "open")?;
    let len = archive_len(&src)?;
    if len == 0 {
        return Err("nothing to protect".into());
    }
    let lay = Layout { len, pct };
    if !ReedSolomonEncoder::supports(GROUP as usize, lay.recov(0) as usize) {
        return Err("that much parity is not supported".into());
    }
    let mut out = BufWriter::with_capacity(1 << 20, match output {
        Some(p) => {
            let mut dst = io(File::create(p), "create output")?;
            let mut from = io(File::open(input), "open")?.take(len);
            io(std::io::copy(&mut from, &mut dst), "copy")?;
            dst
        }
        None => {
            io(src.set_len(len), "cut off the old parity")?;
            let mut dst = io(src.try_clone(), "clone")?;
            io(dst.seek(SeekFrom::Start(len)), "seek")?;
            dst
        }
    });
    let threads = pipe::threads();
    let mut data_crc: Vec<u32> = Vec::with_capacity(lay.shards() as usize);
    let mut rec_crc: Vec<u32> = Vec::new();
    pipe::run(
        threads,
        2 * threads,
        (0..lay.groups()).map(Ok),
        |_, g: u64| -> Res<(Vec<u32>, Vec<Vec<u8>>)> {
            let count = lay.count(g) as usize;
            let mut enc = ReedSolomonEncoder::new(count, lay.recov(g) as usize, SHARD)
                .map_err(|e| format!("parity: {e}"))?;
            let mut crcs = Vec::with_capacity(count);
            for k in 0..count as u64 {
                let s = data_shard(&src, &lay, g * GROUP + k)?;
                crcs.push(crc32fast::hash(&s));
                enc.add_original_shard(&s).map_err(|e| format!("parity: {e}"))?;
            }
            let r = enc.encode().map_err(|e| format!("parity: {e}"))?;
            Ok((crcs, r.recovery_iter().map(<[u8]>::to_vec).collect()))
        },
        |_, (crcs, rec): (Vec<u32>, Vec<Vec<u8>>)| {
            data_crc.extend(crcs);
            for r in rec {
                rec_crc.push(crc32fast::hash(&r));
                io(out.write_all(&r), "write parity")?;
            }
            Ok(())
        },
    )?;
    let at = len + rec_crc.len() as u64 * SHARD as u64;
    let mut h = Vec::new();
    h.extend(PMAGIC);
    h.extend(len.to_le_bytes());
    h.extend(pct.to_le_bytes());
    h.extend((SHARD as u32).to_le_bytes());
    for c in data_crc.iter().chain(&rec_crc) {
        h.extend(c.to_le_bytes());
    }
    h.extend(crc32fast::hash(&h).to_le_bytes());
    h.extend(at.to_le_bytes());
    h.extend(PMAGIC);
    io(out.write_all(&h), "write parity header")?;
    io(out.flush(), "flush")?;
    Ok(rec_crc.len() as u64 * SHARD as u64 + h.len() as u64)
}

#[derive(Debug, Default, Clone)]
pub struct Health {
    pub shards: u64,
    pub recovery: u64,
    /// Data shards whose CRC is wrong (after `repair`: that still are).
    pub bad: Vec<u64>,
    /// Recovery shards whose CRC is wrong.
    pub bad_recovery: Vec<u64>,
    /// Groups with more damage than they have parity for.
    pub lost_groups: Vec<u64>,
    /// Shards `repair` wrote back (data and recovery).
    pub fixed: u64,
}

impl Health {
    pub fn clean(&self) -> bool {
        self.bad.is_empty() && self.bad_recovery.is_empty()
    }
}

/// Which shards of group `g` fail their CRC: `(data indexes, recovery indexes)`, both global.
fn damaged(f: &File, h: &Header, g: u64) -> Res<(Vec<u64>, Vec<u64>)> {
    let (mut bad, mut bad_rec) = (Vec::new(), Vec::new());
    for k in 0..h.lay.count(g) {
        let i = g * GROUP + k;
        if crc32fast::hash(&data_shard(f, &h.lay, i)?) != h.data_crc[i as usize] {
            bad.push(i);
        }
    }
    let first = h.lay.recov_before(g);
    for k in 0..h.lay.recov(g) {
        let r = first + k;
        let s = pread(f, h.lay.len + r * SHARD as u64, SHARD as u64)?;
        if crc32fast::hash(&s) != h.rec_crc[r as usize] {
            bad_rec.push(r);
        }
    }
    Ok((bad, bad_rec))
}

fn scan(f: &File, h: &Header, repair: bool) -> Res<Health> {
    let mut health = Health {
        shards: h.lay.shards(),
        recovery: h.lay.recov_total(),
        ..Health::default()
    };
    let threads = pipe::threads();
    pipe::run(
        threads,
        2 * threads,
        (0..h.lay.groups()).map(Ok),
        |_, g: u64| -> Res<(u64, Vec<u64>, Vec<u64>, bool, u64)> {
            let (mut bad, mut bad_rec) = damaged(f, h, g)?;
            let (mut lost, mut fixed) = (false, 0u64);
            if repair && (!bad.is_empty() || !bad_rec.is_empty()) {
                (lost, fixed) = mend(f, h, g, &mut bad, &mut bad_rec)?;
            }
            Ok((g, bad, bad_rec, lost, fixed))
        },
        |_, (g, bad, bad_rec, lost, fixed): (u64, Vec<u64>, Vec<u64>, bool, u64)| {
            health.bad.extend(bad);
            health.bad_recovery.extend(bad_rec);
            if lost {
                health.lost_groups.push(g);
            }
            health.fixed += fixed;
            Ok(())
        },
    )?;
    Ok(health)
}

/// Rebuild the damaged shards of group `g` from the good ones and write them back; what is still bad stays in
/// `bad`/`bad_rec`. `(group lost, shards written)`.
fn mend(f: &File, h: &Header, g: u64, bad: &mut Vec<u64>, bad_rec: &mut Vec<u64>) -> Res<(bool, u64)> {
    let (count, rcount) = (h.lay.count(g) as usize, h.lay.recov(g) as usize);
    let first = h.lay.recov_before(g);
    let mut fixed = 0u64;
    if !bad.is_empty() {
        if bad.len() > rcount - bad_rec.len().min(rcount) {
            return Ok((true, 0));
        }
        let mut dec = ReedSolomonDecoder::new(count, rcount, SHARD).map_err(|e| format!("parity: {e}"))?;
        for k in 0..count {
            let i = g * GROUP + k as u64;
            if !bad.contains(&i) {
                dec.add_original_shard(k, data_shard(f, &h.lay, i)?)
                    .map_err(|e| format!("parity: {e}"))?;
            }
        }
        for k in 0..rcount {
            let r = first + k as u64;
            if !bad_rec.contains(&r) {
                let s = pread(f, h.lay.len + r * SHARD as u64, SHARD as u64)?;
                dec.add_recovery_shard(k, s).map_err(|e| format!("parity: {e}"))?;
            }
        }
        let done = dec.decode().map_err(|e| format!("parity: {e}"))?;
        let mut still = Vec::new();
        for (k, shard) in done.restored_original_iter() {
            let i = g * GROUP + k as u64;
            if crc32fast::hash(shard) != h.data_crc[i as usize] {
                still.push(i);
                continue;
            }
            let off = i * SHARD as u64;
            let n = (h.lay.len - off).min(SHARD as u64) as usize;
            io(f.write_all_at(&shard[..n], off), "write a repaired shard")?;
            fixed += 1;
        }
        drop(done);
        if !still.is_empty() {
            *bad = still;
            return Ok((true, fixed));
        }
        bad.clear();
    }
    if !bad_rec.is_empty() {
        // the data is whole again: the recovery shards are made again from it
        let mut enc = ReedSolomonEncoder::new(count, rcount, SHARD).map_err(|e| format!("parity: {e}"))?;
        for k in 0..count as u64 {
            enc.add_original_shard(data_shard(f, &h.lay, g * GROUP + k)?)
                .map_err(|e| format!("parity: {e}"))?;
        }
        let r = enc.encode().map_err(|e| format!("parity: {e}"))?;
        let mut still = Vec::new();
        for (k, shard) in r.recovery_iter().enumerate() {
            let i = first + k as u64;
            if !bad_rec.contains(&i) {
                continue;
            }
            if crc32fast::hash(shard) != h.rec_crc[i as usize] {
                still.push(i);
                continue;
            }
            io(f.write_all_at(shard, h.lay.len + i * SHARD as u64), "write a repaired shard")?;
            fixed += 1;
        }
        *bad_rec = still;
    }
    Ok((false, fixed))
}

/// Check every shard against its CRC without changing anything. `Err` for a file without parity.
pub fn verify(path: &Path) -> Res<Health> {
    let f = io(File::open(path), "open")?;
    let h = read_header(&f)?.ok_or("this archive has no parity: protect it first")?;
    scan(&f, &h, false)
}

/// Verify, then rebuild every damaged shard the parity can and write it back in place. A rebuilt shard is written
/// only after it matches its CRC.
pub fn repair(path: &Path) -> Res<Health> {
    let f = io(OpenOptions::new().read(true).write(true).open(path), "open")?;
    let h = read_header(&f)?.ok_or("this archive has no parity: protect it first")?;
    let health = scan(&f, &h, true)?;
    io(f.sync_all(), "sync")?;
    Ok(health)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("fineseed-parity-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&d);
        d
    }

    #[test]
    fn repairs_rot_and_refuses_too_much() {
        let p = scratch("a");
        let data: Vec<u8> = (0..700_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        std::fs::write(&p, &data).unwrap();
        protect(&p, None, 20).unwrap();
        assert_eq!(archive_len(&File::open(&p).unwrap()).unwrap(), data.len() as u64);
        assert!(verify(&p).unwrap().clean());
        // rot in three spots of different shards
        let mut b = std::fs::read(&p).unwrap();
        for at in [10usize, 100_000, 650_000] {
            b[at] ^= 0x55;
        }
        std::fs::write(&p, &b).unwrap();
        let h = verify(&p).unwrap();
        assert_eq!(h.bad.len(), 3);
        let h = repair(&p).unwrap();
        assert!(h.lost_groups.is_empty() && h.fixed == 3, "{h:?}");
        assert_eq!(&std::fs::read(&p).unwrap()[..data.len()], &data[..]);
        assert!(verify(&p).unwrap().clean());
        // far more than the parity holds
        let mut b = std::fs::read(&p).unwrap();
        for k in 0..9 {
            b[k * SHARD + 7] ^= 1;
        }
        std::fs::write(&p, &b).unwrap();
        assert!(!repair(&p).unwrap().lost_groups.is_empty());
        let _ = std::fs::remove_file(&p);
    }
}
