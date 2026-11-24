//! Reading one member, or a range of one, without restoring the others: the blocks of a member are indexed from their
//! headers (a few bytes each, skipping over the coded parts), and a block that copies or XORs bytes of an earlier
//! member reads exactly the blocks of that member it needs. This is what `cat`, `extract` and `mount` stand on.
//!
//! Decoded blocks are kept in a size-bounded cache; two threads that want the same block decode it once.
use super::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

/// Where one block is and how it is coded, from its header alone.
struct BlockRef {
    off: u64,
    len: u32,
    kind: u8,
    xor: Option<(u32, u64)>,
    dup: Option<(u32, u64)>,
    /// `(method, position in the archive, coded length)` per stream.
    parts: Vec<(u8, u64, u32)>,
}

#[derive(Default)]
struct Slot {
    block: OnceLock<Result<Arc<Vec<u8>>, String>>,
    counted: std::sync::atomic::AtomicUsize,
}

struct Cache {
    map: HashMap<(u32, u32), Arc<Slot>>,
    order: VecDeque<(u32, u32)>,
    bytes: usize,
    cap: usize,
}

pub struct Archive {
    file: File,
    path: PathBuf,
    members: Vec<Member>,
    names: Vec<String>,
    index: Vec<OnceLock<Result<Vec<BlockRef>, String>>>,
    cache: Mutex<Cache>,
}

/// The names a member goes by: the entries inside a nested archive are put under the nested archive's name.
fn full_names(members: &[Member]) -> Vec<String> {
    let mut out = Vec::with_capacity(members.len());
    let mut stack: Vec<(u32, String)> = Vec::new();
    for m in members {
        while stack.last().is_some_and(|(d, _)| *d >= m.depth) {
            stack.pop();
        }
        let name = match stack.last() {
            Some((_, p)) => format!("{p}/{}", m.name),
            None => m.name.clone(),
        };
        if m.nest.is_some() {
            stack.push((m.depth, name.clone()));
        }
        out.push(name);
    }
    out
}

impl Archive {
    /// Open an archive and read its table. `cache_bytes` bounds the decoded blocks kept (at least one block).
    pub fn open(path: &Path, cache_bytes: usize) -> Res<Archive> {
        let mut file = io(File::open(path), "open")?;
        let Table { members, .. } = read_table(&mut file)?;
        for m in &members {
            if !safe_name(&m.name) {
                return Err(format!("refusing the member name {:?}", m.name));
            }
            if m.blocks as u64 > m.len {
                return Err("a damaged table".into());
            }
        }
        Ok(Archive {
            file,
            path: path.to_path_buf(),
            names: full_names(&members),
            index: members.iter().map(|_| OnceLock::new()).collect(),
            members,
            cache: Mutex::new(Cache {
                map: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
                cap: cache_bytes.max(BLOCK as usize),
            }),
        })
    }

    pub fn members(&self) -> &[Member] {
        &self.members
    }

    /// The path of entry `i`, nested entries under the name of the archive they are in.
    pub fn name(&self, i: usize) -> &str {
        &self.names[i]
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    fn blocks(&self, i: usize) -> Res<&[BlockRef]> {
        let cell = self.index.get(i).ok_or("no such member")?;
        cell.get_or_init(|| self.scan(i))
            .as_ref()
            .map(|v| v.as_slice())
            .map_err(|e| e.clone())
    }

    fn scan(&self, i: usize) -> Res<Vec<BlockRef>> {
        let m = &self.members[i];
        let mut r = BufReader::with_capacity(8192, io(File::open(&self.path), "open")?);
        io(r.seek(SeekFrom::Start(m.body)), "seek")?;
        let mut out = Vec::with_capacity(m.blocks as usize);
        let (mut pos, mut next) = (m.body, 0u64);
        for _ in 0..m.blocks {
            let [kind, flags] = read_u::<2, _>(&mut r)?;
            let off = u64::from_le_bytes(read_u(&mut r)?);
            let len = u32::from_le_bytes(read_u(&mut r)?);
            pos += 14;
            let pair = |r: &mut BufReader<File>, pos: &mut u64| -> Res<(u32, u64)> {
                let a = u32::from_le_bytes(read_u(r)?);
                let b = u64::from_le_bytes(read_u(r)?);
                *pos += 12;
                Ok((a, b))
            };
            let xor = if flags & 1 == 1 { Some(pair(&mut r, &mut pos)?) } else { None };
            let dup = if flags & 2 == 2 { Some(pair(&mut r, &mut pos)?) } else { None };
            let unit = split::unit(kind);
            if unit == 0
                || len > BLOCK as u32
                || !(len as usize).is_multiple_of(unit)
                || flags > 3
                || (xor.is_some() && dup.is_some())
                || off != next
            {
                return Err("a damaged block header".into());
            }
            let mut parts = Vec::new();
            if dup.is_none() {
                for _ in 0..split::streams(kind) {
                    let [method] = read_u::<1, _>(&mut r)?;
                    let clen = u32::from_le_bytes(read_u(&mut r)?);
                    pos += 5;
                    if clen as usize > 2 * len as usize + 1024 {
                        return Err("a damaged part length".into());
                    }
                    parts.push((method, pos, clen));
                    io(r.seek_relative(i64::from(clen)), "seek")?;
                    pos += u64::from(clen);
                }
            }
            next += u64::from(len);
            out.push(BlockRef {
                off,
                len,
                kind,
                xor,
                dup,
                parts,
            });
        }
        if next != m.len {
            return Err("the blocks of a member do not add up to its length".into());
        }
        Ok(out)
    }

    /// Decode block `k` of member `i`, reading what it copies or XORs from the earlier member.
    fn decode_block(&self, i: usize, k: usize) -> Res<Vec<u8>> {
        let b = &self.blocks(i)?[k];
        let side = match (b.dup, b.xor) {
            (Some((j, off)), _) | (None, Some((j, off))) => {
                if j as usize >= i {
                    return Err("a block refers to a member that is not before it".into());
                }
                let mut buf = vec![0u8; b.len as usize];
                if self.read_range(j as usize, off, &mut buf)? != buf.len() {
                    return Err("a block refers past the end of an earlier member".into());
                }
                Some(buf)
            }
            _ => None,
        };
        if b.dup.is_some() {
            return side.ok_or_else(|| "a copy without its source".into());
        }
        let mut parts = Vec::with_capacity(b.parts.len());
        for &(method, pos, clen) in &b.parts {
            parts.push((method, pread(&self.file, pos, u64::from(clen))?));
        }
        let raw = Raw {
            kind: b.kind,
            xor: b.xor.is_some(),
            off: b.off,
            len: b.len as usize,
            base_off: 0,
            parts,
        };
        decode(&raw, side.as_deref())
    }

    /// Block `k` of member `i` from the cache, decoded if it is not there.
    fn block(&self, i: usize, k: usize) -> Res<Arc<Vec<u8>>> {
        let key = (i as u32, k as u32);
        let slot = {
            let mut c = self.cache.lock().map_err(|_| "a reader panicked")?;
            match c.map.get(&key) {
                Some(s) => s.clone(),
                None => {
                    let s = Arc::new(Slot::default());
                    c.map.insert(key, s.clone());
                    c.order.push_back(key);
                    s
                }
            }
        };
        let r = slot
            .block
            .get_or_init(|| self.decode_block(i, k).map(Arc::new))
            .clone();
        let mut c = self.cache.lock().map_err(|_| "a reader panicked")?;
        match &r {
            Ok(b) => {
                if slot
                    .counted
                    .compare_exchange(0, b.len(), Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    c.bytes += b.len();
                }
                while c.bytes > c.cap && c.order.len() > 1 {
                    let Some(old) = c.order.pop_front() else { break };
                    if old == key {
                        c.order.push_back(old);
                        continue;
                    }
                    if let Some(s) = c.map.remove(&old) {
                        c.bytes -= s.counted.load(Ordering::Acquire);
                    }
                }
            }
            Err(_) => {
                c.map.remove(&key);
                c.order.retain(|x| *x != key);
            }
        }
        r
    }

    /// Fill `buf` from member `i` at `off`; the number of bytes read, short only at the end of the member.
    pub fn read_range(&self, i: usize, off: u64, buf: &mut [u8]) -> Res<usize> {
        let m = self.members.get(i).ok_or("no such member")?;
        if m.nest.is_some() {
            return Err(format!("{} is a nested archive: read the entries inside it", self.names[i]));
        }
        if off >= m.len || buf.is_empty() {
            return Ok(0);
        }
        let want = (buf.len() as u64).min(m.len - off) as usize;
        let refs = self.blocks(i)?;
        let mut k = refs.partition_point(|b| b.off + u64::from(b.len) <= off);
        let (mut done, mut at) = (0usize, off);
        while done < want {
            let r = refs.get(k).ok_or("a read past the last block")?;
            let data = self.block(i, k)?;
            let from = (at - r.off) as usize;
            let n = (data.len() - from).min(want - done);
            buf[done..done + n].copy_from_slice(&data[from..from + n]);
            done += n;
            at += n as u64;
            k += 1;
        }
        Ok(done)
    }

    /// Write all of member `i` to `out`, blocks decoded on every core, and check its blake3 at the end.
    pub fn write_member(&self, i: usize, out: &mut dyn Write) -> Res<()> {
        let m = self.members.get(i).ok_or("no such member")?;
        if m.nest.is_some() {
            return Err(format!("{} is a nested archive: read the entries inside it", self.names[i]));
        }
        let n = self.blocks(i)?.len();
        let threads = pipe::threads();
        let (htx, hrx) = std::sync::mpsc::sync_channel::<Arc<Vec<u8>>>(threads);
        let (ran, digest) = std::thread::scope(|s| {
            let hasher = s.spawn(move || {
                let mut h = blake3::Hasher::new();
                for d in hrx {
                    h.update(&d);
                }
                *h.finalize().as_bytes()
            });
            let ran = pipe::run(
                threads,
                2 * threads,
                (0..n).map(Ok),
                |_, k: usize| self.decode_block(i, k).map(Arc::new),
                |_, d: Arc<Vec<u8>>| {
                    io(out.write_all(&d), "write")?;
                    htx.send(d).map_err(|_| "the hashing thread stopped".to_string())
                },
            );
            drop(htx);
            (ran, hasher.join())
        });
        ran?;
        let digest = digest.map_err(|_| "the hashing thread panicked".to_string())?;
        io(out.flush(), "flush")?;
        if digest != m.hash {
            return Err(format!(
                "{}: the bytes differ from the original (blake3 mismatch)",
                self.names[i]
            ));
        }
        Ok(())
    }

    /// The file entries that `name` stands for: the entry itself, or every file entry under it when it is a folder
    /// or a nested archive. Empty when nothing is called that.
    pub fn select(&self, name: &str) -> Vec<usize> {
        let name = name.trim_end_matches('/');
        let under = format!("{name}/");
        (0..self.members.len())
            .filter(|&i| {
                self.members[i].nest.is_none()
                    && (name.is_empty() || self.names[i] == name || self.names[i].starts_with(&under))
            })
            .collect()
    }

    /// Write the entries `names` select (all of them for none) under `dir`, each checked against its blake3, without
    /// restoring anything else. The number of files written.
    pub fn extract(&self, names: &[String], dir: &Path) -> Res<u32> {
        let mut chosen: Vec<usize> = if names.is_empty() {
            self.select("")
        } else {
            let mut v = Vec::new();
            for n in names {
                let s = self.select(n);
                if s.is_empty() {
                    return Err(format!("no entry called {n}"));
                }
                v.extend(s);
            }
            v
        };
        chosen.sort_unstable();
        chosen.dedup();
        for &i in &chosen {
            let target = dir.join(&self.names[i]);
            if let Some(parent) = target.parent() {
                io(std::fs::create_dir_all(parent), "create dir")?;
            }
            let mut out = BufWriter::with_capacity(1 << 20, io(File::create(&target), "create member")?);
            self.write_member(i, &mut out)?;
        }
        Ok(chosen.len() as u32)
    }

    /// Read all of member `i` and compare it with its blake3, writing it nowhere.
    pub fn verify_member(&self, i: usize) -> Res<()> {
        self.write_member(i, &mut std::io::sink())
    }
}

use std::sync::atomic::Ordering;
