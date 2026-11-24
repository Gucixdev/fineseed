//! The `.fseed` file: many files in one archive, like a zip, with the savings that only many models together give.
//! Everything is decided per tensor and per block by what the bytes are, never by file or tensor name:
//!
//! * a block that is byte-identical to a block of any earlier member is not stored again (a frozen layer, a
//!   copy, the same embedding in two models);
//! * a tensor may be coded against the most similar tensor of the same type and size in ANY earlier member
//!   (found by sampling): blocks become `block ^ reference block` where that is smaller, which is where a
//!   fine-tune, a merge or a model of the same family pays off;
//! * a file whose layout is not read (TensorFlow data files, TFLite, a PyTorch zip, anything) is cut into blocks and
//!   each block is coded as bfloat16, float16, float32 or plain bytes, whichever a sample says (`guess::guess`);
//! * a member that is itself a `.fseed` is opened: its members become members of this archive (so they share
//!   blocks and references with everything else), and the inner file is built again byte for byte when this archive
//!   is unpacked. That is checked when packing; an inner file that does not come back identical (another version
//!   wrote it) is kept as plain bytes instead. Nesting may go as deep as `MAX_DEPTH`.
//!
//! Both sharing kinds are restored by reading bytes of an already restored member, so `restore` writes members in
//! order and every member's blake3 is checked after it is written.
//!
//! Layout, little endian: magic `FSEEDA\x04\0`, then the blocks of all members, then the table, then the table
//! offset `u64` and the magic again. A block is `kind u8` (see `split`), `flags u8` (bit 0: xor with earlier
//! bytes, bit 1: copy of earlier bytes), `offset u64`, `len u32`, `member u32` and `source offset u64` for each
//! set flag (xor first), and unless bit 1 one part per stream: `method u8`, `coded length u32`, bytes.
//! The table: entry count `u32`, top level entry count `u32`, zstd level `u8`, then the entries in pre-order, each
//! name (`u16` length + UTF-8, `/` separated, never absolute or with `..`), role `u8` (0 file, 1 nested archive),
//! original length `u64`, blake3 (32), block count `u32`, offset of its first block `u64`; a nested archive has no
//! blocks and goes on with its zstd level `u8` and the number of its direct entries `u32`, which follow it.
use super::*;
use crate::pipe;
pub mod access;
use std::collections::HashMap;
use std::io::BufReader;
use std::path::PathBuf;

const AMAGIC: &[u8; 8] = b"FSEEDA\x04\0";
/// Deepest nesting of one `.fseed` in another that is read or written.
pub const MAX_DEPTH: u32 = 16;
/// A run of an unread layout shorter than this is stored as plain bytes without a look at its type.
const AUTO_MIN: u64 = 64 << 10;
/// A tensor smaller than this is not worth looking for a reference for.
const REF_MIN: u64 = 64 << 10;
/// Equal bytes in the sample at which a tensor counts as similar to a reference.
const SIMILAR: f64 = 0.30;
const SAMPLE: u64 = 8 << 10;
const CANDIDATES: usize = 8;

/// A writer that counts what went through it, so block offsets need no seeking.
struct Count<W: Write> {
    inner: W,
    n: u64,
}

impl<W: Write> Write for Count<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let k = self.inner.write(buf)?;
        self.n += k as u64;
        Ok(k)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[derive(Debug, Clone)]
pub struct Member {
    pub name: String,
    pub len: u64,
    pub hash: [u8; 32],
    pub blocks: u32,
    pub body: u64,
    /// How many `.fseed` files this member lies inside of.
    pub depth: u32,
    /// For a nested archive: its zstd level and the number of its direct entries (which follow it in the table).
    pub nest: Option<(u8, u32)>,
}

/// What goes into an archive: a file, or a `.fseed` that is opened and built again on unpack.
#[derive(Debug, Clone)]
pub enum Node {
    File {
        name: String,
        path: PathBuf,
    },
    Nest {
        name: String,
        level: u8,
        len: u64,
        hash: [u8; 32],
        children: Vec<Node>,
    },
    /// Standard input, read once and cut into blocks as it comes; `kind` says what the bytes are when the caller
    /// knows (`split::BF16` and so on), else each block is guessed.
    Stream {
        name: String,
        kind: Option<u8>,
    },
}

/// Files to store: `(archive name, path)`. A directory is walked, names start with its own name.
pub fn collect(paths: &[PathBuf]) -> Res<Vec<(String, PathBuf)>> {
    fn walk(prefix: &str, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> Res<()> {
        let mut entries: Vec<_> = io(std::fs::read_dir(dir), "read dir")?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let name = format!("{prefix}/{}", e.file_name().to_string_lossy());
            let ty = io(e.file_type(), "stat")?;
            if ty.is_dir() {
                walk(&name, &e.path(), out)?;
            } else if ty.is_file() {
                out.push((name, e.path()));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    for p in paths {
        if p == Path::new("-") {
            out.push(("stdin".to_string(), p.clone()));
            continue;
        }
        let meta = io(
            std::fs::symlink_metadata(p),
            &format!("stat {}", p.display()),
        )?;
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .ok_or("a path without a name")?;
        if meta.is_dir() {
            walk(&name, p, &mut out)?;
        } else if meta.is_file() {
            out.push((name, p.clone()));
        } else {
            return Err(format!("{}: not a regular file or directory", p.display()));
        }
    }
    let mut seen = std::collections::HashSet::new();
    for (n, _) in &out {
        if !seen.insert(n.clone()) {
            return Err(format!("two members would be named {n}"));
        }
    }
    Ok(out)
}

/// Share of equal bytes in samples of two byte ranges of the same length.
fn sample_equal(fa: &mut File, a: u64, fb: &mut File, b: u64, size: u64) -> Res<f64> {
    let n = size.min(SAMPLE);
    let (mut equal, mut total) = (0u64, 0u64);
    for at in [0, (size - n) / 2, size - n] {
        let (x, y) = (read_at(fa, a + at, n)?, read_at(fb, b + at, n)?);
        equal += x.iter().zip(&y).filter(|(p, q)| p == q).count() as u64;
        total += n;
    }
    Ok(equal as f64 / total as f64)
}

/// Tensors of earlier members, by `(kind, size)`: `(member, begin, name)`, the newest last.
type Index = HashMap<(u8, u64), Vec<(u32, u64, String)>>;

/// The earlier tensor most like `seg` among those of the same kind and size, if one is alike enough.
fn choose_ref(
    seg: &format::Seg,
    f: &mut File,
    paths: &[Option<&Path>],
    index: &Index,
    open: &mut HashMap<u32, File>,
) -> Res<Option<(u32, u64)>> {
    let size = seg.end - seg.begin;
    if size < REF_MIN {
        return Ok(None);
    }
    let Some(list) = index.get(&(seg.kind, size)) else {
        return Ok(None);
    };
    // the same name first (most likely the same layer), then the newest others
    let mut order: Vec<&(u32, u64, String)> = list.iter().filter(|c| c.2 == seg.name).collect();
    order.extend(list.iter().rev().filter(|c| c.2 != seg.name));
    let mut best: (f64, Option<(u32, u64)>) = (SIMILAR, None);
    for c in order.into_iter().take(CANDIDATES) {
        if let std::collections::hash_map::Entry::Vacant(v) = open.entry(c.0) {
            let p = paths[c.0 as usize].ok_or("a reference member has no file")?;
            v.insert(io(File::open(p), "open earlier member")?);
        }
        let other = open.get_mut(&c.0).ok_or("an earlier member is not open")?;
        let s = sample_equal(f, seg.begin, other, c.1, size)?;
        if s > best.0 {
            best = (s, Some((c.0, c.1)));
            if s > 0.6 {
                break;
            }
        }
    }
    Ok(best.1)
}

fn write_member_block<W: Write>(out: &mut W, e: &Encoded, dup: Option<(u32, u64)>) -> Res<()> {
    let mut head = vec![e.kind, u8::from(e.xor) | u8::from(dup.is_some()) << 1];
    head.extend(e.off.to_le_bytes());
    head.extend(e.len.to_le_bytes());
    if e.xor {
        head.extend(e.base_member.to_le_bytes());
        head.extend(e.base_off.to_le_bytes());
    }
    if let Some((m, off)) = dup {
        head.extend(m.to_le_bytes());
        head.extend(off.to_le_bytes());
    }
    io(out.write_all(&head), "write")?;
    if dup.is_none() {
        for p in &e.parts {
            io(out.write_all(&[p.method]), "write")?;
            io(
                out.write_all(&(p.bytes.len() as u32).to_le_bytes()),
                "write",
            )?;
            io(out.write_all(&p.bytes), "write")?;
        }
    }
    Ok(())
}

#[derive(Default, Debug, Clone)]
pub struct Report {
    pub members: u32,
    pub original: u64,
    pub packed: u64,
    pub dup_blocks: u32,
    pub xor_blocks: u32,
    pub blocks: u32,
    /// Nested `.fseed` files opened (packing) or built again (unpacking).
    pub nested: u32,
    pub seconds: f64,
}


fn flatten<'a>(nodes: &'a [Node], out: &mut Vec<&'a Node>) {
    for n in nodes {
        out.push(n);
        if let Node::Nest { children, .. } = n {
            flatten(children, out);
        }
    }
}

/// How a run of `n` bytes starting at a block boundary is cut once a sample says it looks like `kind` with the first
/// whole value at `shift`: a plain head, the values, a plain tail. `(length, kind)` pieces, none empty.
fn auto_pieces(n: u64, kind: u8, shift: u64) -> Vec<(u64, u8)> {
    let unit = split::unit(kind) as u64;
    if kind == split::PLAIN || n <= shift {
        return vec![(n, split::PLAIN)];
    }
    let body = (n - shift) / unit * unit;
    [
        (shift, split::PLAIN),
        (body, kind),
        (n - shift - body, split::PLAIN),
    ]
    .into_iter()
    .filter(|p| p.0 > 0)
    .collect()
}

/// The sample of a block of `n` bytes that decides its type: from the middle, at a multiple of 8 so the alignment
/// found holds for the block start. `(offset in the block, length)`.
fn auto_sample(n: u64) -> (u64, u64) {
    let s = n.min(guess::GUESS_SAMPLE as u64);
    (((n - s) / 2) & !7, s)
}

/// Cut `[begin, end)` of a file whose layout is not read into blocks, each as the type a sample of it looks like.
fn cut_auto(plans: &mut Vec<Plan>, f: &mut File, begin: u64, end: u64) -> Res<()> {
    let mut at = begin;
    while at < end {
        let n = (end - at).min(BLOCK);
        if n < AUTO_MIN {
            cut(plans, at, at + n, split::PLAIN, None);
        } else {
            let (from, s) = auto_sample(n);
            let (kind, shift) = guess::guess(&read_at(f, at + from, s)?);
            let mut p = at;
            for (len, k) in auto_pieces(n, kind, shift as u64) {
                cut(plans, p, p + len, k, None);
                p += len;
            }
        }
        at += n;
    }
    Ok(())
}

/// What one block came to: the coded block, or the earlier bytes it is a copy of.
struct Done {
    plan: Plan,
    enc: Option<Encoded>,
    dup: Option<(u32, u64)>,
    key: [u8; 32],
    data: Vec<u8>,
}

/// A block to code: its place and, for a stream that cannot be read again, its bytes.
type Job = (Plan, Option<Vec<u8>>);

/// What a member read from standard input comes out as: the blocks of 4 MiB read one after the other, each cut by a
/// sample (or all as `kind` when the caller says what the bytes are).
struct StreamJobs<R: Read> {
    input: R,
    at: u64,
    kind: Option<u8>,
    queue: std::collections::VecDeque<Job>,
    done: bool,
}

impl<R: Read> Iterator for StreamJobs<R> {
    type Item = Res<Job>;
    fn next(&mut self) -> Option<Res<Job>> {
        while self.queue.is_empty() && !self.done {
            let mut chunk = vec![0u8; BLOCK as usize];
            let mut have = 0usize;
            while have < chunk.len() {
                match self.input.read(&mut chunk[have..]) {
                    Ok(0) => {
                        self.done = true;
                        break;
                    }
                    Ok(k) => have += k,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => return Some(Err(format!("read input: {e}"))),
                }
            }
            chunk.truncate(have);
            if have == 0 {
                break;
            }
            let n = have as u64;
            let pieces = match self.kind {
                Some(k) => {
                    let u = split::unit(k) as u64;
                    let body = n / u * u;
                    [(body, k), (n - body, split::PLAIN)]
                        .into_iter()
                        .filter(|p| p.0 > 0)
                        .collect()
                }
                None if n < AUTO_MIN => vec![(n, split::PLAIN)],
                None => {
                    let (from, s) = auto_sample(n);
                    let (kind, shift) =
                        guess::guess(&chunk[from as usize..(from + s) as usize]);
                    auto_pieces(n, kind, shift as u64)
                }
            };
            let mut p = 0usize;
            for (len, k) in pieces {
                let bytes = chunk[p..p + len as usize].to_vec();
                self.queue.push_back((
                    Plan {
                        off: self.at + p as u64,
                        len,
                        kind: k,
                        base: None,
                    },
                    Some(bytes),
                ));
                p += len as usize;
            }
            self.at += n;
        }
        self.queue.pop_front().map(Ok)
    }
}

/// Code one member: its blocks go through the pipeline, are written in order, and the member's blake3 is taken on a
/// thread of its own so neither hashing nor writing holds the coders up. Returns `(blake3, blocks, length)` and the
/// blocks it added, `(key, offset, length)`, for the later members to copy from.
#[allow(clippy::type_complexity)]
fn code_member<W: Write, I: Iterator<Item = Res<Job>> + Send>(
    out: &mut Count<W>,
    rep: &mut Report,
    jobs: I,
    file: Option<&File>,
    seen: &HashMap<[u8; 32], (u32, u64, u32)>,
    open: &HashMap<u32, File>,
    level: i32,
) -> Res<([u8; 32], u32, u64, Vec<([u8; 32], u64, u32)>)> {
    let threads = pipe::threads();
    let mut fresh = Vec::new();
    let (mut blocks, mut total) = (0u32, 0u64);
    let (htx, hrx) = std::sync::mpsc::sync_channel::<Vec<u8>>(threads);
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
            jobs,
            |_, (plan, data): Job| -> Res<Done> {
                let data = match data {
                    Some(d) => d,
                    None => pread(file.ok_or("a block without a file to read")?, plan.off, plan.len)?,
                };
                let key = *blake3::hash(&data).as_bytes();
                let dup = seen
                    .get(&key)
                    .filter(|(_, _, l)| u64::from(*l) == plan.len)
                    .map(|(m, o, _)| (*m, *o));
                let enc = if dup.is_some() {
                    None
                } else {
                    let base = match plan.base {
                        Some((m, off)) => {
                            let bf = open.get(&m).ok_or("a reference member is not open")?;
                            Some(pread(bf, off, plan.len)?)
                        }
                        None => None,
                    };
                    Some(encode(&plan, &data, base.as_deref(), level)?)
                };
                Ok(Done {
                    plan,
                    enc,
                    dup,
                    key,
                    data,
                })
            },
            |_, d: Done| {
                match &d.enc {
                    Some(enc) => {
                        write_member_block(out, enc, None)?;
                        rep.xor_blocks += u32::from(enc.xor);
                    }
                    None => {
                        let enc = Encoded {
                            kind: d.plan.kind,
                            xor: false,
                            off: d.plan.off,
                            len: d.plan.len as u32,
                            base_member: 0,
                            base_off: 0,
                            parts: Vec::new(),
                        };
                        write_member_block(out, &enc, d.dup)?;
                        rep.dup_blocks += 1;
                    }
                }
                fresh.push((d.key, d.plan.off, d.plan.len as u32));
                blocks += 1;
                total += d.plan.len;
                htx.send(d.data).map_err(|_| "the hashing thread stopped".to_string())
            },
        );
        drop(htx);
        (ran, hasher.join())
    });
    ran?;
    let digest = digest.map_err(|_| "the hashing thread panicked".to_string())?;
    Ok((digest, blocks, total, fresh))
}

/// Write `nodes` as an archive (to standard output when `output` is `-`). Pure: it opens no archive, so it is also
/// how a nested one is built again.
fn write_archive(nodes: &[Node], output: &Path, level: i32) -> Res<Report> {
    if !(1..=22).contains(&level) {
        return Err("the level is 1 to 22".into());
    }
    let started = std::time::Instant::now();
    let mut flat = Vec::new();
    flatten(nodes, &mut flat);
    let paths: Vec<Option<&Path>> = flat
        .iter()
        .map(|n| match n {
            Node::File { path, .. } => Some(path.as_path()),
            _ => None,
        })
        .collect();
    let sink: Box<dyn Write> = if output == Path::new("-") {
        Box::new(std::io::stdout().lock())
    } else {
        Box::new(io(File::create(output), "create output")?)
    };
    let mut out = Count {
        inner: BufWriter::with_capacity(1 << 20, sink),
        n: 0,
    };
    io(out.write_all(AMAGIC), "write")?;
    let mut table: Vec<Member> = Vec::new();
    let mut seen: HashMap<[u8; 32], (u32, u64, u32)> = HashMap::new();
    let mut index: Index = HashMap::new();
    let mut open: HashMap<u32, File> = HashMap::new();
    let mut rep = Report::default();
    for (i, node) in flat.iter().enumerate() {
        let (name, path) = match node {
            Node::Nest {
                name,
                level: l,
                len,
                hash,
                children,
            } => {
                table.push(Member {
                    name: name.clone(),
                    len: *len,
                    hash: *hash,
                    blocks: 0,
                    body: out.n,
                    depth: 0,
                    nest: Some((*l, children.len() as u32)),
                });
                rep.nested += 1;
                continue;
            }
            Node::Stream { name, kind } => {
                let body = out.n;
                let jobs = StreamJobs {
                    input: std::io::stdin(),
                    at: 0,
                    kind: *kind,
                    queue: Default::default(),
                    done: false,
                };
                let (hash, blocks, len, fresh) =
                    code_member(&mut out, &mut rep, jobs, None, &seen, &open, level)?;
                for (k, off, l) in fresh {
                    seen.entry(k).or_insert((i as u32, off, l));
                }
                rep.blocks += blocks;
                rep.original += len;
                rep.members += 1;
                table.push(Member {
                    name: name.clone(),
                    len,
                    hash,
                    blocks,
                    body,
                    depth: 0,
                    nest: None,
                });
                continue;
            }
            Node::File { name, path } => (name, path),
        };
        let mut f = io(File::open(path), &format!("open {}", path.display()))?;
        let len = io(f.metadata(), "stat")?.len();
        let segs = format::layout(&mut f, len).unwrap_or_default();
        let mut plans = Vec::new();
        let mut pos = 0u64;
        for seg in &segs {
            cut_auto(&mut plans, &mut f, pos, seg.begin)?;
            let r = choose_ref(seg, &mut f, &paths, &index, &mut open)?;
            if seg.kind == split::PLAIN && r.is_none() {
                cut_auto(&mut plans, &mut f, seg.begin, seg.end)?;
            } else {
                cut(&mut plans, seg.begin, seg.end, seg.kind, r);
            }
            pos = seg.end;
        }
        cut_auto(&mut plans, &mut f, pos, len)?;
        let body = out.n;
        let count = plans.len() as u32;
        let jobs = plans.into_iter().map(|p| Ok((p, None)));
        let (hash, _, _, fresh) =
            code_member(&mut out, &mut rep, jobs, Some(&f), &seen, &open, level)?;
        for (k, off, l) in fresh {
            seen.entry(k).or_insert((i as u32, off, l));
        }
        for seg in segs {
            let list = index.entry((seg.kind, seg.end - seg.begin)).or_default();
            list.push((i as u32, seg.begin, seg.name));
            if list.len() > 4 * CANDIDATES {
                list.remove(0);
            }
        }
        rep.blocks += count;
        rep.original += len;
        rep.members += 1;
        table.push(Member {
            name: name.clone(),
            len,
            hash,
            blocks: count,
            body,
            depth: 0,
            nest: None,
        });
    }
    let table_at = out.n;
    let mut t = Vec::new();
    t.extend((table.len() as u32).to_le_bytes());
    t.extend((nodes.len() as u32).to_le_bytes());
    t.push(level as u8);
    for m in &table {
        t.extend((m.name.len() as u16).to_le_bytes());
        t.extend(m.name.as_bytes());
        t.push(u8::from(m.nest.is_some()));
        t.extend(m.len.to_le_bytes());
        t.extend(m.hash);
        t.extend(m.blocks.to_le_bytes());
        t.extend(m.body.to_le_bytes());
        if let Some((l, k)) = m.nest {
            t.push(l);
            t.extend(k.to_le_bytes());
        }
    }
    t.extend(table_at.to_le_bytes());
    t.extend(AMAGIC);
    io(out.write_all(&t), "write")?;
    io(out.flush(), "flush")?;
    rep.packed = out.n;
    rep.seconds = started.elapsed().as_secs_f64();
    Ok(rep)
}

fn is_archive(path: &Path) -> bool {
    let mut head = [0u8; 8];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut head))
        .is_ok()
        && &head == AMAGIC
}

/// Build every nested archive of `nodes` again and compare it with the bytes it had; `false` if one differs.
fn verify(nodes: &[Node], tmp: &Path, counter: &mut u32) -> Res<bool> {
    for n in nodes {
        let Node::Nest {
            level,
            hash,
            children,
            ..
        } = n
        else {
            continue;
        };
        if !verify(children, tmp, counter)? {
            return Ok(false);
        }
        *counter += 1;
        let built = tmp.join(format!("b{counter}"));
        write_archive(children, &built, i32::from(*level))?;
        let same = file_hash(&built)? == *hash;
        io(std::fs::remove_file(&built), "remove a built archive")?;
        if !same {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A `.fseed` file opened into the nodes it holds, if building it again gives its bytes back exactly.
fn expand(path: &Path, name: &str, tmp: &Path, counter: &mut u32) -> Res<Option<Node>> {
    if !is_archive(path) {
        return Ok(None);
    }
    *counter += 1;
    let dir = tmp.join(format!("x{counter}"));
    let Ok((level, children, _)) = restore_tree(path, &dir, false) else {
        return Ok(None);
    };
    let node = Node::Nest {
        name: name.to_string(),
        level,
        len: io(std::fs::metadata(path), "stat")?.len(),
        hash: file_hash(path)?,
        children,
    };
    Ok(verify(std::slice::from_ref(&node), tmp, counter)?.then_some(node))
}

/// Store `files` in `output`. `level` is the zstd level, 1 to 22. A file that is a `.fseed` is opened (see the
/// module doc).
pub fn store(files: &[(String, PathBuf)], output: &Path, level: i32) -> Res<Report> {
    store_as(files, output, level, None)
}

/// `store`, where a file named `-` is standard input and `kind` (if any) is what its bytes are.
pub fn store_as(
    files: &[(String, PathBuf)],
    output: &Path,
    level: i32,
    kind: Option<u8>,
) -> Res<Report> {
    let tmp = PathBuf::from(format!("{}.tmp", output.display()));
    let mut counter = 0u32;
    let mut nodes = Vec::new();
    let built = (|| -> Res<Report> {
        for (name, path) in files {
            if path == Path::new("-") {
                nodes.push(Node::Stream {
                    name: name.clone(),
                    kind,
                });
                continue;
            }
            nodes.push(match expand(path, name, &tmp, &mut counter)? {
                Some(n) => n,
                None => Node::File {
                    name: name.clone(),
                    path: path.clone(),
                },
            });
        }
        write_archive(&nodes, output, level)
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    built
}

struct Table {
    level: u8,
    tops: u32,
    members: Vec<Member>,
}

/// Check the entries make a tree of the stated shape, and set each one's depth.
fn walk(t: &mut [Member], at: &mut usize, count: u32, depth: u32) -> Res<()> {
    if depth > MAX_DEPTH {
        return Err("archives nested deeper than this reads".into());
    }
    for _ in 0..count {
        let i = *at;
        let m = t.get_mut(i).ok_or("a damaged table")?;
        m.depth = depth;
        *at += 1;
        if let Some((_, k)) = m.nest {
            walk(t, at, k, depth + 1)?;
        }
    }
    Ok(())
}

fn read_table(f: &mut File) -> Res<Table> {
    // the archive proper: what a parity trailer (`parity`) follows is not part of it
    let total = crate::parity::archive_len(f)?;
    if total < (AMAGIC.len() * 2 + 12) as u64 {
        return Err("not a fineseed archive".into());
    }
    io(f.seek(SeekFrom::Start(0)), "seek")?;
    let head: [u8; 8] = read_u(f)?;
    io(f.seek(SeekFrom::Start(total - 16)), "seek")?;
    let at = u64::from_le_bytes(read_u(f)?);
    let tail: [u8; 8] = read_u(f)?;
    if &head != AMAGIC || &tail != AMAGIC || at >= total - 16 {
        return Err("not a fineseed archive, or cut short".into());
    }
    let mut t = vec![0u8; (total - 16 - at) as usize];
    io(f.seek(SeekFrom::Start(at)), "seek")?;
    io(f.read_exact(&mut t), "read table")?;
    let mut r = &t[..];
    let n = u32::from_le_bytes(read_u(&mut r)?);
    let tops = u32::from_le_bytes(read_u(&mut r)?);
    let [level] = read_u::<1, _>(&mut r)?;
    if !(1..=22).contains(&level) {
        return Err("a damaged table".into());
    }
    let mut members = Vec::new();
    for _ in 0..n {
        let nl = u16::from_le_bytes(read_u(&mut r)?) as usize;
        if nl > r.len() {
            return Err("a damaged table".into());
        }
        let name =
            String::from_utf8(r[..nl].to_vec()).map_err(|_| "a member name that is not UTF-8")?;
        r = &r[nl..];
        let [role] = read_u::<1, _>(&mut r)?;
        let len = u64::from_le_bytes(read_u(&mut r)?);
        let hash: [u8; 32] = read_u(&mut r)?;
        let blocks = u32::from_le_bytes(read_u(&mut r)?);
        let body = u64::from_le_bytes(read_u(&mut r)?);
        let nest = match role {
            0 => None,
            1 if blocks == 0 => {
                let [l] = read_u::<1, _>(&mut r)?;
                let k = u32::from_le_bytes(read_u(&mut r)?);
                if !(1..=22).contains(&l) {
                    return Err("a damaged table".into());
                }
                Some((l, k))
            }
            _ => return Err("a damaged table".into()),
        };
        members.push(Member {
            name,
            len,
            hash,
            blocks,
            body,
            depth: 0,
            nest,
        });
    }
    let mut at = 0usize;
    walk(&mut members, &mut at, tops, 0)?;
    if at != members.len() {
        return Err("a damaged table".into());
    }
    Ok(Table {
        level,
        tops,
        members,
    })
}

/// Every entry, nested ones included, in the order of the table (a nested archive, then what it holds).
pub fn list(archive: &Path) -> Res<Vec<Member>> {
    Ok(read_table(&mut io(File::open(archive), "open")?)?.members)
}

fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name.contains('\\')
        && name
            .split('/')
            .all(|c| !c.is_empty() && c != "." && c != "..")
        && name.split('/').next() != Some(STAGE)
}

/// Where a nested archive's members are put while the archive is built from them.
const STAGE: &str = ".fseed-stage";

struct Blk {
    raw: Raw,
    base_member: u32,
    dup: Option<(u32, u64)>,
}

fn read_member_block<R: Read>(r: &mut R) -> Res<Blk> {
    let [kind, flags] = read_u::<2, _>(r)?;
    let off = u64::from_le_bytes(read_u(r)?);
    let len = u32::from_le_bytes(read_u(r)?) as usize;
    let xor = flags & 1 == 1;
    let is_dup = flags & 2 == 2;
    let (base_member, base_off) = if xor {
        (
            u32::from_le_bytes(read_u(r)?),
            u64::from_le_bytes(read_u(r)?),
        )
    } else {
        (0, 0)
    };
    let dup = if is_dup {
        let m = u32::from_le_bytes(read_u(r)?);
        Some((m, u64::from_le_bytes(read_u(r)?)))
    } else {
        None
    };
    let unit = split::unit(kind);
    if unit == 0
        || len > BLOCK as usize
        || !len.is_multiple_of(unit)
        || flags > 3
        || (is_dup && xor)
    {
        return Err("a damaged block header".into());
    }
    let mut parts = Vec::new();
    if !is_dup {
        for _ in 0..split::streams(kind) {
            let [method] = read_u::<1, _>(r)?;
            let clen = u32::from_le_bytes(read_u(r)?) as usize;
            if clen > 2 * len + 1024 {
                return Err("a damaged part length".into());
            }
            let mut bytes = vec![0u8; clen];
            io(r.read_exact(&mut bytes), "read part")?;
            parts.push((method, bytes));
        }
    }
    Ok(Blk {
        raw: Raw {
            kind,
            xor,
            off,
            len,
            base_off,
            parts,
        },
        base_member,
        dup,
    })
}

struct Ctx {
    f: File,
    table: Vec<Member>,
    /// Where each entry restored so far lies, for the blocks that read earlier bytes.
    paths: Vec<Option<PathBuf>>,
    regenerate: bool,
    stage: PathBuf,
    rep: Report,
}

/// Restore the blocks of file entry `i` to `target`; its blake3 is checked as soon as it is written.
///
/// The blocks are parsed one after the other (that is a sequential read of the archive anyway), decoded on every
/// core, and written in order while a thread of its own hashes what was written.
fn restore_file(ctx: &mut Ctx, i: usize, target: &Path) -> Res<()> {
    let m = ctx.table[i].clone();
    io(ctx.f.seek(SeekFrom::Start(m.body)), "seek")?;
    let mut rd = BufReader::with_capacity(1 << 20, io(ctx.f.try_clone(), "clone")?);
    let out = io(File::create(target), "create member")?;
    let paths = &ctx.paths;
    let rep = &mut ctx.rep;
    let files: std::sync::Mutex<HashMap<u32, std::sync::Arc<File>>> = Default::default();
    let earlier = |j: u32| -> Res<std::sync::Arc<File>> {
        let path = paths
            .get(j as usize)
            .and_then(|p| p.as_ref())
            .filter(|_| (j as usize) < i)
            .ok_or("a block refers to a member that is not restored yet")?;
        let mut g = files.lock().map_err(|_| "a worker panicked")?;
        if g.len() > 48 {
            g.clear();
        }
        if let Some(f) = g.get(&j) {
            return Ok(f.clone());
        }
        let f = std::sync::Arc::new(io(File::open(path), "open an earlier member")?);
        g.insert(j, f.clone());
        Ok(f)
    };
    let threads = pipe::threads();
    let (mut written, mut seen) = (0u64, 0u32);
    let (htx, hrx) = std::sync::mpsc::sync_channel::<Vec<u8>>(threads);
    let (ran, digest) = std::thread::scope(|s| {
        let hasher = s.spawn(move || {
            let mut h = blake3::Hasher::new();
            for d in hrx {
                h.update(&d);
            }
            *h.finalize().as_bytes()
        });
        let jobs = (0..m.blocks).map(|_| read_member_block(&mut rd));
        let ran = pipe::run(
            threads,
            2 * threads,
            jobs,
            |_, b: Blk| -> Res<(Vec<u8>, u64, bool, bool)> {
                let src = match (b.dup, b.raw.xor) {
                    (Some(d), _) => Some(d),
                    (None, true) => Some((b.base_member, b.raw.base_off)),
                    _ => None,
                };
                let side = match src {
                    Some((j, off)) => Some(pread(&*earlier(j)?, off, b.raw.len as u64)?),
                    None => None,
                };
                let bytes = match (b.dup.is_some(), side) {
                    (true, Some(copy)) => copy,
                    (_, side) => decode(&b.raw, side.as_deref())?,
                };
                // each worker writes its own block where it belongs: one writer at the end of the pipeline
                // was the floor of the whole unpack (page cache fill, 2 GB/s) however many cores there were
                io(std::os::unix::fs::FileExt::write_all_at(&out, &bytes, b.raw.off), "write")?;
                Ok((bytes, b.raw.off, b.dup.is_some(), b.raw.xor))
            },
            |_, (bytes, off, dup, xor): (Vec<u8>, u64, bool, bool)| {
                if off != written {
                    return Err("blocks out of order".to_string());
                }
                written += bytes.len() as u64;
                rep.dup_blocks += u32::from(dup);
                rep.xor_blocks += u32::from(xor);
                seen += 1;
                htx.send(bytes).map_err(|_| "the hashing thread stopped".to_string())
            },
        );
        drop(htx);
        (ran, hasher.join())
    });
    ran?;
    let digest = digest.map_err(|_| "the hashing thread panicked".to_string())?;
    if written != m.len || seen != m.blocks || digest != m.hash {
        return Err(format!(
            "{}: the restored bytes differ from the original (blake3 mismatch)",
            m.name
        ));
    }
    rep.members += 1;
    rep.original += m.len;
    rep.blocks += m.blocks;
    Ok(())
}

/// Restore `count` entries, starting at entry `*next`, under `dir`.
fn run(ctx: &mut Ctx, next: &mut usize, count: u32, dir: &Path) -> Res<Vec<Node>> {
    let mut out = Vec::new();
    for _ in 0..count {
        let i = *next;
        *next += 1;
        let m = ctx.table[i].clone();
        let target = dir.join(&m.name);
        if let Some(parent) = target.parent() {
            io(std::fs::create_dir_all(parent), "create dir")?;
        }
        match m.nest {
            None => {
                restore_file(ctx, i, &target)?;
                ctx.paths[i] = Some(target.clone());
                out.push(Node::File {
                    name: m.name,
                    path: target,
                });
            }
            Some((level, k)) => {
                let children = run(ctx, next, k, &ctx.stage.join(i.to_string()))?;
                if ctx.regenerate {
                    write_archive(&children, &target, i32::from(level))?;
                    if io(std::fs::metadata(&target), "stat")?.len() != m.len
                        || file_hash(&target)? != m.hash
                    {
                        return Err(format!(
                            "{}: the nested archive did not come back identical (blake3 mismatch)",
                            m.name
                        ));
                    }
                    ctx.paths[i] = Some(target);
                }
                ctx.rep.nested += 1;
                out.push(Node::Nest {
                    name: m.name,
                    level,
                    len: m.len,
                    hash: m.hash,
                    children,
                });
            }
        }
    }
    Ok(out)
}

/// With `regenerate`, a nested archive is built again as a file; without it the files it holds are left in the
/// stage directory under `dir` and the tree of them is returned (what packing another archive reads).
fn restore_tree(
    archive: &Path,
    dir: &Path,
    regenerate: bool,
) -> Res<(u8, Vec<Node>, Report)> {
    let started = std::time::Instant::now();
    let mut f = io(File::open(archive), "open")?;
    let Table {
        level,
        tops,
        members,
    } = read_table(&mut f)?;
    for m in &members {
        if !safe_name(&m.name) {
            return Err(format!("refusing the member name {:?}", m.name));
        }
        if m.blocks as u64 > m.len {
            return Err("a damaged table".into());
        }
    }
    io(std::fs::create_dir_all(dir), "create dir")?;
    let mut ctx = Ctx {
        f,
        paths: vec![None; members.len()],
        table: members,
        regenerate,
        stage: dir.join(STAGE),
        rep: Report::default(),
    };
    let done = run(&mut ctx, &mut 0, tops, dir);
    if regenerate {
        let _ = std::fs::remove_dir_all(&ctx.stage);
    }
    let nodes = done?;
    ctx.rep.packed = io(std::fs::metadata(archive), "stat")?.len();
    ctx.rep.seconds = started.elapsed().as_secs_f64();
    Ok((level, nodes, ctx.rep))
}

/// Restore every file under `dir`, nested archives built again; each is checked against its blake3 as it is written.
pub fn restore(archive: &Path, dir: &Path) -> Res<Report> {
    Ok(restore_tree(archive, dir, true)?.2)
}
