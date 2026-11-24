//! `fineseed mount IN.fseed DIR`: the archive as a read-only folder. Nothing is unpacked: a read decodes only the
//! blocks it touches (`archive::access`), keeps them in a bounded cache, and the kernel keeps the pages (the archive
//! never changes under the mount, so the cache is not dropped on open). A nested `.fseed` shows as a folder holding
//! its entries. Needs `fusermount3` (fuse3) on the machine; nothing is linked against libfuse.
use crate::Res;
use crate::archive::access::Archive;
use fuser::{
    Config, Errno, FileAttr, FileHandle, FileType, Filesystem, FopenFlags, Generation, INodeNo,
    LockOwner, MountOption, OpenFlags, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry,
    ReplyOpen, ReplyStatfs, Request,
};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::time::{Duration, SystemTime};

/// The archive never changes while it is mounted, so what the kernel learns can stay.
const TTL: Duration = Duration::from_secs(3600);

enum Kind {
    Dir(BTreeMap<String, u64>),
    File(usize),
}

struct Node {
    parent: u64,
    kind: Kind,
}

struct Fs {
    ar: Archive,
    nodes: Vec<Node>,
    uid: u32,
    gid: u32,
    time: SystemTime,
}

/// The folder tree of the entries; inode `n` is `nodes[n - 1]`, the root is 1.
fn tree(ar: &Archive) -> Vec<Node> {
    let mut nodes = vec![Node {
        parent: 1,
        kind: Kind::Dir(BTreeMap::new()),
    }];
    for (i, m) in ar.members().iter().enumerate() {
        let name = ar.name(i).to_string();
        let parts: Vec<&str> = name.split('/').collect();
        let mut at = 1u64;
        for (k, part) in parts.iter().enumerate() {
            let last = k + 1 == parts.len();
            let known = match &nodes[at as usize - 1].kind {
                Kind::Dir(c) => c.get(*part).copied(),
                Kind::File(_) => break,
            };
            at = match known {
                Some(n) => n,
                None => {
                    let ino = nodes.len() as u64 + 1;
                    let kind = if last && m.nest.is_none() {
                        Kind::File(i)
                    } else {
                        Kind::Dir(BTreeMap::new())
                    };
                    nodes.push(Node { parent: at, kind });
                    if let Kind::Dir(c) = &mut nodes[at as usize - 1].kind {
                        c.insert((*part).to_string(), ino);
                    }
                    ino
                }
            };
        }
    }
    nodes
}

impl Fs {
    fn attr(&self, ino: u64) -> Option<FileAttr> {
        let n = self.nodes.get(ino.checked_sub(1)? as usize)?;
        let (kind, size, perm, nlink) = match &n.kind {
            Kind::Dir(c) => (FileType::Directory, 0, 0o555, 2 + c.len() as u32),
            Kind::File(i) => (FileType::RegularFile, self.ar.members()[*i].len, 0o444, 1),
        };
        Some(FileAttr {
            ino: INodeNo(ino),
            size,
            blocks: size.div_ceil(512),
            atime: self.time,
            mtime: self.time,
            ctime: self.time,
            crtime: self.time,
            kind,
            perm,
            nlink,
            uid: self.uid,
            gid: self.gid,
            rdev: 0,
            flags: 0,
            blksize: 1 << 20,
        })
    }
}

impl Filesystem for Fs {
    fn lookup(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        let found = self
            .nodes
            .get((u64::from(parent) as usize).wrapping_sub(1))
            .and_then(|n| match &n.kind {
                Kind::Dir(c) => c.get(name.to_str()?).copied(),
                Kind::File(_) => None,
            })
            .and_then(|ino| self.attr(ino));
        match found {
            Some(a) => reply.entry(&TTL, &a, Generation(0)),
            None => reply.error(Errno::ENOENT),
        }
    }

    fn getattr(&self, _req: &Request, ino: INodeNo, _fh: Option<FileHandle>, reply: ReplyAttr) {
        match self.attr(u64::from(ino)) {
            Some(a) => reply.attr(&TTL, &a),
            None => reply.error(Errno::ENOENT),
        }
    }

    fn open(&self, _req: &Request, ino: INodeNo, _flags: OpenFlags, _kill: bool, reply: ReplyOpen) {
        match self.nodes.get((u64::from(ino) as usize).wrapping_sub(1)).map(|n| &n.kind) {
            Some(Kind::File(_)) => reply.opened(FileHandle(0), FopenFlags::FOPEN_KEEP_CACHE),
            Some(Kind::Dir(_)) => reply.error(Errno::EISDIR),
            None => reply.error(Errno::ENOENT),
        }
    }

    fn read(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock: Option<LockOwner>,
        reply: ReplyData,
    ) {
        let Some(Kind::File(i)) = self
            .nodes
            .get((u64::from(ino) as usize).wrapping_sub(1))
            .map(|n| &n.kind)
        else {
            return reply.error(Errno::ENOENT);
        };
        let mut buf = vec![0u8; size as usize];
        match self.ar.read_range(*i, offset, &mut buf) {
            Ok(n) => reply.data(&buf[..n]),
            Err(e) => {
                eprintln!("fineseed mount: {}: {e}", self.ar.name(*i));
                reply.error(Errno::EIO);
            }
        }
    }

    fn readdir(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        mut reply: ReplyDirectory,
    ) {
        let ino = u64::from(ino);
        let Some(Node {
            parent,
            kind: Kind::Dir(children),
        }) = self.nodes.get((ino as usize).wrapping_sub(1))
        else {
            return reply.error(Errno::ENOTDIR);
        };
        let dots = [(ino, ".".to_string()), (*parent, "..".to_string())];
        let all = dots
            .into_iter()
            .map(|(n, s)| (n, FileType::Directory, s))
            .chain(children.iter().map(|(name, &c)| {
                let ty = match self.nodes[c as usize - 1].kind {
                    Kind::Dir(_) => FileType::Directory,
                    Kind::File(_) => FileType::RegularFile,
                };
                (c, ty, name.clone())
            }));
        for (k, (n, ty, name)) in all.enumerate().skip(offset as usize) {
            if reply.add(INodeNo(n), (k + 1) as u64, ty, name) {
                break;
            }
        }
        reply.ok();
    }

    fn statfs(&self, _req: &Request, _ino: INodeNo, reply: ReplyStatfs) {
        let total: u64 = self.ar.members().iter().map(|m| m.len).sum();
        reply.statfs(total.div_ceil(512), 0, 0, self.nodes.len() as u64, 0, 512, 255, 0);
    }
}

/// Serve `archive` read-only at `dir` until it is unmounted (`fusermount3 -u DIR`, or Ctrl-C). `cache_mib` bounds the
/// decoded blocks kept in memory.
pub fn mount(archive: &Path, dir: &Path, cache_mib: usize) -> Res<()> {
    let meta = std::fs::metadata(archive).map_err(|e| format!("stat: {e}"))?;
    let ar = Archive::open(archive, cache_mib << 20)?;
    let nodes = tree(&ar);
    let time = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let fs = Fs {
        ar,
        nodes,
        uid: meta.uid(),
        gid: meta.gid(),
        time,
    };
    let mut cfg = Config::default();
    cfg.mount_options = vec![
        MountOption::RO,
        MountOption::NoAtime,
        MountOption::FSName("fineseed".into()),
    ];
    cfg.n_threads = Some(crate::pipe::threads().min(8));
    cfg.clone_fd = true;
    fuser::mount(fs, dir, &cfg).map_err(|e| format!("mount {}: {e}", dir.display()))
}
