//! Canonical Huffman coder for one plane of bytes, code lengths limited to 15 bits.
//!
//! Coded form: 128 bytes of code lengths (two symbols per byte, 4 bits each, 0 = symbol absent) followed by the
//! bit stream, most significant bit first, padded with zero bits. The number of symbols is not stored: the caller
//! knows the plane length. `decode` refuses lengths that do not form a prefix code and streams that run dry.

const MAX_LEN: u32 = 15;
pub const TABLE_BYTES: usize = 128;

fn counts(data: &[u8]) -> [u64; 256] {
    // four tables, so consecutive equal bytes (an exponent plane is mostly that) do not wait on one counter
    let mut c = [0u64; 256];
    // 2^32 would overflow a u32 counter: fold in slices well below that
    for part in data.chunks(1 << 30) {
        let mut t = [[0u32; 256]; 4];
        let (quads, rest) = part.as_chunks::<4>();
        for q in quads {
            t[0][q[0] as usize] += 1;
            t[1][q[1] as usize] += 1;
            t[2][q[2] as usize] += 1;
            t[3][q[3] as usize] += 1;
        }
        for &b in rest {
            t[0][b as usize] += 1;
        }
        for s in 0..256 {
            c[s] += u64::from(t[0][s]) + u64::from(t[1][s]) + u64::from(t[2][s]) + u64::from(t[3][s]);
        }
    }
    c
}

/// Code lengths for `counts`, none above `MAX_LEN`; absent symbols get 0, a lone symbol gets 1.
fn lengths(counts: &[u64; 256]) -> [u8; 256] {
    let mut c = *counts;
    loop {
        let l = build(&c);
        if l.iter().all(|&x| u32::from(x) <= MAX_LEN) {
            return l;
        }
        for v in c.iter_mut().filter(|v| **v > 0) {
            *v = (*v / 2).max(1);
        }
    }
}

fn build(counts: &[u64; 256]) -> [u8; 256] {
    // nodes: (weight, index); leaves are 0..256, inner nodes follow, `parent` links give the depths
    let mut parent = vec![usize::MAX; 512];
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize)>> =
        std::collections::BinaryHeap::new();
    for (s, &w) in counts.iter().enumerate() {
        if w > 0 {
            heap.push(std::cmp::Reverse((w, s)));
        }
    }
    let mut out = [0u8; 256];
    if heap.len() == 1 {
        if let Some(std::cmp::Reverse((_, s))) = heap.pop() {
            out[s] = 1;
        }
        return out;
    }
    let mut next = 256;
    while heap.len() > 1 {
        let (Some(std::cmp::Reverse((wa, a))), Some(std::cmp::Reverse((wb, b)))) =
            (heap.pop(), heap.pop())
        else {
            break;
        };
        parent[a] = next;
        parent[b] = next;
        heap.push(std::cmp::Reverse((wa + wb, next)));
        next += 1;
    }
    for s in 0..256 {
        if counts[s] == 0 {
            continue;
        }
        let (mut d, mut n) = (0u32, s);
        while parent[n] != usize::MAX {
            n = parent[n];
            d += 1;
        }
        out[s] = d.min(255) as u8;
    }
    out
}

/// Canonical codes: shorter first, then by symbol. `None` if the lengths overfill the code space.
fn codes(len: &[u8; 256]) -> Option<[u16; 256]> {
    let mut by_len = [0u32; 17];
    for &l in len {
        if u32::from(l) > MAX_LEN {
            return None;
        }
        by_len[l as usize] += 1;
    }
    by_len[0] = 0;
    let mut next = [0u32; 17];
    let mut code = 0u32;
    for l in 1..=MAX_LEN as usize {
        code = (code + by_len[l - 1]) << 1;
        next[l] = code;
        if code + by_len[l] > 1 << l {
            return None;
        }
    }
    let mut out = [0u16; 256];
    for s in 0..256 {
        let l = len[s] as usize;
        if l > 0 {
            out[s] = next[l] as u16;
            next[l] += 1;
        }
    }
    Some(out)
}

/// The code lengths for a plane and the size `encode_with` would give for it, from one pass over the plane.
pub struct Table {
    len: [u8; 256],
    pub size: usize,
}

pub fn table(data: &[u8]) -> Table {
    if data.is_empty() {
        return Table {
            len: [0; 256],
            size: 0,
        };
    }
    let c = counts(data);
    let len = lengths(&c);
    let bits: u64 = (0..256).map(|s| c[s] * u64::from(len[s])).sum();
    Table {
        len,
        size: TABLE_BYTES + bits.div_ceil(8) as usize,
    }
}

/// The coded bytes, or `None` for an empty plane.
pub fn encode(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }
    encode_with(data, &table(data))
}

/// `encode` with the lengths `table(data)` already found.
pub fn encode_with(data: &[u8], t: &Table) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }
    let sym = symbols(&t.len)?;
    let mut out = Vec::with_capacity(t.size.max(TABLE_BYTES) + 8);
    header(&t.len, &mut out);
    out.extend_from_slice(&encode_one(data, &sym));
    Some(out)
}

fn header(len: &[u8; 256], out: &mut Vec<u8>) {
    for pair in len.as_chunks::<2>().0 {
        out.push(pair[0] | pair[1] << 4);
    }
}

/// Symbol code and length in one word (code << 8 | length), so the loop does one load per byte.
fn symbols(len: &[u8; 256]) -> Option<[u32; 256]> {
    let code = codes(len)?;
    let mut sym = [0u32; 256];
    for s in 0..256 {
        sym[s] = u32::from(code[s]) << 8 | u32::from(len[s]);
    }
    Some(sym)
}

/// One stream: the bits most significant first, padded with zero bits.
///
/// Without a branch or a call in the loop: after every two symbols (at most 30 bits on top of 7 pending) the word
/// `acc` is written at the byte where the pending bits start, and the position moves by the whole bytes. What sits
/// past the position is stale and gets written over by the next word.
fn encode_one(data: &[u8], sym: &[u32; 256]) -> Vec<u8> {
    let mut out = vec![0u8; data.len() * 2 + 16];
    let (mut acc, mut nb, mut pos) = (0u64, 0u32, 0usize);
    // four codes of at most 14 bits and the 7 pending bits fit in 64; with longer codes it is two at a time
    let quad = sym.iter().all(|e| e & 0xff <= 14);
    let rest = if quad {
        let (quads, rest) = data.as_chunks::<4>();
        for q in quads {
            let e = [sym[q[0] as usize], sym[q[1] as usize], sym[q[2] as usize], sym[q[3] as usize]];
            acc = (((acc << (e[0] & 0xff) | u64::from(e[0] >> 8)) << (e[1] & 0xff) | u64::from(e[1] >> 8))
                << (e[2] & 0xff)
                | u64::from(e[2] >> 8))
                << (e[3] & 0xff)
                | u64::from(e[3] >> 8);
            nb += (e[0] & 0xff) + (e[1] & 0xff) + (e[2] & 0xff) + (e[3] & 0xff);
            out[pos..pos + 8].copy_from_slice(&acc.wrapping_shl(64 - nb).to_be_bytes());
            pos += (nb >> 3) as usize;
            nb &= 7;
        }
        rest
    } else {
        let (pairs, rest) = data.as_chunks::<2>();
        for p in pairs {
            let (e0, e1) = (sym[p[0] as usize], sym[p[1] as usize]);
            let (l0, l1) = (e0 & 0xff, e1 & 0xff);
            acc = (acc << l0 | u64::from(e0 >> 8)) << l1 | u64::from(e1 >> 8);
            nb += l0 + l1;
            out[pos..pos + 8].copy_from_slice(&acc.wrapping_shl(64 - nb).to_be_bytes());
            pos += (nb >> 3) as usize;
            nb &= 7;
        }
        rest
    };
    for &b in rest {
        let e = sym[b as usize];
        acc = acc << (e & 0xff) | u64::from(e >> 8);
        nb += e & 0xff;
        out[pos..pos + 8].copy_from_slice(&acc.wrapping_shl(64 - nb).to_be_bytes());
        pos += (nb >> 3) as usize;
        nb &= 7;
    }
    if nb > 0 {
        out[pos] = (acc << (8 - nb)) as u8;
        pos += 1;
    }
    out.truncate(pos);
    out
}

/// Symbols per chunk of the chunked form. Each chunk is a stream of its own, so four of them decode side by side
/// (four dependency chains instead of one) and a GPU can give each to a thread.
pub const CHUNK: usize = 1 << 14;

pub(crate) fn read_lengths(coded: &[u8]) -> Result<[u8; 256], String> {
    if coded.len() < TABLE_BYTES {
        return Err("huffman table cut short".into());
    }
    let mut len = [0u8; 256];
    for (i, b) in coded[..TABLE_BYTES].iter().enumerate() {
        len[2 * i] = b & 15;
        len[2 * i + 1] = b >> 4;
    }
    Ok(len)
}

pub(crate) const TABLE_LEN: usize = 1 << MAX_LEN;
/// An entry no code points at: one bit long and marked, so a step still moves on and the marker is read afterwards.
const UNUSED: u16 = 1 << 12 | 1;

/// The decoding table, indexed by the longest code in use, not by 15 bits: a mantissa plane needs 9 or 10 and the
/// part of the table that is read then stays in the L1 cache. Entry = symbol << 4 | length, `UNUSED` for no code.
/// Its size is fixed so that the index (masked with `TABLE_LEN - 1`) needs no bounds check.
pub(crate) fn decoding_table(len: &[u8; 256]) -> Result<(Box<[u16; TABLE_LEN]>, u32), String> {
    let code = codes(len).ok_or("huffman lengths overfill the code space")?;
    let top = u32::from(len.iter().copied().max().unwrap_or(1)).max(1);
    let mut table: Box<[u16; TABLE_LEN]> = vec![0u16; TABLE_LEN]
        .into_boxed_slice()
        .try_into()
        .map_err(|_| "table")?;
    table[..1 << top].fill(UNUSED);
    for s in 0..256 {
        let l = u32::from(len[s]);
        if l == 0 {
            continue;
        }
        let lo = usize::from(code[s]) << (top - l);
        let hi = lo + (1 << (top - l));
        if hi > 1 << top {
            return Err("huffman code out of range".into());
        }
        table[lo..hi].fill((s as u16) << 4 | l as u16);
    }
    Ok((table, top))
}

/// One stream being decoded into one slice. The state is the bit position alone: every step loads the eight bytes
/// around it, which leave at least 57 bits, and takes `per` symbols from them, so there is no refill to wait for.
struct Lane<'a> {
    stream: &'a [u8],
    out: &'a mut [u8],
    bit: usize,
    at: usize,
    bad: u32,
}

impl<'a> Lane<'a> {
    fn new(stream: &'a [u8], out: &'a mut [u8]) -> Self {
        Lane {
            stream,
            out,
            bit: 0,
            at: 0,
            bad: 0,
        }
    }

    /// Room for a step without any check: eight bytes to read and `per` symbols to write.
    #[inline(always)]
    fn can(&self, per: usize) -> bool {
        self.at + per <= self.out.len() && (self.bit >> 3) + 8 <= self.stream.len()
    }

    #[inline(always)]
    fn step(&mut self, table: &[u16; TABLE_LEN], top: u32, per: usize) {
        let b = self.bit >> 3;
        let mut w = u64::from_be_bytes(self.stream[b..b + 8].try_into().unwrap_or([0; 8])) << (self.bit & 7);
        let (mut used, mut bad) = (0usize, 0u16);
        for o in &mut self.out[self.at..self.at + per] {
            let e = table[(w >> (64 - top)) as usize & (TABLE_LEN - 1)];
            let l = e & 15;
            bad |= e;
            w <<= l;
            used += usize::from(l);
            *o = (e >> 4) as u8;
        }
        self.bit += used;
        self.at += per;
        self.bad |= u32::from(bad >> 12 & 1);
    }

    /// The rest, with every code checked against the bits that are left.
    fn finish(&mut self, table: &[u16; TABLE_LEN], top: u32, per: usize) -> Result<(), String> {
        while self.can(per) && self.bad == 0 {
            self.step(table, top, per);
        }
        if self.bad != 0 {
            return Err("huffman stream damaged or cut short".into());
        }
        let total = self.stream.len() * 8;
        while self.at < self.out.len() {
            let b = self.bit >> 3;
            let mut w = 0u64;
            for i in 0..8 {
                w = w << 8 | u64::from(self.stream.get(b + i).copied().unwrap_or(0));
            }
            w <<= self.bit & 7;
            let e = table[(w >> (64 - top)) as usize & (TABLE_LEN - 1)];
            let l = usize::from(e & 15);
            if e & UNUSED & 1 << 12 != 0 || self.bit + l > total {
                return Err("huffman stream damaged or cut short".into());
            }
            self.out[self.at] = (e >> 4) as u8;
            self.at += 1;
            self.bit += l;
        }
        Ok(())
    }
}

/// The number of symbols a step takes: the 57 bits it has, over the longest code.
fn per_step(top: u32) -> usize {
    (57 / top) as usize
}

/// Bits one lookup of the multi-symbol table reads, and lookups per step. The exponent plane of a weight file has
/// 2 to 3 bits of information per symbol, so a 12-bit window holds three symbols and one lookup gives all of them.
/// A step reads 57 bits at least: every lookup uses at most `WIN`, and the one that meets a longer code (the slow
/// path, up to 15 bits) ends the step.
const WIN: u32 = 12;
const STEPS: usize = 4;
const _: () = assert!(STEPS as u32 * WIN <= 57 && (STEPS as u32 - 1) * WIN + MAX_LEN <= 57);
/// Below this many symbols the table (4096 entries) costs more to build than it saves.
const MULTI_FROM: usize = 1 << 16;
const LONG: u32 = 0;
const BAD: u32 = 1 << 31 | 1 << 28 | 1 << 24;

/// The single-symbol table plus one that gives up to three symbols for each `WIN` bits: symbols in bytes 0 to 2, bits
/// used in bits 24 to 27, the number of symbols in bits 28 and 29, bit 31 for bits no code starts with. `LONG` (zero)
/// means the window starts a code longer than `WIN`.
struct Multi {
    tab: Box<[u32; 1 << WIN]>,
    one: Box<[u16; TABLE_LEN]>,
    top: u32,
}

fn multi_table(len: &[u8; 256]) -> Result<Multi, String> {
    let (one, top) = decoding_table(len)?;
    let mut tab: Box<[u32; 1 << WIN]> = vec![0u32; 1 << WIN].into_boxed_slice().try_into().map_err(|_| "table")?;
    for (p, slot) in tab.iter_mut().enumerate() {
        let (mut avail, mut used, mut count, mut syms, mut bad) = (WIN, 0u32, 0u32, 0u32, false);
        while count < 3 {
            // the bits still known, padded with zeros to the width of the single-symbol table: a code that is in
            // them whole is the one at that index, whatever the padding is
            let v = p & ((1 << avail) - 1);
            let e = one[if avail >= top { v >> (avail - top) } else { v << (top - avail) }];
            if e & (1 << 12) != 0 {
                bad = count == 0;
                break;
            }
            let l = u32::from(e & 15);
            if l > avail {
                break;
            }
            syms |= u32::from(e >> 4) << (8 * count);
            avail -= l;
            used += l;
            count += 1;
        }
        *slot = if bad {
            BAD
        } else if count == 0 {
            LONG
        } else {
            syms | used << 24 | count << 28
        };
    }
    Ok(Multi { tab, one, top })
}

impl Lane<'_> {
    /// Room for a step of `step_multi` without any check: up to three symbols for each lookup, and a four byte store.
    #[inline(always)]
    fn can_multi(&self) -> bool {
        self.at + 3 * STEPS < self.out.len() && (self.bit >> 3) + 8 <= self.stream.len()
    }

    #[inline(always)]
    fn step_multi(&mut self, m: &Multi) {
        let b = self.bit >> 3;
        let mut w = u64::from_be_bytes(self.stream[b..b + 8].try_into().unwrap_or([0; 8])) << (self.bit & 7);
        let (mut used, mut at, mut bad) = (0usize, self.at, 0u32);
        for _ in 0..STEPS {
            let e = m.tab[(w >> (64 - WIN)) as usize & ((1 << WIN) - 1)];
            if e == LONG {
                let e1 = m.one[(w >> (64 - m.top)) as usize & (TABLE_LEN - 1)];
                bad |= u32::from(e1 >> 12 & 1) << 31;
                self.out[at] = (e1 >> 4) as u8;
                at += 1;
                used += usize::from(e1 & 15);
                break;
            }
            bad |= e;
            w <<= (e >> 24) & 15;
            used += ((e >> 24) & 15) as usize;
            // the fourth byte is not a symbol: it is written over by the next store or by `finish`
            self.out[at..at + 4].copy_from_slice(&e.to_le_bytes());
            at += ((e >> 28) & 3) as usize;
        }
        self.bit += used;
        self.at = at;
        self.bad |= bad >> 31;
    }
}

/// Exactly `n` bytes, or an error on a damaged table or stream.
pub fn decode(coded: &[u8], n: usize) -> Result<Vec<u8>, String> {
    let len = read_lengths(coded)?;
    let mut out = vec![0u8; n];
    if n < MULTI_FROM {
        let (table, top) = decoding_table(&len)?;
        Lane::new(&coded[TABLE_BYTES..], &mut out).finish(&table, top, per_step(top))?;
    } else {
        let m = multi_table(&len)?;
        let mut lane = Lane::new(&coded[TABLE_BYTES..], &mut out);
        while lane.can_multi() && lane.bad == 0 {
            lane.step_multi(&m);
        }
        lane.finish(&m.one, m.top, per_step(m.top))?;
    }
    Ok(out)
}

/// `encode_with` in chunks: the code lengths, then the byte length of each chunk (u32, little endian), then the
/// chunks, each padded to a byte. The number of chunks is `n.div_ceil(CHUNK)` and the caller knows `n`.
pub fn encode_chunked(data: &[u8], t: &Table) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }
    let sym = symbols(&t.len)?;
    let coded: Vec<Vec<u8>> = data.chunks(CHUNK).map(|c| encode_one(c, &sym)).collect();
    let mut out = Vec::with_capacity(TABLE_BYTES + 4 * coded.len() + coded.iter().map(Vec::len).sum::<usize>());
    header(&t.len, &mut out);
    for c in &coded {
        out.extend_from_slice(&(c.len() as u32).to_le_bytes());
    }
    for c in &coded {
        out.extend_from_slice(c);
    }
    Some(out)
}

/// Where each chunk's stream sits in a part made by `encode_chunked`: byte offsets (start, end) into `coded`.
pub(crate) fn chunk_ranges(coded: &[u8], n: usize) -> Result<Vec<(usize, usize)>, String> {
    let count = n.div_ceil(CHUNK);
    let head = TABLE_BYTES + 4 * count;
    if coded.len() < head {
        return Err("huffman chunk lengths cut short".into());
    }
    let mut at = head;
    let mut ranges = Vec::with_capacity(count);
    for i in 0..count {
        let l = u32::from_le_bytes(coded[TABLE_BYTES + 4 * i..][..4].try_into().unwrap_or([0; 4])) as usize;
        let end = at
            .checked_add(l)
            .filter(|&e| e <= coded.len())
            .ok_or("a huffman chunk runs past the part")?;
        ranges.push((at, end));
        at = end;
    }
    if at != coded.len() {
        return Err("a huffman part has bytes after its last chunk".into());
    }
    Ok(ranges)
}

/// The inverse of `encode_chunked`: exactly `n` bytes, or an error on a damaged table or stream.
pub fn decode_chunked(coded: &[u8], n: usize) -> Result<Vec<u8>, String> {
    let len = read_lengths(coded)?;
    let ranges = chunk_ranges(coded, n)?;
    let mut out = vec![0u8; n];
    let multi = if n >= MULTI_FROM { Some(multi_table(&len)?) } else { None };
    let single = if multi.is_none() { Some(decoding_table(&len)?) } else { None };
    let mut lanes: Vec<Lane> = out
        .chunks_mut(CHUNK)
        .zip(&ranges)
        .map(|(dst, &(a, b))| Lane::new(&coded[a..b], dst))
        .collect();
    for group in lanes.chunks_mut(4) {
        if let (Some(m), [a, b, c, d]) = (&multi, &mut *group) {
            while a.can_multi() && b.can_multi() && c.can_multi() && d.can_multi() {
                a.step_multi(m);
                b.step_multi(m);
                c.step_multi(m);
                d.step_multi(m);
            }
        } else if let (Some((table, top)), [a, b, c, d]) = (&single, &mut *group) {
            let per = per_step(*top);
            while a.can(per) && b.can(per) && c.can(per) && d.can(per) {
                a.step(table, *top, per);
                b.step(table, *top, per);
                c.step(table, *top, per);
                d.step(table, *top, per);
            }
        }
        for lane in group {
            match (&multi, &single) {
                (Some(m), _) => lane.finish(&m.one, m.top, per_step(m.top))?,
                (_, Some((table, top))) => lane.finish(table, *top, per_step(*top))?,
                _ => {}
            }
        }
    }
    drop(lanes);
    Ok(out)
}

/// Size in bytes `encode` would give, from the counts alone (for choosing a method without coding twice).
pub fn estimate(data: &[u8]) -> usize {
    table(data).size
}
