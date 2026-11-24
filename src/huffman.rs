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
    let len = &t.len;
    let code = codes(len)?;
    let mut out = Vec::with_capacity(t.size.max(TABLE_BYTES) + 8);
    for pair in len.as_chunks::<2>().0 {
        out.push(pair[0] | pair[1] << 4);
    }
    // symbol code and length in one word, so the loop does one load per byte
    let mut sym = [0u32; 256];
    for s in 0..256 {
        sym[s] = u32::from(code[s]) << 8 | u32::from(len[s]);
    }
    // the low `nb` bits of `acc` are pending; 32 of them are written at once
    let (mut acc, mut nb) = (0u64, 0u32);
    for &b in data {
        let e = sym[b as usize];
        let l = e & 0xff;
        acc = acc << l | u64::from(e >> 8);
        nb += l;
        if nb >= 32 {
            out.extend_from_slice(&((acc >> (nb - 32)) as u32).to_be_bytes());
            nb -= 32;
        }
    }
    while nb >= 8 {
        out.push((acc >> (nb - 8)) as u8);
        nb -= 8;
    }
    if nb > 0 {
        out.push(((acc << (8 - nb)) & 0xff) as u8);
    }
    Some(out)
}

/// Exactly `n` bytes, or an error on a damaged table or stream.
pub fn decode(coded: &[u8], n: usize) -> Result<Vec<u8>, String> {
    if coded.len() < TABLE_BYTES {
        return Err("huffman table cut short".into());
    }
    let mut len = [0u8; 256];
    for (i, b) in coded[..TABLE_BYTES].iter().enumerate() {
        len[2 * i] = b & 15;
        len[2 * i + 1] = b >> 4;
    }
    let code = codes(&len).ok_or("huffman lengths overfill the code space")?;
    // entry = symbol << 4 | length, 0 = no code
    let mut table = vec![0u16; 1 << MAX_LEN];
    for s in 0..256 {
        let l = u32::from(len[s]);
        if l == 0 {
            continue;
        }
        let lo = usize::from(code[s]) << (MAX_LEN - l);
        let hi = lo + (1 << (MAX_LEN - l));
        if hi > table.len() {
            return Err("huffman code out of range".into());
        }
        table[lo..hi].fill((s as u16) << 4 | l as u16);
    }
    let stream = &coded[TABLE_BYTES..];
    let mut out = vec![0u8; n];
    // `buf` holds the next bits from its top; bits past `have` are the stream's next bits or zero
    let (mut buf, mut have, mut pos, mut at) = (0u64, 0u32, 0usize, 0usize);
    let mask = (1usize << MAX_LEN) - 1;
    // fast: eight bytes in per refill leave at least 56 bits, so three symbols of at most 15 bits need no checks
    while at + 3 <= n && pos + 8 <= stream.len() {
        let w = u64::from_be_bytes([
            stream[pos],
            stream[pos + 1],
            stream[pos + 2],
            stream[pos + 3],
            stream[pos + 4],
            stream[pos + 5],
            stream[pos + 6],
            stream[pos + 7],
        ]);
        buf |= w >> have;
        let adv = (63 - have) >> 3;
        pos += adv as usize;
        have += adv << 3;
        let mut bad = 0u32;
        for k in 0..3 {
            let e = table[(buf >> (64 - MAX_LEN)) as usize & mask];
            let l = u32::from(e & 15);
            bad |= u32::from(l == 0);
            buf <<= l;
            have = have.wrapping_sub(l);
            out[at + k] = (e >> 4) as u8;
        }
        if bad != 0 {
            return Err("huffman stream damaged or cut short".into());
        }
        at += 3;
    }
    // slow: byte by byte, and a code longer than the bits left is an error
    while at < n {
        while have <= 56 && pos < stream.len() {
            buf |= u64::from(stream[pos]) << (56 - have);
            pos += 1;
            have += 8;
        }
        let e = table[(buf >> (64 - MAX_LEN)) as usize & mask];
        let l = u32::from(e & 15);
        if l == 0 || l > have {
            return Err("huffman stream damaged or cut short".into());
        }
        out[at] = (e >> 4) as u8;
        at += 1;
        buf <<= l;
        have -= l;
    }
    Ok(out)
}

/// Size in bytes `encode` would give, from the counts alone (for choosing a method without coding twice).
pub fn estimate(data: &[u8]) -> usize {
    table(data).size
}
