//! Canonical Huffman coder for one plane of bytes, code lengths limited to 15 bits.
//!
//! Coded form: 128 bytes of code lengths (two symbols per byte, 4 bits each, 0 = symbol absent) followed by the
//! bit stream, most significant bit first, padded with zero bits. The number of symbols is not stored: the caller
//! knows the plane length. `decode` refuses lengths that do not form a prefix code and streams that run dry.

const MAX_LEN: u32 = 15;
pub const TABLE_BYTES: usize = 128;

fn counts(data: &[u8]) -> [u64; 256] {
    let mut c = [0u64; 256];
    for &b in data {
        c[b as usize] += 1;
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

/// The coded bytes, or `None` for an empty plane.
pub fn encode(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }
    let len = lengths(&counts(data));
    let code = codes(&len)?;
    let mut out = Vec::with_capacity(TABLE_BYTES + data.len() / 2);
    for pair in len.as_chunks::<2>().0 {
        out.push(pair[0] | pair[1] << 4);
    }
    let (mut acc, mut bits) = (0u64, 0u32);
    for &b in data {
        let l = u32::from(len[b as usize]);
        acc = acc << l | u64::from(code[b as usize]);
        bits += l;
        while bits >= 8 {
            out.push((acc >> (bits - 8)) as u8);
            bits -= 8;
        }
        acc &= (1 << bits) - 1;
    }
    if bits > 0 {
        out.push((acc << (8 - bits)) as u8);
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
    let mut out = Vec::with_capacity(n);
    let (mut acc, mut bits, mut pos) = (0u64, 0u32, 0usize);
    while out.len() < n {
        while bits < MAX_LEN && pos < stream.len() {
            acc = acc << 8 | u64::from(stream[pos]);
            pos += 1;
            bits += 8;
        }
        // past the end the stream reads as zero bits; a code that needs them is caught below
        let window = if bits >= MAX_LEN {
            (acc >> (bits - MAX_LEN)) as usize & ((1 << MAX_LEN) - 1)
        } else {
            ((acc << (MAX_LEN - bits)) as usize) & ((1 << MAX_LEN) - 1)
        };
        let e = table[window];
        let l = u32::from(e & 15);
        if l == 0 || l > bits {
            return Err("huffman stream damaged or cut short".into());
        }
        out.push((e >> 4) as u8);
        bits -= l;
        acc &= (1 << bits) - 1;
    }
    Ok(out)
}

/// Size in bytes `encode` would give, from the counts alone (for choosing a method without coding twice).
pub fn estimate(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }
    let c = counts(data);
    let len = lengths(&c);
    let bits: u64 = (0..256).map(|s| c[s] * u64::from(len[s])).sum();
    TABLE_BYTES + bits.div_ceil(8) as usize
}
