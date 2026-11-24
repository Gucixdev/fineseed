// Chunked Huffman decode, one invocation per chunk (see src/huffman.rs, `encode_chunked`).
// `stream` is the whole coded part as bytes, read as big-endian bits at any bit position.
struct Params { top: u32, count: u32, n: u32, chunk: u32 }

@group(0) @binding(0) var<storage, read> stream: array<u32>;
@group(0) @binding(1) var<storage, read> spans: array<u32>;    // per chunk: first byte, end byte
@group(0) @binding(2) var<storage, read> table: array<u32>;    // symbol << 4 | length, bit 12 = unused code
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@group(0) @binding(4) var<storage, read_write> err: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> p: Params;

const SHARED: u32 = 4096u;
var<workgroup> tab: array<u32, 4096>;

fn bswap(x: u32) -> u32 {
    return (x << 24u) | ((x & 0xff00u) << 8u) | ((x >> 8u) & 0xff00u) | (x >> 24u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // the first 4096 table entries (all of them for a code of up to 12 bits) are copied to workgroup memory
    let size = min(1u << p.top, SHARED);
    for (var i = lid.x; i < size; i += 64u) { tab[i] = table[i]; }
    workgroupBarrier();

    let c = gid.x;
    if (c >= p.count) { return; }
    let first = c * p.chunk;
    var left = min(p.chunk, p.n - first);
    var bit = spans[2u * c] * 8u;
    let end = spans[2u * c + 1u] * 8u;
    var o = first / 4u;
    var bad = 0u;
    var cached = 0xffffffffu;
    var a = 0u;
    var b = 0u;
    loop {
        if (left == 0u) { break; }
        let take = min(4u, left);
        var word = 0u;
        for (var k = 0u; k < take; k++) {
            let wi = bit >> 5u;
            if (wi != cached) {
                a = bswap(stream[wi]);
                b = bswap(stream[wi + 1u]);
                cached = wi;
            }
            let sh = bit & 31u;
            var win = a << sh;
            if (sh != 0u) { win |= b >> (32u - sh); }
            let idx = win >> (32u - p.top);
            var e = 0u;
            if (idx < SHARED) { e = tab[idx]; } else { e = table[idx]; }
            bad |= e;
            bit += e & 15u;
            word |= ((e >> 4u) & 255u) << (8u * k);
        }
        out[o] = word;
        o += 1u;
        left -= take;
    }
    if (((bad >> 12u) & 1u) != 0u || bit > end) { atomicStore(&err[0], 1u); }
}
