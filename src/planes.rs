//! Byte planes: the bytes of an array of `w`-byte values, regrouped so that byte `k` of every value is
//! together. The high bytes of floats (sign and exponent) repeat a lot and compress; the low bytes of the
//! mantissa do not, and now sit apart where a coder can leave them alone.

/// `data` is `w * n` bytes; plane `k` holds byte `k` of each value.
pub fn split(data: &[u8], w: usize) -> Vec<Vec<u8>> {
    debug_assert!(w > 0 && data.len().is_multiple_of(w));
    let n = data.len() / w;
    let mut planes = vec![vec![0u8; n]; w];
    for (i, value) in data.chunks_exact(w).enumerate() {
        for (k, byte) in value.iter().enumerate() {
            planes[k][i] = *byte;
        }
    }
    planes
}

/// The inverse of `split`: `out` is `planes.len() * planes[0].len()` bytes.
pub fn merge(planes: &[Vec<u8>], out: &mut [u8]) {
    let w = planes.len();
    for (k, plane) in planes.iter().enumerate() {
        for (i, byte) in plane.iter().enumerate() {
            out[i * w + k] = *byte;
        }
    }
}

/// `a ^= b` over the shorter of the two.
pub fn xor_into(a: &mut [u8], b: &[u8]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= *y;
    }
}
