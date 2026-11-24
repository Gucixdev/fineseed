//! The GPU decoder gives the bytes the CPU decoder gives. Skips when this machine has no usable GPU.
#![cfg(feature = "gpu")]

use fineseed::huffman::{CHUNK, decode_chunked, encode_chunked, table};

fn lcg(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 33) as u8
        })
        .collect()
}

#[test]
fn gpu_matches_cpu_and_refuses_damage() {
    let Some(gpu) = fineseed::gpu::Gpu::new() else {
        eprintln!("no GPU, skipped");
        return;
    };
    for n in [1, 5, CHUNK - 1, CHUNK, CHUNK + 3, 7 * CHUNK + 11] {
        for data in [
            lcg(1, n).iter().map(|b| 0x3c + b % 5).collect::<Vec<u8>>(),
            lcg(2, n),
            vec![9; n],
        ] {
            let coded = encode_chunked(&data, &table(&data)).expect("non-empty");
            assert_eq!(decode_chunked(&coded, n).expect("cpu"), data);
            assert_eq!(gpu.decode_chunked(&coded, n).expect("gpu"), data, "n = {n}");
        }
    }
    let data: Vec<u8> = lcg(3, 3 * CHUNK).iter().map(|b| b % 11).collect();
    let coded = encode_chunked(&data, &table(&data)).expect("non-empty");
    assert!(gpu.decode_chunked(&coded[..coded.len() - 40], data.len()).is_err());
}
