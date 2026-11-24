# fineseed

**Like zip, but made for AI model files.** Same weights in, the exact same bytes out, just smaller. Every unpack is checked
against a BLAKE3 hash of the original and fails loudly instead of handing you a wrong file.

```
fineseed pack Qwen3-1.7B.safetensors     # 4.06 GB -> 2.29 GB (56 %)
fineseed unpack Qwen3-1.7B.safetensors.fseed
```

**Why not just zip it?** A BF16 weight is two bytes: the first (sign and exponent) repeats a lot, the second is noise. zip
mixes them (zstd alone gets about 77 %). fineseed pulls them apart, codes each on its own, and lands at 56 to 67 %. Keep the same
model in several formats, quantizations or folders and it stores the shared part once: **Goedel-8B as safetensors and as
GGUF costs 10.9 GB, not 32.8 GB raw.**

## Try it

```
fineseed pack models/ extra.onnx -o all.fseed   # many files or folders in one archive
fineseed list all.fseed                         # what is inside
fineseed cat all.fseed models/a.gguf > a.gguf   # ONE file out, nothing else is decoded
fineseed extract all.fseed models/a.gguf -o out/
fineseed mount all.fseed /mnt/models            # browse it as a read-only folder (FUSE)
fineseed unpack all.fseed -o out/               # everything, each file verified
fineseed check all.fseed                        # unpack to scratch, verify, clean up
```

Streams work too: `curl ... | fineseed pack - --dtype bf16 -o dump.fseed` (also `-o -` for stdout). `-l N` sets the
zstd level (1 to 22, default 3); `--fast` skips zstd on numbers. It never quantizes, it only compresses, losslessly.

Protect an archive against bit rot (Reed-Solomon parity, like par2 but inside the file):

```
fineseed protect all.fseed --parity 5    # +5 %: survives damaged 64 KiB pieces anywhere
fineseed scrub all.fseed                 # check every piece against its CRC32
fineseed repair all.fseed                # rebuild damaged pieces in place
```

## Measured

Real runs of `fineseed pack`, default settings, sizes in decimal GB, "packed" is the `.fseed` as a share of the input.

| model | format | size | packed |
|---|---|---|---|
| Qwen3-1.7B | safetensors BF16 | 4.06 GB | 2.29 GB, **56.3 %** |
| Qwen2.5-3B | safetensors BF16 | 6.17 GB | 4.13 GB, **66.9 %** |
| Goedel-Prover-8B | safetensors BF16 | 16.38 GB | 10.91 GB, **66.6 %** |
| Goedel-Prover-8B | GGUF Q4_K_M | 5.03 GB | 4.82 GB, **95.9 %** |
| Goedel-Prover-8B | GGUF IQ3_XS | 3.63 GB | 3.52 GB, **97.1 %** |

Quantized GGUF is close to random bits: 3 to 4 % is all there is. BF16 is where it pays.

| many models in one archive | input | packed |
|---|---|---|
| Goedel-8B as safetensors **and** GGUF (same weights) | 32.77 GB | 10.92 GB, **33.3 %** (2nd copy costs 2 MB) |
| Goedel-8B: safetensors BF16, GGUF BF16, Q4_K_M, IQ3_XS | 41.42 GB | 18.73 GB, **45.2 %** |
| three unrelated models | 26.62 GB | 17.33 GB, **65.1 %** (nothing to share, nothing lost) |

**Against the others, on BF16** (not the same machine or data, so read it as a ballpark):

| | size | speed |
|---|---|---|
| zstd -3 (measured here, 512 MiB) | 78.0 % | - |
| ZipNN (its README, one thread) | 66.3 % | 1120 MB/s pack, 1660 MB/s unpack |
| DFloat11 (its README: "about 30 % smaller") | about 70 % | GPU decode |
| fineseed (128 MiB synthetic BF16) | 66.2 % | one core: pack 320 MB/s (350 with `--fast`), unpack 670 |

On one file fineseed is where ZipNN is: the mantissa is noise (the two bytes alone have 66.5 % of entropy). Side by side
here (i7-8700, one core, SD 1.5 VAE in fp16): ZipNN 0.5.4 gives 85.0 % at 147 MB/s pack and 772 unpack, fineseed 85.4 %
at 208 and 374 (hash check and file writes included). ZipNN has no archive: shared blocks across formats and models. Two
different fine-tunes do not share much: SD 1.5 and dreamshaper-8 together come out 2.3 % under the two packed alone.

## What it does

* **Formats.** SafeTensors, GGUF (v2, v3) and ONNX are read tensor by tensor, with bounds on every length. Zip containers
  (DDUF, PyTorch `.pt`) are opened and each stored entry is read as a file of its own. Anything else (TensorFlow, TFLite,
  npz, ...) is cut into blocks and each block is typed as BF16, F16, F32 or plain bytes from a 16 KiB sample.
* **Split by what the bytes are.** BF16, F16, F32 by exponent and mantissa, integers by byte plane, and the ggml quant
  types (Q4_0 to Q8_0, K-quants, IQ, TQ) by their fields. A wrong guess only costs ratio, never correctness.
* **Own Huffman coder** next to zstd and "stored raw" (smallest wins per stream); each tensor is also tried as
  `block XOR block of the most similar earlier tensor`, which pays when a model is a small step from its base.
* **Many models, one file.** Identical blocks are stored once across members and formats; a tensor is coded against the
  most similar earlier one. A `.fseed` among the inputs is opened and nested (up to 16 levels), rebuilt byte for byte.
* **Random access.** `cat`, `extract` and `mount` decode only the blocks a read touches, plus what those refer to.
* **Library and proofs.** `fineseed::api` is bytes in, bytes out, hostile input gives an `Error`, never a panic. `lean4/`
  proves split, merge, XOR and block cutting round-trip and diffs the Rust code against the Lean oracle.

## Not done
A mantissa-given-exponent context coder for BF16; anything KV-cache specific; FP8 splitting (no gain); TFLite and
TensorFlow index readers. The parity header cannot be repaired. `mount` shows a nested `.fseed` as a folder. **GPU:**
`--features gpu` decodes the Huffman chunks with wgpu (Vulkan), checked against the CPU. On a GTX 1050 Ti it does 1 to
1.5 GB/s of one plane with the transfers, so 12 CPU threads beat it, and no command uses it yet.

## Build and test
```
cargo build --release          # `mount` needs fusermount3; --no-default-features skips it
cargo test --release
```

Rust 2024, MIT. Dependencies: `blake3`, `serde_json`, `zstd`, `crc32fast`, `reed-solomon-simd`, `fuser` (`mount`, optional).
