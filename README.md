# fineseed

A lossless packer for AI model files. Whatever goes in comes back **byte for byte**; every unpack is checked against a
BLAKE3 hash of the original and fails loudly instead of handing you a wrong file.

It exists because model files are a bad fit for zip: the numbers inside are 2-byte floats whose first byte (sign and exponent)
repeats a lot and whose second byte is noise. fineseed pulls those apart, codes each part on its own, and also looks for what
several models share.

## Use

One file kind, `.fseed`, for one model or many, like a zip:

```
fineseed pack model.safetensors                      # -> model.safetensors.fseed
fineseed pack models/ extra.onnx -o all.fseed        # many files or folders in one archive
fineseed pack all.fseed new-model.gguf -o more.fseed # an .fseed is opened: its members share blocks with the rest
fineseed list more.fseed                             # every entry, nested ones indented
fineseed unpack more.fseed -o out/                   # every file, nested archives built again, each checked
fineseed check more.fseed                            # unpack to a scratch folder, verify, delete it
```

`-l N` sets the zstd level (1 to 22, default 3). Quantizing is not what this does: it only compresses.

## What it does

* **Formats.** SafeTensors, GGUF (v2, v3) and ONNX (initializers) are read tensor by tensor, with bounds on every length.
  Every other file (TensorFlow data files, TFLite, a PyTorch zip, npz, anything) is cut into blocks and each block is coded as
  bfloat16, float16, float32 or plain bytes, whichever a 16 KiB sample of it says (a fast zstd stands in for the coder, so
  text and code are not mistaken for weights). It still comes back exact, and a file that is not a model costs only the look.
* **Split by what the bytes are.** BF16, F16 and F32 are cut so the exponent sits apart from the mantissa, integers by byte
  plane, and every quantized ggml type this knows (Q4_0 to Q8_0, the K-quants Q2_K to Q6_K, the IQ types and TQ1_0/TQ2_0) by its fields (scales, mins, nibbles). A wrong
  guess only costs ratio, never correctness: the split is bijective and every byte is covered exactly once.
* **Own Huffman coder** next to zstd and "stored raw": the smallest of the three wins per stream.
* **Against a base.** Each tensor is also tried as `block XOR block of the most similar earlier tensor`.
* **Archive of many models.** `pack` keeps a block identical to an earlier one (from any member, any format) only once, and
  codes a tensor against the most similar earlier tensor of the same type and size, from any earlier member. So a GGUF of a
  model already in the archive as SafeTensors costs almost nothing. The table of contents is at the end; `list` reads only that.
* **Nesting.** A `.fseed` among the inputs is opened and its members become members of the new archive, so they share blocks
  and references with everything else (a model that sits in two inner archives is stored once). Unpack builds the inner file again
  byte for byte; packing proves it can (it builds every inner archive again and compares), and an inner file that would not
  come back identical, because another version wrote it, is kept as plain bytes instead. Up to 16 levels.
* **Library.** `fineseed::api` is bytes in, bytes out (`compress`/`decompress` for a whole file, `encode_block`/`decode_block`
  for one tensor with an optional reference), no files and no global state, hostile input gives an `Error`, never a panic.
* **Proofs.** `lean4/` proves the split, merge, XOR and block cutting round-trip, and runs the Rust code against the Lean
  oracle through the devframework differential test (see `lean4/README.md`).

## Measured

| input | size | result |
|---|---|---|
| Goedel-Prover BF16 safetensors (4 shards) | 16.4 GB | **66.6 %** (the floor for BF16 weights on their own is about 65 to 66 %) |
| the same model as GGUF Q4_K_M | - | 95.9 % (quantized weights are near-random; entropy coding alone cannot do more) |
| the same model as GGUF IQ3_XS | - | 97.1 % |
| safetensors + the GGUF of the same model, one archive | 32.8 GB | **33.3 %**, 3906 blocks stored once, restored and checked |

zstd alone on BF16 gives about 77 %. 24 s to pack and 18 s to unpack 1.6 GB on 6 threads.

## Not done yet

Extracting one member without restoring the others, random access into a packed file, a mantissa-given-exponent context coder
for the last half point on BF16, readers for the formats' own metadata (TFLite and TensorFlow index files) so their tensors are cut
exactly instead of found by sampling.

## Build and test

```
cargo build --release
cargo test --release
```

Rust 2024, dependencies: `blake3`, `serde_json`, `zstd`.
