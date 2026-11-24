# examples

Small measuring tools, not part of the program. Each takes a model file: `cargo run --release --example NAME FILE`.

| example | what it shows |
|---|---|
| `layout` | what the reader finds in a model file: tensors, kinds, whether they tile the data area |
| `entropy` | bits per BF16 value for several ways of modelling the biggest tensor, to see what is worth building |
| `fragments` | `api::encode_auto` on pieces cut at random places (64 to 256 KiB), against zstd level 3 |
| `huff` | Huffman encode and decode speed per plane, one thread, single stream against chunks |
| `phases` | time of each phase of coding one BF16 block: split, table, encode, decode, merge, BLAKE3, zstd |
| `phases16` | the same for unpacking F16 data (three planes), decode best of five |
| `gpu_huff` | chunked Huffman decode on the GPU against the CPU; needs `--features gpu` |
