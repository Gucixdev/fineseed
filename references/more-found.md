# also found while searching (titles and numbers from search results, papers not read)
- FM-Delta (NeurIPS 2024): a pre-trained base stored uncompressed, each fine-tune as an integer delta; 20B fine-tunes 423 GB -> 205 GB.
- TensorDex / TStore / TensorHub (arXiv 2604.17104): tensor-centric storage; 70.5% less on 2,890 Hugging Face models; picks the
  reference model by bit-level similarity because family metadata is missing for most repos -> take the AUTO BASE idea.
- Brevis (arXiv 2608.02162, Aug 2026): a tensor program synthesized per tensor; up to 30.87% smaller than ZipNN and DFloat11.
  Far later; shows the ceiling is above plain planes+zstd.
- DFloat11: entropy-coded exponents of BF16 with GPU decode (the idea behind exponent Huffman).
