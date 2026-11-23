# zipnn (verified: arXiv 2411.05239, IBM; repo zipnn/zipnn, license NOASSERTION: read nothing until it is clear)

What it is: lossless compressor for model files. Idea: the exponent byte of weights is very skewed (about 12 of 256 values
make up 99.9%), so regroup bytes by position ("byte grouping"), code the exponent group with Huffman or zstd, and leave the
mantissa group alone when it does not compress. Reports about 33% smaller BF16 models, and over half for "clean" models.
A 2025 follow-up extends it to FP8 and FP4.

Take: byte grouping per dtype (fineseed `planes.rs` is this idea written from the paper's description, before reading it);
leave a plane raw when the coder gains nothing; Huffman-only on the exponent plane for speed.
Gap: fineseed uses zstd for every plane. An own Huffman coder for 8-bit planes is the "wlasna kompresja" part.
