# zipllm (verified: arXiv 2505.06252, NSDI 2026; code github.com/ds2-lab/ZipLLM, Apache-2.0, not read yet)

What it is: a store for many LLMs. Deduplicates by TENSOR (boundaries from safetensors, not content-defined chunks), then
compresses what is left with BitX: XOR a fine-tune against its base model, then zstd or pcodec. Reports 54.1% less storage
on 3,048 sampled models. Finding behind it: fine-tunes of one family differ by sparse, structured parameter deltas.

Take: tensor hash as the unit of dedup across files; XOR delta against a base (fineseed has this per block, tried against
the plain coding and kept only if smaller); pcodec as a coder to try on the XORed numbers.
Leave: the multi-model service around it.
Not in fineseed yet: dedup of identical tensors across files; picking the base automatically.
