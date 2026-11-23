# seedlm (NOT seedpack: seedpack is a lossless GGUF archiver, see seedpack.md)

SeedLM (arXiv 2410.10714, ICLR 2025): each block of weights is stored as the seed of a pseudo-random generator plus a few
coefficients, and rebuilt with an LFSR at inference. It is LOSSY (a quantization method) and data-free.

Take: nothing for the lossless core. If he wants a lossy mode, it is a separate command and a separate decision, and
every lossy step needs a before/after generation check (his lobotomy rule).
