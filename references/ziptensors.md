# ziptensor = Vasyanator/ziptensors (found 2026-11-22, Apache-2.0, README read, code not read)

What it is: lossless `.ziptensors` container for safetensors, GGUF, ONNX, PyTorch zip. "Typically 8 to 33% smaller", every
restored file bit-exact (BLAKE3 of the original; the writer decodes and checks before it replaces anything). Rust core, CLI,
Python module, C ABI, read-only mount. Closest existing thing to what fineseed is for.

Features he lists, and what fineseed does with each:
- per-tensor transform chosen by format: byte-plane split for floats, FIELD split for quantized GGUF blocks (scales apart from
  quants) -> take both; fineseed has planes only; the field split for Q4/Q8 blocks is the missing piece for GGUF.
- independent chunks of about 1 MiB, so one tensor or byte range decodes alone, in parallel -> take (fineseed blocks are 32 MiB
  and carry no index; random access needs an index and smaller chunks).
- one container, many files, dedup by content hash across members; an F32 file that holds only F16 values shares storage with
  its F16 sibling -> take (many models in one archive, like a zip).
- delta blobs: finetune stored as XOR residual against its base, mode max -> have it per block; missing: auto base choice.
- modes (balanced default, max for packs), edit/merge/compact/recover on a container, `--max-total-decoded` budget for
  containers from elsewhere (decompression-bomb guard) -> take the budget and the verify-before-replace habit.
Benchmark against it: not done (installing it was blocked); not run; numbers belong here once it is.
