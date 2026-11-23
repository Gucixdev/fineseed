# qstore (verified: arXiv 2505.04081, PVLDB 19(3) 2025, UIUC)

What it is: storing one model at two precisions. Keep the low-precision model and only the residual needed to rebuild the
high-precision one EXACTLY. Reports up to 2.2x smaller than storing both, and a fast load of the low-precision one.

Take: for a bf16 model and its own quantization (GGUF Q8, Q4) kept side by side: pack the pair as quantized + residual.
This is a bigger idea than the byte planes and needs the quantizer to be reproducible; later phase, after GGUF reading.
Leave: the database setting.
