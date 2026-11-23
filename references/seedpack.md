# seedpack = glmn-ai/seedpack / NeuroSquad (found 2026-11-22, Apache-2.0, README read, code not read). NOT SeedLM.

What it is: LOSSLESS archiver for a whole GGUF repository. The BF16 model is entropy-coded once; every quant (IQ1..Q8) is not
stored but "grown": llama.cpp quantizers are deterministic, so the archive reruns the quantization on the GPU from the BF16
(plus the repo's imatrix, with a bit-exact copy of ggml-quants.c) and stores only the difference from the published file,
coded with the quantizer's own probabilities. Claimed (author's own numbers, no independent benchmark): all of
unsloth/Qwen3.5-9B-GGUF 147.81 GB -> 12.40 GB (8.4%); Qwen3-0.6B repo 10.95 GB -> 781.7 MB; a single Q4_K_M only 93.8%,
a single BF16 65.2% (so the whole win is the quants-from-base idea). Dynamic quants that cannot be re-derived exactly store
more (Qwen3.8-27B 9.6%). Needs an NVIDIA GPU (CUDA 13) or a slow `--cpu` mode that compiles the kernels.

For fineseed: the biggest saving by far for a model with many quants, but it needs a bit-exact port of the ggml quantizers
(and imatrix handling) = a project of its own. Phase 3, only if he keeps bf16+quant sets (today: goedel-bf16.gguf alone,
qwen2.5-3b Q4_K_M alone; no pair on disk).
