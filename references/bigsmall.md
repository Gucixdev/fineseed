# bigsmall = wpferrell/Bigsmall (found 2026-11-22; license Elastic-2.0 = source-available: README read, CODE NOT READ, not run)

What it is: lossless model compressor, "about 34% smaller", `.bs` container, md5 per tensor checked on decompress.
Ideas worth taking (ideas are free, code is not):
- delta patches for fine-tunes where "the engine measures both codings per tensor and never ships a delta larger than
  standalone" (fineseed per block does the same).
- FP8 weights also shrink (about 17%).
- "Ferrell Duo": one file holding a small lossy INT4 model AND the lossless residual that rebuilds the exact original, about
  3 points of the raw size more than lossless alone. This is the QStore idea in a product form and the clean way to have a
  lossy mode that never loses the original -> candidate for a lossy mode.
- streaming run of a model bigger than RAM, hardware autopilot -> out of scope (a packer, not a runtime).
- pairs: delta size is pair-dependent, under 1% of full size for the best >=7B SFT pairs up to 61% for small full tunes.
Leave: the runtime, Python, anything under that license.
