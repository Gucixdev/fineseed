# fineseed model proofs

Lean 4 model of what `src/split.rs` does to the bytes of a block (cut them into streams and put them back) and of how
a range is cut into blocks. The pack is lossless only if putting the streams back gives the block back whatever the
coder does with the streams in between, so that is what is proved.

## Proved (`FineSplit.lean`)

| theorem | says |
| --- | --- |
| `bf16_roundtrip`, `f16_roundtrip`, `f32_roundtrip` | Cutting a float into exponent, sign and mantissa bytes and joining them again gives the same bytes, for every bf16, f16 and f32 value. |
| `bf16_bytes`, `f16_bytes`, `f32_bytes` | Every stream byte is a byte (below 256). |
| `nib_roundtrip`, `nib_symbols`, `nib_dec_roundtrip` | A byte as two 4-bit symbols and back is the byte; both symbols are below 16; two symbols below 16 join to a byte and split back to the same two. |
| `merge_split`, `split_lengths`, `bf16_stream`, `nib_stream` | If one value survives its own cut, a whole stream of values does; the streams have the length of the input. |
| `xor_cancel`, `xor_length`, `xor_byte` | Storing `block ^ reference` and xoring the reference in again returns the block; the result is bytes. |
| `lens_sum` | The blocks a range is cut into add up to the range: no byte lost or added. |
| `lens_bounds`, `lens_full`, `lens_unit` | Every block is non-empty and at most one step; all but the last are full; every block is a whole number of units when the step and the range are. |

Axioms: `propext`, `Classical.choice`, `Quot.sound` at most. No `sorry`, no `native_decide`, no `bv_decide`.

## Tested, not proved (`FineModel.lean`, `Oracle.lean`)

The byte-plane layout and the field layouts of the ggml quantized blocks are index bookkeeping. `FineModel.lean`
computes them the other way round from `split.rs` (for each stream, which byte it holds, instead of writing each
field out), and the Rust code equals it on 200 runs of 1000 operations (100 seeds, release build and debug build with
overflow checks) plus random seeds for 10 more seconds, including damaged nibble streams that must be refused and the
cutting of ranges into blocks. The Rust source is compiled from `split.rs` and `planes.rs` as they stand.

Five hand-made breaks of `split.rs` (a mask, a shift, the nibble order, the block step, the nibble check) were each
caught at the first seed. A wrong *table* of block layouts is not caught here, because the harness passes the table to
the oracle; `covers()` and the roundtrip tests in `tests/formats.rs` check the table.

## Run

```bash
./check    # lake build, axiom audit, source scan
./run      # devframework's differential test of split.rs against FineModel.lean
```

`./run` uses the steps of devframework (`src/test/steps.sh`, path in `$DEVFRAMEWORK`).
