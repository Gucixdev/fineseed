/-!
# What `src/split.rs` and `src/lib.rs` do to bytes, as arithmetic, with the proofs

A byte is a `Nat` below 256. The Rust code takes a block of bytes apart into streams (`split`), codes each stream,
and on unpack puts the streams back (`merge`). For the archive to be lossless, `merge (split x) = x` for every `x`
whatever else happens to the streams in between. This file proves that for the float splits and the 4-bit split,
and for the cutting of a range into blocks; the byte-plane and quantized-block layouts, which are index
bookkeeping, are replayed against the Rust code by the differential test (`Oracle.lean`, `./run`).
-/

set_option linter.unusedVariables false

namespace FineSplit

/-! ## One value -/

/-- BF16 (little endian bytes `lo`, `hi`): the exponent as one byte, then sign and the 7 low mantissa bits. -/
def bf16Enc (lo hi : Nat) : Nat × Nat := ((hi % 128) * 2 + lo / 128, hi / 128 * 128 + lo % 128)
def bf16Dec (e s : Nat) : Nat × Nat := ((e % 2) * 128 + s % 128, s / 128 * 128 + e / 2)

theorem bf16_roundtrip {lo hi : Nat} (hlo : lo < 256) (hhi : hi < 256) :
    bf16Dec (bf16Enc lo hi).1 (bf16Enc lo hi).2 = (lo, hi) := by
  simp only [bf16Enc, bf16Dec, Prod.mk.injEq]
  omega

theorem bf16_bytes {lo hi : Nat} (hlo : lo < 256) (hhi : hi < 256) :
    (bf16Enc lo hi).1 < 256 ∧ (bf16Enc lo hi).2 < 256 := by
  simp only [bf16Enc]
  omega

/-- F16 (value `x`): the high byte (sign, exponent, top 2 mantissa bits), then the low byte. -/
def f16Enc (x : Nat) : Nat × Nat := (x / 256, x % 256)
def f16Dec (h l : Nat) : Nat := h * 256 + l

theorem f16_roundtrip {x : Nat} (hx : x < 65536) :
    f16Dec (f16Enc x).1 (f16Enc x).2 = x := by
  simp only [f16Enc, f16Dec]
  omega

theorem f16_bytes {x : Nat} (hx : x < 65536) :
    (f16Enc x).1 < 256 ∧ (f16Enc x).2 < 256 := by
  simp only [f16Enc]
  omega

/-- F32 (value `x`): exponent, sign with the top 7 mantissa bits, the middle and the low mantissa byte. -/
def f32Enc (x : Nat) : Nat × Nat × Nat × Nat :=
  (x / 8388608 % 256, x / 2147483648 * 128 + x / 65536 % 128, x / 256 % 256, x % 256)
def f32Dec (e s m1 m0 : Nat) : Nat := s / 128 * 2147483648 + e * 8388608 + s % 128 * 65536 + m1 * 256 + m0

theorem f32_roundtrip {x : Nat} (hx : x < 4294967296) :
    f32Dec (f32Enc x).1 (f32Enc x).2.1 (f32Enc x).2.2.1 (f32Enc x).2.2.2 = x := by
  simp only [f32Enc, f32Dec]
  omega

theorem f32_bytes {x : Nat} (hx : x < 4294967296) :
    (f32Enc x).1 < 256 ∧ (f32Enc x).2.1 < 256 ∧ (f32Enc x).2.2.1 < 256 ∧ (f32Enc x).2.2.2 < 256 := by
  simp only [f32Enc]
  omega

/-- A byte of a 4-bit field as two symbols, low then high. -/
def nibEnc (b : Nat) : Nat × Nat := (b % 16, b / 16)
def nibDec (lo hi : Nat) : Nat := lo + hi * 16

theorem nib_roundtrip {b : Nat} (hb : b < 256) : nibDec (nibEnc b).1 (nibEnc b).2 = b := by
  simp only [nibEnc, nibDec]
  omega

/-- Both symbols are below 16, which is what `merge` insists on when it reads them back. -/
theorem nib_symbols {b : Nat} (hb : b < 256) : (nibEnc b).1 < 16 ∧ (nibEnc b).2 < 16 := by
  simp only [nibEnc]
  omega

/-- The other way: two symbols below 16 make a byte below 256 and split back into the same two. -/
theorem nib_dec_roundtrip {lo hi : Nat} (hlo : lo < 16) (hhi : hi < 16) :
    nibDec lo hi < 256 ∧ nibEnc (nibDec lo hi) = (lo, hi) := by
  simp only [nibDec, nibEnc, Prod.mk.injEq]
  omega

/-! ## A whole stream of values -/

/-- Cut every value in two parts and collect the parts in two streams. -/
def splitWith (enc : α → β × γ) (vs : List α) : List β × List γ := (vs.map enc).unzip

/-- Put the streams back together value by value. -/
def mergeWith (dec : β → γ → α) (p : List β × List γ) : List α := (p.1.zip p.2).map fun q => dec q.1 q.2

/-- Whatever the values are, if one value survives its own cut, the whole stream does. -/
theorem merge_split {enc : α → β × γ} {dec : β → γ → α} {vs : List α}
    (h : ∀ v ∈ vs, dec (enc v).1 (enc v).2 = v) : mergeWith dec (splitWith enc vs) = vs := by
  unfold mergeWith splitWith
  rw [List.zip_unzip, List.map_map]
  conv => rhs; rw [← List.map_id vs]
  apply List.map_congr_left
  intro v hv
  simpa using h v hv

/-- The two streams have the length of the input. -/
theorem split_lengths (enc : α → β × γ) (vs : List α) :
    (splitWith enc vs).1.length = vs.length ∧ (splitWith enc vs).2.length = vs.length := by
  simp [splitWith]

/-- BF16 values given as `(lo, hi)` byte pairs. -/
theorem bf16_stream {vs : List (Nat × Nat)} (h : ∀ v ∈ vs, v.1 < 256 ∧ v.2 < 256) :
    mergeWith bf16Dec (splitWith (fun v => bf16Enc v.1 v.2) vs) = vs := by
  apply merge_split
  intro v hv
  obtain ⟨h1, h2⟩ := h v hv
  have := bf16_roundtrip h1 h2
  simpa using this

/-- 4-bit symbols of a stream of bytes. -/
theorem nib_stream {vs : List Nat} (h : ∀ v ∈ vs, v < 256) :
    mergeWith nibDec (splitWith nibEnc vs) = vs := by
  apply merge_split
  intro v hv
  exact nib_roundtrip (h v hv)

/-! ## XOR against a reference block -/

/-- Byte-wise xor of two blocks: `planes::xor_into`. -/
def xorBlock (a b : List Nat) : List Nat := List.zipWith (· ^^^ ·) a b

/-- Applying the same reference twice gives the block back, so a block stored as `block ^ reference` is restored by
xoring the reference in again. -/
theorem xor_cancel : ∀ {a b : List Nat}, a.length = b.length → xorBlock (xorBlock a b) b = a
  | [], [], _ => rfl
  | x :: a, y :: b, h => by
    have ih := xor_cancel (a := a) (b := b) (by simpa using h)
    simp only [xorBlock, List.zipWith_cons_cons] at ih ⊢
    rw [ih, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

theorem xor_length (a b : List Nat) : (xorBlock a b).length = min a.length b.length := by
  simp [xorBlock]

theorem xor_byte {a b : Nat} (ha : a < 256) (hb : b < 256) : a ^^^ b < 256 := by
  have : (256 : Nat) = 2 ^ 8 := by decide
  rw [this] at *
  exact Nat.xor_lt_two_pow ha hb

/-! ## Cutting a range into blocks -/

/-- The block lengths of `n` bytes cut by steps of `step` (`split::block_lens`); `fuel` bounds the number of blocks. -/
def lens (step : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | fuel + 1, n => if n = 0 then [] else min n step :: lens step fuel (n - min n step)

/-- No byte is lost or added: the blocks add up to the range. -/
theorem lens_sum {step : Nat} (hs : 0 < step) : ∀ {fuel n : Nat}, n ≤ fuel → (lens step fuel n).sum = n
  | 0, n, h => by simp [lens]; omega
  | fuel + 1, n, h => by
    unfold lens
    by_cases hn : n = 0
    · simp [hn]
    · simp only [hn, ite_false, List.sum_cons]
      have ih := lens_sum hs (fuel := fuel) (n := n - min n step) (by omega)
      rw [ih]
      omega

/-- Every block is non-empty and at most `step` bytes. -/
theorem lens_bounds {step : Nat} (hs : 0 < step) : ∀ {fuel n : Nat}, ∀ l ∈ lens step fuel n, 0 < l ∧ l ≤ step
  | 0, n, l, h => by simp [lens] at h
  | fuel + 1, n, l, h => by
    unfold lens at h
    by_cases hn : n = 0
    · simp [hn] at h
    · simp only [hn, ite_false, List.mem_cons] at h
      rcases h with rfl | h
      · omega
      · exact lens_bounds hs l h

/-- A block is a whole number of `unit`s when the step and the range are. -/
theorem lens_unit {step unit : Nat} (hu : unit ∣ step) : ∀ {fuel n : Nat}, unit ∣ n → ∀ l ∈ lens step fuel n, unit ∣ l
  | 0, n, hn, l, h => by simp [lens] at h
  | fuel + 1, n, hn, l, h => by
    unfold lens at h
    by_cases hz : n = 0
    · simp [hz] at h
    · simp only [hz, ite_false, List.mem_cons] at h
      have hm : unit ∣ min n step := by
        rcases Nat.le_total n step with h1 | h1
        · rw [Nat.min_eq_left h1]; exact hn
        · rw [Nat.min_eq_right h1]; exact hu
      rcases h with rfl | h
      · exact hm
      · exact lens_unit hu (Nat.dvd_sub hn hm) l h

/-- With fuel to spare every block but the last is full, so only the last can be a partial unit. -/
theorem lens_full {step : Nat} (hs : 0 < step) :
    ∀ {fuel n : Nat}, n ≤ fuel → ∀ l ∈ (lens step fuel n).dropLast, l = step
  | 0, n, h, l, hl => by simp [lens] at hl
  | fuel + 1, n, h, l, hl => by
    unfold lens at hl
    by_cases hn : n = 0
    · simp [hn] at hl
    · simp only [hn, ite_false] at hl
      rcases Nat.eq_zero_or_pos (n - min n step) with hm | hm
      · have : lens step fuel (n - min n step) = [] := by
          rw [hm]; cases fuel <;> simp [lens]
        rw [this] at hl
        simp at hl
      · have hf : 0 < fuel := by omega
        obtain ⟨f', rfl⟩ : ∃ f', fuel = f' + 1 := ⟨fuel - 1, by omega⟩
        have hne : lens step (f' + 1) (n - min n step) ≠ [] := by
          unfold lens
          have : n - min n step ≠ 0 := by omega
          simp [this]
        rw [List.dropLast_cons_of_ne_nil hne, List.mem_cons] at hl
        rcases hl with rfl | hl
        · omega
        · exact lens_full hs (by omega) l hl

end FineSplit
