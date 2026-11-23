import FineSplit

/-!
# The generic layouts, written the other way round

`split.rs` fills the streams by walking the blocks and writing each field out (a scatter). The model below gets
the same streams by asking, for each stream, which byte of which block it holds (a gather), and the merge by asking,
for each byte of a block, which stream and position it comes from. Two different ways to the same bytes: the
differential test (`Oracle.lean`) replays them against the Rust code.
-/

namespace FineModel

/-- A field of a quantized block, as in `split::Field`: byte planes, one byte stream, one 4-bit stream. -/
inductive Field where
  | planes (off len : Nat)
  | bytes (off len : Nat)
  | nibs (off len : Nat)
  deriving Repr

def Field.off : Field → Nat | .planes o _ | .bytes o _ | .nibs o _ => o
def Field.len : Field → Nat | .planes _ l | .bytes _ l | .nibs _ l => l

/-- The blocks of `size` bytes of `data`. -/
def blocks (size : Nat) (data : List Nat) : List (List Nat) :=
  if size = 0 then [] else (List.range (data.length / size)).map fun i => (data.drop (i * size)).take size

/-- The streams of one field. -/
def fieldStreams (bs : List (List Nat)) : Field → List (List Nat)
  | .planes off len => (List.range len).map fun j => bs.map fun b => b.getD (off + j) 0
  | .bytes off len => [bs.flatMap fun b => (b.drop off).take len]
  | .nibs off len => [bs.flatMap fun b => ((b.drop off).take len).flatMap fun x => [x % 16, x / 16]]

/-- All streams of `data` cut into blocks of `size` bytes by `fields`. -/
def splitQuant (size : Nat) (fields : List Field) (data : List Nat) : List (List Nat) :=
  let bs := blocks size data
  fields.flatMap (fieldStreams bs)

def width : Field → Nat | .planes _ l => l | _ => 1

/-- Where a stream that serves byte `p` of a block starts, and the field itself. -/
def fieldAt (fields : List Field) (p : Nat) : Option (Nat × Field) :=
  let rec go : List Field → Nat → Option (Nat × Field)
    | [], _ => none
    | f :: fs, base => if f.off ≤ p ∧ p < f.off + f.len then some (base, f) else go fs (base + width f)
  go fields 0

/-- Byte `p` of block `i`, read from the streams; `none` for a nibble symbol above 15. -/
def byteOf (fields : List Field) (ss : List (List Nat)) (i p : Nat) : Option Nat :=
  match fieldAt fields p with
  | none => some 0
  | some (base, .planes off _) => some ((ss.getD (base + (p - off)) []).getD i 0)
  | some (base, .bytes off len) => some ((ss.getD base []).getD (i * len + (p - off)) 0)
  | some (base, .nibs off len) =>
    let lo := (ss.getD base []).getD ((i * len + (p - off)) * 2) 0
    let hi := (ss.getD base []).getD ((i * len + (p - off)) * 2 + 1) 0
    if lo < 16 ∧ hi < 16 then some (FineSplit.nibDec lo hi) else none

/-- Put `n` blocks of `size` bytes back together; `none` when a nibble symbol is above 15. -/
def mergeQuant (size : Nat) (fields : List Field) (ss : List (List Nat)) (n : Nat) : Option (List Nat) :=
  (List.range n).flatMapM fun i => (List.range size).mapM fun p => byteOf fields ss i p

/-- Planes of `w`-byte values: stream `k` holds byte `k` of every value. -/
def planes (w : Nat) (data : List Nat) : List (List Nat) :=
  (List.range w).map fun k => (List.range (data.length / w)).map fun i => data.getD (i * w + k) 0

def unplanes (w : Nat) (ss : List (List Nat)) (n : Nat) : List Nat :=
  (List.range n).flatMap fun i => (List.range w).map fun k => (ss.getD k []).getD i 0

end FineModel
