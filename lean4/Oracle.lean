import FineSplit
import FineModel

/-!
The oracle of devframework's differential test for `src/split.rs` (and `planes::xor_into`): the Rust harness prints
one line per operation, `OP ARGS => RESULT`, bytes as lowercase hex (`-` for none); this program reads the lines,
recomputes each RESULT from the model (`FineSplit.lean`, `FineModel.lean`) and prints the lines it would have
written. The two outputs must be equal.
-/

open FineSplit FineModel

def hexVal (c : Char) : Nat := if c.isDigit then c.toNat - 48 else c.toNat - 87

def unhex (s : String) : List Nat :=
  if s == "-" then [] else go (s.toList.map hexVal)
where
  go : List Nat → List Nat
    | a :: b :: r => (a * 16 + b) :: go r
    | _ => []

def hexByte (b : Nat) : String :=
  let d := "0123456789abcdef".toList
  String.ofList [d.getD (b / 16) '0', d.getD (b % 16) '0']

def hex (bs : List Nat) : String := if bs.isEmpty then "-" else String.join (bs.map hexByte)

def streamsText (ss : List (List Nat)) : String := " ".intercalate (ss.map hex)

/-- Little endian values of `w` bytes. -/
def values (w : Nat) (d : List Nat) : List Nat :=
  (List.range (d.length / w)).map fun i =>
    (List.range w).foldr (fun k acc => acc * 256 + d.getD (i * w + k) 0) 0

def bytesOf (w : Nat) (x : Nat) : List Nat := (List.range w).map fun k => x / 256 ^ k % 256

def splitKind (k : Nat) (d : List Nat) : Option (List (List Nat)) :=
  if 1 ≤ k ∧ k ≤ 8 then some (planes k d)
  else if k = 10 then
    let vs := (values 2 d).map fun x => bf16Enc (x % 256) (x / 256)
    some [vs.map (·.1), vs.map (·.2)]
  else if k = 11 then
    let vs := (values 2 d).map f16Enc
    some [vs.map (·.1), vs.map (·.2.1), vs.map (·.2.2)]
  else if k = 12 then
    let vs := (values 4 d).map f32Enc
    some [vs.map (·.1), vs.map (·.2.1), vs.map (·.2.2.1), vs.map (·.2.2.2)]
  else none

def mergeKind (k n : Nat) (ss : List (List Nat)) : Option (List Nat) :=
  let g := fun i j => (ss.getD i []).getD j 0
  if 1 ≤ k ∧ k ≤ 8 then some (unplanes k ss n)
  else if k = 10 then
    some ((List.range n).flatMap fun i => let p := bf16Dec (g 0 i) (g 1 i); [p.1, p.2])
  else if k = 11 then
    some ((List.range n).flatMap fun i => bytesOf 2 (f16Dec (g 0 i) (g 1 i) (g 2 i)))
  else if k = 12 then
    some ((List.range n).flatMap fun i => bytesOf 4 (f32Dec (g 0 i) (g 1 i) (g 2 i) (g 3 i)))
  else none

/-- `p0.4,b4.12,n16.128`: the fields of a block type. -/
def parseFields (s : String) : Option (List Field) :=
  (s.splitOn ",").mapM fun w =>
    match w.toList with
    | c :: rest =>
      match (String.ofList rest).splitOn "." |>.filterMap String.toNat? with
      | [o, l] => if c = 'p' then some (.planes o l) else if c = 'b' then some (.bytes o l)
                  else if c = 'n' then some (.nibs o l) else none
      | _ => none
    | [] => none

def nums (ws : List String) : List Nat := ws.filterMap String.toNat?

def answer : List String → Option String
  | ["split", k, d] =>
    match nums [k] with
    | [k] => (splitKind k (unhex d)).map streamsText
    | _ => none
  | "merge" :: k :: n :: ss =>
    match nums [k, n] with
    | [k, n] => (mergeKind k n (ss.map unhex)).map hex
    | _ => none
  | ["qsplit", size, fields, d] =>
    match nums [size], parseFields fields with
    | [s], some fs => some (streamsText (splitQuant s fs (unhex d)))
    | _, _ => none
  | "qmerge" :: size :: fields :: n :: ss =>
    match nums [size, n], parseFields fields with
    | [s, n], some fs =>
      some ((mergeQuant s fs (ss.map unhex) n).elim "err" hex)
    | _, _ => none
  | ["xor", a, b] => some (hex (xorBlock (unhex a) (unhex b)))
  | ["blocks", b, e, u, k] =>
    match nums [b, e, u, k] with
    | [b, e, u, k] =>
      let step := max (k / max u 1) 1 * max u 1
      let n := e - b
      let ls := lens step n n
      some (if ls.isEmpty then "-" else " ".intercalate (ls.map toString))
    | _ => none
  | _ => none

partial def loop (h : IO.FS.Stream) : IO Unit := do
  let line ← h.getLine
  if line.isEmpty then return
  match line.trimAscii.toString.splitOn " => " with
  | [lhs, _] =>
    match answer ((lhs.splitOn " ").filter (· ≠ "")) with
    | some r => IO.println s!"{lhs} => {r}"
    | none => IO.eprintln s!"bad line: {line}"; IO.Process.exit 1
  | _ => IO.eprintln s!"bad line: {line}"; IO.Process.exit 1
  loop h

def main : IO Unit := do loop (← IO.getStdin)
