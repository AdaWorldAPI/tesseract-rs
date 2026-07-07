# CLAUDE.md — tesseract-rs

Read first, every session. The repo's commits + PRs are the durable record of
prior sessions; **this file is the awareness that would otherwise reset with the
session** — the rules, the proven method, and what's next.

## What this is

A **pure-Rust transcode** of Tesseract OCR — NOT a binding. The antimatter15 FFI
wrapper (`tesseract-sys` / `tesseract-plumbing`) was deleted 2026-06-18 per the
operator directive: *transcode Tesseract into Rust, do NOT wrap libtesseract;
delete the C++ residue.* Virtual workspace; the OCR is rebuilt leaf-by-leaf, each
leaf **byte-parity-proven against the C++ original before it lands.**

## Core-First doctrine (non-negotiable)

Transcoded logic lives in the **OGAR Core** = `lance-graph-contract` (sibling repo
`../lance-graph`). `tesseract-core` **CONSUMES** it; it never re-implements. The
char set is `CharSet = lance_graph_contract::unicharset::UniCharSet`. A new
primitive is shaped + proven in the Core, then surfaced here with a
consumer-boundary test. **Never build a parallel object model here.**
Full doctrine: `../lance-graph/.claude/knowledge/core-first-transcode-doctrine.md`.

## What's shipped (all byte-parity vs libtesseract on real `eng` data)

| Primitive | Proven in Core (EPIPHANIES) | Parity | Surfaced here |
|---|---|---|---|
| `UNICHARSET` id↔unichar | E-CPP-PARITY-1 | 112/112 | `CharSet::{id_to_unichar,unichar_to_id}`, `ids_to_text` |
| `UNICHAR` UTF-8 codec | E-CPP-PARITY-2 | 268/268 | `unichar::{utf8_step,utf8_to_utf32}` |
| properties | E-CPP-PARITY-3 | 112/112 | `CharSet::get_is{alpha,lower,upper,digit,punctuation,ngram}` |
| script table (interned) | E-CPP-PARITY-4 | 112/112 | `CharSet::{get_script,script_of,get_script_table_size,...}` |
| other_case (case pair) | E-CPP-PARITY-5 | 112/112 | `CharSet::get_other_case` |
| direction + mirror | E-CPP-PARITY-6 | 112/112 | `CharSet::{get_direction,get_mirror}` |
| recoder (`UNICHARCOMPRESS` load side) | E-CPP-PARITY-7 | 112 enc + 112 dec | `Recoder`, `recoded_to_text` (codes→ids→text) |

`ids_to_text` (the recognizer's id→text walk) is the first OCR-facing step in
`tesseract-core`; `recoded_to_text` is the recoder-fed variant (codes→decode→ids→text).
Cross-ref the Core's `EPIPHANIES.md` E-CPP-PARITY-1..7 +
E-CPP-KEYSTONE-1 (classid→ClassView→adapter dispatch).

## The proven method — self-validating oracle

Each leaf is proven this way (the `/tmp` artifacts are ephemeral — rebuild them):

1. C++ source: `AdaWorldAPI/Tesseract` (this arc used `/tmp/tesseract`, **5.5.0**).
2. Build a tiny oracle that dumps BOTH the id↔unichar **bijection** (a proven
   112/112 reference) AND the new field, linking the installed `-ltesseract`:
   `g++ -std=c++17 oracle.cpp -I<src>/src/ccutil -I<src>/include -I/usr/include/leptonica $(pkg-config --cflags --libs tesseract) $(pkg-config --libs lept)`.
   Namespace in 5.5.0: `using tesseract::UNICHARSET;`.
3. **ABI-skew gotcha:** the in-env lib is **5.3.4**, the source headers **5.5.0**,
   and no tesseract dev headers are installed. Mixing them is unsafe — so the
   oracle dumps the bijection too: if the bijection diff is **0**, the object
   layout is sound for the fields read and the new field's diff is trustworthy.
   Always check the bijection half first.
4. Rust side (committed, durable): `cargo run -p lance-graph-contract --example
   unicharset_dump -- <unicharset> {properties|script|other_case}`; `diff` the two.
   eng data = a trained `eng.lstm-unicharset` (`combine_tessdata -u`).

## Iron rules (learned this arc — do not relearn the hard way)

1. **NEVER `cargo --all` / `--all-targets` / `cargo fmt --all` from this repo.**
   `tesseract-core` path-deps `lance-graph-contract`, so `--all` follows the path
   INTO the lance-graph workspace and rebuilds/reformats ~30 unrelated files (a
   real disaster this session). **Always scope `-p tesseract-core`.** CI
   (`.github/workflows/rust.yml`) is already scoped and sibling-checks-out
   lance-graph.
2. **Consume the Core, never re-implement.** A needed primitive that doesn't exist
   → add it to `lance-graph-contract`, prove it there, surface here.
3. **Board hygiene lands in lance-graph** (where the Core change is): EPIPHANIES +
   LATEST_STATE. tesseract-rs commits are the consumer wiring + this file.
4. No libtesseract/leptonica at runtime — they are only the *oracle's* link deps,
   never in the Rust path (the unicharset path is pure text, never touches `Pix`).

## Next leaf

**The UNICHARSET *varied-field* surface is COMPLETE** — every field that carries
varied, falsifiable information on the real `eng.lstm-unicharset` is transcoded +
byte-parity-proven 112/112: bijection, properties, script, other_case, direction,
mirror. `direction`/`mirror` were read by continuing the token walk past the
optional bbox+stats CSV (one whitespace token → fixed offsets, no bespoke 5-tier
detector needed), and their green parity **proves the CSV-skip is correct.**

**Deferred (weak falsifier on this data, NOT a gap):** the bbox ints
(`get_top_bottom`), the 6 float stats, and `normed` sit *inside* that CSV. On the
LSTM unicharset they are **uniform** — 111/111 CSV lines are identically
`0,255,0,255,0,0,0,0,0,0` and `normed` ≈ the unichar — so a byte-parity diff would
be all-uniform and prove nothing the CSV-skip hasn't already shown. Transcribing
them is mechanical but should be gated on a **legacy (non-LSTM) `eng.unicharset`
with real bbox/stats** so the diff can actually falsify. (Note `get_top_bottom`'s
out-of-range default is `0,256,0,256` — 256, not 255 — and `set_top_bottom` clips
to `[0,255]`; `unicharset.h:586-606`.)

**The recoder is DONE** (`unicharcompress.{h,cpp}`, load side) — byte-parity
green on real `eng.lstm-recoder` (E-CPP-PARITY-7): `UnicharCompress`
(`DeSerialize` → `from_le_bytes`; `EncodeUnichar`/`DecodeUnichar`/`code_range`)
in `lance-graph-contract`, surfaced here as `Recoder` + `recoded_to_text`
(codes→decode→ids→`ids_to_text`). It was the first BINARY leaf (`TFile` LE; the
1012 B = `4 + 112·9` on-disk size was a first-principles pre-registration of a
correct parse), and `kMaxCodeLen = 9` (the plan summary's "3" was wrong —
Hangul/Han USE length-3, the array is sized 9). The routing verdict held
(content-store tier, NOT `emit_rust`) — re-verified LIVE against OGAR's
SURREAL-AST-TRAP-PREFLIGHT + OGAR-AS-IR §3. `0x08` OCR is now MINTED (OGAR #148:
`recoder`=0x0802, mirrored in `ogar_codebook`), so the recoder keystone
(`invoke_recoder`, the E-CPP-KEYSTONE-1 analog) is unblocked but deferred — the
`classid→ClassView→content` dispatch is already proven generically.

**The recognizer is UNDERWAY — Leaves 1-6 shipped** (`tesseract-recognizer`, the
COMPUTE tier — a NEW crate, deps `ndarray`). `matrix_dot_vector` transcodes the
base int8 `IntSimdMatrix::MatrixDotVector` by consuming
`ndarray::simd_runtime::matmul_i8_to_i32` (the hardware acceleration — the
recognizer NEVER re-implements SIMD, per the `simd-savant` "all SIMD from
`ndarray::simd`" invariant); byte-parity green vs libtesseract on synthetic
int8, two shapes (`E-OCR-MATDOTVEC-1`, integer-combined diff so it is
`TFloat`-agnostic; the in-env lib is FAST_FLOAT). The **two-foundations** split
is now real: `tesseract-recognizer` (deps ndarray) = compute, `tesseract-core`
(deps lance-graph-contract) = content. **Toolchain: always bump to 1.95** (ndarray
manifest gate); CI sibling-checks-out ndarray now. **Leaf 2 shipped:**
`WeightMatrix::DeSerialize` (int-mode load + f32 `forward`, byte-parity green on
f32 bit-patterns vs libtesseract, `E-OCR-WEIGHTMATRIX-1`). **Leaf 3:** activations
(LUT `tanh`/`logistic` + `relu`/`clip`/`softmax`, byte-parity on a 4096-pt sweep,
`E-OCR-ACTIVATION-1`). **Leaf 4:** `FullyConnected::Forward` (int8 path) =
`activation(WeightMatrix·input)` — the first COMPLETE layer, composing the two
proven halves; byte-parity green across all 7 activations + 2 shapes vs a
libtesseract oracle running the REAL `MatrixDotVector`+`FuncInplace`
(`E-OCR-FULLYCONNECTED-1`; `fully_connected_forward` + `FcActivation`, the
compute-side activation vocab, mapped from the Core `NetworkType` ordinal — no
Core dep). **Leaf 5:** `LSTM::Forward` (1-D int8) — the recurrent layer, the
hardest leaf. `Lstm::from_le_bytes` (`i32 na_` + 4 gate `WeightMatrix`es
CI/GI/GF1/GO, `ns=CI.num_outputs`, `ni=na_−ns`) + `forward`: the 4 gates via
`fully_connected_forward` (CI=tanh, GI/GF1/GO=logistic), cell
`c=clip(GF1·c+CI·GI, ±100)`, output `h=tanh(c)·GO`, and the **int8-quantized
recurrence** (`h`→int8 `clip(round(x·127),±127)` into the next timestep's
source). Byte-parity green across 3 shapes incl. ns=48/ni=36 × 8 timesteps vs a
libtesseract oracle running the REAL `MatrixDotVector`+`FuncInplace`+vector-ops
+`WriteTimeStepPart` quant (`E-OCR-LSTM-1`; no FMA discrepancy — separate mul+add
matches). Added `WeightMatrix::from_le_bytes_prefix` (returns bytes consumed) to
chain the 4 gates. **Leaf 6:** the graph walk — `graph::Layer` (`Lstm` / `FullyConnected`
/ `Reversed` / `Series` / `Parallel`), the compute-side execution tree (the
`invoke_network` counterpart; the Core describes the tree *structure*, this crate
*runs* it). `Series` chains sub-layers with the **inter-layer int8 requant** (the
intermediate NetworkIO is int_mode → `quantize_i8`); `Reversed` (XREVERSED) =
reverse→inner→reverse. Byte-parity green: `Series[LSTM,FC]` across 4 shapes incl.
ns=96/ni=192/no=111 (eng.lstm's LSTM192→Fc111 tail) vs a libtesseract oracle
chaining the REAL per-layer bodies + the REAL `WriteTimeStep` requant
(`E-OCR-GRAPHWALK-1`). **Leaf 7 DONE** — the recognizer now spans **logits →
text**: `7a` = the recoder `SetupDecoder` beam maps (`is_valid_start_`/
`final_codes_`/`next_codes_`) in the Core, byte-parity green (`E-OCR-RECODER-BEAM-1`,
lance-graph PR #647); `7b` = `RecodeBeamSearch::Decode` (the non-dict CTC beam,
`recodebeam.cpp` 1382 lines) in `tesseract-core`, byte-parity green across 4
configs (`E-OCR-RECODEBEAM-1`, tesseract-rs PR #7). So the chain int8 features →
graph forward → softmax logits → beam decode → labels → `recoded_to_text` → string
is complete.

**The 2-D front-end A1-A5 + the network loader B1 are DONE — the recognizer
runs the REAL model image-grid → logits, byte-parity green.** A1-A5
(`tesseract-recognizer`: `NetworkIo`/`StrideMap`/`TRand`, `Convolve`/`Maxpool`/
`Reconfig`/`Txy`) shipped byte-parity on synthetic grids. **B1** is a NEW
assembly crate `tesseract-ocr` (deps BOTH foundations — recognizer for compute +
core for the Core network surface): `Network::from_le_bytes` transcodes
`Network::CreateFromFile` + `Plumbing::DeSerialize` (recursive children +
`learning_rates_` skip when `NF_LAYER_SPECIFIC_LR`), loading the REAL eng.lstm
into a runnable `Node` tree; `Node::forward_io` composes the A1-A5 grid ops + the
proven Leaf-4/5/6 compute. **The full composed forward — Convolve+TRand-noise →
FcTanh → Maxpool → XYTranspose → LstmSummary → Lstm → XReversed → Lstm → Lstm →
FcSoftmax — reproduces libtesseract's `net->Forward` BIT-FOR-BIT** (softmax f32
output; **8/8** synthetic image widths 6..63 incl. odd widths stressing the
ragged Maxpool-3×3/Convolve-3×3/Txy chain; `num_weights` self-check 385807 ==
libtesseract). Header parse is Core-First (the Core's proven `NetworkHeader` /
`E-OCR-NETWORK-SINK-1`; the wire discriminant is the `kTypeNames` **string** after
an `i8` NT_NONE tag, NOT a raw ordinal). Oracle: `/tmp/network_forward_oracle.cpp`
(public-API-only — `CreateFromFile`/`SetRandomizer`/`Forward`, dodges the
5.3.4/5.5.0 ABI skew; source banked in the v2 plan §B1) vs `cargo run -p
tesseract-ocr --example network_dump`. Board: lance-graph `E-OCR-NETWORK-FORWARD-1`.

**B2 is DONE too — the full recognizer loads from disk, byte-parity green.**
`tesseract-ocr/src/lstm_recognizer.rs` (`LstmRecognizer::from_components`)
transcodes `LSTMRecognizer::DeSerialize` for the `include_charsets == false`
split-traineddata path: after the B1 network, the lstm component's 81-byte tail
is `network_str_` + 4×i32 (`training_flags`=65, `training_iteration`,
`sample_iteration`, `null_char`=110) + 3×f32 (`adam_beta`/`learning_rate`/
`momentum`); the unicharset (TEXT) + recoder (binary) load from their own
components (both already `E-CPP-PARITY-1..7`). The 8 trailing-parse fields are
**byte-identical** vs a public-API oracle (`Network::CreateFromFile` +
`TFile::DeSerialize`); assembly cross-checks (network 385807, charset 112,
recoder code_range 111, null 110, int-mode+recoding) all consistent. Board:
lance-graph `E-OCR-RECOGNIZER-LOAD-1`.

**A6a is DONE — the pixel → int8 grid step, byte-parity green.**
`tesseract-recognizer/src/input.rs` (`from_grey_pix`) transcodes
`NetworkIO::FromPix` → `FromPixes`→`Copy2DImage`→`SetPixel` for the 8-bit grey
2-D path (eng): `ComputeBlackWhite` middle-row local-extrema → `STATS(0,255)` →
`black=mins.ile(0.25)`/`white=maxes.ile(0.75)`, then
`clip(round(128·((pixel−black)/contrast−1)), ±127)` (**×128 = INT8_MAX+1, NOT
the ×127 of write_time_step** — a real gotcha). Byte-identical vs a public-API
`FromPix` oracle on **8/8** widths (3..64, incl. odd + the width=3 minimum).
Board: lance-graph `E-OCR-FROMPIX-1`.

**Next: A6b + B3 — close image→text.** The canonical continuation plan (proven
state + the byte-parity method + every remaining leaf with C++ ref / oracle /
crate / order) is **`.claude/plans/recognizer-image-to-text-v2.md`** — START
THERE. **A6b** = image file → decode → `pixConvertTo8` → **`pixScale` to height
36** — the leptonica-coupled commodity front of the front-end; per the founding
directive ("no leptonica at runtime; delete the C++ residue") it is pure-Rust
(image decode via `image`-rs; `pixScale` byte-parity is leptonica's resampling
algorithm — a hard, separate problem — so the pragmatic boundary is a
pre-scaled-8-bit-grey input, with A6a proving the Tesseract-specific
normalization). **B3** = the `RecognizeLine` glue threading A6(a/b) →
`network.forward` (B1, DONE) → `RecodeBeamSearch::Decode` (`E-OCR-RECODEBEAM-1`,
DONE) → `ExtractBestPathAsUnicharIds` (C2, DONE) → `recoded_to_text`
(`E-CPP-PARITY-7`, DONE). Everything except A6b's image decode+scale is proven.
Then dict beam + CJK trie (C1/C3, accuracy). (Still deferred, unchanged: the
bbox/stats sub-leaf, gated on a legacy non-LSTM `eng.unicharset`; the 2-D LSTM /
softmax-LSTM paths — eng.lstm is 1-D non-softmax.)

## Network structure — ruff→OGAR sink onto V3 SoA (Core-side, byte-parity proven)

The recognizer's polymorphic `Network` subclass tree is sunk onto the Core the
**right** way — NOT a hand-rolled `enum NetworkKind` (that draft was rejected as
the parallel-object-model anti-pattern). Operator directive: *"6x8:8, 16 B tenant
= classid + 12 B, ruff>OGAR transpiler sink-in."* Executed:

1. **Harvest** — `ruff/crates/ruff_cpp_spo/examples/harvest_network.rs` (committed)
   walks the 11 network headers via libclang → the `has_function`/
   `virtually_overrides` SPO manifest (62 classes, 5060 triples). The `Forward`
   override set = the compute-leaf list; the `DeSerialize` set = the binary-leaf
   list. This IS the `classid → ClassView` method-resolution table.
2. **Base-header leaf** — `lance_graph_contract::network` (`NetworkType` 27 types +
   `NetworkHeader::from_le_bytes` = the shared prefix `Network::CreateFromFile`
   reads, `network.cpp:214-248`) sinks each node onto `facet::FacetCascade` (16 B
   = classid + 6×8:8, `CascadeShape::G6D2`). `facet_classid =
   compose_classid(network_layer=0x0804, ntype)`. **Byte-parity GREEN** on real
   `/tmp/eng.lstm`: `Series ni=36 no=111 num_weights=385807` == libtesseract
   `Network::CreateFromFile`; oracle `spec()` == the model spec string.
   Oracle `/tmp/network_spec_oracle.cpp` (built `-DFAST_FLOAT`); example
   `network_dump.rs`. Board: EPIPHANIES `E-OCR-NETWORK-SINK-1`.

Deferred: per-subclass payload + tree recursion (Plumbing children → `EdgeBlock`,
weights → out-of-line Lance column); the `invoke_network` keystone; the recognizer
COMPUTE leaves below. Plan: `.claude/plans/network-ruff-ogar-sink-v1.md`. The
recognizer-side binary reader (`crates/tesseract-recognizer/src/io.rs`) is written,
awaiting Leaf 4's Network loader (uncommitted until wired).

## Branch / PR / merge order

This arc's dev branch: `claude/happy-hamilton-0azlw4` → base `master`. **PR #3** =
"pure-Rust transcode workspace + UNICHARSET consumer surface." The companion Core
PR is **lance-graph #556**.

> **Merge #556 (lance-graph) FIRST.** CI here checks out lance-graph's *default
> branch* (main) as the path dep, so the consumer tests (`get_script`,
> `get_other_case`, …) only compile once those accessors are on lance-graph main.
> Expect PR #3 CI to be red until #556 merges.

## Prior art (read before re-exploring)

- `.claude/plans/tesseract-rs-ast-dll-codegen-v1.md` — codegen / adapter-body half.
- `.claude/plans/tesseract-rs-receive-contract-v1.md` — the consume-the-Core contract.
- `.claude/handovers/2026-06-16-*` — cpp-spo corpus + headstone exploration.
