# Plan — recognizer: image → text (v2, the continuation handover)

**Read this first if you are continuing the Tesseract-transcode arc.** It is
self-contained: current proven state, the byte-parity method (with the hard-won
gotchas), and every remaining leaf with its C++ ref, oracle strategy, crate
placement, and ordering. Supersedes `recognizer-decode-frontend-v1.md` (which was
the original 7a/7b scoping — now EXECUTED); the 2-D-front-end section there is
still valid and expanded here.

**Branch:** develop on `claude/tesseract-recognizer-decode` (the live PR #7
branch) or a fresh `claude/<slug>` off `master` if #7 has merged — the merged-PR
rule: never stack on merged history; restart from the latest default branch.

---

## Where we are — Leaves 1-7 EXECUTED, byte-parity green

The recognizer now spans **logits → text**. Every leaf below is byte-parity-proven
against a live libtesseract **5.3.4** oracle (`-DFAST_FLOAT`, so `TFloat = float`).

| Leaf | What | Crate | EPIPHANIES (in lance-graph board) |
|---|---|---|---|
| 1 | `matrix_dot_vector` (int8 GEMM via `ndarray::simd_runtime::matmul_i8_to_i32`) | `tesseract-recognizer` | `E-OCR-MATDOTVEC-1` |
| 2 | `WeightMatrix::from_le_bytes` + `forward` (int mode) | `tesseract-recognizer` | `E-OCR-WEIGHTMATRIX-1` |
| 3 | activations (tanh/logistic LUT + relu/clip/softmax) | `tesseract-recognizer` | `E-OCR-ACTIVATION-1` |
| 4 | `FullyConnected::Forward` = `activation(W·u)` | `tesseract-recognizer` | `E-OCR-FULLYCONNECTED-1` |
| 5 | `LSTM::Forward` (1-D int8, gates + int8 recurrence) | `tesseract-recognizer` | `E-OCR-LSTM-1` |
| 6 | graph walk (`Series`/`Reversed`/`Parallel` + int8 requant) | `tesseract-recognizer` | `E-OCR-GRAPHWALK-1` |
| — | network structure sink (`NetworkType`, `NetworkHeader`, FacetCascade) | `lance-graph-contract` | `E-OCR-NETWORK-SINK-1` |
| 7a | recoder `SetupDecoder` beam maps (`is_valid_start_`/`final_codes_`/`next_codes_`) | `lance-graph-contract` | `E-OCR-RECODER-BEAM-1` |
| 7b | `RecodeBeamSearch::Decode` (non-dict CTC beam) → `ExtractBestPathAsLabels` | `tesseract-core` | `E-OCR-RECODEBEAM-1` |

Earlier UNICHARSET / UNICHAR / recoder-load leaves: `E-CPP-PARITY-1..7`,
`E-CPP-KEYSTONE-1` (all in `lance-graph-contract`).

**The pipeline that exists today:** int8 feature sequence → `graph::Layer::forward`
(Leaves 1-6) → 111-class softmax logits → `RecodeBeamSearch::decode` +
`extract_best_path_as_labels` (7a maps + 7b) → codes → `recoded_to_text`
(`E-CPP-PARITY-7`) → **string**.

**The gap to image → text:** the 2-D front-end (image `Pix` → int8 feature
sequence) is NOT built. The eng.lstm spec is
`[1,36,0,1[C3,3Ft16]Mp3,3TxyLfys48Lfx96RxLrx96Lfx192Fc111]` — a `Convolve` +
`Maxpool` + `XYTranspose` front-end feeds the 1-D LSTM core that Leaves 5-6 run.

### PR state + merge order (as of this handover)

- **lance-graph #647** — Core side: 7a `SetupDecoder` maps + `RecodedCharId::from_codes` + board. **Merge FIRST.**
- **tesseract-rs #7** — Leaves 5, 6, 7b + review fixes + this plan. CI builds the `lance-graph-contract` path dep against lance-graph `main`, so #7 is red until #647 merges.

---

## The proven method — how every leaf lands (do not deviate)

This is the discipline that made all 7 leaves byte-parity-clean. A fresh session
MUST follow it.

1. **Read the C++ in full first** (the iron rule): the leaf's `.h` + `.cpp`
   method bodies, not snippets. The C++ source for this arc lives at
   **`/tmp/tesseract`** (5.5.0 headers — rebuild the checkout if the `/tmp`
   artifacts are gone: it is the `AdaWorldAPI/Tesseract` mirror). The **installed
   lib is 5.3.4** (`pkg-config --modversion tesseract`), FAST_FLOAT.

2. **Build a self-validating oracle** that links `-ltesseract` and dumps the
   REAL output for a synthetic (or real-eng) input:
   ```sh
   g++ -std=c++17 -DFAST_FLOAT <oracle>.cpp \
     -I/tmp/tesseract/src/lstm -I/tmp/tesseract/src/ccstruct \
     -I/tmp/tesseract/src/ccutil -I/tmp/tesseract/src/dict \
     -I/tmp/tesseract/src/classify -I/tmp/tesseract/src/arch \
     -I/tmp/tesseract/src/viewer -I/tmp/tesseract/src/textord \
     -I/tmp/tesseract/src/cutil -I/tmp/tesseract/include \
     $(pkg-config --cflags tesseract) -o <oracle> \
     $(pkg-config --libs tesseract) $(pkg-config --libs lept)
   ```
   `-DFAST_FLOAT` is MANDATORY (the lib is FAST_FLOAT; omit it and `TFloat`
   becomes double and every f32 diff fails).

3. **Dodge the 5.5.0-header / 5.3.4-lib ABI skew** — the single most important
   lesson. Two safe patterns, in preference order:
   - **Public-API-only oracle (BEST, the 7b pattern):** construct the object via
     its public ctor, call only public methods, dump only public outputs. Never
     read a private member from the oracle TU — if the header's field layout
     disagrees with the lib's, a private read is silent garbage. 7b constructed
     `RecodeBeamSearch(recoder, null_char, simple, nullptr)` and used only
     `Decode` + `ExtractBestPathAsLabels`. Zero layout risk.
   - **Bijection self-check (the recoder pattern):** if you must read state, ALSO
     dump a known-good invariant (the UNICHARSET id↔unichar bijection) in the
     same run; if its diff is 0, the object layout is sound for the fields read.

4. **Shared-input `.bin` for float leaves:** the Rust side GENERATES the synthetic
   input, WRITES it to a `.bin` (LE), runs its transcode, prints the result; the
   oracle READS the same `.bin`. This makes the input byte-identical (no
   cross-language FP generation drift). 7b: `beam_dump.rs` writes `i32 T, i32 N,
   T·N f32 LE`; the oracle reads it. Leaf 2 inverted it (Rust writes the weight
   bytes, libtesseract's `DeSerialize` reads them) — an independent wire-layout
   proof.

5. **Diff must be byte-identical.** f32 dumps use `{:08x}` of `.to_bits()`.
   Non-zero diff = not done. (The `cd /tmp` false-green trap: run the Rust dump
   from the repo dir so cargo has a manifest; verify non-empty line counts.)

6. **Where a leaf lands (the three-tier placement):**
   - **`tesseract-recognizer`** (deps `ndarray`) — SIMD/compute leaves (GEMM,
     WeightMatrix, activations, FC, LSTM, graph, **the whole 2-D front-end**).
     Never re-implement SIMD — consume `ndarray::simd_runtime` (the `simd-savant`
     invariant).
   - **`tesseract-core`** (deps `lance-graph-contract`) — recoder-coupled content
     + SIMD-free decode (recoder surface, `recoded_to_text`, the CTC beam).
   - **`lance-graph-contract`** (the OGAR Core, sibling repo) — pure content
     tables / structure the Core owns (UNICHARSET, UNICHAR, recoder, NetworkType).
     A new Core primitive is shaped + proven THERE, then surfaced in the consumer.
     Board hygiene (EPIPHANIES) lands in lance-graph.

7. **Cargo scope is `-p <crate>`, NEVER `--all` / `--all-targets` at the
   workspace root** — `tesseract-*` path-deps `lance-graph-contract` (which
   path-deps `ndarray`), so `--all` follows INTO those workspaces and
   rebuilds/reformats ~30 unrelated files. Toolchain: **1.95** (ndarray's manifest
   gate). CI already sibling-checks-out ndarray + lance-graph.

8. **Arena for pointer lattices (the 7b pattern):** C++ borrowed-`prev`-pointer
   structures (beam nodes, StrideMap indices) become a `Vec<Node>` + `Option<u32>`
   index in Rust — no `unsafe`, no dangling across `Vec` growth. Read Copy fields
   out into locals before any `arena.push` to satisfy the borrow checker.

---

## Remaining leaves — the arc to image → text

### Phase A — the 2-D front-end (image features → LSTM input)

The eng.lstm front-end: `Input → [C3,3Ft16] (Convolve+Reconfig) → Mp3,3 (Maxpool)
→ Txy (XYTranspose) → …LSTM…`. All are COMPUTE → `tesseract-recognizer`.

- **A1 — `NetworkIO` + `StrideMap` (FOUNDATIONAL, do first).**
  `src/lstm/networkio.{h,cpp}` + `src/lstm/stridemap.{h,cpp}`. The multi-dim
  int8/f32 SoA with the `(batch, y, x) ↔ timestep t` index map + `AddOffset(x,
  FD_WIDTH)` neighbour access. Every 2-D layer needs it; Leaves 5-6 used plain
  `&[&[i8]]` (a 1-D degenerate NetworkIO). **Scope:** the int8 + f32 storage,
  `StrideMap` construction from `(width, height)`, `Index`/`AddOffset`,
  `CopyTimeStepFrom`/`WriteTimeStep`. **Oracle:** build a `NetworkIO` at known
  dims, set values, dump the flat backing store + a few `Index`/`AddOffset`
  lookups; diff. Arena/flat-Vec model (no raw pointers). This is the biggest
  single design leaf — get the StrideMap indexing byte-exact before any layer.

- **A2 — `Convolve::Forward`** (`src/lstm/convolve.{h,cpp}`). Stacks
  `x_scale × y_scale × ni` inputs per output timestep (sliding window) via
  `CopyTimeStepGeneral` + StrideMap offsets → a wider NetworkIO the next layer
  consumes. **Oracle:** synthetic int8 NetworkIO in, compare the convolved
  NetworkIO out.

- **A3 — `Maxpool::Forward`** (`src/lstm/maxpool.{h,cpp}`). Downscale by
  `x_scale × y_scale`, max per window (`MaxpoolTimeStep`); also records the argmax
  for the backward pass (forward-only here). **Oracle:** synthetic in, max-pooled
  out.

- **A4 — `Reconfig::Forward`** (`src/lstm/reconfig.{h,cpp}`). The `Ft` scale-and-
  deepen (stacks windows WITHOUT maxing) — eng's `C3,3Ft16` pairs a Convolve with
  a Reconfig. **Oracle:** synthetic in/out.

- **A5 — `XYTranspose`** (`src/lstm/reversed.{h,cpp}`, `CopyWithXYTranspose`).
  The `Txy` transpose that reorients the grid so the LSTM scans the other axis.
  (Note Leaf 6 already did XREVERSED; this is the 2-D sibling.) **Oracle:**
  synthetic grid in, transposed grid out.

- **A6 — `Input::Forward`** (`src/lstm/input.{h,cpp}`, `networkio.cpp:293`:
  `(INT8_MAX + 1) · pixel`). The image `Pix` → int8 `NetworkIO`. **The biggest
  external-dep leaf — leptonica-heavy.** Defer to its own sub-leaf. **Oracle:**
  a tiny synthetic `Pix` (or a PNG decoded via leptonica in the oracle) → int8
  NetworkIO; the Rust side needs a leptonica-free `Pix` decode (either vendor a
  minimal grayscale reader or gate this leaf on a leptonica-rs decision — raise
  it to the operator; it is the one place the "no leptonica at runtime" rule is
  in tension). Until A6, the front-end is provable on synthetic NetworkIO inputs
  (A1-A5 need no image).

### Phase B — assemble the full network + recognizer

- **B1 — `Network::CreateFromFile` → build the `Layer` tree.**
  `src/lstm/network.cpp:214-248` (the shared header) + `plumbing.cpp` /
  per-subclass `DeSerialize`. Wire the Core's proven network sink
  (`lance_graph_contract::network`, `E-OCR-NETWORK-SINK-1`, which already parses
  the eng.lstm spec into a FacetCascade tree byte-parity green) to the
  recognizer's `graph::Layer` builder: the Core describes the tree STRUCTURE, the
  recognizer INSTANTIATES the runnable `Layer` per node (Series/Parallel/Reversed/
  LSTM/FC/Convolve/Maxpool/…). The per-subclass weight `DeSerialize` (`WeightMatrix`
  Leaf 2 for FC/LSTM; conv/pool params) loads each node's payload. **Oracle:**
  load real `/tmp/eng.lstm`, compare the built tree's `num_weights` / spec string
  vs libtesseract `Network::CreateFromFile` (`E-OCR-NETWORK-SINK-1` already did
  the top level — extend to per-node payloads).

- **B2 — `LSTMRecognizer` load** (`src/lstm/lstmrecognizer.{h,cpp}`,
  `DeSerialize`). The `.lstm` component of `eng.traineddata` bundles: the network
  (B1), the recoder (`E-CPP-PARITY-7` + 7a), the unicharset (`E-CPP-PARITY-1..6`),
  `null_char_`, `training_flags_`, `SimpleTextOutput()`. **null_char note:** it is
  a network-output class (one of the `code_range` codes), NOT `code_range` itself
  (eng's `Fc111` → 111 outputs = `code_range`); the real value is serialized in
  the recognizer, so read it from the loaded `.lstm` rather than guessing.

- **B3 — `RecognizeLine` end-to-end** (`lstmrecognizer.cpp` `RecognizeLine`).
  image `Pix` → `Input::Forward` (A6) → `network_->Forward` (B1 tree) →
  `RecodeBeamSearch::Decode` (7b) → the best-path extract → text. **Two
  milestones, in order:** (i) a **labels-only string** via the already-shipped
  `extract_best_path_as_labels` → `recoded_to_text` (valid for the eng
  single-code recoder, needs NO C2) — the first "reads a line" checkpoint; then
  (ii) the **full `RecognizeLine`** producing words with unichar-ids + certs,
  which calls `ExtractBestPathAsUnicharIds` — so **C2 lands before this second
  milestone** (scheduled in step 4 of the order, NOT with the C1/C3 accuracy
  layer). Oracle: run libtesseract on a real line image, diff the recognized
  string (milestone i) then the per-char certs (milestone ii). This composes
  every leaf.

### Phase C — completeness (C2 is a B3 prereq; C1/C3 are the deferred accuracy waves)

- **C1 — dict / language-model beam.** The dawg machinery skipped in 7b:
  `ContinueDawg`, `PushInitialDawgIfBetter`, `DawgPositionVector`, the
  `is_dawg` beams, `worst_dict_cert`/`dict_ratio` scoring (`recodebeam.cpp:1057-
  1164`). Needs the `Dict` + dawg load (`src/dict/`). Turns the non-dict CTC core
  into dictionary-corrected output. Biggest remaining subsystem after the
  front-end.
- **C2 — `ExtractBestPathAsUnicharIds`** (`recodebeam.cpp:224-329`).
  **SCHEDULED BEFORE B3 (order step 4), not deferred** — B3's full words-with-certs
  output calls it. Groups the best-path codes into complete `RecodedCharId`s →
  `DecodeUnichar` → unichar-ids + certs + ratings + xcoords. Required for
  multi-code (Han/Hangul) text and for per-char confidence. The already-shipped
  `extract_best_path_as_labels` (codes only) is the single-code labels path that
  the B3 milestone (i) uses; C2 is the general one B3 milestone (ii) needs. (The
  `certainty` field was dropped from the beam `RecodeNode` — re-add it when C2
  lands.)
- **C3 — multi-code (CJK) recoder trie exercise.** The `next_codes_` trie (built
  + proven structurally in 7a, but `next_codes_` is empty for eng pass-through)
  needs a non-eng traineddata (e.g. `chi_sim.lstm-recoder`, code length 3) to
  byte-parity the length>1 beam paths. Gate on obtaining that data.
- **Deferred sub-leaves already noted:** the bbox/stats CSV fields
  (`get_top_bottom` + 6 float stats + `normed`) — need a legacy non-LSTM
  `eng.unicharset` with real bbox to falsify (`tesseract-rs/CLAUDE.md` "Next
  leaf"); the 2-D LSTM / softmax-LSTM paths (eng is 1-D non-softmax).

---

## Recommended order + gating

1. **A1 (NetworkIO/StrideMap)** — foundational; nothing 2-D moves without it.
2. **A2-A5 (Convolve/Maxpool/Reconfig/XYTranspose)** — provable on synthetic
   NetworkIO, no image needed. Ship the front-end compute.
3. **B1 (network tree build)** — wire the Core sink to the recognizer graph;
   now the full network runs on a synthetic NetworkIO.
4. **A6 (Input, leptonica) + B2 (recognizer load) + C2 (`ExtractBestPathAsUnicharIds`)
   + B3 (RecognizeLine)** — closes image → text. **C2 is scheduled here, before
   B3, not with the accuracy layer:** B3's full output (words with unichar-ids +
   certs) calls C2, so C2 is a B3 prerequisite, not a deferral. (A minimal
   labels-only RecognizeLine — a plain string via the already-shipped
   `extract_best_path_as_labels` → `recoded_to_text`, valid for the eng
   single-code recoder — is a legitimate intermediate milestone that does NOT
   need C2; the full words-with-certs milestone does.) A6 is the external-dep
   decision point — raise leptonica-rs vs vendored-decoder to the operator
   before starting it.
5. **C1 + C3** — the true accuracy layer, each its own wave: C1 (dict / LM beam)
   and C3 (CJK multi-code trie). Both are optional over a working non-dict,
   single-code RecognizeLine.

Each leaf: read C++ full → oracle `-DFAST_FLOAT` → byte-parity diff → `-p` gates
(`cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`) → commit → EPIPHANIES
entry in lance-graph → push. One leaf per commit; board hygiene in the same PR.

---

## Iron rules (unchanged — repeated so a fresh session cannot miss them)

1. Every leaf byte-parity vs libtesseract 5.3.4 before it lands; oracle built `-DFAST_FLOAT`.
2. Consume the Core, never re-implement; a needed primitive → add to `lance-graph-contract`, prove there, surface here.
3. Never re-implement SIMD — `ndarray::simd_runtime` only (`simd-savant`).
4. Scope cargo `-p <crate>`; NEVER `--all` (path-dep walks into lance-graph/ndarray). Toolchain 1.95.
5. No libtesseract/leptonica at runtime — they are the ORACLE's link deps only. A6 is the one tension point → operator decision.
6. Board hygiene (EPIPHANIES) lands in lance-graph (where the Core change is); tesseract-rs commits are the consumer wiring + the plan.
7. Merged-PR rule: never stack on merged history — restart the branch from the latest default, keep unmerged commits by rebasing them onto the new base.

## Prior art / references (read before re-exploring)

- `recognizer-decode-frontend-v1.md` — the original 7a/7b + front-end scoping (7a/7b now EXECUTED).
- `recognizer-core-shape-v1.md` — the recognizer↔ndarray int8-GEMM seam design.
- `network-ruff-ogar-sink-v1.md` — the network structure → V3 SoA sink (`E-OCR-NETWORK-SINK-1`).
- `tesseract-rs/CLAUDE.md` — the shipped-leaf table, the proven method, iron rules, "Next leaf" (bbox/stats deferral).
- lance-graph `.claude/board/EPIPHANIES.md` — `E-OCR-*` + `E-CPP-PARITY-*` findings (each leaf's proof record).
- `../lance-graph/.claude/knowledge/core-first-transcode-doctrine.md` — the Core-First doctrine.

---

## B1 WIRE-FORMAT FACTS (read 2026-07-05 from /tmp/tesseract source — verbatim, for the loader)

**Per-node header** (`Network::CreateFromFile`, `network.cpp:214-248`; the Core's
proven `NetworkHeader::from_le_bytes` already parses this):
`i8 type` (if == NT_NONE(0): u32-len string type-name looked up in kTypeNames) ·
`i8 training` · `i8 needs_to_backprop` · `i32 network_flags` · `i32 ni` ·
`i32 no` · `i32 num_weights` · `string name` (u32 len + bytes) → then the
subclass `DeSerialize`:

- **Plumbing (Series/Parallel/Reversed — none override it):** `u32 count` +
  `count ×` recursive `CreateFromFile`; THEN `learning_rates_` (f32 vec) ONLY if
  `network_flags & NF_LAYER_SPECIFIC_LR`. A `Reversed` is a Plumbing with 1
  child; its Forward wraps the child with `CopyWithXReversal` (XREVERSED),
  `CopyWithYReversal` (YREVERSED) or `CopyWithXYTranspose` (XYTRANSPOSE) —
  all three NetworkIo ops are A1-proven.
- **Input:** `StaticShape` = 5×i32 `batch,height,width,depth,loss_type`
  (`static_shape.h` DeSerialize).
- **FullyConnected:** one `WeightMatrix` (Leaf-2 proven,
  `WeightMatrix::from_le_bytes_prefix`). Activation = the node's NetworkType
  (NT_TANH/RELU/SOFTMAX/... → FcActivation mapping proven in Leaf 4).
- **Convolve:** `i32 half_x, i32 half_y` (`convolve.cpp:42-51`);
  `no = ni·(2hx+1)·(2hy+1)` recomputed.
- **Reconfig / Maxpool:** `i32 x_scale, i32 y_scale`; Maxpool sets `no = ni`.
- **LSTM** (`lstm.cpp::DeSerialize`): `i32 na_`; `nf_ = no` (LSTM_SOFTMAX) /
  `ceil_log2(no)` (SOFTMAX_ENCODED) / `0` (plain — eng); gates loop `w in
  0..WT_COUNT` SKIPPING `GFS` when `!Is2D()`; after CI: `ns_ = CI.num_outputs`,
  `is_2d_ = (na_ - nf_ == ni_ + 2·ns_)`; if SOFTMAX*: one recursive
  `CreateFromFile` softmax child. eng.lstm is plain 1-D NT_LSTM (+ possibly
  NT_LSTM_SUMMARY nodes — `Lfys48`); Leaf-5 `Lstm::from_le_bytes` matches the
  plain-1-D payload exactly (na + 4 gates CI,GI,GF1,GO).

**B1 design decision (per Core-First + two-foundations + the sweep findings):**
the loader + runnable tree live in a NEW assembly crate `crates/tesseract-ocr`
(deps: `tesseract-recognizer` compute + `tesseract-core` content) — the place
B2 (`LSTMRecognizer` load: network ‖ charset ‖ recoder ‖ null_char) and B3
(`RecognizeLine`) also live, since only the assembly tier sees both foundations.
Node headers parse via the Core's proven `NetworkHeader` (re-export through
tesseract-core like unicharset); payloads via the recognizer's proven parsers.
Tree type: a local `Node` enum (Input/Series/Parallel/Reversed{X,Y,Txy}/
Convolve/Maxpool/Reconfig/Lstm{summary?}/Fc) with
`forward_io(&NetworkIo, &mut TRand) -> NetworkIo` — NOT a contortion of the
1-D `graph::Layer` (which stays the proven 1-D core). STILL TO READ before
coding the walk: `lstm.cpp::Forward`'s NetworkIO framing (src/dest index walks,
the NT_LSTM_SUMMARY final-step-only output via `ResizeXTo1`), `series.cpp::
Forward` (scratch chaining + int-mode inheritance — the Leaf-6 semantic over
NetworkIo), `parallel.cpp::Forward` (CopyPacking), `reversed.cpp::Forward`.
Oracle: per-node spec/num_weights on real `/tmp/eng.lstm` (extend the Core's
proven `network_dump`), then a full-tree `Forward` diff on a synthetic
NetworkIo, then RecognizeLine (B3) on a real line image.

---

## B1 EXECUTED (2026-07-07) — network loader + runnable forward, BYTE-PARITY GREEN

**Shipped** in the NEW assembly crate `crates/tesseract-ocr` (deps both
foundations, exactly as the design decision above called for):

- `src/network.rs` — `Network::from_le_bytes` (= `Network::CreateFromFile` +
  `Plumbing::DeSerialize`) → a runnable `Node` tree; `Node::forward_io` composes
  A1-A5 grid ops + Leaf-4/5/6 compute. `InputShape`, `ReverseKind{X,Y,Txy}`,
  `NetError`.
- `examples/network_dump.rs` — loads real `/tmp/eng.lstm`, prints the tree +
  `nw`/`ni`/`no`, runs a full `forward` over a synthetic grid (width arg), dumps
  `oshape` + per-timestep softmax f32-bit `o` lines; writes the shared
  `net_input.bin`.
- `tesseract-core` re-exports `lance_graph_contract::network` (Core-First
  header surface).

**Two corrections the execution nailed (bank these):**

1. **The type discriminant is the `kTypeNames` STRING, not a raw ordinal.**
   `Network::Serialize` writes `i8 tag = NT_NONE(0)` THEN `string type_name`
   (u32 len + bytes) — the Core's `NetworkHeader::from_le_bytes` parses exactly
   this. A synthetic-byte test that pushes a single type ordinal will NOT parse.
   Mirror the Core's `header_bytes(type_name, ni, no, num_weights, name)` helper.
2. **`NF_LAYER_SPECIFIC_LR` (bit 64) is NOT a reject.** The real eng.lstm outer
   Series carries it; `Plumbing::DeSerialize` reads a trailing `learning_rates_`
   (`u32 count` + `count×f32`) AFTER the children. `skip_layer_lr(cur, flags)`
   reads past it (only after Plumbing + Reversed; leaf nodes never serialize it).

**Byte-parity result:** the full composed forward — Convolve+TRand-noise →
FcTanh → Maxpool → XYTranspose → LstmSummary → Lstm → XReversed → Lstm → Lstm →
FcSoftmax — is **bit-identical** to libtesseract's `net->Forward` on **8/8**
synthetic image widths (6, 8, 11, 17, 24, 31, 40, 63 — odd widths stress the
ragged Maxpool-3×3 / Convolve-3×3 / Txy chain). `num_weights` self-check
385807 == libtesseract; softmax f32 output. Board: lance-graph
`E-OCR-NETWORK-FORWARD-1`.

**Build/run the parity (the `/tmp` artifacts are ephemeral — rebuild):**
```sh
# Rust side (writes net_input.bin + the o-lines):
cargo run -q -p tesseract-ocr --example network_dump -- /tmp/eng.lstm /tmp/net_input.bin 24 > /tmp/rust_net.tsv
# Oracle (public-API only — dodges the 5.3.4-lib / 5.5.0-header ABI skew):
g++ -std=c++17 -DFAST_FLOAT /tmp/network_forward_oracle.cpp \
    -I/tmp/tesseract/src/lstm -I/tmp/tesseract/src/ccutil -I/tmp/tesseract/src/ccstruct \
    -I/tmp/tesseract/include $(pkg-config --cflags --libs tesseract lept) \
    -o /tmp/network_forward_oracle
/tmp/network_forward_oracle /tmp/eng.lstm /tmp/net_input.bin > /tmp/oracle_net.tsv
diff <(grep '^o' /tmp/oracle_net.tsv) <(grep '^o' /tmp/rust_net.tsv)   # empty => green
```

### Oracle source (banked — `/tmp/network_forward_oracle.cpp`, public-API only)
```cpp
// Byte-parity oracle for the FULL network-tree forward pass vs the Rust
// transcode's `Network::forward` (tesseract-ocr/examples/network_dump.rs).
//
// Loads the real eng.lstm via the REAL public `Network::CreateFromFile`,
// builds the same synthetic int8 input grid (read from the shared
// net_input.bin, written in exact StrideMap walk order), seeds the REAL
// `TRand` identically (seed 1, no warm-up draw), and calls the REAL
// polymorphic `Network::Forward` — which dispatches through Series/
// Convolve/Maxpool/Txy(Reversed)/LSTM/FullyConnected exactly as the C++
// library always has. Dumps the same `oshape` / `o` lines as the Rust side
// for a byte-identical diff.
//
//   ./network_forward_oracle [eng.lstm] [net_input.bin]
#include "network.h"
#include "networkio.h"
#include "networkscratch.h"
#include "stridemap.h"
#include "helpers.h"
#include "serialis.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace tesseract;

int main(int argc, char **argv) {
  const char *lstm_path = argc > 1 ? argv[1] : "/tmp/eng.lstm";
  const char *bin_path = argc > 2 ? argv[2] : "/tmp/net_input.bin";

  // ---- Load the network (the REAL recursive Network::CreateFromFile) ----
  std::vector<char> data;
  if (!LoadDataFromFile(lstm_path, &data)) {
    fprintf(stderr, "load fail: %s\n", lstm_path);
    return 1;
  }
  TFile fp;
  if (!fp.Open(data.data(), data.size())) {
    fprintf(stderr, "TFile::Open failed\n");
    return 1;
  }
  Network *net = Network::CreateFromFile(&fp);
  if (!net) {
    fprintf(stderr, "CreateFromFile failed\n");
    return 1;
  }
  printf("nw\t%d\n", net->num_weights());
  printf("ni\t%d\tno\t%d\n", net->NumInputs(), net->NumOutputs());

  // ---- Load the shared input .bin: i32 batch,height,width,depth then the
  // f32s in the exact StrideMap walk order (written by the Rust side). ----
  FILE *bf = fopen(bin_path, "rb");
  if (!bf) {
    fprintf(stderr, "open %s failed\n", bin_path);
    return 1;
  }
  auto read_i32 = [&](int32_t *v) {
    if (fread(v, sizeof(int32_t), 1, bf) != 1) {
      fprintf(stderr, "bin truncated (header)\n");
      exit(1);
    }
  };
  int32_t batch = 0, height = 0, width = 0, depth = 0;
  read_i32(&batch);
  read_i32(&height);
  read_i32(&width);
  read_i32(&depth);

  StrideMap map;
  std::vector<std::pair<int, int>> hw = {{static_cast<int>(height), static_cast<int>(width)}};
  map.SetStride(hw);

  NetworkIO input;
  input.ResizeToMap(true, map, depth);

  {
    StrideMap::Index idx(map);
    std::vector<float> vals(static_cast<size_t>(depth));
    do {
      int t = idx.t();
      if (depth > 0 &&
          fread(vals.data(), sizeof(float), static_cast<size_t>(depth), bf) !=
              static_cast<size_t>(depth)) {
        fprintf(stderr, "bin truncated at t=%d\n", t);
        exit(1);
      }
      input.WriteTimeStep(t, vals.data());
    } while (idx.Increment());
  }
  fclose(bf);

  // ---- Randomizer: seed 1, NO warm-up draw — matches the Rust side exactly.
  // Set BEFORE Forward: Convolve pulls out-of-image noise from it. Plumbing
  // ::SetRandomizer recurses into every child (Series/Reversed/...), so one
  // call at the root reaches the Convolve node wherever it sits in the tree.
  TRand trand;
  trand.set_seed(1);
  net->SetRandomizer(&trand);

  // ---- Forward: the REAL polymorphic dispatch through the whole tree. ----
  NetworkScratch scratch;
  NetworkIO output;
  net->Forward(false, input, nullptr, &scratch, &output);

  // ---- Dump: byte-identical format to the Rust side's oshape/o lines. ----
  printf("oshape\t%d\t%d\t%d\t%d\t%d\n", output.stride_map().Size(FD_BATCH),
         output.stride_map().Size(FD_HEIGHT), output.stride_map().Size(FD_WIDTH),
         output.Width(), output.NumFeatures());
  for (int t = 0; t < output.Width(); ++t) {
    printf("o\t%d", t);
    if (output.int_mode()) {
      const int8_t *row = output.i(t);
      for (int f = 0; f < output.NumFeatures(); ++f) {
        printf("\t%d", static_cast<int>(row[f]));
      }
    } else {
      const float *row = output.f(t);
      for (int f = 0; f < output.NumFeatures(); ++f) {
        uint32_t u;
        float v = row[f];
        std::memcpy(&u, &v, 4);
        printf("\t%08x", u);
      }
    }
    printf("\n");
  }
  return 0;
}
```

---

## B2 EXECUTED (2026-07-07) — LSTMRecognizer load, trailing-field parity GREEN

**Shipped** in `crates/tesseract-ocr/src/lstm_recognizer.rs` (`LstmRecognizer` +
`RecognizerError`) + `examples/lstm_recognizer_dump.rs`. `from_components(lstm,
unicharset_text, recoder)` = `LSTMRecognizer::DeSerialize` for the
`include_charsets == false` path (how `combine_tessdata -u` split eng.lstm):
after `Network::from_le_bytes` (B1) the lstm component's 81-byte tail is
`network_str_` (u32-len string) + **4×i32** (`training_flags_`,
`training_iteration_`, `sample_iteration_`, `null_char_`) + **3×f32**
(`adam_beta_`, `learning_rate_`, `momentum_`); the unicharset (TEXT,
`UniCharSet::load_from_str`, `E-CPP-PARITY-1..6`) + recoder (binary,
`UnicharCompress::from_le_bytes`, `E-CPP-PARITY-7`) come from their own
components.

**Byte-parity GREEN** on real `/tmp/eng.lstm`: the 8 trailing-parse lines
byte-identical vs a public-API oracle (`Network::CreateFromFile` +
`TFile::DeSerialize` in the exact `lstmrecognizer.cpp:144-166` order — no private
access, no ABI skew):
```
netstr [1,36,0,1Ct3,3,16Mp3,3Lfys48Lfx96Lrx96Lfx192O1c1]
tflags 65   titer 6352400   siter 6352704   null 110
abeta 3f7fbe77   lrate 3a83126f   moment 3f000000
```
`training_flags=65 = TF_INT_MODE(1) | TF_COMPRESS_UNICHARSET(64)` → `is_int_mode`
+ `is_recoding` both true (eng is an int8 recoded LSTM). Assembly cross-checks
(not in the parity diff — each already proven): network `num_weights=385807`,
charset `size=112`, recoder `code_range=111`, `null_char=110` — all consistent
with B1 + E-CPP-PARITY-1..7. `null_char=110` is exactly the beam null the
`RecodeBeamSearch` (`E-OCR-RECODEBEAM-1`) expects. Board: lance-graph
`E-OCR-RECOGNIZER-LOAD-1`.

**Build/run the parity:**
```sh
cargo run -q -p tesseract-ocr --example lstm_recognizer_dump -- \
    /tmp/eng.lstm /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder > /tmp/rust_lstmrec.tsv
g++ -std=c++17 -DFAST_FLOAT /tmp/lstm_recognizer_oracle.cpp \
    -I/tmp/tesseract/src/lstm -I/tmp/tesseract/src/ccstruct -I/tmp/tesseract/src/ccutil \
    -I/tmp/tesseract/src/arch -I/tmp/tesseract/src/viewer -I/tmp/tesseract/src/dict \
    -I/tmp/tesseract/src/classify -I/tmp/tesseract/include \
    $(pkg-config --cflags tesseract) -o /tmp/lstm_recognizer_oracle \
    $(pkg-config --libs tesseract) $(pkg-config --libs lept)
/tmp/lstm_recognizer_oracle /tmp/eng.lstm > /tmp/oracle_lstmrec.tsv
diff <(grep -E '^(netstr|tflags|titer|siter|null|abeta|lrate|moment)' /tmp/oracle_lstmrec.tsv) \
     <(grep -E '^(netstr|tflags|titer|siter|null|abeta|lrate|moment)' /tmp/rust_lstmrec.tsv)   # empty => green
```

**Next: A6 + B3 — the leptonica image front-end.** `RecognizeLine`
(`lstmrecognizer.cpp:236-291`): image `Pix` → `Input::Forward` (**A6** — the
leptonica decision point, raise to operator: pure-Rust image decode vs a
leptonica dep) → `network_->Forward` (B1 tree, DONE) → the softmax logits →
`RecodeBeamSearch::Decode` (`E-OCR-RECODEBEAM-1`, DONE) →
`ExtractBestPathAsUnicharIds` (C2, DONE) → `recoded_to_text` (`E-CPP-PARITY-7`,
DONE). So B3 is: the leptonica `Input` (A6) + the `RecognizeLine` glue that
threads the already-proven pieces. Everything downstream of the image grid is
proven; A6 is the last unproven leaf and the one architectural fork.

### Oracle source (banked — `/tmp/lstm_recognizer_oracle.cpp`, public-API only)
```cpp
// Byte-parity oracle for `LSTMRecognizer::DeSerialize`'s trailing-field read
// (recognizer leaf B2), vs the Rust transcode.
//
// Loads the real eng.lstm via the REAL public `Network::CreateFromFile` (this
// advances the TFile cursor past the whole network tree, exactly as
// LSTMRecognizer::DeSerialize does at lstmrecognizer.cpp:135), then reads the
// 8 trailing fields via the REAL `TFile::DeSerialize` overloads, in the exact
// order lstmrecognizer.cpp:144-166 reads them (the `include_charsets == false`
// path — no inline unicharset/recoder, matching how /tmp/eng.lstm was split
// out of a traineddata):
//
//   network_str_ (std::string) -> training_flags_ (i32) ->
//   training_iteration_ (i32) -> sample_iteration_ (i32) ->
//   null_char_ (i32) -> adam_beta_ (f32) -> learning_rate_ (f32) ->
//   momentum_ (f32)
//
//   ./lstm_recognizer_oracle [eng.lstm]
#include "network.h"
#include "serialis.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace tesseract;

int main(int argc, char **argv) {
  const char *lstm_path = argc > 1 ? argv[1] : "/tmp/eng.lstm";

  // ---- Load the network (the REAL recursive Network::CreateFromFile) ----
  std::vector<char> data;
  if (!LoadDataFromFile(lstm_path, &data)) {
    fprintf(stderr, "load fail: %s\n", lstm_path);
    return 1;
  }
  TFile fp;
  if (!fp.Open(data.data(), data.size())) {
    fprintf(stderr, "TFile::Open failed\n");
    return 1;
  }
  Network *net = Network::CreateFromFile(&fp);
  if (!net) {
    fprintf(stderr, "CreateFromFile failed\n");
    return 1;
  }

  // ---- Read the 8 trailing fields, exact order of
  // LSTMRecognizer::DeSerialize (lstmrecognizer.cpp:144-166). ----
  std::string network_str;
  if (!fp.DeSerialize(network_str)) {
    fprintf(stderr, "DeSerialize(network_str_) failed\n");
    return 1;
  }
  int32_t training_flags = 0;
  if (!fp.DeSerialize(&training_flags)) {
    fprintf(stderr, "DeSerialize(training_flags_) failed\n");
    return 1;
  }
  int32_t training_iteration = 0;
  if (!fp.DeSerialize(&training_iteration)) {
    fprintf(stderr, "DeSerialize(training_iteration_) failed\n");
    return 1;
  }
  int32_t sample_iteration = 0;
  if (!fp.DeSerialize(&sample_iteration)) {
    fprintf(stderr, "DeSerialize(sample_iteration_) failed\n");
    return 1;
  }
  int32_t null_char = 0;
  if (!fp.DeSerialize(&null_char)) {
    fprintf(stderr, "DeSerialize(null_char_) failed\n");
    return 1;
  }
  float adam_beta = 0.0f;
  if (!fp.DeSerialize(&adam_beta)) {
    fprintf(stderr, "DeSerialize(adam_beta_) failed\n");
    return 1;
  }
  float learning_rate = 0.0f;
  if (!fp.DeSerialize(&learning_rate)) {
    fprintf(stderr, "DeSerialize(learning_rate_) failed\n");
    return 1;
  }
  float momentum = 0.0f;
  if (!fp.DeSerialize(&momentum)) {
    fprintf(stderr, "DeSerialize(momentum_) failed\n");
    return 1;
  }

  // ---- Dump: tab-separated, f32 as raw LE bit-pattern hex. ----
  uint32_t abeta_bits, lrate_bits, moment_bits;
  std::memcpy(&abeta_bits, &adam_beta, 4);
  std::memcpy(&lrate_bits, &learning_rate, 4);
  std::memcpy(&moment_bits, &momentum, 4);

  printf("netstr\t%s\n", network_str.c_str());
  printf("tflags\t%d\n", training_flags);
  printf("titer\t%d\n", training_iteration);
  printf("siter\t%d\n", sample_iteration);
  printf("null\t%d\n", null_char);
  printf("abeta\t%08x\n", abeta_bits);
  printf("lrate\t%08x\n", lrate_bits);
  printf("moment\t%08x\n", moment_bits);
  return 0;
}
```

---

## A6a EXECUTED (2026-07-07) — NetworkIO::FromPix (pixel → int8 grid), byte-parity GREEN

**Shipped** in `crates/tesseract-recognizer/src/input.rs` (`from_grey_pix` +
private `Stats`/`compute_black_white`) + a `set_pixel` method on `NetworkIo` +
`examples/from_pix_dump.rs`. Transcodes `NetworkIO::FromPix` →
`FromPixes`→`Copy2DImage`→`SetPixel` (`networkio.cpp:127-297`) for the 8-bit
**grey 2-D path** (eng `[1,36,0,1…]`, depth=1, height=36>1):
- `ComputeBlackWhite` — middle row (y=height/2) local minima/maxima → two
  `STATS(0,255)` histograms → `black=mins.ile(0.25)`, `white=maxes.ile(0.75)`.
- `STATS::ile` — bucket walk + linear interp (`statistc.cpp:172-197`), exact.
- `SetPixel` — `clip(round((INT8_MAX+1)·((pixel−black)/contrast − 1)), ±127)`.
  **The ×(INT8_MAX+1)=×128 constant is distinct from `write_time_step`'s ×127**
  — a real gotcha; reusing the Leaf-5 quantizer here would be silently wrong.

**Byte-parity GREEN** on **8/8** synthetic image widths (3, 5, 12, 17, 24, 33,
48, 64 — width=3 is the min for the `width>=3` extrema branch; odd widths
included) vs a public-API oracle (`NetworkIO::FromPix` on a leptonica `Pix`
built from the shared `frompix_input.bin`; `set_int_mode(true)` first, matching
`RecognizeLine` `lstmrecognizer.cpp:345`). Up to 2304 int8 cells per case,
all identical.

**t-order note (the structural risk, verified):** `Copy2DImage` walks y-outer /
x-inner with a plain `t++`; the recognizer `StrideMap` packs **width innermost**
(`t_increments[WIDTH]=1`, `t(0,y,x)=y·W+x`), so linear `t++` tracks `index.t()`
exactly — no per-index recompute needed.

**A6b (the leptonica fork) is deliberately OUT of this leaf** and is the one
architectural decision the founding directive settles toward pure-Rust ("no
leptonica at runtime; delete the C++ residue"): image file → decode → depth-
convert (`pixConvertTo8`) → **`pixScale` to target height 36**. `pixScale`
byte-parity is leptonica's resampling algorithm, NOT a Tesseract algorithm — a
separate, hard commodity problem. The pragmatic boundary: the consumer supplies
a pre-decoded 8-bit grey image at height 36 (via `image`-rs for decode; the
scale is either a documented approximation or a later dedicated leaf), and A6a
proves the Tesseract-specific normalization exactly. **B3** (`RecognizeLine`
glue) then threads A6a → `network.forward` (B1) → beam decode
(`E-OCR-RECODEBEAM-1`) → `ExtractBestPathAsUnicharIds` (C2) → `recoded_to_text`
(`E-CPP-PARITY-7`). Board: lance-graph `E-OCR-FROMPIX-1`.

**Build/run the parity:**
```sh
cargo run -q -p tesseract-recognizer --example from_pix_dump -- /tmp/frompix_input.bin 24 > /tmp/rust_frompix.tsv
g++ -std=c++17 -DFAST_FLOAT /tmp/frompix_oracle.cpp \
    -I/tmp/tesseract/src/lstm -I/tmp/tesseract/src/ccstruct -I/tmp/tesseract/src/ccutil \
    -I/tmp/tesseract/src/arch -I/tmp/tesseract/src/viewer -I/tmp/tesseract/src/dict \
    -I/tmp/tesseract/src/classify -I/tmp/tesseract/include \
    $(pkg-config --cflags tesseract) -o /tmp/frompix_oracle \
    $(pkg-config --libs tesseract) $(pkg-config --libs lept)
/tmp/frompix_oracle /tmp/frompix_input.bin > /tmp/oracle_frompix.tsv
diff /tmp/oracle_frompix.tsv /tmp/rust_frompix.tsv   # empty => green
```

### Oracle source (banked — `/tmp/frompix_oracle.cpp`, public-API only)
```cpp
// Byte-parity oracle for the recognizer leaf A6a: `NetworkIO::FromPix` (the
// image-pixel -> int8 grid step) vs the Rust transcode's FromPix path in
// tesseract-recognizer.
//
// Reads a synthetic 8-bit grey image from a shared .bin (i32 width, i32
// height, then height*width row-major u8 pixels — row 0 left-to-right, then
// row 1, ...), builds a REAL leptonica Pix at that size, puts the NetworkIO
// into int8 mode (matches LSTMRecognizer::RecognizeLine, lstmrecognizer.cpp:345
// `inputs->set_int_mode(IsIntMode());` called BEFORE
// Input::PreparePixInput -> NetworkIO::FromPix, input.cpp:142), and calls the
// REAL public `NetworkIO::FromPix(StaticShape, Image, TRand*)`
// (networkio.cpp:163) which dispatches to FromPixes -> ComputeBlackWhite ->
// Copy2DImage -> SetPixel exactly as the C++ library always has. Dumps the
// resulting int8 grid for a byte-identical diff against the Rust side.
//
// The pix is built at the network's target height (36) with shape.width()==0
// (dynamic), so FromPix does NOT scale and Copy2DImage's fixed-width
// clip/pad-with-noise branch never fires: this is the pure
// FromPixes -> Copy2DImage -> SetPixel normalization path, no randomizer
// draws expected.
//
//   ./frompix_oracle [frompix_input.bin]
#include "networkio.h"
#include "static_shape.h"
#include "stridemap.h"
#include "helpers.h"

#include <allheaders.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace tesseract;

int main(int argc, char **argv) {
  const char *bin_path = argc > 1 ? argv[1] : "/tmp/frompix_input.bin";

  // ---- Read the shared input .bin: i32 width, i32 height, then
  // height*width row-major u8 grey pixels (written by the Rust side). ----
  FILE *bf = fopen(bin_path, "rb");
  if (!bf) {
    fprintf(stderr, "open %s failed\n", bin_path);
    return 1;
  }
  auto read_i32 = [&](int32_t *v) {
    if (fread(v, sizeof(int32_t), 1, bf) != 1) {
      fprintf(stderr, "bin truncated (header)\n");
      exit(1);
    }
  };
  int32_t width = 0, height = 0;
  read_i32(&width);
  read_i32(&height);
  if (width <= 0 || height <= 0) {
    fprintf(stderr, "bad dims %d x %d\n", width, height);
    return 1;
  }
  std::vector<uint8_t> pixels(static_cast<size_t>(width) * static_cast<size_t>(height));
  if (fread(pixels.data(), 1, pixels.size(), bf) != pixels.size()) {
    fprintf(stderr, "bin truncated (pixels)\n");
    fclose(bf);
    return 1;
  }
  fclose(bf);

  // ---- Build an 8-bit grey Pix from the pixel buffer (row-major, row 0
  // first). ----
  Image pix = pixCreate(width, height, 8);
  if (pix == nullptr) {
    fprintf(stderr, "pixCreate failed\n");
    return 1;
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      pixSetPixel(pix, x, y, pixels[static_cast<size_t>(y) * width + x]);
    }
  }

  // ---- Build the input StaticShape: batch=1, height=36 (fixed — the
  // network's input height; matches the pix height here so FromPix does NOT
  // scale, since FromPix/FromPixes themselves never scale — only
  // Input::PreparePixInput does, and we bypass that to test FromPix in
  // isolation), width=0 (dynamic, so the real pix width drives the
  // StrideMap and Copy2DImage's noise-pad branch never fires), depth=1
  // (8-bit grey, single feature). ----
  StaticShape shape;
  shape.set_batch(1);
  shape.set_height(36);
  shape.set_width(0);
  shape.set_depth(1);

  // ---- Randomizer: seed 1, no warm-up draw. Only consulted by
  // Copy2DImage's Randomize() call if the fixed target_width exceeds the
  // real pix width, which cannot happen here (shape.width()==0). ----
  TRand rand;
  rand.set_seed(1);

  // ---- NetworkIO in int8 mode (matches LSTMRecognizer::RecognizeLine's
  // inputs->set_int_mode(IsIntMode()) called BEFORE FromPix). ----
  NetworkIO netio;
  netio.set_int_mode(true);
  netio.FromPix(shape, pix, &rand);
  pix.destroy();

  // ---- Dump: shape line then one "i" line per timestep. ----
  printf("shape\t%d\t%d\t%d\t%d\n", netio.stride_map().Size(FD_HEIGHT),
         netio.stride_map().Size(FD_WIDTH), netio.NumFeatures(), netio.Width());
  for (int t = 0; t < netio.Width(); ++t) {
    printf("i\t%d", t);
    const int8_t *row = netio.i(t);
    for (int f = 0; f < netio.NumFeatures(); ++f) {
      printf("\t%d", static_cast<int>(row[f]));
    }
    printf("\n");
  }
  return 0;
}
```
