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
