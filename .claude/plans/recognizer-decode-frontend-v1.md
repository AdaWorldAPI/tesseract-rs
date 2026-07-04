# Plan — recognizer Leaf 7 (recodebeam) + the 2-D front-end → end-to-end OCR

**Status:** Leaves 1-6 + **Leaf 7 (7a maps + 7b CTC beam) EXECUTED** + byte-parity
green — the recognizer now spans **logits → text** (`Decode` → labels →
`recoded_to_text`). This plan's REMAINING scope is the **2-D front-end** (image →
features) + the dict/LM beam. **Branch:** `claude/happy-hamilton-0azlw4`
(tesseract-rs PR #5).

## ✅ Leaf 7 EXECUTED (2026-07-04)

- **7a** — recoder `SetupDecoder` beam maps (`is_valid_start_`/`final_codes_`/
  `next_codes_`) in the Core `UnicharCompress`, byte-parity green on real
  `eng.lstm-recoder` (114-line `dump_beam` diff). `E-OCR-RECODER-BEAM-1`,
  lance-graph PR #647.
- **7b** — `RecodeBeamSearch::Decode` (non-dict CTC beam) in `tesseract-core`,
  byte-parity green across 4 configs (`null_char ∈ {110,0,42}` × fold/simple)
  via the public `Decode`→`ExtractBestPathAsLabels` oracle. `E-OCR-RECODEBEAM-1`,
  tesseract-rs PR #5. + `RecodedCharId::from_codes` in the Core.

The rest of this doc is the ORIGINAL scoping (retained); the "Recommended order"
steps 1-2 are now done, steps 3-5 (2-D front-end, image `Input`, dict beam) remain.

## Where we are (proven)

| Leaf | What | Proof |
|---|---|---|
| 1 | `matrix_dot_vector` (int8 GEMM via ndarray) | `E-OCR-MATDOTVEC-1` |
| 2 | `WeightMatrix::from_le_bytes` + `forward` | `E-OCR-WEIGHTMATRIX-1` |
| 3 | activations (tanh/logistic/relu/clip/softmax LUT) | `E-OCR-ACTIVATION-1` |
| 4 | `FullyConnected::Forward` = `activation(W·u)` | `E-OCR-FULLYCONNECTED-1` |
| 5 | `LSTM::Forward` (gates + int8 recurrence) | `E-OCR-LSTM-1` |
| 6 | graph walk (`Series`/`Reversed`/`Parallel` + int8 requant) | `E-OCR-GRAPHWALK-1` |

The recognizer runs the 1-D core `Series[…LSTM192, Fc111-softmax]` from int8
feature sequences → the 111-class softmax logits per timestep. Everything above
is byte-parity vs libtesseract 5.3.4 (`-DFAST_FLOAT`).

## Leaf 7 — `recodebeam` (the CTC decode, logits → text) — THE HARD ONE

`recodebeam.cpp` is **1382 lines** — a full CTC beam search, NOT a greedy argmax
(a greedy decode would not be byte-parity with Tesseract). Do it in two sub-leaves.

### Leaf 7a — recoder `SetupDecoder` beam maps (Core, `lance-graph-contract`)

The deferred piece from the recoder leaf (`E-CPP-PARITY-7`): `UnicharCompress::SetupDecoder`
(`unicharcompress.cpp:396-434`) builds the beam-search trie maps `next_codes_`
(code → valid next codes), `final_codes_` (code → codes that can end a sequence),
and `is_valid_start_`. For the eng.lstm recoder (112 pass-through, all length-1)
these are near-trivial, but they MUST be transcoded + byte-parity-proven (extend
`recoder_oracle.cpp` to dump the maps). Bounded, provable, unblocks 7b. Lands in
the Core (merged), so a NEW lance-graph PR.

### Leaf 7b — `RecodeBeamSearch::Decode` (recognizer, the beam search)

- `ComputeTopN(output.f(t), num_features, kBeamWidths[0])` — per-timestep top-N.
- `DecodeStep(...)` — the beam step: `ContinueContext` walks the recoder's
  `next_codes_`/`final_codes_` maps, extends beam nodes, applies the CTC null
  (`kBeamWidths`), scores. **No dictionary** for the first pass (`dict_ = nullptr`
  is a supported mode — `IsSpaceDelimitedLang` gate) — dict integration is a
  later sub-leaf.
- `ExtractBestPaths` → the code sequence → `recoded_to_text` (already proven,
  `E-CPP-PARITY-7`) → the string.
- **Byte-parity approach:** run libtesseract's `RecodeBeamSearch::Decode` on the
  SAME synthetic softmax output my Leaf 6 produces, compare the decoded code
  sequence (and the final string). The oracle constructs a `RecodeBeamSearch`
  with the loaded recoder + a `GENERIC_2D_ARRAY<float>` of my logits, calls
  `Decode`, and `ExtractBestPathAsUnicharIds`.
- Deferred within 7b: the dictionary / language-model path, `DecodeSecondaryBeams`
  (LSTM choice modes), `worst_dict_cert` scoring — the non-dict greedy-CTC beam is
  the falsifiable core.

## The 2-D front-end (image → features) — the OTHER remaining gap

The eng.lstm spec `[1,36,0,1[C3,3Ft16]Mp3,3TxyLfys48Lfx96RxLrx96Lfx192Fc111]` has
a 2-D front-end before the 1-D LSTM core:

- **`NetworkIO` + `StrideMap`** (foundational) — the multi-dim int8/f32 SoA with
  the `(batch, y, x) ↔ timestep t` index map + `AddOffset(x, FD_WIDTH)` neighbor
  access. Every 2-D layer needs it. The 1-D leaves used plain `&[&[i8]]`; the 2-D
  layers need the grid.
- **`Convolve::Forward`** (`convolve.cpp`) — stack `x_scale × y_scale × ni` inputs
  per output timestep (sliding window); `CopyTimeStepGeneral` + StrideMap offsets.
- **`Maxpool::Forward`** (`maxpool.cpp`) — downscale by `x_scale × y_scale`, max
  per window (`MaxpoolTimeStep`).
- **`Reconfig::Forward`** (`reconfig.cpp`) — the `Ft`/scale-and-deepen (stacks
  windows without maxing).
- **`XYTranspose`** (`reversed.cpp` `CopyWithXYTranspose`) — the `Txy` transpose.
- **`Input::Forward`** — the image `Pix` → int8 `NetworkIO` (`networkio.cpp:293`
  `(INT8_MAX+1)·pixel`). **Leptonica-heavy** — needs a `Pix` decode; the biggest
  external-dep gap.

## Recommended order

1. **Leaf 7a** (recoder SetupDecoder maps) — bounded, Core, unblocks the decode.
2. **Leaf 7b** (non-dict CTC beam) — logits → text; the recognizer produces text
   from feature sequences (deferring the image front-end + dict). This is the
   "recognizer works" milestone.
3. **NetworkIO/StrideMap** + Convolve/Maxpool/Reconfig/XYTranspose — the 2-D
   front-end (image-features → LSTM input).
4. **Input** (leptonica `Pix` decode) — closes image → text.
5. Dict / language-model beam scoring — the accuracy layer.

## Iron rules (unchanged)

- Every leaf byte-parity vs libtesseract before it lands; oracle built `-DFAST_FLOAT`.
- Compute leaves in `tesseract-recognizer` (deps ndarray); recoder/content in the
  Core (`lance-graph-contract`). Board hygiene (EPIPHANIES) lands in lance-graph.
- Scope cargo `-p tesseract-recognizer`; never `--all` (path-dep walks into
  lance-graph/ndarray).
