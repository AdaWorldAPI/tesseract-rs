# Recognizer Core-shape design pass — v1 (plan, not the design)

> **Status:** PLAN (drafted 2026-07-04, after the recoder landed as
> E-CPP-PARITY-7). The next module after the recoder — and the first
> **compute** leaf, where the operator's sanity check bites: *OCR without
> hardware acceleration isn't smart.* The LSTM recognizer IS the arithmetic;
> this plan fixes the two-foundations architecture, the clean int8 seam onto
> ndarray, the leaf scoping, and the oracle/parity approach. It does NOT
> pre-draw the Rust types.

## 1. What the recognizer is, and why it is next

Tesseract's LSTM recognizer turns a normalized line image into the recoded-code
lattice that `tesseract-core::recoded_to_text` (E-CPP-PARITY-7) already decodes
to text. Its hot path is **8-bit integer matrix arithmetic** — `src/lstm/` +
`src/arch/`:

- `arch/intsimdmatrix.{h,cpp}` — `IntSimdMatrix`, the int8 matrix×vector
  primitive (base scalar reference + AVX2/SSE/NEON/RVV SIMD subclasses).
- `lstm/weightmatrix.{h,cpp}` — `WeightMatrix`: stores weights as int8 (`wi_`)
  + a per-output float `scales_`; `ConvertToInt()` quantizes (row max-abs →
  `INT8_MAX`, store the reproducing scale); `MatrixDotVector(int8 u, TFloat v)`
  runs the int-mode forward step.
- `lstm/{network,series,parallel,lstm,fullyconnected,convolve,maxpool}.cpp` —
  the network graph (the DAG of layers).
- `lstm/recodebeam.{h,cpp}` — the CTC beam search over the code lattice →
  `RecodedCharID` sequences → (the recoder) → unichar-ids → text.

**This is where ndarray stops being "unused" and becomes load-bearing.** The
recoder/unicharset leaves were codec *tables* (correctly zero-dep). The
recognizer is *compute*: dense int8 GEMM, the exact thing ndarray's SIMD
foundation exists for.

## 2. The two-foundations architecture (the correction)

The earlier framing "the OCR transcode is ndarray-free" was wrong at the system
level. Corrected:

```
lance-graph-contract   = the CONTENT foundation  (codec tables: unicharset,
                          unichar, recoder — proven, zero-dep)
ndarray                = the COMPUTE foundation   (int8/bf16 SIMD GEMM, AMX,
                          AVX-512/AVX2-VNNI, NEON — already shipped)
tesseract-core         = content-tier consumer    (CharSet, Recoder,
                          ids_to_text, recoded_to_text — zero-dep)
tesseract-recognizer   = compute-tier consumer    (NEW; deps ndarray +
                          tesseract-core) — WeightMatrix, network graph,
                          recodebeam → the code lattice recoded_to_text eats
```

**Routing verdict (confirm live at session start):** the recognizer is NOT a
`lance-graph-contract` (zero-dep) citizen — compute is not content, and the
int8 GEMM already exists in ndarray. Core-First is satisfied by **consuming
ndarray's proven GEMM, never re-transcoding SIMD** (mirrors the `simd-savant`
iron rule: all SIMD comes from `ndarray::simd`). So the recognizer lands as a
**new workspace crate `crates/tesseract-recognizer`** depending on `ndarray`
(path, fork) + `tesseract-core` (for `Recoder`/`CharSet`/`recoded_to_text`).
`tesseract-core` stays zero-dep; the ndarray dep is quarantined to the compute
crate.

## 3. The clean int8 seam — no Core gap (the reason this is tractable)

Tesseract's int8 forward step maps **one-to-one** onto primitives ndarray
already ships (verified 2026-07-04):

| Tesseract | ndarray |
|---|---|
| `WeightMatrix::ConvertToInt` (row max-abs → INT8_MAX + float scale) | `simd_amx::quantize_energy_i8(&[f64], &mut [i8])` |
| `IntSimdMatrix::MatrixDotVector` base (int8 W × int8 u → i32) | `simd_runtime::matmul_i8_to_i32` (AMX TDPBUSD → VPDPBUSD-zmm → -ymm → scalar, sign-shift bias trick) |
| per-output scale i32 → TFloat (`v[o] = scales[o]·dot`) | `simd_amx::dequantize_result_f64(&[i32], &mut [f64], scale)` |
| bias `w(i, num_in) · INT8_MAX` (**not `· 1`** — the input's imaginary `1.0` is int8-quantized to 127; intsimdmatrix.cpp:101) | add `(w(i,num_in) as i32) · 127` to the i32 accumulate before scaling |

**The exact base formula (intsimdmatrix.cpp:78-117 — the oracle spec):**
`v[i] = (Σ_j w(i,j)·u[j] + w(i,num_in)·INT8_MAX) · scales[i]`, where the C++
chunks outputs ×4 for auto-vectorization (arithmetic-irrelevant). The bias
column is `w(i, num_in)` (the last, `dim2()-1`, weight column); `scales[i] =
max_abs_row / INT8_MAX` from `ConvertToInt` (weightmatrix.cpp:198).

**Key parity property:** int8×int8→i32 accumulation is **exact and
order-independent**, so AMX / AVX512-VNNI / AVX2-VNNI / scalar all yield the
IDENTICAL i32 — the recognizer's integer matmul is bit-reproducible across every
SIMD tier (unlike float/BF16 GEMM). Only the final per-row scale is a float
op; with `TFloat` matched (Tesseract `tesstypes.h` — float or double, confirm
the build) it is a single deterministic multiply.

**Consequence:** the recognizer does NOT transcode a matmul — it consumes
ndarray's. The transcoded leaf is the recognizer-specific glue (quantize +
matmul + bias + scale + weight reshaping), byte-parity-proven against the C++
**base** `IntSimdMatrix::MatrixDotVector` (the scalar reference in
`intsimdmatrix.cpp`, which uses the un-reshaped row-major weights — the natural
oracle). If a needed capability turns out missing in ndarray, that is an
ndarray Core gap → file it and extend ndarray deliberately (per its W1a
consumer contract), never hack the recognizer crate.

## 4. Leaf scoping (each byte-parity-proven before it lands)

1. **Leaf 1 — `MatrixDotVector` (the hardware-acceleration leaf).** Adapter:
   int8 weights + per-row float scales + int8 input → TFloat output, via
   ndarray `matmul_i8_to_i32` + bias-add + `dequantize`. Oracle: the base C++
   `IntSimdMatrix::MatrixDotVector(w, scales, u, v)` on **synthetic** int8
   (no `Pix`). Parity: i32 accumulate EXACT (integer must match to the bit);
   TFloat output bit-exact if `TFloat` matches, else a pinned ε with the
   integer half asserted exact. **This is the first shippable leaf and needs no
   image.**
2. **Leaf 2 — `WeightMatrix::DeSerialize` (int mode).** The binary load of
   `wi_` (int8) + `scales_` (float) from a real `.lstm` network component
   (`TFile`, same little-endian discipline the recoder proved). Byte-parity on
   a real trained component (`combine_tessdata -u eng.traineddata`).
3. **Leaf 3+ — the network graph forward pass.** `Series` / `Parallel` /
   `FullyConnected` / `LSTM` / `Convolve` / `Maxpool` `Forward()` composed over
   `NetworkIO`. Multi-leaf; each layer type its own parity check against the
   C++ `Network::Forward`. (The activation LUTs — `functions.cpp`
   `generate_lut.py` — are their own small table leaves.)
4. **Leaf N — `recodebeam` (CTC beam decode).** The lattice → `RecodedCharID`
   sequences, closing the loop into `recoded_to_text`. This is where an
   end-to-end "image → text" parity becomes possible.

**Deferred (front of the pipeline, leptonica-heavy):** image input + line
normalization (`Pix` → `lstm/input.cpp`). That is the one place leptonica is
unavoidable at the *oracle* boundary; the Rust path stays image-lib-agnostic
(takes a normalized `NetworkIO`-shaped input). Gate on reaching Leaf 3.

## 5. Oracle + parity approach (same self-validating discipline)

- **Leaf 1 oracle:** a tiny C++ harness linking libtesseract, calling
  `tesseract::IntSimdMatrix::MatrixDotVector` on seeded synthetic int8 W (m×n),
  int8 u (n), float scales (m); dump the TFloat output. Rust side: a committed
  example on `tesseract-recognizer` running the ndarray adapter on the SAME
  seeded inputs; diff. Because the inputs are synthetic and in-process, this is
  simpler than the recoder oracle (no traineddata component needed for Leaf 1).
- **ABI note:** `IntSimdMatrix` is a public `TESS_API` struct — no internal
  header skew games needed the way UNICHARSET required; still dump a trivial
  known-answer case (e.g. identity-ish W) as the layout self-check.
- **Float discipline:** assert the i32 accumulate exact FIRST (the integer
  contract), then the scaled float; if `TFloat` precision differs, pin ε and
  say so (certification-officer discipline — never launder a float diff as
  "byte-identical").

## 6. Gates to load at session start (in order)

1. `../lance-graph/.claude/knowledge/core-first-transcode-doctrine.md`
2. ndarray `.claude/knowledge/vertical-simd-consumer-contract.md` (the W1a
   consumer contract — recognizer is a consumer; if it needs a new ndarray
   `pub fn`, this governs it) + ndarray `simd-savant` "all SIMD from
   `ndarray::simd`" invariant.
3. tesseract-rs `CLAUDE.md` iron rules (scope `-p`, consume-the-Core; **now
   also**: `-p tesseract-recognizer`, and NEVER re-implement SIMD).
4. Tesseract `src/arch/intsimdmatrix.cpp` (the base `MatrixDotVector` — the
   oracle) + `src/lstm/weightmatrix.cpp` (`ConvertToInt` + the int-mode
   `MatrixDotVector` call site + bias handling).

## 7. Open questions / Core-gap watch (resolve in the leaf session)

- **`TFloat` = `double` by default, `float` iff `FAST_FLOAT`** (tesstypes.h:36-40).
  The oracle links libtesseract 5.3.4, so the parity float type is whatever THAT
  build used (FAST_FLOAT flag unknown — a Leaf-1 probe: feed a `W·u` whose
  double-vs-float rounding diverges and see which the lib produces, or inspect
  the package build flags). If double → `dequantize_result_f64`, bit-exact
  achievable; if float → an f32 scale + pinned ε. The **integer** half (the i32
  `total`, incl. the `·INT8_MAX` bias) is exact regardless and MUST match to the
  bit — assert it first, independent of the float question.
- **Weight reshaping.** The base `MatrixDotVector` uses un-reshaped row-major
  weights (the clean oracle); the SIMD subclasses use `Init()`-reshaped
  `shaped_w_`. Transcode against the BASE (un-reshaped) — ndarray's matmul owns
  its own internal tiling. Do NOT transcode Tesseract's `Init()` reshaping
  (that's SIMD-layout glue ndarray replaces).
- **NetworkIO int8 quantization of the INPUT.** The input `u` to an int layer
  is the int8-quantized previous output; confirm where that quantization + its
  scale live (`networkio.cpp`) — Leaf 1 takes int8 `u` as given; Leaf 3 wires
  the inter-layer quantization.
- **Crate wiring.** `crates/tesseract-recognizer` needs `ndarray = { path =
  "../../../ndarray", default-features = false, features = [...] }` — pick the
  minimal feature set for int8 GEMM (avoid dragging the full HPC tree). Confirm
  the fork path + that the int8 matmul is reachable without heavy features.
- **`0x08` OCR mint.** The recognizer container kind (if it warrants a classid)
  would mint in OGAR alongside `unicharset`/`recoder`/`charset` (0x0801–0x0803);
  defer until a keystone actually needs it (same posture as the recoder).
