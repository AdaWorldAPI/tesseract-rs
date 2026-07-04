# Recoder Core-shape design pass — v1 (plan, not the design)

> **Status:** EXECUTED (2026-07-04) — landed as E-CPP-PARITY-7. The load-side
> recoder (`UnicharCompress` in `lance-graph-contract` + `Recoder` /
> `recoded_to_text` in `tesseract-core`) is byte-parity green on real
> `eng.lstm-recoder` (encode 112/112 + decode 112/112 + code_range=111).
> **Correction to §1 below:** `kMaxCodeLen = 9` (not 3 — Hangul/Han USE
> length-3, but the array is sized 9). The routing verdict (content-store tier,
> NOT `emit_rust`) was re-verified LIVE against OGAR
> (SURREAL-AST-TRAP-PREFLIGHT + OGAR-AS-IR §3) — held. `0x08` OCR is now minted
> (OGAR #148: `recoder`=0x0802), so the recoder keystone is unblocked but
> deferred (dispatch already proven generically). The original plan (drafted
> 2026-07-01, post-OGAR #85–#145 survey) is preserved below.

## 1. What the recoder is, and why it is next

`ccutil/unicharcompress.{h,cpp}` — `UNICHARCOMPRESS` re-encodes each
unichar-id as a short sequence of small codes (CJK radical-stroke, Indic
grapheme pieces, ligature dissection; pass-through for simple scripts).
The LSTM recognizer's output lattice speaks **recoded codes, not raw
unichar-ids** — so `ids_to_text` only becomes real OCR output once
`DecodeUnichar` exists. Key surface (unicharcompress.h):

- `RecodedCharID` — the code-sequence holder (`length`, up to 3 codes,
  `self_normalized`, equality + hash).
- `EncodeUnichar(unichar_id) -> RecodedCharID` / `DecodeUnichar(code) ->
  unichar_id` — the two runtime lookups (table-backed, O(1)-ish).
- `code_range()` — 1 + max code value (the lattice width).
- `Serialize/DeSerialize` — binary I/O via `serialis.h` `TFile`.
- `ComputeEncoding(unicharset, null_id, radical_stroke_table)` — the
  TRAINING-side table builder.

**Scope decision (mirror of the unicharset `load_via_fgets`-only scope):**
transcode the **load side only** — `DeSerialize` + `EncodeUnichar` +
`DecodeUnichar` + `code_range`. `ComputeEncoding` is training-side; it is a
later, separate leaf if ever needed. Falsifier data exists on disk:
`/tmp/eng.lstm-recoder` (1012 B, real trained eng recoder — regenerate via
`combine_tessdata -u` if reaped).

## 2. The nuanced routing decision (the reason this plan exists)

OGAR #85–#145 landed two things that could plausibly claim this module:

1. **`ogar-from-ruff` transpile lane** — per-class minting + `emit_rust`
   (pull-back codegen, proven on a Rails-lifted `CompiledClass`).
2. **OGAR-as-IR framing** (`docs/OGAR-AS-IR.md`) — six IR-shape tests
   gating any IR-surface change.

**Routing verdict (drafted here, confirm in the design session):** the
recoder is a **content-store tier**, same category as `UniCharSet` — a
loaded codec table (id ↔ code-sequence bijection + bounds), data-shaped, no
lifecycle vocabulary, no effects, no AR shape. Applying the OGAR-as-IR §3
tests: it adds **no** field to `Class`, no `ActionDef` variant, no
`KausalSpec` slot — it is *not IR-surface*, so per test routing ("rerouted,
not rejected") it belongs where `UniCharSet` lives: a zero-dep module in
`lance-graph-contract`, dispatched through the existing keystone
(`methods_for` gate → content-store trait → adapter leaf). The
`ogar-from-ruff` emit lane targets AR-shaped producers; a C++ leaf codec is
not one. **Do NOT route this through emit_rust just because the lane is
new.** If the design session finds the recoder needs a Core capability the
content tier can't carry, that is a Core gap → `core-gap-auditor` rules
EXTEND-CORE vs ADAPTER-HACK, per doctrine.

**SURREAL-AST-TRAP-PREFLIGHT (five questions, answers drafted):**
- Q1 WHAT am I reading FROM? — C++ binary table serialization (TFile), not
  an AR producer. No DDL anywhere near it.
- Q2 LIFECYCLE VOCABULARY? — none. Pure data tables.
- Q3 TARGET IR? — none; target is a contract content-store module (below
  IR). No `Class`/`ActionDef` emission.
- Q4 ARROW DIRECTION? — C++ artifact → Rust reader (pull-in of *data*, not
  behavior).
- Q5 WOULD THE INVERSE RECOVER BEHAVIOR? — N/A; there is no behavior arm,
  only lookups. Round-trip Encode∘Decode = identity IS the falsifier.

Re-run these live at session start; if any answer changes, stop and reroute.

## 3. First binary-format leaf — what actually changes

Every prior leaf parsed text. The recoder is **binary** (`serialis.h`
conventions). The design session's first read is therefore
`unicharcompress.cpp::DeSerialize` + the `TFile` primitives it calls
(endianness, length-prefix conventions). The Rust side gets a
`RecoderError`-typed binary reader — same lenient-vs-strict decisions the
unicharset parser made, but byte-level. Budget most of the session here.

## 4. Falsifier (same self-validating oracle discipline)

1. Extend `/tmp/uniprops_oracle.cpp` (rebuild if reaped; recipe in
   tesseract-rs `CLAUDE.md` §method): load unicharset + recoder via
   libtesseract, dump per-id `EncodeUnichar` sequences (`encode` mode) and
   the decode round-trip + `code_range` (`decode` mode). Keep the bijection
   mode as the layout self-check (5.5.0-header / 5.3.4-lib skew, proven
   sound for prior leaves — re-verify, don't assume, since UNICHARCOMPRESS
   is a *new object layout* not yet covered by the bijection check; if the
   skew bites here, build the oracle against a source-built lib instead).
2. Rust side: committed example (`recoder_dump`, modes `encode|decode`) in
   lance-graph-contract, diff both dumps — 112/112 on eng, plus
   Encode∘Decode identity across the full id range.
3. Wire into `tesseract-core`: `ids_to_text` gains the real decode path
   (`codes → unichar_ids → text`), one consumer-boundary test.

## 5. Gates to load at session start (in order)

1. `../lance-graph/.claude/knowledge/core-first-transcode-doctrine.md`
2. OGAR `docs/SURREAL-AST-TRAP-PREFLIGHT.md` (5 Q live re-run)
3. OGAR `docs/OGAR-AS-IR.md` §3 (six tests — routing confirmation)
4. tesseract-rs `CLAUDE.md` iron rules (scoped `-p`, consume-the-Core)

## 6. Deferred / cross-repo notes (do not act in the leaf session)

- **`0x08` OCR domain**: named in OGAR's APP‖class core codebook,
  **not yet minted** in `ogar-vocab` (mints so far: 0x07 OSINT, 0x09
  Health, 0x0B Auth, 0x0C Automation, commerce extensions). When the
  unicharset/recoder keystone graduates from test-registry classids to
  canonical ones, the mint happens in OGAR (`ogar-vocab` + PortSpec if a
  port is warranted) — a separate cross-repo PR, per the consumer
  best-practices doc (classid is pure address; pull, never re-declare).
  **Classid half-order FLIPPED (2026-07-02)** — canon HIGH / APP-render
  prefix LOW, read `domain:appid:classview` (V3 marker currently 1000):
  lance-graph #628 (`CanonHigh` + compat reader) + OGAR #147
  (`ogar_vocab::app` lockstep). The #95 layout doc's original order
  description is superseded — when minting, take the order from
  `ogar_vocab::app` / contract `CanonHigh`, never from prose.
- **EdgeBlock-superseded flag**: OGAR ADR work states canon =
  `key(16)+value(496)`, EdgeBlock superseded (F-5) — divergent from
  lance-graph CANON `key(16)|edges(16)|value(480)`. Cross-repo
  coordination owns this (COUNT_FUSE dependency contract); the recoder
  content tier touches neither layout.
- bbox/stats/normed sub-leaf stays gated on a legacy (non-LSTM)
  unicharset (weak falsifier on LSTM data — see CLAUDE.md).
