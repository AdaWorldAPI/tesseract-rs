# Handover: `tesseract-rs` as C++ SPO corpus — and the reverted runtime direction

> **Origin:** session in `AdaWorldAPI/bardioc` (`session_01VysoWJ6vsyg3wEGc5v7T5v`), 2026-06-16.
> **Status:** handover only. No code touched in this repo. Companion to the harvester proposal at `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-spo-handover.md`.
> **Why this is here:** `AdaWorldAPI/tesseract-rs` is the fork that *would* be the natural surface for a Tesseract C++ harvester corpus, and a future session opening this repo deserves to land on the current architectural status (reverted-runtime + corpus-walk-possible) rather than re-derive it.

---

## 0. TL;DR

- **The runtime direction (`tesseract-rs` as a Rust wrapper around the original Tesseract C++ engine) was reverted upstream in `AdaWorldAPI/lance-graph` PR #498.** Quote from #498's body: *"The tesseract-rs cross-repo wiring explored mid-session was **reverted** (board reflects it) — hand-wrapping the original Tesseract C engine is the wrong direction. Pure-Rust OCR via `ocrs` + `rten` (ONNX-adjacent) is the chosen path, parked pending scope."*
- **`tesseract-rs` still has architectural value as a C++ source-tree corpus** for an analogous `ruff_cpp_spo` harvester (in the AdaWorldAPI/ruff fork), mirroring how `ruff_ruby_spo` walks Rails-codebase Ruby sources for SPO-triplet extraction.
- **The harvester proposal lives in `AdaWorldAPI/ruff`** under `.claude/handovers/2026-06-16-ruff-cpp-spo-handover.md`. This doc cross-links to it and names what `tesseract-rs` contributes if someone picks the harvester up.

---

## 1. The runtime direction is reverted — what this means

From `AdaWorldAPI/lance-graph` PR #498 (merged 2026-06-16 04:36Z), the OCR direction was decided:

> *"The tesseract-rs cross-repo wiring explored mid-session was **reverted** (board reflects it) — hand-wrapping the original Tesseract C engine is the wrong direction. Pure-Rust OCR via `ocrs` + `rten` (ONNX-adjacent) is the chosen path, parked pending scope."*

And from PR #498's `2fa7fcb0` commit (OCR `LayoutBlock → NodeRow` transcode POC):

> *"Engine-agnostic — not Tesseract-coupled."*

So:
- **Don't build a Tesseract-runtime crate here.** `ocrs` + `rten` are the chosen runtime OCR path; `OcrProvider` is the engine-agnostic trait that consumers code against.
- **`AdaWorldAPI/lance-graph` PR #497** is the v2 transcode plan (LSTM hosted via `embedanything` → `candle` → `ndarray` AMX, layout 1:1 transcoded). The PR landed but `#498`'s body supersedes its direction; #497 is design-spec on record, not the live path.

If the runtime direction is ever un-parked, this repo is the natural home for it. Until then: **dormant runtime, live corpus.**

---

## 2. `tesseract-rs` as C++ SPO corpus — the surviving angle

The pattern established in `AdaWorldAPI/ruff`:

```
language-specific AST parser  →  frontend-local IR  →  ModelGraph (shared)  →  expand() → Vec<Triple>  →  ndjson  →  lance-graph SPO store
```

For C++ this is an unbuilt opportunity. The harvester would:

1. Use `clang` crate (libclang FFI) as the parser — same role `lib-ruby-parser` plays for `ruff_ruby_spo`.
2. Walk Tesseract sources (this repo's `master` HEAD, or a pinned tagged version) into a `CppClass.declarations: Vec<Declaration>` IR.
3. Unpack into shared `ModelGraph` slots in `ruff_spo_triplet`.
4. `expand()` emits SPO triples into ndjson; ndjson loads into `lance-graph`'s SPO store.

**`tesseract-rs`'s contribution:** the corpus is reachable, intrusively-templated, large enough to exercise every C++ harvester predicate, and already cloned in the workspace at `AdaWorldAPI/tesseract-rs`. No third-party Tesseract clone is required.

The runtime work (vendor-wrapping the C++ Tesseract API surface in safe Rust) is **a different problem** from the corpus-walk work (extracting Tesseract's class structure into SPO triples for graph reasoning). The corpus walk is additive even when the runtime path stays parked.

---

## 3. Decision points for a session picking this up

| Q | Reading |
|---|---|
| Should the runtime direction be un-parked? | Currently NO per `lance-graph` #498. Operator-pinned. |
| Should `ruff_cpp_spo` proceed with Tesseract as corpus? | See `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-spo-handover.md` §4. Reading 1 (corpus-walk is independently useful) is the only one cleanly additive. |
| Which Tesseract commit pins the corpus? | Open — `tesseract-rs` `master` HEAD or a tagged release. Decide before walking. |
| Where do Tesseract-specific predicates live? | Domain predicates (`loads_traineddata`, `has_recognizer`, `outputs_glyph`, `consumes_layout_block`) are project-specific — NOT in `ruff_spo_triplet::Predicate`'s closed vocab. They live in a project-analysis layer above the harvester. |
| Should the existing `tesseract-rs` source (whatever's at `master` HEAD) be preserved? | YES. Don't delete the fork's contents — corpus walking needs a baseline. The runtime direction can be marked DEPRECATED in a top-level README without dropping the source. |

---

## 4. Cross-references

- **Companion handover (the harvester proposal):**
  - `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-spo-handover.md`
- **Upstream context:**
  - `AdaWorldAPI/lance-graph` PR #497 — `Tesseract → tesseract-rs 1:1 transcode v2` plan (six new plan docs).
  - `AdaWorldAPI/lance-graph` PR #498 — `GUID decode→read-mode keystone + helix Signed360 right-size + OCR→NodeRow transcode`. Body explicitly reverts the tesseract-rs runtime direction.
  - `AdaWorldAPI/lance-graph` `.claude/plans/tesseract-rs-ast-dll-codegen-v1.md` — the `clang → IR → Rust via ruff` codegen plan.
  - `AdaWorldAPI/lance-graph` `.claude/plans/ocr-canonical-soa-integration-v1.md` — `OcrProvider` trait + OCR `LayoutBlock → NodeRow` mapping.
- **Established harvester template:**
  - `AdaWorldAPI/ruff` PR #4 — `ruff_spo_triplet` + `ruff_ruby_spo` scaffold (the structural template for `ruff_cpp_spo`).
  - `AdaWorldAPI/ruff` PR #5 — predicate vocab 7 → 34; `Provenance::OpenProjectExtracted` calibration; the `predicate_count_locked_at_N` gate pattern.

---

_Authored by an external session (`AdaWorldAPI/bardioc` `session_01VysoWJ6vsyg3wEGc5v7T5v`). Posted under `.claude/handovers/` so the session that owns this repo can pick up with grounded context. No code, no PR, no changes to this repo's source — only a forward-pointer + reverted-direction record._
