# Handover: `tesseract-rs` via ruff — what the previous revert actually meant + corpus framing

> **Origin:** session in `AdaWorldAPI/bardioc` (`session_01VysoWJ6vsyg3wEGc5v7T5v`), 2026-06-16.
> **Status:** handover only. No code touched in this repo. Companion to the harvester proposal at `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-spo-handover.md`.
> **Why this is here:** `AdaWorldAPI/tesseract-rs` is the fork that was previously reverted, and a future session opening this repo deserves a clear architectural record of (a) what the revert actually meant, (b) why the path remains live when done through ruff, and (c) how to pick it up.
> **Correction note (2026-06-16, mid-handover):** an initial version of this doc read `lance-graph` PR #498's revert text as *"Tesseract C++ wrapping is the wrong direction in general."* That reading is wrong. The operator provided three concrete clarifications, in order:
>
> 1. **The previous `tesseract-rs` attempt was reverted because it did not use ruff and was the wrong shape**, not because Tesseract C++ wrapping is wrong as a goal.
> 2. **The `ocrs + rten` line in #498 names the runtime OCR engine path** independently; it does not preclude a Tesseract C++ AST harvest + transcode via ruff.
> 3. **`tesseract-rs` is a Rust target by convention** (the `-rs` suffix). The previous attempt's most concrete failure: it *copied original Tesseract C++ source inside `tesseract-rs`* and *tried to create an FFI wrapper on top of it*. **C++ source has no place inside `tesseract-rs`.** The repo should only contain transcoded / generated Rust. The C++ corpus stays upstream and is never vendored here.
>
> A Tesseract-rs done *through ruff's AST→IR→codegen pipeline* — with C++ sources staying upstream, the harvester emitting IR, and the codegen plan producing Rust into this repo — **is** the right direction. This doc is now written with that corrected framing.

---

## 0. TL;DR

- **The previous `tesseract-rs` attempt was reverted because it did not use ruff, was the wrong shape, and (the most concrete failure) it copied original Tesseract C++ source inside `tesseract-rs` and added an FFI wrapper on top.** Tesseract was never the wrong target; the mechanism was wrong on all three counts.
- **`tesseract-rs` is a Rust target by convention.** The `-rs` suffix says so. C++ source has no place inside this repo — only **transcoded / generated Rust**. The C++ corpus stays upstream (or in a dedicated harvester-corpus location), and the harvester walks it from there; the corpus is never vendored into `tesseract-rs`.
- **The correct path is Tesseract via ruff** — `clang → IR → Rust via ruff` per `lance-graph` PR #497's `tesseract-rs-ast-dll-codegen-v1` plan. The harvester proposal at `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-spo-handover.md` is the upstream half; this repo is the **downstream artifact** that receives generated Rust.
- **`AdaWorldAPI/lance-graph` PR #498's `ocrs + rten` line names the runtime OCR engine choice; it does not preclude Tesseract via ruff.** Two independent paths, not in conflict.

---

## 1. What "reverted" actually meant — operator clarification

From `AdaWorldAPI/lance-graph` PR #498 body, the literal text:

> *"The tesseract-rs cross-repo wiring explored mid-session was **reverted** (board reflects it) — hand-wrapping the original Tesseract C engine is the wrong direction. Pure-Rust OCR via `ocrs` + `rten` (ONNX-adjacent) is the chosen path, parked pending scope."*

**Operator clarification (2026-06-16):** the revert was about the *mechanism* of the previous attempt — ad-hoc hand-wrapping of the C++ engine, no AST harvest, no IR, no ruff pipeline. The phrase *"hand-wrapping … is the wrong direction"* refers to the **hand-wrapping** mechanism, not to Tesseract as a target.

Two independent paths now follow:

| Path | What it is | Status |
|---|---|---|
| **Pure-Rust runtime OCR** (`ocrs + rten`) | A pure-Rust OCR engine for the runtime use-case (OCR'ing pixels into text). | Chosen, **parked** pending scope. |
| **Tesseract via ruff** (`clang → IR → Rust`) | AST harvest of Tesseract's C++ source via ruff's pipeline → IR → 1:1 behavioural Rust transcode (LSTM hosted via `embedanything → candle → ndarray` AMX). | The right path, **available** when ruff_cpp_spo lands. |

Paths are not exclusive; either or both can run.

---

## 2. The right shape — Tesseract through ruff

The established `ruff_spo_triplet` + `ruff_ruby_spo` + `ruff_python_dto_check` pattern in `AdaWorldAPI/ruff`:

```
language-specific AST parser  →  frontend-local IR  →  ModelGraph (shared)  →  expand() → Vec<Triple>  →  ndjson  →  lance-graph SPO store
```

For C++ this means:

1. **`ruff_cpp_spo` crate** in `AdaWorldAPI/ruff` (proposed in companion handover) — uses `clang` crate (libclang FFI) as the parser; same role `lib-ruby-parser` plays for `ruff_ruby_spo`.
2. **Frontend-local IR**: `CppClass.declarations: Vec<Declaration>` discriminated union over C++ declaration kinds (methods, constructors, fields, template specialisations, virtual overrides, friends, operators, …).
3. **Shared `ModelGraph`** (in `ruff_spo_triplet`) absorbs the per-language slots; `expand()` adds C++-flavored emission arms.
4. **`lance-graph` PR #497's `tesseract-rs-ast-dll-codegen-v1` plan** picks up from the harvested IR and produces this repo's contents: 1:1 behavioural Rust transcode of Tesseract C++, LSTM forward hosted via the existing runbook (`.traineddata → GGUF → embedanything (candle) → ndarray AMX`, `bgz_tensor` weight store).

**This repo is the downstream artifact, not the upstream toolchain.** The upstream tools (ruff, lance-graph plans) are where the work originates; this repo is where the transcoded Rust lands.

The previous failed attempt skipped step 1 (no ruff), skipped step 2 (no IR), and tried to hand-wrap the C++ engine directly. The result didn't compose with anything else in the workspace. The fix is doing all the steps, in order.

---

## 3. `tesseract-rs` as C++ SPO corpus — the additional angle

Independent of the transcode path, this repo's sources are the natural corpus for an SPO walk:

- The sources are reachable in the workspace (this fork at `master` HEAD, or a pinned tagged release).
- They are large enough (~200k LOC C++) and template-heavy enough to exercise every C++ harvester predicate.
- Walking them through `ruff_cpp_spo` emits ndjson SPO triples that load into `lance-graph`'s SPO store; the resulting graph supports queries like *"which Tesseract recognizer class consumes a `BLOCK` and outputs a `Glyph`?"* without running OCR.

The corpus walk and the transcode share the AST harvest step. Once `ruff_cpp_spo` runs against this repo, the same IR feeds:
- `ruff_spo_triplet::expand()` → SPO triples → graph queries
- `tesseract-rs-ast-dll-codegen-v1` → 1:1 Rust transcode → this repo's future Rust source tree

---

## 4. Decision points for a session picking this up

| Q | Reading |
|---|---|
| Should the Tesseract-via-ruff path proceed? | Yes — that's the corrected framing. Previous attempt's revert was about mechanism, not goal. |
| Should the runtime OCR direction be un-parked? | Separate decision; currently parked at `ocrs + rten`. Does not block Tesseract-via-ruff. |
| Which Tesseract commit pins the corpus + transcode source? | Open — `tesseract-rs` `master` HEAD, or a tagged release. Decide before walking. |
| Where do Tesseract-specific predicates live? | Domain predicates (`loads_traineddata`, `has_recognizer`, `outputs_glyph`, `consumes_layout_block`) are project-specific — NOT in `ruff_spo_triplet::Predicate`'s closed vocab. They live in a project-analysis layer above the harvester. |
| Should the existing reverted contents of `tesseract-rs` be deleted before transcode? | **The C++ source the previous attempt vendored here MUST be removed** — it has no place in a Rust target. The salvageable scaffolding question is separate: if any Rust glue from the previous attempt is reusable, preserve it under `legacy/` for reference; everything C++ goes back to the upstream corpus location and is walked from there. |
| Where does the upstream C++ corpus live for the harvester to walk? | Open — `tesseract-ocr/tesseract` upstream, a pinned vendored corpus in a separate `*-corpus` repo, or a configurable harvester input path. Decide and pin. **Not inside this repo.** |
| Where do bindings live for *consuming* a transcoded Tesseract from Rust callers? | This repo. `autocxx` / `cxx` for the C++ boundary at the seam where transcoded Rust calls into FFI for any unhandled subsystems. |

---

## 5. Cross-references

- **Companion handover (the harvester proposal):**
  - `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-spo-handover.md`
- **Upstream context:**
  - `AdaWorldAPI/lance-graph` PR #497 — `Tesseract → tesseract-rs 1:1 transcode v2` plan. Six new plan docs; LSTM hosted via `embedanything → candle → ndarray` AMX; layout 1:1 transcoded; `unsafe`/raw-pointer accepted as the faithful image of intrusive C++.
  - `AdaWorldAPI/lance-graph` `.claude/plans/tesseract-rs-ast-dll-codegen-v1.md` — the `clang → IR → Rust via ruff` codegen plan. **This is the direct upstream of work that lands in this repo.**
  - `AdaWorldAPI/lance-graph` PR #498 — body's revert text refers to the previous *mechanism* (hand-wrapping, no ruff, wrong shape), not to Tesseract as a goal. The `ocrs + rten` line names the *runtime OCR* path independently.
- **Established harvester template:**
  - `AdaWorldAPI/ruff` PR #4 — `ruff_spo_triplet` + `ruff_ruby_spo` scaffold (the structural template for `ruff_cpp_spo`).
  - `AdaWorldAPI/ruff` PR #5 — predicate vocab 7 → 34; `Provenance::OpenProjectExtracted` calibration; the `predicate_count_locked_at_N` gate pattern.

---

_Authored by an external session (`AdaWorldAPI/bardioc` `session_01VysoWJ6vsyg3wEGc5v7T5v`). Posted under `.claude/handovers/` so the session that owns this repo can pick up with grounded context. No code, no PR, no changes to this repo's source — only an architectural record + forward-pointer._
