# tesseract-rs Headstone Exploration — AdaWorldAPI/tesseract-rs

## Purpose

This document is a headstone exploration for the full line of thought connecting:

```text
upstream Tesseract C++ corpus  (tesseract-ocr/tesseract @ pinned commit, NEVER vendored here)
ruff_cpp_spo                   (AdaWorldAPI/ruff — libclang AST harvest)
ruff_spo_triplet               (closed-vocab SPO grammar, ModelGraph IR)
ndjson                         (the wire format)
tesseract-rs-ast-dll-codegen-v1  (AdaWorldAPI/lance-graph plan — IR → Rust codegen)
AdaWorldAPI/tesseract-rs       (THIS REPO — Rust target, receives generated source only)
lance-graph SPO store          (parallel consumer of the same IR for graph queries)
ocrs + rten                    (parallel runtime OCR path; orthogonal, parked)
```

The goal is to preserve the architectural synthesis for *what this repo IS when complete*, before the previous attempt's structural mistakes get rediscovered and re-made.

---

## Capstone thesis

```text
tesseract-rs is a Rust target.
The -rs suffix says so. The directory contains Rust, never C++.

The upstream Tesseract C++ corpus lives at tesseract-ocr/tesseract.
It does not move. It does not get copied here.

ruff_cpp_spo walks the upstream corpus via libclang.
It emits a ModelGraph IR in the shared SPO grammar.
The IR is the contract — between the harvester and every downstream consumer.

tesseract-rs-ast-dll-codegen-v1 (a lance-graph plan) consumes the IR.
It produces Rust source. The Rust source lands here.

lance-graph SPO store consumes the same IR in parallel.
The graph answers structural queries without OCR running.

This repo's job is to BE the Rust target.
Its source tree is generated, not authored.
The provenance for every file traces back to a commit of the upstream corpus
and a version of the codegen plan.

ocrs + rten is the runtime OCR engine path. It is independent.
This repo's transcoded Rust is not a runtime OCR engine — it is the
faithful Rust image of Tesseract's machinery, queryable, testable,
and useful for cross-checking ocrs output against Tesseract baseline.
```

---

## The three-layer architecture

### Layer 0 — Upstream C++ corpus

The Tesseract source lives at its upstream home (`tesseract-ocr/tesseract`) or in a pinned corpus location external to this repo. A specific commit or tagged release is pinned by the codegen plan. The corpus is **never vendored into this repo**.

This layer answers:

```text
which commit of Tesseract is being transcoded
where the upstream corpus physically lives
who owns its evolution (upstream Tesseract maintainers)
```

### Layer 1 — Transcode pipeline (off-repo)

Two off-repo deliverables drive the transcode:

1. **`ruff_cpp_spo`** in `AdaWorldAPI/ruff` — uses `clang` crate (libclang FFI) to walk the upstream corpus, produce `CppClass.declarations` IR, project into the shared `ModelGraph` (`ruff_spo_triplet`).
2. **`tesseract-rs-ast-dll-codegen-v1`** in `AdaWorldAPI/lance-graph/.claude/plans/` — consumes the harvested IR, emits Rust source files according to the codegen target spec, lands the files into this repo's source tree.

This layer answers:

```text
how upstream C++ becomes ModelGraph IR
how ModelGraph IR becomes Rust source
which classes / templates / methods get transcoded vs hosted
where the LSTM forward gets routed (embedanything → candle → ndarray AMX, per PR #497 v2)
which preprocessor macros expand at codegen time vs stay as Rust constants
```

### Layer 2 — This repo, the Rust target

`AdaWorldAPI/tesseract-rs` receives the codegen output. The repo's source tree is **generated, not authored.** Every `.rs` file's header carries provenance: which upstream commit, which codegen plan version, which IR snapshot.

This layer answers:

```text
what generated Rust is available to consumers
which Rust crates compose to recreate Tesseract's structure
what FFI boundary exists (if any) for unhandled subsystems
how a consumer Rust application calls into transcoded Tesseract internals
how `cargo test` verifies the transcode against Tesseract baseline outputs
```

---

## Why the previous attempt was retired (and what to NOT repeat)

The previous tesseract-rs attempt failed structurally on three mechanisms:

1. **It did not use ruff.** No AST harvest, no IR, no participation in the workspace's SPO grammar.
2. **It was the wrong shape.** What it produced didn't compose with anything else in the workspace — no classid-resolved consumption, no `ValueSchema` integration, no SPO graph entries.
3. **It vendored original Tesseract C++ inside this repo and added an FFI wrapper on top.** This is the most concrete failure: `-rs` repos hold **Rust**, never C++. The previous attempt put the upstream corpus *inside* the Rust target, which is structurally wrong before any code is written.

`AdaWorldAPI/lance-graph` PR #498's body recorded the revert verbatim: *"hand-wrapping the original Tesseract C engine is the wrong direction."* The operator clarified that this refers to the **hand-wrapping mechanism**, not to Tesseract as a goal. A correctly-shaped Tesseract-rs — driven by ruff's harvester, populated by codegen output, with C++ staying upstream — *is* the right direction.

The current state of this repo (whatever was committed before the revert) should be evaluated against the four-rule test:

- Does any C++ source live inside this repo? → must be removed before the correct-shape transcode lands.
- Does any FFI wrapper around the C++ engine exist here? → must be retired; the correct shape generates Rust, not wraps C++.
- Is any Rust glue from the previous attempt salvageable? → preserve under `legacy/` if so; otherwise let the codegen plan regenerate fresh.
- Is the repo prepared to receive *generated* source? → CI / formatting / lint config should match what the codegen plan emits.

---

## Why `ocrs + rten` alone is not enough

`ocrs + rten` (per `lance-graph` PR #498) is the chosen pure-Rust OCR engine path for the runtime use-case. It is **orthogonal** to this repo, not a replacement for it.

Two distinct concerns:

| Path | Question it answers | This repo's role |
|---|---|---|
| `ocrs + rten` | *"How do I OCR pixels into text in pure Rust at runtime?"* | None. Different engine, different runtime. |
| Tesseract via ruff (this repo) | *"What is Tesseract's internal structure as queryable + transcoded Rust?"* | **Direct.** The repo IS the transcoded Rust image. |

A future production setup may use `ocrs + rten` for the runtime OCR engine AND consume the SPO graph emitted from this repo's source for *"which recognizer produces which glyph under which conditions"* analytical queries. The two coexist; neither retires the other.

---

## Invariants

These are what the substrate enforces; this repo inherits them.

1. **No C++ source inside this repo.** The single most concrete rule. The previous attempt failed it. `-rs` is a Rust target.
2. **No `ValueSchema::Tesseract` (or `Cpp`) enum variant.** Per `lance-graph` PR #500's enforced contract test `ocr_schema_fit_rides_existing_preset_no_new_variant`: transcoded Tesseract rows ride existing presets via `classid → ClassView`.
3. **`HelixResidue` width is 6 bytes** (the stored `Signed360` place index). Any pre-#498 documentation citing 48 bytes is a bits-bytes slip; do not propagate.
4. **Generated, not authored.** Source files carry codegen-plan-version + upstream-commit provenance in their headers. Hand-edits to generated source either get folded into the codegen plan or get isolated under `legacy/` with explicit non-regeneration markers.
5. **`OcrProvider` engine-agnostic boundary** (per `lance-graph` PR #498) — any consumer Rust here exposes capability through that trait, not via Tesseract-specific surface.
6. **Five-specialist drift-catching pass** before any FINDING-grade claim about transcode fidelity (per `lance-graph` PR #500's `cascade-architect / family-codec-smith / palette-engineer / dto-soa-savant / truth-architect` framing).
7. **Gating probes before FINDING.** The transcode's big claims (`int8-exact LSTM`, `bit-reproducible diff against C++ Tesseract output`, `~200k-LOC 1:1 layout`) are CONJECTURE until measured. Pattern: `lance-graph/.claude/plans/ocr-probes-v1.md`.

---

## What "complete" looks like

The headstone is reached when:

1. **The repo contains no C++ source.** All Tesseract C++ from the previous attempt has been retired; the corpus reference points upstream.
2. **`ruff_cpp_spo` is shipped in `AdaWorldAPI/ruff`** and produces a `ModelGraph` for Tesseract that round-trips through libclang determinism (CPP-AST-RT probe green).
3. **`tesseract-rs-ast-dll-codegen-v1` in `lance-graph`** consumes the IR and produces Rust source into this repo, with provenance headers naming the upstream commit and codegen-plan version.
4. **Gating probes pass:** `int8-exact LSTM forward` matches Tesseract baseline within tolerance; `bit-reproducible diff` against the C++ Tesseract output for a fixed corpus; the layout transcode's coverage matches the upstream ~200k LOC baseline.
5. **`OcrProvider` consumer integration** — a transcoded Tesseract `OcrProvider` implementation lives in this repo, engine-agnostic at the trait surface, callable from any consumer that already speaks `OcrProvider` (the same trait `ocrs` will implement).
6. **`lance-graph` SPO store contains queryable Tesseract structure** — the same IR that drives this repo's codegen has populated the substrate graph, so cross-checks like *"does this transcoded method have a corresponding upstream method?"* are graph-resolvable.

When these six hold, this repo has fulfilled its purpose as the Rust image of Tesseract's machinery.

---

## Headstone state — what the era closes

```text
The era that closes:
  - Vendoring upstream C++ source inside Rust-target repos.
  - Hand-wrapping C++ engines via ad-hoc unsafe FFI.
  - Authoring Rust source by hand that's claimed to mirror Tesseract C++.
  - Reading lance-graph #498's revert text as "Tesseract is the wrong goal."

The era that opens:
  - The Rust target receives generated source; corpora stay upstream.
  - Provenance: every generated file traces to an upstream commit + plan version.
  - The same IR feeds two consumers: codegen → Rust here; SPO graph → lance-graph.
  - ocrs + rten and transcoded-Tesseract-via-ruff coexist as parallel paths,
    not as alternatives.
  - "Wrong shape" is the failure mode for the next correctly-shaped attempt
    to avoid — not "wrong goal."
```

The capstone thesis at the top of this doc is the one-line restatement of the open-era state for this repo.

---

## Cross-references

### This repo (`AdaWorldAPI/tesseract-rs`)
- Companion tactical handover: `.claude/handovers/2026-06-16-cpp-spo-corpus-handover.md` — what the previous tesseract-rs revert actually meant + corpus framing + post-#500 corrections.

### Sibling repo (`AdaWorldAPI/ruff`)
- `.claude/handovers/2026-06-16-ruff-cpp-headstone-exploration.md` — the harvester-side headstone (same shape, focused on `ruff_cpp_spo` as the upstream half of the pipeline that lands generated Rust into this repo).
- `.claude/handovers/2026-06-16-ruff-cpp-spo-handover.md` — the tactical evaluation + scaffold proposal for `ruff_cpp_spo`.

### Upstream architecture context (`AdaWorldAPI/lance-graph`)
- PR #496 — `ValueSchema` presets + §0 anti-invention guardrail.
- PR #497 — `Tesseract → tesseract-rs 1:1 transcode v2` plans; LSTM hosted via `embedanything → candle → ndarray` AMX; layout 1:1 transcoded; `unsafe`/raw-pointer accepted as the faithful image of intrusive C++.
- PR #498 — `GUID decode→read-mode keystone + helix Signed360 right-size + OCR→NodeRow transcode`. Body records the revert; operator clarified the revert was mechanism, not goal.
- PR #500 (open at time of writing) — rebaseline of #497 OCR plans; enforced no-new-variant contract test; 5-specialist drift-catching framing; gating probes pattern; HelixResidue 48 B → 6 B propagated.
- `.claude/plans/tesseract-rs-ast-dll-codegen-v1.md` — the **direct upstream consumer** of `ruff_cpp_spo` IR. This plan produces what lands in this repo.
- `.claude/plans/tesseract-rs-transcode-master-v1.md` — master transcode plan (v2).
- `.claude/plans/ocr-canonical-soa-integration-v1.md` — OCR canonical-SoA wiring; the analog of what the C++ transcode produces for this repo.
- `.claude/plans/ocr-probes-v1.md` — gating probes template.

### Other workspace headstones (for shape reference)
- `AdaWorldAPI/lance-graph/.claude/plans/3DGS-Cesium-BindSpace4-headstone-exploration.md` — the headstone shape this document follows.
- `AdaWorldAPI/bardioc/ROADMAP_RUST_PRIMARY_HEADSTONE.md` — Phase A→I migration headstone.

---

_Authored by an external session (`AdaWorldAPI/bardioc` `session_01VysoWJ6vsyg3wEGc5v7T5v`). Headstone shape — preserves the architectural synthesis for what this repo IS when complete. Companion tactical handover at `2026-06-16-cpp-spo-corpus-handover.md` carries the corpus framing + post-#500 corrections. No code, no PR — synthesis-preservation only._
