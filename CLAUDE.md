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

`ids_to_text` (the recognizer's id→text walk) is the first OCR-facing step in
`tesseract-core`. Cross-ref the Core's `EPIPHANIES.md` E-CPP-PARITY-1..6 +
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

**The real next module: the recoder** (`unicharcompress.{h,cpp}`) — then the
recognizer. That is a new Core type (not another UNICHARSET accessor), so it
starts with a read of `unicharcompress.h` + a Core-shape design pass, not a quick
leaf.

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
