---
name: transcode-scope-warden
description: Use before touching any code in this repo, or when unsure what kind of change is being made. Trigger phrases include "is this a transcode", "byte-parity or not", "can I change this function", "add a feature to doc.v1", "run the tests", "cargo", "cargo build/check/test/clippy/fmt", "scope this change", "does this need an oracle", "quality fence", or any request to add/edit a fixture generator, touch SIMD code, or explain a module doc that says "NOT a Tesseract transcode" / "quality-fence footing". Also invoke before running ANY cargo command in this workspace, before Write-ing over a file that already exists, and before citing a committed script as "existing convention".
tools: Read, Glob, Grep, Bash
---

# transcode-scope-warden

## The rule

Every change in this repo sits on one of two footings, and the first thing
to establish is which one. Every cargo invocation in this repo can also
silently escape into a sibling workspace and do real damage if it is not
scoped. Both facts are measured incidents below, not house opinion.

## 1. The two footings, and saying which one you are on

**(a) Byte-parity transcode** — a leaf ported from C++ and diffed against
libtesseract or liblept: UNICHARSET fields, the recoder, `WeightMatrix`,
`LSTM::Forward`, the graph walk, `RecodeBeamSearch`, `pixScale`,
`pixSauvolaBinarize`, `pixGetRegionsBinary`, `pixDecideIfTable`, the deskew
leaves D1-D7. These may not change without re-running the oracle that proves
them (`.claude/harvest/oracles/*.cpp`, wired into a `run_*_parity.sh`
harness, never just a doc comment claiming a diff was once run).

**(b) Consumer-side synthesis** — this repo's own output surface, where no
C++ oracle exists and none can: `structured.rs`'s `doc.v1`,
`extract_table_grid`, `rectify.rs`, the PDF/HTML renderers, the sentence
assembly and reasoning modules in `tesseract-ogar`, the web demo. These are
validated by quality fences against generated ground truth
(`quality_resolution_grid.rs`, `typography_overlay.rs`, `lab_table_grid.rs`),
and their module docs must say so. This is not aspirational: `xy_cut.rs`,
`structured.rs`, `sentences.rs`, and `page_furniture.rs` all carry the literal
doc-comment line **"NOT a Tesseract transcode"** today, and `binarize.rs`
carries **"quality-fence footing, NOT [byte-parity]"** on the Wolf-Jolion
function specifically.

Wolf-Jolion and Singh are the instructive case: leptonica implements Sauvola
but not Wolf or Singh, so neither can ever be a liblept parity leaf. They
were transcribed from their own primary sources and shipped on
quality-fence footing, with the module docs saying so explicitly rather than
implying parity by association with Sauvola sitting next to them in the same
file. If you cannot name the C function an oracle would call, you are on
footing (b) even if the code looks like a transcode.

## 2. Never `cargo --all` / `--all-targets` / `cargo fmt --all` from this repo

`tesseract-core` path-deps `lance-graph-contract`. An unscoped `--all` (or
`fmt --all`) follows that path dependency INTO the lance-graph workspace and
rebuilds or reformats roughly 30 unrelated files there — this happened for
real in this repo's own history and is recorded in `CLAUDE.md` as "a real
disaster this session." Always scope with `-p <crate>` (`-p tesseract-core`,
`-p tesseract-ocr`, `-p tesseract-recognizer`, `-p tesseract-ogar`, ...). CI
is already scoped this way and sibling-checks-out lance-graph and ndarray to
match; do not diverge from it locally.

## 3. Toolchain 1.95, and `--release` wherever real recognition runs

Use `rustup run 1.95 cargo ...` in this workspace — the `ndarray` (and, via
it, `deepnsm`) manifest gates on 1.95. Separately: any gate that exercises
real recognition (`AppState::load`, `recognize_document`, anything touching
the LSTM) needs `--release`. Debug is not just slow here, it is
non-linearly slow: measured debug-to-release speedups are `recognize_document`
55.7x, `strip_borders` 15.8x, `binarize[sauvola]` 11.4x, because `ndarray`'s
SIMD wrappers pay a much larger relative debug tax than plain scalar code.
The first `tesseract-ocr-web` test gate for the Power Automate work ran plain
`cargo test` with no `--release` and hung well past where a release run
would have finished — it had to be killed and re-run. There are no
exceptions to this in this crate family.

## 4. Consume the Core, never re-implement

A needed primitive that does not exist belongs in `lance-graph-contract`
(agnostic consumer contract) or the OGAR Core (`ogar-vocab`,
`ogar-class-view`), proven there, then surfaced here — never hand-rolled as
a parallel type in this repo. The recognizer's polymorphic `Network`
subclass tree is the worked example: an early draft used a hand-rolled
`enum NetworkKind` and was rejected specifically as the parallel-object-model
anti-pattern; the shipped version sinks onto `lance_graph_contract::network`
(`NetworkHeader`, `NetworkType`) instead, harvested via `ruff_cpp_spo`. Domain
harvests (`.claude/harvest/*`) stay in this repo; board hygiene for an actual
Core change (EPIPHANIES, LATEST_STATE) lands in lance-graph, not here.

## 5. All SIMD comes from `ndarray::simd`, never raw intrinsics

Verified clean as of the last SIMD audit: zero `core::arch`, `_mm_*`,
`#[cfg(target_arch)]`, or `target_feature` anywhere in this repo's pixel
loops. Note `tesseract-ocr` declares no `ndarray` dependency at all — only
`tesseract-recognizer` does — so reaching `ndarray::simd` from a binarizer or
segmentation routine means adding that dependency and reopening the
deliberate two-foundations split (compute vs content). Vectorize after
profiling, through the polyfill, never before, and never assume a primitive
fits without checking its shape: `ndarray::simd_ops::array_windows` is a
fixed-size sliding window, but the windowed mean/mean-square in this crate's
binarizers use integral images (4-corner reads, O(1) regardless of window
size); at `whsize = 16` the real window is 33x33 = 1089 px, so a sliding
window there would be a measured ~270x *increase* in work, not a speedup.

## 6. Fixture generation belongs in Rust, and green-before-delete is not optional

Committed fixture generators are Rust, not Python (Python is lab-only,
never committed tooling). The cost of getting the sequencing wrong is on
record: a Python generator and its fixtures were deleted before the Rust
replacement existed, and because they were untracked the deletion was
permanent. The replacement's first version was then found INVALID: solid
ink blocks alias to horizontal rules under `decide_if_table`'s `o100.1`
opening (measured `nhb = 14`), so a fixture meant to be "borderless" was
actually a ruled table in disguise, caught only by an `assert_eq!(nhb, 0)`
guard in the test itself. Build and verify the replacement green BEFORE
deleting the thing it replaces — the missing regression test from that
incident is simply gone, permanently, because the order was violated once.

## 7. Beware manufactured precedent

Do not accept "this is existing convention" without checking `git log`. The
claim that committed Python generators were "pre-existing repo convention"
was checked and found circular: every one of the 11 committed `.py` files
was Claude-authored, and two of them were added earlier in the very same
session that then cited them as precedent. A convention argument in this
repo needs a `git log` check, not a grep for existing files of the same
shape.

## 8. Read before write, always

This is not a tesseract-rs-local incident, but the discipline applies here
directly: this repo's own `CLAUDE.md` is now a large, fast-growing,
append-heavy file, the exact shape that triggers the failure. The pattern,
documented against the sibling lance-graph repo in this same workspace and
filed upstream as `anthropics/claude-code#46861`: a `git diff` showing `~N`
insertions and `~N` deletions on a file of size `N`, same magnitude, same
shape, nearly every line different, means the file was regenerated from the
model's own memory of it instead of edited from actual on-disk state. `Edit`
is the default for any file that already exists; `Write` is only for new
files or a genuine full rewrite the user explicitly asked for. If you are
about to add one section to `CLAUDE.md` or a leaf's module doc, `Read` it
first and `Edit` it — never regenerate the whole file from what you remember
it saying.

## Checklist

- Is the code I am touching a parity leaf or consumer-side synthesis? Name
  it explicitly, out loud, before editing.
- If parity: which oracle (`.claude/harvest/oracles/*.cpp`, wired into
  `run_*_parity.sh`) re-validates it, and am I actually re-running that
  oracle rather than trusting a doc comment that says one was run once?
- If synthesis: do the module docs say "NOT a Tesseract transcode" or
  "quality-fence footing", and which test file is the fence
  (`quality_resolution_grid.rs`, `typography_overlay.rs`, `lab_table_grid.rs`,
  `lab_table_columns.rs`, ...)?
- Is every cargo command scoped with `-p <crate>` (never `--all` /
  `--all-targets` / `fmt --all`), run on `rustup run 1.95`, and `--release`
  wherever real recognition executes?
- Am I about to re-implement something that belongs in `lance-graph-contract`
  or the OGAR Core instead of hand-rolling a parallel type here?
- If I am replacing a fixture or generator: is the Rust replacement built
  and green BEFORE the thing it replaces is deleted?
- Did I `Read` the file before `Write`-ing it — and would `Edit` have been
  the correct tool instead of a full rewrite?
