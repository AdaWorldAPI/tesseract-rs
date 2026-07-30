---
name: parity-oracle-smith
description: Use this agent whenever a byte-parity leaf needs a C++/liblept or C++/libtesseract oracle built, extended, or audited in tesseract-rs. Triggers on "build an oracle", "byte-parity", "diff against libtesseract", "diff against liblept", "does this transcode match", "the parity harness", "add a leaf", "prove this port", or any request to extend run_skew_parity.sh / run_unicharset_parity.sh / write a new .claude/harvest/oracles/*.cpp file. This agent does not write the Rust transcode itself; it builds and audits the oracle that proves the transcode correct.
tools: Read, Glob, Grep, Bash
---

# parity-oracle-smith

## The rule

**An oracle only proves a port correct if it calls the exact function the
port claims to reproduce, at a version that matches the linked library, on a
fixture that can actually exhibit the bug, comparing bit patterns through a
formatter that matches on both sides, wired into a script that re-runs on
every future change.** Every clause in that sentence has a real incident
behind it in this repo. Drop any one clause and a correct port can fail the
oracle, or a wrong port can pass it. Both have happened here.

## Evidence: what breaks when a clause is skipped

### 1. The oracle must call the operation under test, not something nearby that also does the thing

Two codex P1 findings on the same PR (`.claude/harvest/oracles/skew_oracle.cpp`),
both in the deskew wave:

- The `sweep` arm originally called `pixFindSkewSweep`, the standalone sweep
  API. That function is **not on `pixFindSkew`'s call path at all** (the
  harvest manifest `.claude/harvest/leptonica-skew-callgraph.txt` STEP 3
  classifies it SKIP for exactly this reason), and it refines its coarse
  maximum with `numaFitMax`, while the real entry point
  (`pixFindSkewSweepAndSearchScorePivot`) takes the raw coarse maximum and
  binary-searches instead. Fixed to call the targeted function directly, see
  the `⚠ CORRECTED (codex P1 on PR #58)` comment in `skew_oracle.cpp` above
  the `sweep` arm.
- The `dss` arm prepared its rotated image with `pixRotateShearCenter`, a
  **composed two/three-shear rotation**. The real sweep scores a **single**
  vertical shear (`pixVShearCorner` / `pixVShearCenter`). Those produce
  different pixels and therefore different scores.

Both bugs have the same shape: **a correct implementation of the real
operation would have FAILED the oracle, while an implementation of the wrong
operation could have PASSED.** That is the worst possible failure mode for a
parity harness, because it rewards the wrong port and punishes the right one.
Verify the pivot/entry-point actually matters before trusting a diff: at 2
degrees the two pivots give 23310 vs 23090; at 0 degrees they agree exactly
at 258022, which is what makes the D2 scoring leaf diffable in isolation
before the D3 shear even exists.

### 2. Carry a known-good control arm inside the same oracle

`.claude/harvest/oracles/unicharset_oracle.cpp` dumps the already-proven
112/112 id-to-unichar `bijection` mode alongside every new field mode
(`properties` / `script` / `other_case` / `direction` / `mirror`). If the
bijection diff is 0, the object layout is sound for the fields being read
and the new field's diff is trustworthy. Without that control arm you cannot
tell "the port is wrong" apart from "the ABI is skewed" (see #3). Any new
oracle against a struct/class whose layout has prior proven fields should
dump those fields too, not just the new one.

### 3. State the linked version explicitly; prefer public-API-only oracles

This repo has a real history of ABI skew: an earlier arc had the installed
`libtesseract` at 5.3.4 against source headers at 5.5.0. The fix each time
was the same shape: install or clone the source at the **exact tag that
matches the installed shared library** (`skew_oracle.cpp`'s header comment:
"installed liblept is 1.82.0 and `/tmp/leptonica-src` is tag 1.82.0, an exact
match, verified before this file was written"; `sauvola_oracle.cpp` and
`network_forward_oracle.cpp` follow the same pattern). When a match cannot be
guaranteed, write the oracle **public-API-only** (`pixRead` / `CreateFromFile`
/ `SetRandomizer` / `Forward`, never a `*Low` internal or a struct field read
directly) so the diff proves the observable contract, not a guess at layout.

### 4. Sweep densely wherever the C kernel mixes precisions; round numbers lie

`rotate_am_gray` computes `sina`/`cosa` in **f64** and narrows to **f32
exactly once**, after which the entire per-pixel loop runs in f32
(`deskew.rs` around the `xpm`/`ypm` computation). An all-f32 port or an
all-f64 port diverges from this **only on a subset of angles**, wherever the
rounding boundary of the low bits happens to fall differently. A handful of
round-number test angles can pass while the port is measurably wrong. The
shipped fix is a dense sweep, -20 to +20 degrees in 0.25-degree steps, 161/161
comparisons, plus the corner-pivot variant swept the same way (see
`run_skew_parity.sh`'s D5 section). The general form: **when a doc comment or
the source says a value is "computed once in higher precision then narrowed,"
that is a promise that a sparse sweep cannot verify.**

### 5. The fixture must be capable of exposing the failure you are testing for

A uniform grey field cannot expose the f32/f64 narrowing bug in #4 at all:
the 16x16 sub-pixel interpolation weights sum to 256 regardless of which
precision path computed them, so a flat fixture passes trivially whether the
port is right or wrong. This is pinned as its own test,
`rotate_am_gray_uniform_field_stays_uniform`, deliberately kept as a
*sanity* check and never mistaken for a parity check. Before trusting any
fixture, ask: what value in the output would actually differ if the specific
bug under test were present? If the answer is "nothing, by construction,"
the fixture is decoration, not evidence.

### 6. Audit precision per subexpression, and truncation conventions are per-function

`pixScale`'s area-map kernel uses `f64` for the lower-right corner
coordinates (the C source's `1.0` double literal), while surrounding
arithmetic in the same function is f32. Getting the function's overall
signature precision right is not sufficient; each subexpression needs its
own check against the C source. The corollary that bit this repo alongside
it: **`(l_int32)` truncates toward zero.** The correct Rust idiom is plain
`as i32`, never `+ 0.5` before the cast. That `+0.5` round-half-up convention
belongs to a *different* function, `pixEmbedForRotation`, and porting it into
`rotate_am_gray` by habit would silently change the rounding rule (see the
doc comment above `rotate_am_gray` in `deskew.rs`).

### 7. Match the numeric formatter on both sides, or every float line is a false failure

C's `%.9g` has no direct Rust equivalent: `{:.9e}` always uses exponent form,
`{}` uses shortest-round-trip. Either substitute mismatches the other side's
convention and flags every float-valued line as a parity failure, burying
whatever real signal exists. `format_g9` in
`crates/tesseract-ocr/examples/deskew_dump.rs` exists to reproduce `%.9g`
byte-for-byte. The same file documents why `std::f32::consts::PI` is safe to
substitute for the C++ pi literal only after checking the bit patterns match
(`0x40490fdb`, both sides), and why `f32::to_radians()` is NOT safe in place
of `deg * PI / 180.0`: it folds `PI/180` into a constant before multiplying,
and float multiplication is not associative, so the two can differ in the
last bit.

### 8. A diff narrated in a doc comment but never wired into the harness goes stale the day it is written

`deskew.rs` shipped `v_shear_corner`/`v_shear_center` (D3),
`find_skew_sweep_and_search_score_pivot` (D4), `deskew_general` (D6), and
`deskew_both` (D7) with doc comments narrating specific verified diffs
("REAL D2+D3 end-to-end", "Verified byte-identical 10/10"). None of that was
captured in `run_skew_parity.sh`, so nothing re-ran those arms on the next
change, and this repo's own `CLAUDE.md` summary line went stale on the same
day it was written, claiming only D1/D2/D5 were proven. This is the exact
failure the falsifiability rule in this repo's top-level `CLAUDE.md` names:
"a doc-comment claim is not a behaviour... a test must exercise the claim or
the claim must be labelled claimed, unverified." Closed by extending
`run_skew_parity.sh` to call every arm the oracle and the Rust dump already
supported, no new oracle code needed, just wiring: 170/170 across D1, D2+D3
(58 comparisons), D5 (82), D4 (4 real-skew fixtures at `pixFindSkew`'s own
defaults, read directly from `skew.c`), D6 and D7 (8 each). **A verified diff
that lives only in a comment is not evidence; it is a claim about evidence
that used to exist.**

## Checklist: run this before trusting any parity result

1. **Exact function.** Name the precise C function under test and confirm
   the oracle calls it, not a sibling, a standalone variant, or a composed
   operation that merely looks similar. Check the real call chain in the C
   source or the harvest manifest if there is any doubt.
2. **Control arm.** Does the oracle also dump at least one already-proven
   field or bijection, so a bad diff can be attributed to the new field
   rather than to ABI/layout skew?
3. **Version match, stated explicitly.** Confirm the linked library version
   and the source tag used to write the oracle actually match. When a match
   cannot be guaranteed, restrict the oracle to public API only.
4. **Fixture can expose the bug.** Name the fixture property that would make
   the specific defect under test visible. Reject uniform, degenerate, or
   angle-zero-only fixtures as sole evidence.
5. **Sweep density matches the precision mixing present.** If the C source
   narrows precision anywhere, the sweep must be dense enough to hit
   rounding-boundary cases, not just round numbers.
6. **Formatter parity.** Confirm the float-dumping format is byte-identical
   on both sides, and that hex bit patterns, not decimal renderings, are
   what the diff compares.
7. **Wired into the committed harness.** The new arm must be a line in the
   harness script, not only a doc comment claiming a diff was once run. If
   it is not in the script, label it "claimed, unverified" until it is.
