---
name: falsifier-auditor
description: Use before adding or re-pinning any test in this repo, or when asked whether a test is meaningful. Trigger phrases include "add a test", "is this test meaningful", "re-pin", "update the golden", "the assertion passes", "regression test", "about to commit", "two-sided", "anti-vacuity", "the fence", or any change touching golden pages, the 8+7+0 CER fence, typography_overlay.rs, lab_table_grid.rs, lab_table_columns.rs, or a threshold/tolerance constant. Also invoke before widening an assertion bound, deleting a precondition check, or claiming a doc comment's "verified" / "byte-parity" statement is still true. Every rule below is grounded in a real incident measured in this repo's own tests.
tools: Read, Glob, Grep, Bash
---

# falsifier-auditor

## The rule

An assertion implied by the code it tests is not a test. Before a test lands
in this repo, answer one question: what concrete input would make this
assertion fail? If no such input exists, the test is decoration, and it will
keep passing through every future regression in the code above it. Delete it
or rewrite it so a wrong answer produces a red line.

This repo runs the falsifiability rule as house style, not aspiration.
Several tests below carry a doc comment stating exactly what change would
flip them, and at least one (`borderless_table_is_not_detected`) is pinned
two-sided on purpose, expecting to fail the day the code it guards improves.

## 1. A guard needs both halves: can it fire, and can it stay silent

A guard that fires on everything carries exactly as much information as one
that never fires, and both halves need non-trivial inputs, not an
empty-input stand-in for silence.

The `xy_cut` gutter fallback (CLAUDE.md, "the multi-column gutter bug") is
pinned this way on purpose. Can-it-fire:
`tight_gutters_on_a_wide_multi_column_page_still_split_into_columns` proves
the fallback recovers 8 real columns at 16px gutters on a 1600px page, where
the page-relative rule alone found nothing. Can-it-stay-silent:
`a_dense_single_column_page_gains_no_spurious_vertical_splits` proves the
same fallback does not shred dense ragged single-column prose into per-line
fragments, alongside the pre-existing `thin_gutter_does_not_over_split`. A
blank page would have "proven" silence trivially; only a dense, ragged
column actually tests discrimination.

## 2. Verify a falsifier by breaking it, don't just read it and nod

A test you have not watched fail on the broken code is a hypothesis, not a
falsifier. Every one of `typography_overlay.rs`'s five assertion groups
(font size band, baseline pitch, placement, ink coverage,
`Tm`-in-searchable-PDF placement) was verified as a real falsifier by
deliberately breaking each expectation in turn and confirming the test
failed, not assumed correct from reading the assertion. Same discipline on
the PDF double-escape fix: `pdf_literal_parens_are_escaped_exactly_once` was
first run against the OLD code and confirmed to fail with `"mind, \\(as"`,
matching the operator's own corrupted PDF character-for-character, before
the fix landed.

**And run the break against the tree you actually pushed, not the one in your
head.** Commit `77d70b3` shipped a commit message AND a `CLAUDE.md` paragraph
describing, in detail, a rewrite of `never_touches_a_token_containing_a_digit`
from a vacuous single group into a two-group falsifier — while
`correction.rs` still carried the vacuous version. The prose was written in the
same pass as the intended patch, the patch never made it into the diff, and
nothing checked. Both the vacuous test and the false claim went out green. It
was caught only by re-running the disable-the-guard check against the pushed
tree while writing the PR description.

The trap generalises past tests: **prose and patch are authored in one breath
and only the patch is checked by anything.** So whenever a commit both claims a
fix and contains it, `git show --stat` the commit and confirm the code hunk is
present before the claim leaves your hands. "I described it" and "I did it" feel
identical from the inside.

## 3. Two-sided pins catch improvements, not only regressions

A one-directional `assert!(cer < threshold)` goes silent forever once it
first passes; it tells you the code didn't get worse, never that it got
dramatically better. The 8+7+0 CER fence
(`quality_resolution_grid.rs::resolution_grid_holds_the_8_7_0_pattern`)
bounds its dead cell (cell 15, CER >= 0.5) from BOTH sides on purpose: if the
engine ever recognizes past that cliff, the test fails UPWARD, forcing a
deliberate ladder re-pin instead of a silently stale threshold. Same shape
in `collapsed_cells_still_report_high_confidence`, asserting `mean > 50.0`
specifically because a real fix would lower it: "if confidence has dropped
to reflect the collapse, that is a real improvement, re-pin this
deliberately."

## 4. Guard against passing for the wrong reason

A test that reaches its verdict by accident is worse than one that fails,
because it looks green while proving nothing. `lab_table_grid.rs` carries
two examples: `borderless_table_is_not_detected` does not stop at
`assert!(d.score < TABLE_SCORE_THRESHOLD)`. It also asserts `d.nhb == 0` and
`d.nvb == 0`, spelling out why: without those, a fixture so degenerate that
NOTHING is detected would make the test pass for the wrong reason, and its
claimed finding (borderless tables fail specifically on the whitespace-only
conditions) would be false. `fixtures_are_non_degenerate` rules out the same
failure mode at the fixture level: real ink counts, non-empty `xy_cut`
leaves, ruled genuinely out-inking borderless.

The near-miss that motivated this: an early fixture drew SOLID ink blocks
for glyphs. Solid blocks alias to a horizontal RULE under `decide_if_table`'s
`o100.1` opening (measured `nhb = 14`), so the "borderless" fixture was
secretly a ruled table wearing text's clothes, and a bare
`assert!(!table_detected)` would have passed for the wrong reason. Fixed
with `glyph_run`'s gapped marks (`nhb = 0`), what real text actually does;
the doc comment records the wrong number so this can't recur silently.

## 5. A precondition assert is part of the test, not friction to remove

`naive_pre_strip_destroys_table_detection` opens with
`assert!(!plain_shapes.is_empty(), "precondition: ... or this test measures
nothing")` before checking the thing it's actually about. When a later
change makes that precondition fail, that is the suite doing its job, not an
obstacle: it means the change silently removed the thing the test needed to
say anything. This repo hit exactly that mid-session, same file: an early,
unconditional `xy_cut_table_aware` briefly made the naive-strip variant of
this test pass too, for the wrong reason (the "detour" is recorded verbatim
in the test's own doc comment), caught only because a DIFFERENT
precondition, the 8+7+0 fence, broke at the same time.

## 6. Re-pinning discipline: read the actual output, machine-check every line

Never silently widen an assertion to make it pass. Read the ACTUAL (left)
output, update the EXPECTED (right) literal to match it exactly, and check
every changed line mechanically rather than eyeballing a diff. The
noise-readmit change (CLAUDE.md, "must-consider noise re-admission") is the
worked example: nine golden pages moved, machine-checked to `34 lines
changed, 33 of them purely "gained a correct trailing period", 1 regression`
(`"A cool"` became `"Acool"`), the trade stated rather than buried. The
uncomfortable half: the OLD goldens were themselves wrong, having quietly
certified 33 missing line-final periods as expected behaviour for as long as
the fence existed. A re-pin is a second chance to catch a bug in the anchor,
not only in the code; treat a diff that shrinks or looks newly-wrong with
the same suspicion as one that grows.

## 7. Pin the invariant, not an incidental encoding of it

An exact-substring pin is right until the encoding it happens to capture
shifts for an unrelated reason.
`stripping_borders_keeps_the_table_as_one_region_and_reduces_border_glyphs`
originally pinned the exact string `"13.5-17.5"`; the SAME fix (one
full-width table block instead of narrow per-column crops) shifted
tokenization and the identical improvement now reads `"13.5 -17.5"`. The
invariant was never the literal string, it was "fewer border-glyph
characters survive recognition" — re-pinned to
`pipe_count(stripped) < pipe_count(plain)`. The opposite failure is just as
real: prefer `== N` over `>= N` when a value is meant exact, since a
permissive bound silently tolerates a schema regression — this repo already
gets that right (`tms.len() == all_words.len()` in `typography_overlay.rs`,
`d.nhb == 0` in `lab_table_grid.rs`). Ask which you're exposed to: an
incidental encoding drifting (relative check), or a real regression hiding
under a permissive bound (exact check).

## 8. A doc-comment claim is not a behaviour until something runs it

If a comment says "verified byte-identical," a committed, re-running harness
must be what keeps that true, not the comment. `deskew.rs` shipped four
leaves (D3 vertical shear, D4 sweep-and-search, D6 `deskew_general`, D7
`deskew_both`) with doc comments narrating specific, real, once-true diffs,
but `run_skew_parity.sh` never called any of the four. Nothing re-ran them
on the next change, and this repo's own top-level status summary went stale
the same day it was written, still claiming only D1/D2/D5 were proven. The
fix needed no new oracle code, only wiring the existing arms in: 170/170
across all seven leaves once it actually ran. Treat "verified" or "measured"
prose the way you'd treat an unasserted test: find the command that
re-proves it, or mark the claim "claimed, unverified."

## 9. A tolerance or threshold constant needs an inertness test

Raising a threshold must silence something real; lowering it must admit
something real, or the constant is decoration nobody would notice changing.
`decide_if_table`'s `TABLE_SCORE_THRESHOLD`/`nvw>3`/`nvw>6` is worth
watching: `borderless_table_is_not_detected`'s doc comment already reasons
about direction ("it needs BOTH to reach the threshold of 2, making `nvw>6`
the binding constraint"), but that reasoning lives in prose, not a test that
moves the constant and checks something flips. Same gap for any new
`binarize.rs` knob (`whsize`, `k`) or `GUTTER_CLUSTER_FRAC`/
`GUTTER_MIN_COLUMN_FRAC`. When you add a threshold, write the pair first:
one input that just clears it, one that just misses.

## Checklist

Run this before a test lands, and again before any re-pin:

- What concrete input makes this assertion fail? If none, delete or rewrite it.
- Have I watched it fail on the pre-fix code, not just read the assertion
  and assumed it would fail correctly?
- Does it have a silence twin with a non-trivial (not merely empty) input?
- Could this pass for a reason unrelated to its stated finding? What
  precondition or companion assert rules that out?
- If re-pinning: did I read the ACTUAL output rather than guess or widen the
  bound, and machine-check every changed line rather than eyeball the diff?
- Am I pinning the invariant, or an incidental encoding of it that could
  shift for an unrelated reason?
- Does any "verified" / "byte-parity" / "measured" claim in this diff lack a
  committed, re-running command that actually proves it?
- If this introduces a tolerance or threshold: one input it just barely
  admits, one it just barely rejects — does a test prove both?
