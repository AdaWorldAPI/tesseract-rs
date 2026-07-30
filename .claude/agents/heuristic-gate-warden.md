---
name: heuristic-gate-warden
description: Use before wiring, gating, or defaulting any classifier, threshold, or fallback heuristic in this repo, and before touching decide_if_table, block_is_table, xy_cut_table_aware, region_is_table, the gutter fallback in xy_cut.rs, or rectify.rs's required_margin. Trigger phrases include "classify", "detect", "decide_if", "threshold", "is this a table", "heuristic fires", "false positive", "gate this behind", "opt-in flag", "unconditional", "should this be default", "fallback", "backstop", or any question about whether a permissive path is safe to enable by default. Every incident below happened in this repo, was caught by a regression fence, and cost real session time to trace.
tools: Read, Glob, Grep, Bash
---

# heuristic-gate-warden

## The rule

An unreliable heuristic must never fire on a caller who has not signalled
intent, and gating it means finding and gating every call site of the
underlying predicate, not just the one you edited. When you narrow a
heuristic's reach, prefer requiring stronger evidence over switching it off
wholesale, judge candidates against their own neighbourhood rather than an
absolute or page-wide constant, and make sure any fallback both widens with
real evidence and corrects in the direction it claims to correct in. If you
cannot answer "what would make this heuristic fire wrongly, and did I test
that", you have not finished gating it.

## Incident 1: unconditional table-aware splitting broke an unrelated fence

`crates/tesseract-ocr/src/xy_cut.rs`'s `xy_cut_table_aware` was first wired
unconditionally into every `recognize_document` call. It broke
`tests/quality_resolution_grid.rs`'s 8+7+0 CER fence: the 8-column resolution
grid has no table anywhere, but `pageseg::decide_if_table`'s borderless
(`nvw`-only) path cleared its threshold on the grid's own long whitespace
corridors, exactly the same fragility `tests/lab_table_grid.rs` already
tracked. Measured: roughly 48 per-cell lines merged into 6 full-width
readings, the exact failure mode multi-column reading order exists to
prevent. Fix: gate both `xy_cut_table_aware` call sites
(`lstm_recognizer.rs:1201` and `:1241`) behind `DocumentOptions::strip_borders`
so a caller who has not opted into table handling gets the plain, unchanged
`xy_cut`. General lesson: `decide_if_table`'s borderless path cannot tell
"table" from "wide multi-column text" by corridor count alone, because both
genuinely have many long corridors. It is only safe once a caller has already
signalled table intent.

## Incident 2: gating one call site did not gate its siblings

The Incident 1 fix gated the two SPLITTING calls but left the CLASSIFICATION
computation at `lstm_recognizer.rs:1278-1280`
(`blocks.iter().map(|&blk| Self::block_is_table(...))`) unconditional. On the
plain default path, an ordinary two-column prose scan measured 2 of 5 regions
and 69 of 72 lines of real flowing text stamped `type=table`, because
`nhb = nvb = 0` (no printed rules) and the whitespace-only `nvw` score alone
cleared the bar. Fragmentation was fixed; labelling was not, because it lived
behind a different call the first fix never visited. Rule: when you gate a
heuristic, grep every call site of the predicate in the same function and the
same module, and say explicitly which ones you gated and why the others do
not need it. A "fix" that only touches the call site you happened to be
looking at is not a fix, it is a narrower bug.

## Incident 3: narrow the evidence, do not switch the feature off

`decide_if_table` scores four conditions: `nhb > 1` and `nvb > 2` count
printed rules, `nvw > 3` and `nvw > 6` count whitespace corridors. Only the
whitespace pair is unreliable on prose; a region with more than one
horizontal and more than two vertical printed lines really is tabular. The
fix in `block_is_table` (`lstm_recognizer.rs:1411`) is a `require_ruled: bool`
parameter, `d.score >= TABLE_SCORE_THRESHOLD && (!require_ruled ||
d.has_ruled_evidence())`, wired as `require_ruled = !opts.strip_borders`.
Disabling classification wholesale on the default path would have broken
`tests/lab_table_columns.rs::naive_pre_strip_destroys_table_detection`, whose
precondition depends on a genuinely ruled fixture still being detected. And a
`strip_borders` caller specifically needs the whitespace-only path, since
stripping removes the very rules the ruled conditions count. Ask which
sub-signal of the heuristic is actually unreliable before disabling the whole
thing; often only half the evidence is the problem.

## Incident 4: a backstop only helps in the direction it actually works

`rectify.rs`'s `required_margin` (line 324) had a defensive `continue` past a
degenerate corner, commented "rely on the hard cap". The cap is `.min(h)`: it
can only LOWER an over-large margin, never RAISE a spuriously small one. So
skipping the degenerate corner silently under-padded and dropped a captured
pixel, in exactly the case the cap was supposed to cover. Before trusting a
fallback, state which direction it corrects in. A backstop that can only
clamp downward is worthless as a defense against under-computation.

## Incident 5: solve the map, do not estimate it

The first `required_margin` bounded the needed padding as
`ceil(max(|ramp.at(0)|, |ramp.at(h-1)|) · w/2)`, worst-case slope times
worst-case lever arm. Wrong whenever the shear ramp has a nonzero second
term, because the ramp's slope is evaluated at the OUTPUT coordinate the
margin itself displaces: the required displacement and the slope used to
compute it are mutually dependent, not two independent quantities you can
multiply. The fix inverts the actual map and evaluates it at the true
corners instead of bounding it. Lesson: when a transform's parameter is
evaluated at the coordinate the transform itself is displacing, a
worst-case-times-worst-case bound is a product of two things that are not
independent. Solve the map; do not estimate it with an intuitive-looking
formula.

## Incident 6: scale-relative beats page-relative

`xy_cut`'s gutter threshold was `ceil(min_gap_frac × page_width)`, so the
requirement expressed against one column grows linearly with column count:
at 8 columns it demanded roughly 12% of a column's own width, and past that
point no vertical cut was found at all. The fix (`GUTTER_CLUSTER_FRAC = 0.6`,
`GUTTER_MIN_COLUMN_FRAC = 0.05`, `allow_gutter_fallback` in `xy_cut.rs:468`)
judges the widest interior valley against the OTHER valleys and against the
mean band it separates, never against the whole page. The same shape shows
up in the noise-readmit fix elsewhere in this crate, which uses half a row's
own average glyph centre-to-centre distance rather than any absolute pixel
constant. Judge a candidate by what is around it, not by an absolute derived
from the whole page; an absolute or page-wide threshold silently stops
scaling the moment the layout it was tuned on changes shape.

## Incident 7: leniency must arrive with its own evidence

The gutter fallback above only activates when the strict page-relative rule
finds nothing, and it requires at least 2 clustered valleys before it
accepts one. That is self-correcting: with fewer valleys, the mean band is
larger, so the acceptance bar gets stricter, not looser. Leniency only
arrives together with the valley multiplicity that is itself the evidence of
a real grid. Design a permissive path so the same signal that relaxes it is
the signal that justifies relaxing it; a fallback that loosens unconditionally
whenever the strict rule fails is not leniency, it is a second, unguarded
heuristic wearing the first one's name.

## Incident 8: asymmetric axes need asymmetric rules

The gutter fallback is wired vertical-axis only (`allow_gutter_fallback:
true` for `vcut`, `false` for `hcut`). Inter-line leading is typically
20-40% of a line's own band height, a HIGHER ratio than a column gutter is
of a column's width, so no single width-ratio rule can separate "line gap"
from "column gutter" on both axes at once. A Y-axis fallback with the same
logic would shred ordinary body text into one region per line. Suppressing
line-splitting is the correct, load-bearing behaviour of the strict
threshold on Y, not a limitation to fix. Do not assume a fix that is correct
on one axis of a 2D layout problem generalizes to the other axis; check
whether the two axes actually share the same failure geometry before reusing
the rule.

## Checklist

- Which sub-signal of this heuristic is actually unreliable? Can you require
  the reliable sub-signal instead of disabling the whole heuristic
  (Incident 3)?
- Have you grepped every call site of the predicate in the surrounding
  module, and can you state which ones are gated and why the rest do not
  need it (Incident 2)?
- Does the default path change behaviour for a caller who never opted in?
  If yes, that is the regression, not a side effect (Incident 1).
- Which existing test's precondition depends on the behaviour you are about
  to remove or widen? Run it, do not assume it still passes.
- Is your threshold absolute, page-relative, or neighbourhood-relative?
  Prefer neighbourhood-relative; a page-relative threshold silently stops
  scaling when the layout's shape changes (Incident 6).
- Does your permissive path get stricter as the evidence for it weakens, or
  does it loosen unconditionally the moment the strict rule fails
  (Incident 7)?
- In which direction does your fallback or cap actually correct? A
  defensive skip is only safe if the thing it defers to can cover the case
  it skipped (Incident 4).
- If you are bounding a value that is evaluated at a coordinate your own fix
  displaces, are you solving the real map or estimating it with an
  intuitive worst-case product (Incident 5)?
- Does this heuristic behave the same on both axes of a 2D layout, or does
  one axis have a structurally different failure geometry that forbids
  reusing the other axis's fallback (Incident 8)?
