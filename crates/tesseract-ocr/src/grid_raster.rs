//! Column-lattice inheritance: re-split a degraded, over-merged block at the
//! boundaries a page's OWN well-formed columns already establish.
//!
//! **Consumer-side layer — NOT a Tesseract transcode**, same footing as
//! [`crate::xy_cut`] and [`crate::structured`]. No pixel access, no
//! binarization, no recognition: this module reasons about [`PageRect`]
//! geometry alone, before or after `xy_cut` runs.
//!
//! # The problem this module targets
//!
//! `xy_cut` finds a vertical gutter only where the ink on both sides of it
//! actually clears; a degraded row (weak/broken glyphs bridging what should
//! be a gutter) can therefore fail to split while the *rest of the page*
//! splits correctly. When most of a page's blocks fall into a regular
//! column lattice — same width, evenly spaced — that lattice is strong
//! evidence for where a stray merged block SHOULD have been cut, even
//! though its own local ink didn't support the cut. [`detect_column_raster`]
//! finds that lattice; [`split_nonconforming`] applies it.
//!
//! # ⚠ MEASURED FIRST — the premise this module was scoped against
//!
//! Simulating `xy_cut` (`XyCutParams::default`, including the gutter
//! fallback) over the committed `corpus/quality/resgrid.pgm` fixture (approx
//! Otsu 172) found:
//!
//! 1. resgrid yields **8 leaves, not 16** — full-HEIGHT columns spanning
//!    both grid rows. The row split never fires (the row gutter is rejected
//!    by `xy_cut`'s sliver rule; the neighbouring inked runs are too thin).
//!    Only the vertical k-way cut fires.
//! 2. resgrid does **not** reproduce the reported symptom: both bands
//!    independently carry all 7 gutters at 69-71 px against a ~49 px
//!    threshold — comfortably wide margins, nothing degraded. The operator's
//!    original symptom came from a different re-render; **the falsifiers
//!    below are therefore SYNTHETIC by necessity** — no committed fixture in
//!    this corpus reproduces the reported defect, and the 8+7+0 CER fence is
//!    expected to be a measured NO-OP for this module (nothing here should
//!    move it).
//! 3. On resgrid, a correctly-implemented raster detects 8 columns and
//!    splits NOTHING — every one of the 8 leaves already spans exactly one
//!    lattice column, so [`split_nonconforming`] is a no-op there. That is
//!    the module working as designed, not evidence it does nothing.
//!
//! # ⚠ Mutually exclusive with `strip_borders` (table-aware) segmentation
//!
//! **Never apply this in the `strip_borders` / `xy_cut_table_aware` branch.**
//! A table's whole point is a regular column lattice of cells — exactly the
//! geometry [`detect_column_raster`] is built to find — but
//! `xy_cut_table_aware` deliberately keeps a classified table as ONE
//! unsplit leaf (see `pageseg::region_is_table` / the ingredient-3 table
//! work in `CLAUDE.md`). Running this module there would re-fragment the
//! very block that fix was built to keep whole. The two are mutually
//! exclusive by construction: whichever branch of
//! `recognize_page_blocks_words_with_mode` ran `xy_cut_table_aware` must
//! never also run [`detect_column_raster`] / [`split_nonconforming`] on that
//! branch's output. (Wiring the exclusion is the orchestrator's job in
//! `lstm_recognizer.rs`; this module has no `strip_borders` awareness of its
//! own and cannot enforce the rule itself — it only refuses to be misused by
//! documenting it here.)
//!
//! # What this module does NOT do
//!
//! No pixel access, no binarization, no ink-bbox tightening, no
//! `DocumentOptions` field, no `doc.v1` change, no Y-axis raster (row
//! leading is a much noisier signal than column pitch — see the
//! `xy_cut` gutter-fallback module docs for why the same asymmetry applies
//! there). "Empty column" semantics are absence, exactly like `xy_cut`'s own
//! `ink_bbox`: this module hands back a raw geometric sub-rect for every
//! spanned column; the caller (orchestrator) is responsible for tightening
//! each sub-rect to its own ink bbox against the binary page and dropping
//! any that come back ink-free. That composition is deliberately out of
//! scope here — see `.claude/plans/quality-wave-v1.md` "ORCHESTRATOR WIRING".

use crate::xy_cut::PageRect;

/// Width tolerance for the "modal-width" (conforming) subset: a block's
/// width must fall within `[1 - WIDTH_TOL, 1 + WIDTH_TOL]` of the median
/// width to be treated as a lattice cell. `0.25` — wide enough to absorb
/// ordinary glyph-width raggedness (resgrid's measured pitch deviation was
/// ~0.5%, so this is ~50x that) while still rejecting a merged multi-column
/// block or a full-width headline, both several times a single cell's width.
const WIDTH_TOL_NUM: usize = 25;
const WIDTH_TOL_DEN: usize = 100;

/// Minimum number of modal-width blocks required before a lattice fit is
/// even attempted. Below this, "a lattice" is indistinguishable from
/// coincidence.
const MIN_RASTER_COLS: usize = 3;

/// Position tolerance for the lattice fit: a conforming block's `left` edge
/// must land within `POS_TOL` of its predicted lattice slot, as a fraction
/// of the fitted pitch, to be counted as "on the lattice". `0.15` — resgrid's
/// measured pitch deviation was ~0.5%, so this is ~30x that: comfortably
/// wide for real glyph raggedness, comfortably narrow to reject a genuinely
/// different (irregular) spacing.
const POS_TOL_NUM: usize = 15;
const POS_TOL_DEN: usize = 100;

/// Minimum fraction of lattice slots (`0..=k_max`) that must actually be
/// occupied by an on-lattice block for the fit to be trusted. `0.5` —
/// leniency arrives with its own evidence: the more columns of a wide
/// lattice go missing, the stricter this effectively becomes relative to
/// what's left, because `k_max` (set by the single farthest-right block)
/// does not shrink just because the middle is sparse.
const OCCUPANCY_FRAC_NUM: usize = 1;
const OCCUPANCY_FRAC_DEN: usize = 2;

/// Minimum height, as a fraction of the raster's `cell_h`, a spanning block
/// must have to be split. `0.5` — separates a merged CELL ROW (expected to
/// be close to `cell_h`) from a spanning HEADLINE (expected to be a small
/// fraction of it, e.g. ~0.2x): a neighbourhood-relative distinction, not an
/// absolute pixel constant.
const SPAN_HEIGHT_FRAC_NUM: usize = 1;
const SPAN_HEIGHT_FRAC_DEN: usize = 2;

/// Minimum ratio of fitted pitch to modal width (`pitch >= 0.9 * m`) for the
/// pitch to be accepted as a real column-to-column step rather than noise.
const PITCH_FLOOR_NUM: usize = 9;
const PITCH_FLOOR_DEN: usize = 10;

/// Fraction of a lattice column's own width that a candidate block's overlap
/// with it must clear to count as "spanning" that column, in
/// [`ColumnRaster::spanned`]. `0.5` — the same neighbourhood-relative
/// convention as [`SPAN_HEIGHT_FRAC_NUM`]/[`SPAN_HEIGHT_FRAC_DEN`].
const SPAN_OVERLAP_FRAC_NUM: usize = 1;
const SPAN_OVERLAP_FRAC_DEN: usize = 2;

/// A detected regular column lattice over a set of [`PageRect`] blocks.
///
/// Built by [`detect_column_raster`]; consumed by [`split_nonconforming`]
/// (or directly, via [`ColumnRaster::spanned`] / [`ColumnRaster::boundaries`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnRaster {
    /// The lattice's columns, `(left, right)` in ascending order, `right`
    /// exclusive (matching [`PageRect`]'s own convention). Every slot
    /// `0..=k_max` is present, including slots with no on-lattice block —
    /// occupancy is a *detection-time* gate, not a filter on the output
    /// column list.
    pub columns: Vec<(usize, usize)>,
    /// The fitted pitch: the median left-to-left step between consecutive
    /// lattice columns.
    pub pitch: usize,
    /// The median height of the CONFORMING, on-lattice blocks that
    /// established this raster — the reference a spanning block's own
    /// height is judged against (see [`split_nonconforming`]).
    pub cell_h: usize,
}

impl ColumnRaster {
    /// The gap midpoint between every pair of adjacent columns, ascending.
    /// `boundaries().len() == columns.len() - 1` (empty if there are fewer
    /// than 2 columns).
    #[must_use]
    pub fn boundaries(&self) -> Vec<usize> {
        self.columns
            .windows(2)
            .map(|w| (w[0].1 + w[1].0) / 2)
            .collect()
    }

    /// How many of this raster's columns `b` "spans" — its overlap with a
    /// column covers at least half that column's own width.
    #[must_use]
    pub fn spanned(&self, b: PageRect) -> usize {
        self.columns
            .iter()
            .copied()
            .filter(|&(left, right)| {
                let col_w = right.saturating_sub(left);
                if col_w == 0 {
                    return false;
                }
                let overlap_left = b.left.max(left);
                let overlap_right = b.right.min(right);
                let overlap = overlap_right.saturating_sub(overlap_left);
                SPAN_OVERLAP_FRAC_DEN * overlap >= SPAN_OVERLAP_FRAC_NUM * col_w
            })
            .count()
    }
}

/// The median of `values` (sorted ascending; the average of the two middle
/// elements on an even count, truncating). Integer-only — this module never
/// casts a pixel coordinate to a float, so there is no precision-loss lint
/// to justify anywhere in the lattice fit. `None` on an empty slice.
fn median_usize(values: &[usize]) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2
    } else {
        sorted[mid]
    })
}

/// Rounds `a / b` to the nearest integer (half rounds up), for non-negative
/// `usize` operands via `(2a + b) / (2b)` — integer-only, so no `f32` round
/// or cast is needed anywhere in the lattice fit. Caller guarantees `b > 0`.
fn round_div(a: usize, b: usize) -> usize {
    (2 * a + b) / (2 * b)
}

/// Detects a regular column lattice over `blocks`, if one exists.
///
/// The fit is over the MODAL-WIDTH subset of `blocks` — never
/// `blocks` itself — because a page containing the very defect this module
/// targets always ALSO contains the merged/degraded block as one of its
/// `blocks`, and that block would otherwise poison a naive read of "the"
/// column width or spacing. See the module docs for the full algorithm
/// rationale and the resgrid measurement that grounds each tolerance.
///
/// Returns `None` — silently, never a false positive — when fewer than 3
/// blocks are given, when no modal-width subset of at least
/// [`MIN_RASTER_COLS`] blocks exists, when the fitted pitch is not a real
/// column-to-column step, when fewer than [`MIN_RASTER_COLS`] distinct
/// lattice slots are actually occupied, or when the occupied slots fall
/// below [`OCCUPANCY_FRAC_NUM`]/[`OCCUPANCY_FRAC_DEN`] of the full span.
#[must_use]
pub fn detect_column_raster(blocks: &[PageRect]) -> Option<ColumnRaster> {
    if blocks.len() < MIN_RASTER_COLS {
        return None;
    }

    // Step 1: modal width `m`, and the conforming (modal-width) subset.
    let widths: Vec<usize> = blocks.iter().map(PageRect::width).collect();
    let m = median_usize(&widths)?;
    if m == 0 {
        return None;
    }
    let conforming: Vec<PageRect> = blocks
        .iter()
        .copied()
        .filter(|b| {
            let w = b.width();
            WIDTH_TOL_DEN * w >= (WIDTH_TOL_DEN - WIDTH_TOL_NUM) * m
                && WIDTH_TOL_DEN * w <= (WIDTH_TOL_DEN + WIDTH_TOL_NUM) * m
        })
        .collect();
    if conforming.len() < MIN_RASTER_COLS {
        return None;
    }

    // Step 2: pitch `p` from the median consecutive left-edge step, over the
    // conforming subset sorted by `left`. MEDIAN is load-bearing: with an
    // interior column merged away, the diff sequence looks like
    // `{p, p, k*p, p}` and the median is still `p`.
    // DEDUP FIRST — load-bearing, and its absence was a real defect caught by
    // the orchestrator's disable run. The lattice step is a property of the
    // distinct COLUMN POSITIONS, not of how many blocks stack in each one. On a
    // multi-band grid every column contributes one left per band, so the raw
    // sequence is dominated by ZERO diffs (`bands - 1` of them per column) and
    // the median collapses to 0 — failing the `p > 0` test below and returning
    // `None` for exactly the 2xN grid this module exists to detect. The
    // single-band fixtures could not expose it.
    let mut lefts: Vec<usize> = conforming.iter().map(|b| b.left).collect();
    lefts.sort_unstable();
    lefts.dedup();
    let diffs: Vec<usize> = lefts.windows(2).map(|w| w[1] - w[0]).collect();
    let p = median_usize(&diffs)?;
    if p == 0 {
        return None;
    }
    if PITCH_FLOOR_DEN * p < PITCH_FLOOR_NUM * m {
        return None;
    }

    // Step 3: lattice fit. `x0` is the leftmost conforming block's `left`;
    // every conforming block's slot index `k` is `round((left - x0) / p)`.
    // A block whose `left` misses its predicted slot by more than
    // POS_TOL * p is NOT on the lattice and is excluded from the on-lattice
    // set below (rather than rejecting the whole fit outright) — a single
    // misaligned block should not by itself veto detection when enough
    // OTHER blocks conform; the aggregate MIN_RASTER_COLS / occupancy gates
    // downstream are what actually decide.
    let x0 = lefts[0];
    let mut on_lattice: Vec<(PageRect, usize)> = Vec::new();
    for b in &conforming {
        let diff = b.left - x0;
        let k = round_div(diff, p);
        let predicted = x0 + k * p;
        let pos_diff = predicted.abs_diff(b.left);
        if POS_TOL_DEN * pos_diff <= POS_TOL_NUM * p {
            on_lattice.push((*b, k));
        }
    }

    let mut distinct_k: Vec<usize> = on_lattice.iter().map(|&(_, k)| k).collect();
    distinct_k.sort_unstable();
    distinct_k.dedup();
    if distinct_k.len() < MIN_RASTER_COLS {
        return None;
    }
    let k_max = *distinct_k.last()?;
    if k_max < 2 {
        return None;
    }

    // Step 4: occupancy. Leniency arrives with its own evidence: as more of
    // the `0..=k_max` slots go unoccupied, this gets stricter relative to
    // what remains, since `k_max` itself does not shrink.
    if OCCUPANCY_FRAC_DEN * distinct_k.len() < OCCUPANCY_FRAC_NUM * (k_max + 1) {
        return None;
    }

    // Step 5: build every slot 0..=k_max (including unoccupied ones — see
    // `ColumnRaster::columns` docs) and the reference cell height.
    let heights: Vec<usize> = on_lattice.iter().map(|&(b, _)| b.height()).collect();
    let cell_h = median_usize(&heights)?;
    let columns = (0..=k_max)
        .map(|k| {
            let left = x0 + k * p;
            (left, left + m)
        })
        .collect();

    Some(ColumnRaster {
        columns,
        pitch: p,
        cell_h,
    })
}

/// Re-splits every block in `blocks` that spans `>= 2` of `r`'s columns AND
/// is at least [`SPAN_HEIGHT_FRAC_NUM`]/[`SPAN_HEIGHT_FRAC_DEN`] of `r.cell_h`
/// tall, at `r`'s column boundaries.
///
/// Preserves input order: a split block is replaced IN PLACE by its
/// sub-rects, left-to-right (matching `xy_cut`'s own vertical-cut ordering);
/// every other block passes through unchanged. Only cuts strictly inside
/// `(b.left, b.right)` are used, so a sub-rect is never emitted outside the
/// source block's own x-range, and a block with no boundary strictly inside
/// it (despite nominally spanning `>= 2` columns, e.g. a sliver overlap) is
/// left unchanged rather than dropped or fabricated.
#[must_use]
pub fn split_nonconforming(blocks: &[PageRect], r: &ColumnRaster) -> Vec<PageRect> {
    let boundaries = r.boundaries();
    let mut out = Vec::with_capacity(blocks.len());
    for &b in blocks {
        let spans_enough_columns = r.spanned(b) >= 2;
        let tall_enough = SPAN_HEIGHT_FRAC_DEN * b.height() >= SPAN_HEIGHT_FRAC_NUM * r.cell_h;
        if !(spans_enough_columns && tall_enough) {
            out.push(b);
            continue;
        }

        let mut cuts: Vec<usize> = boundaries
            .iter()
            .copied()
            .filter(|&x| x > b.left && x < b.right)
            .collect();
        cuts.sort_unstable();
        if cuts.is_empty() {
            out.push(b);
            continue;
        }

        let mut left = b.left;
        for cut in cuts {
            out.push(PageRect {
                left,
                top: b.top,
                right: cut,
                bottom: b.bottom,
            });
            left = cut;
        }
        out.push(PageRect {
            left,
            top: b.top,
            right: b.right,
            bottom: b.bottom,
        });
    }
    out
}

// ORCHESTRATOR EXECUTES — disable-the-fix table for falsifiers 1-4 below.
// Each row names the guard the falsifier depends on and the SPECIFIC,
// re-runnable change that must flip the assertion from pass to fail if
// disabled. None of these toggles exist at runtime (the constants above are
// `const`, not parameters) — the orchestrator re-runs each falsifier against
// a LOCAL, throwaway edit of this file, confirms the predicted failure, then
// discards the edit. This is how each falsifier is confirmed to be a real
// falsifier rather than incidentally always passing.
//
// | # | falsifier | guard under test | disable | predicted failure |
// |---|---|---|---|---|
// | 1 | `raster_resegments_a_degraded_row_at_the_inherited_boundaries` | `split_nonconforming` actually splits | make `split_nonconforming` return `blocks.to_vec()` unconditionally | `output.len()` assertion fails at 9 (8 blocks in, nothing split) instead of 15 |
// | 2 | `two_column_prose_geometry_yields_no_raster` | `MIN_RASTER_COLS` | set `MIN_RASTER_COLS = 2` | `detect_column_raster` now returns `Some` (the 2-block conforming subset clears the lowered floor); `is_none()` assertion fails |
// | 3 | `spanning_headline_stays_whole_but_a_full_height_merged_row_splits` | `SPAN_HEIGHT_FRAC` | drop the `tall_enough` check from `split_nonconforming` (treat every spanning block as tall enough) | the headline (spans 3, height 0.2*cell_h) now ALSO splits into 3; `output.contains(&headline)` assertion fails |
// | 4 | `irregular_pitch_is_rejected_uniform_pitch_is_accepted` | `POS_TOL` filtering (dropping off-lattice blocks from `on_lattice`) | remove the `POS_TOL_DEN * pos_diff <= POS_TOL_NUM * p` filter (push every conforming block onto `on_lattice` regardless of `pos_diff`) | the irregular-pitch case's middle block is no longer dropped; `distinct_k` reaches 3 and `detect_column_raster` wrongly returns `Some`; the `is_none()` assertion fails |

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns `n` column `(left, right)` x-ranges: column `k` starts at
    /// `x0 + k * (cell_w + gutter)` and is `cell_w` wide. Shared by every
    /// falsifier below so a test's INPUT fixture and its OWN independently
    /// computed EXPECTED geometry come from the same formula — never via
    /// [`ColumnRaster::boundaries`] or [`ColumnRaster::spanned`], which
    /// would make the assertion tautological (testing the code against
    /// itself).
    fn lattice(x0: usize, cell_w: usize, gutter: usize, n: usize) -> Vec<(usize, usize)> {
        (0..n)
            .map(|k| {
                let left = x0 + k * (cell_w + gutter);
                (left, left + cell_w)
            })
            .collect()
    }

    fn rect(left: usize, top: usize, right: usize, bottom: usize) -> PageRect {
        PageRect {
            left,
            top,
            right,
            bottom,
        }
    }

    // ---- 1: CAN-FIRE ----------------------------------------------------

    #[test]
    fn raster_resegments_a_degraded_row_at_the_inherited_boundaries() {
        let x0 = 100;
        let cell_w = 200;
        let gutter = 40;
        let n = 8;
        let cell_h = 60;
        let cols = lattice(x0, cell_w, gutter, n);

        // Band A: 8 well-formed column blocks, one per lattice column.
        let band_a: Vec<PageRect> = cols.iter().map(|&(l, r)| rect(l, 0, r, cell_h)).collect();

        // Band B: ONE block merging columns 0..=6 (7 columns) at full cell
        // height (a degraded ROW, not a headline). Nothing over column 7.
        let merged_left = cols[0].0;
        let merged_right = cols[6].1;
        let merged = rect(merged_left, cell_h, merged_right, 2 * cell_h);

        let mut blocks = band_a.clone();
        blocks.push(merged);

        let raster = detect_column_raster(&blocks).expect("an 8-column lattice must be detected");
        assert_eq!(raster.columns.len(), 8);
        assert_eq!(raster.pitch, cell_w + gutter);

        let output = split_nonconforming(&blocks, &raster);
        assert_eq!(output.len(), 8 + 7);

        // Expected cut positions: the gutter midpoints between columns
        // 0..=6, computed directly from the drawn geometry.
        let mut expected_cuts: Vec<usize> =
            (0..6).map(|k| (cols[k].1 + cols[k + 1].0) / 2).collect();
        expected_cuts.sort_unstable();

        let sub_rects = &output[8..15];
        assert_eq!(sub_rects.len(), 7);
        let mut left = merged_left;
        for (i, &cut) in expected_cuts.iter().enumerate() {
            assert_eq!(sub_rects[i].left, left);
            assert_eq!(sub_rects[i].right, cut);
            assert_eq!(sub_rects[i].top, merged.top);
            assert_eq!(sub_rects[i].bottom, merged.bottom);
            left = cut;
        }
        assert_eq!(sub_rects[6].left, left);
        assert_eq!(sub_rects[6].right, merged_right);

        // Column 7's territory is covered by exactly band A's own column-7
        // block -- nothing from the split reaches into it.
        let over_col7 = output.iter().filter(|r| r.left >= cols[7].0).count();
        assert_eq!(over_col7, 1);

        // Disable check (see the ORCHESTRATOR EXECUTES table above): a
        // `split_nonconforming` that returns its input unchanged fails this
        // at the `output.len() == 15` assertion (would read 9).
    }

    // ---- 2: STAY-SILENT (prose) ------------------------------------------

    #[test]
    fn two_column_prose_geometry_yields_no_raster() {
        let w1 = 300;
        let w2 = 500; // unequal widths -- not a lattice.
        let gutter = 40;
        let cell_h = 60;
        let left_col = (100, 100 + w1);
        let right_col = (100 + w1 + gutter, 100 + w1 + gutter + w2);

        let band1 = [
            rect(left_col.0, 0, left_col.1, cell_h),
            rect(right_col.0, 0, right_col.1, cell_h),
        ];
        let band2 = [
            rect(left_col.0, cell_h, left_col.1, 2 * cell_h),
            rect(right_col.0, cell_h, right_col.1, 2 * cell_h),
        ];
        let full_width = rect(left_col.0, 2 * cell_h, right_col.1, 3 * cell_h);

        let blocks = vec![band1[0], band1[1], band2[0], band2[1], full_width];

        // Anti-vacuity: prove this fixture HAS a genuine spanning candidate
        // -- the two columns this page actually has, and the full-width
        // block plainly covers both -- so a `None` verdict below means the
        // guard declined a real candidate, not that nothing here could ever
        // span in the first place.
        let hypothetical = ColumnRaster {
            columns: vec![left_col, right_col],
            pitch: right_col.0 - left_col.0,
            cell_h,
        };
        assert_eq!(hypothetical.spanned(full_width), 2);

        assert!(detect_column_raster(&blocks).is_none());

        // ── which guard ACTUALLY declines this, measured ─────────────
        // The comment that stood here claimed "lower MIN_RASTER_COLS to 2 and
        // this wrongly returns Some". That was FALSE, and the orchestrator's
        // disable run proved it: with the floor at 2 the fixture still
        // returns None, because the two conforming (w2) blocks sit at the
        // SAME left, so `distinct_k.len() == 1` and the very same comparison
        // rejects at 1 < 2. Unequal column widths mean the w1 column never
        // reaches the lattice step at all.
        //
        // So THIS fixture is a valid stay-silent for "prose yields no
        // raster", but it isolates nothing: three separate guards would each
        // decline it. The isolating case is the sibling test below.
    }

    /// REGRESSION for the dedup defect: a MULTI-BAND grid must be detected.
    ///
    /// This is the operator's actual #50 shape — a 2x8 sheet — and before the
    /// `lefts.dedup()` in step 2 it returned `None`. Every column contributes
    /// one left per band, so the raw diff sequence held 8 zeros against 7 real
    /// steps; the median landed on 0 and the `p > 0` test rejected the page.
    /// The module would have shipped unable to see the only layout it was
    /// written for, and every single-band fixture passed regardless.
    ///
    /// **ORCHESTRATOR EXECUTES.** Disable: delete `lefts.dedup();` -> this
    /// must turn red with `pitch` unrecoverable (detection returns `None`).
    #[test]
    fn a_two_band_grid_is_detected_despite_repeated_lefts() {
        let cell_w = 300;
        let gutter = 40;
        let pitch = cell_w + gutter;
        let cell_h = 60;
        let x0 = 100;
        let cols = 8;

        // Two full bands: 16 blocks, only 8 distinct lefts.
        let mut blocks = Vec::new();
        for band in 0..2 {
            for k in 0..cols {
                let left = x0 + k * pitch;
                blocks.push(rect(
                    left,
                    band * cell_h,
                    left + cell_w,
                    (band + 1) * cell_h,
                ));
            }
        }

        // Anti-vacuity: the raw (undeduped) diff sequence really is
        // zero-dominated, which is the whole mechanism under test.
        let mut raw: Vec<usize> = blocks.iter().map(|b| b.left).collect();
        raw.sort_unstable();
        let zeros = raw.windows(2).filter(|w| w[1] == w[0]).count();
        let steps = raw.windows(2).filter(|w| w[1] > w[0]).count();
        assert!(
            zeros > steps,
            "fixture must be zero-dominated to exercise the defect \
             (zeros {zeros}, steps {steps})"
        );

        let r = detect_column_raster(&blocks).expect("a 2x8 grid IS a grid");
        assert_eq!(r.columns.len(), cols, "all eight columns recovered");
        assert_eq!(r.pitch, pitch, "pitch from DISTINCT lefts, not repeats");
        assert_eq!(r.cell_h, cell_h);
    }

    /// The column-count rule, ISOLATED — the case the prose fixture above
    /// cannot make: a genuine, perfectly regular TWO-column lattice.
    ///
    /// Both columns are the same width and land on a clean pitch, so they
    /// clear width tolerance, the pitch floor and `POS_TOL`, and produce
    /// `distinct_k == [0, 1]`. What declines them is exactly and only the
    /// not-a-lattice rule — a two-column page is prose, not a grid — which
    /// this crate spells twice: `distinct_k.len() < MIN_RASTER_COLS` and
    /// `k_max < 2`.
    ///
    /// **ORCHESTRATOR EXECUTES.** Disable: set `MIN_RASTER_COLS = 2` *and*
    /// delete the `if k_max < 2 { return None; }` guard -> this must turn
    /// red. Either edit alone leaves it green, which is precisely why both
    /// spellings exist and why the fixture above could not prove it.
    #[test]
    fn a_regular_two_column_lattice_is_still_not_a_grid() {
        let cell_w = 300;
        let gutter = 40;
        let pitch = cell_w + gutter;
        let cell_h = 60;
        let x0 = 100;

        // Two columns, three bands: 6 identical-width blocks on a clean pitch.
        let mut blocks = Vec::new();
        for band in 0..3 {
            for k in 0..2 {
                let left = x0 + k * pitch;
                blocks.push(rect(
                    left,
                    band * cell_h,
                    left + cell_w,
                    (band + 1) * cell_h,
                ));
            }
        }

        // Anti-vacuity: every block IS width-conforming (identical widths, so
        // the median is that width) and the lefts ARE a clean two-slot
        // lattice. Nothing here is malformed; only the column COUNT is short.
        assert!(blocks.iter().all(|b| b.width() == cell_w));
        let lefts: Vec<usize> = {
            let mut v: Vec<usize> = blocks.iter().map(|b| b.left).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        assert_eq!(lefts, vec![x0, x0 + pitch], "exactly two distinct slots");

        assert!(
            detect_column_raster(&blocks).is_none(),
            "two columns are prose, not a grid, however regular they are"
        );
    }

    // ---- 3: STAY-SILENT (headline), two-sided in one test ---------------

    #[test]
    fn spanning_headline_stays_whole_but_a_full_height_merged_row_splits() {
        let x0 = 50;
        let cell_w = 180;
        let gutter = 30;
        let n = 3;
        let cell_h = 100;
        let cols = lattice(x0, cell_w, gutter, n);

        let band: Vec<PageRect> = cols.iter().map(|&(l, r)| rect(l, 0, r, cell_h)).collect();

        let span_left = cols[0].0;
        let span_right = cols[2].1;
        let headline_h = cell_h / 5; // exactly 0.2 * cell_h
        let headline = rect(span_left, cell_h, span_right, cell_h + headline_h);
        let merged = rect(
            span_left,
            cell_h + headline_h,
            span_right,
            cell_h + headline_h + cell_h,
        );

        let mut blocks = band.clone();
        blocks.push(headline);
        blocks.push(merged);

        let raster = detect_column_raster(&blocks).expect("a 3-column lattice must be detected");
        assert_eq!(raster.columns.len(), 3);
        assert_eq!(raster.pitch, cell_w + gutter);

        // Anti-vacuity: the headline genuinely spans all 3 columns -- so its
        // survival below is the SPAN_HEIGHT_FRAC guard actively declining a
        // real candidate, not inertness because it was never a candidate.
        assert_eq!(raster.spanned(headline), 3);

        let output = split_nonconforming(&blocks, &raster);

        // The headline is unsplit: present, byte-identical to the input.
        assert!(output.contains(&headline));

        // The merged full-height row becomes 3 sub-rects; the band + the
        // untouched headline account for the rest.
        assert_eq!(output.len(), band.len() + 1 + 3);

        let expected_cuts: Vec<usize> = (0..2).map(|k| (cols[k].1 + cols[k + 1].0) / 2).collect();
        let sub_rects: Vec<&PageRect> = output.iter().filter(|r| r.top == merged.top).collect();
        assert_eq!(sub_rects.len(), 3);
        let mut left = merged.left;
        for (i, &cut) in expected_cuts.iter().enumerate() {
            assert_eq!(sub_rects[i].left, left);
            assert_eq!(sub_rects[i].right, cut);
            left = cut;
        }
        assert_eq!(sub_rects[2].left, left);
        assert_eq!(sub_rects[2].right, merged.right);

        // Disable check: dropping the `tall_enough` check from
        // `split_nonconforming` (treating every spanning block as eligible)
        // makes the headline ALSO split into 3, so `output.contains(&headline)`
        // fails.
    }

    // ---- 4: REGULARITY ----------------------------------------------------

    #[test]
    fn irregular_pitch_is_rejected_uniform_pitch_is_accepted() {
        let cell_w = 200;
        let cell_h = 80;
        let base_p = 240;

        // Uniform pitch {p, p}: accepted.
        let uniform_lefts = [0usize, base_p, 2 * base_p];
        let uniform_blocks: Vec<PageRect> = uniform_lefts
            .iter()
            .map(|&l| rect(l, 0, l + cell_w, cell_h))
            .collect();
        assert!(detect_column_raster(&uniform_blocks).is_some());

        // Irregular pitch {p, 2.2p}: with only 2 diffs the fitted pitch is
        // their average (1.6p), which misplaces the middle block ~0.6p away
        // from its predicted slot -- past POS_TOL (0.15 * 1.6p = 0.24p) -- so
        // it is dropped from the on-lattice set, leaving only 2 distinct
        // slots: below MIN_RASTER_COLS.
        let irregular_lefts = [0usize, base_p, base_p + (22 * base_p) / 10];
        let irregular_blocks: Vec<PageRect> = irregular_lefts
            .iter()
            .map(|&l| rect(l, 0, l + cell_w, cell_h))
            .collect();
        assert!(detect_column_raster(&irregular_blocks).is_none());

        // Disable check: removing the POS_TOL filter (pushing every
        // conforming block onto `on_lattice` regardless of `pos_diff`)
        // keeps the irregular case's middle block, reaching 3 distinct
        // slots, and `detect_column_raster` wrongly returns `Some`.
    }

    // ---- 5: OCCUPANCY -----------------------------------------------------

    #[test]
    fn a_lattice_supported_on_under_half_its_slots_is_rejected() {
        let cell_w = 200;
        let cell_h = 80;
        let p = 240;

        // Positive control: k = 0, 1, 2, 3 (4 consecutive, fully-occupied
        // slots) IS detected -- proving a subsequent `None` below is
        // specifically an occupancy failure, not an artifact of the fixture
        // never having enough blocks in the first place.
        let dense_lefts = [0usize, p, 2 * p, 3 * p];
        let dense_blocks: Vec<PageRect> = dense_lefts
            .iter()
            .map(|&l| rect(l, 0, l + cell_w, cell_h))
            .collect();
        assert!(detect_column_raster(&dense_blocks).is_some());

        // k = 0, 1, 2, 9 on the same pitch: 4 blocks give 3 diffs
        // {p, p, 7p}, whose median is `p` (robust -- the majority value
        // wins over the one outlier, exactly the robustness the module
        // docs describe), so the fit correctly recovers pitch `p` rather
        // than being dragged toward some average. That correctly-fitted
        // lattice then clears MIN_RASTER_COLS (4 distinct slots) and
        // k_max >= 2, but k_max = 9 means occupancy needs >= 5 occupied
        // slots; only 4 are.
        let sparse_lefts = [0usize, p, 2 * p, 9 * p];
        let sparse_blocks: Vec<PageRect> = sparse_lefts
            .iter()
            .map(|&l| rect(l, 0, l + cell_w, cell_h))
            .collect();
        assert!(detect_column_raster(&sparse_blocks).is_none());
    }

    // ---- 6: ORDER -----------------------------------------------------

    #[test]
    fn split_preserves_block_order_and_inserts_sub_rects_in_place() {
        let x0 = 0;
        let cell_w = 150;
        let gutter = 30;
        let cell_h = 50;
        let cols = lattice(x0, cell_w, gutter, 3);

        let col_a: Vec<PageRect> = cols.iter().map(|&(l, r)| rect(l, 0, r, cell_h)).collect();
        let merged = rect(cols[0].0, cell_h, cols[2].1, 2 * cell_h);
        let trailing = rect(cols[0].0, 2 * cell_h, cols[0].1, 3 * cell_h);

        // Input order: [colA0, colA1, colA2, merged, trailing].
        let blocks = vec![col_a[0], col_a[1], col_a[2], merged, trailing];

        let raster = detect_column_raster(&blocks).expect("a 3-column lattice must be detected");
        let output = split_nonconforming(&blocks, &raster);

        let cuts: Vec<usize> = (0..2).map(|k| (cols[k].1 + cols[k + 1].0) / 2).collect();
        let expected = vec![
            col_a[0],
            col_a[1],
            col_a[2],
            rect(merged.left, merged.top, cuts[0], merged.bottom),
            rect(cuts[0], merged.top, cuts[1], merged.bottom),
            rect(cuts[1], merged.top, merged.right, merged.bottom),
            trailing,
        ];

        assert_eq!(output, expected);
    }

    // ---- 7: DEGENERATE ----------------------------------------------------

    #[test]
    fn degenerate_inputs_yield_no_raster() {
        let empty: Vec<PageRect> = Vec::new();
        assert!(detect_column_raster(&empty).is_none());

        let single = vec![rect(0, 0, 100, 50)];
        assert!(detect_column_raster(&single).is_none());
    }
}
