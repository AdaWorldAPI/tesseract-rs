//! Drop-cap seam recovery (default-on) + loud page-level count.
//!
//! Closes the measured content loss at the drop-cap seam documented in
//! `CLAUDE.md` § "Drop cap — DIAGNOSED, and the obvious fix is MEASURED
//! HARMFUL". Two glyphs are lost today on the reference page: the
//! ornamental initial itself (an 81x72 blob landing in
//! [`crate::blob_filter::FilteredBlobs::large`], which no code in this
//! crate consumes — `segment.rs:146-157` builds `ToBlockCtx` with `blobs` +
//! `noise` only, so `filtered.large` dies there) and the following glyph,
//! whose ink merged into that same oversized component so the row's
//! ink-left starts to its right. This module delivers three pieces:
//!
//! 1. [`detect_drop_caps`] — a shape-qualified detector over
//!    [`crate::blob_filter::FilteredBlobs::large`]: population-relative
//!    height-SD + height-ratio + aspect windows, never an absolute pixel
//!    constant.
//! 2. [`seam_left_extension`] — an **x-only** row-crop extension that
//!    recovers the merged neighbour by widening a row's crop leftward, up
//!    to a bound derived from the row's own glyph spacing.
//! 3. [`count_page_drop_caps`] — a page-level count so the initial's own
//!    loss (never recovered — see "Text splice" below) is at least
//!    *reported*, not silent.
//!
//! ## Why x-only (the y-axis is measured harmful)
//!
//! `CLAUDE.md`'s drop-cap section measured, on the real page, that widening
//! a row's crop **vertically** to reach the cap destroys the entire line —
//! "ye hewn to eet very tired of" against the correct "ice was beginning to
//! get very tired of" — because a drop cap spans roughly 2-3 text-line
//! heights, so including it in the row band forces the whole band (and the
//! prescale that follows it) to ~3x its true height. The y-axis is
//! therefore **never** touched here: [`seam_left_extension`] only ever
//! *decreases* a row's `left` edge, bounded below by the cap's own
//! `bbox.0` and above by one glyph width, and never reads or writes a row's
//! `bottom`/`top`.
//!
//! ## Why the constants are population-relative, not absolute pixel counts
//!
//! Every threshold in [`detect_drop_caps`] is expressed against
//! [`OrdinaryScale`] — the pool's own mean and population standard
//! deviation of glyph height — so the same code classifies a drop cap at
//! any DPI, point size, or typeface without a single hardcoded pixel
//! literal. The measured reference-page cap: `height_sd = 8.01`
//! (`pool mean 25.40, sd 5.82, n 2389`); `MIN_HEIGHT_SD = 4.0` leaves a 2x
//! margin below that measurement (an ordinary capital letter measures
//! roughly 0.5 SD above its own line's mean, nowhere near either bound).
//!
//! [`seam_left_extension`]'s own reach constant is likewise never fixed:
//! the winning seam on the reference page measured 8 px, and
//! [`crate::lstm_recognizer::noise_readmit_reach`] (the sibling seam
//! primitive this one mirrors — same "judge a candidate by what surrounds
//! it, never by an absolute size" correction shape as the `xy_cut` gutter
//! fallback and the table-column bridge) reports an advance-half of
//! roughly 8-10 px at a 25.40 px mean glyph height on the same page. This
//! module accepts `reach` as a caller-supplied parameter for exactly that
//! reason — callers pass a population-relative value (typically
//! [`crate::lstm_recognizer::noise_readmit_reach`]'s own output, or an
//! equivalent row-local yardstick), never a literal.
//!
//! ## Text splice — deliberately NOT shipped here
//!
//! Prepending the cap's own recognized reading (e.g. "A" or "Ai") is out of
//! scope: the glyph alone decodes unstably across A/Ai/Bi depending on how
//! it is cropped, so `"Aiice..."` against the true `"Alice..."` is not
//! obviously an improvement over today's `"ice..."` — it trades a missing
//! opening for a misspelled word a downstream dictionary pass may then
//! "correct" confidently in the wrong direction. That needs `#51`'s
//! cross-text disambiguation plus a second drop-cap fixture before it is
//! attempted; this module ships only the seam recovery and the loud count.
//!
//! ## What this module owns
//!
//! Pure functions over already-classified blob geometry
//! (`(left, bottom, right, top)` tuples, y-UP page space, `top > bottom` —
//! [`crate::blob_filter::FilteredBlobs`]'s convention) plus one page-level
//! entry point that chains [`crate::conncomp::conn_comp_areas`] →
//! [`crate::blob_filter::filter_blobs`]. It calls no other module in this
//! crate, and in particular never touches row/line segmentation
//! (`segment.rs`) or the recognizer (`lstm_recognizer.rs`) — those are
//! wired to this module's output by the caller, not by this module.

/// The pool's own scale: mean and **population** standard deviation of
/// ordinary blob height, plus the sample count that produced them. Every
/// threshold in [`detect_drop_caps`] is expressed relative to this, never
/// as an absolute pixel constant — see the module docs' "Why the constants
/// are population-relative" section.
#[derive(Clone, Copy, Debug)]
pub struct OrdinaryScale {
    /// Mean height (`top - bottom`) over the pool.
    pub mean_h: f32,
    /// Population standard deviation of height over the pool (divides by
    /// `n`, not `n - 1` — this is a description of the pool as observed,
    /// not an estimate of a larger population).
    pub sd_h: f32,
    /// Number of blobs the scale was computed from.
    pub n: usize,
}

/// A blob admitted as a drop cap by [`detect_drop_caps`], carrying the
/// three population-relative measurements that qualified it.
///
/// `bbox` is `(left, bottom, right, top)`, y-UP page space, `top >
/// bottom` — the same convention as
/// [`crate::blob_filter::FilteredBlobs::large`].
#[derive(Clone, Copy, Debug)]
pub struct DropCap {
    /// The cap's bounding box, unchanged from the input `large` tuple.
    pub bbox: (i32, i32, i32, i32),
    /// `(height - mean_h) / sd_h`, or `f32::INFINITY` when `sd_h` is
    /// degenerate (see [`detect_drop_caps`]'s "Degenerate-sd rule").
    pub height_sd: f32,
    /// `height / mean_h`.
    pub height_ratio: f32,
    /// `width / height`.
    pub aspect: f32,
}

/// Minimum pool size before [`ordinary_scale`] will report a scale at all
/// — mirrors [`crate::lstm_recognizer::noise_readmit_reach`]'s own "fewer
/// than two blobs means there is no basis to judge anything by" decline,
/// scaled up for a page-wide population estimate rather than a single
/// row's local spacing.
pub const MIN_POOL: usize = 8;
/// Lower bound on `(height - mean_h) / sd_h`. `4.0` sits at half the
/// measured reference-page cap (`8.01`), a 2x safety margin below the one
/// real measurement this module has.
pub const MIN_HEIGHT_SD: f32 = 4.0;
/// Lower bound on `height / mean_h`.
pub const MIN_HEIGHT_RATIO: f32 = 1.8;
/// Upper bound on `height / mean_h` — excludes figures/halftones, which
/// measure well above this on the reference page.
pub const MAX_HEIGHT_RATIO: f32 = 6.0;
/// Lower bound on `width / height` — excludes rules/hairlines (very wide,
/// very short).
pub const MIN_ASPECT: f32 = 0.25;
/// Upper bound on `width / height` — excludes tall narrow marks.
pub const MAX_ASPECT: f32 = 1.6;
/// Fraction of a row's own band height that a candidate cap must vertically
/// overlap before [`seam_left_extension`] will consider it at all.
pub const MIN_BAND_OVERLAP_FRAC: f32 = 0.5;

/// Mean + **population** standard deviation of `top - bottom` over `pool`.
///
/// Returns `None` when `pool.len() < `[`MIN_POOL`] — mirrors
/// [`crate::lstm_recognizer::noise_readmit_reach`]'s "too few samples to
/// judge anything by" decline. A small pool's mean/sd would otherwise be
/// dominated by whichever handful of glyphs happened to be present, giving
/// [`detect_drop_caps`] no stable population to measure against.
///
/// ```text
/// pool heights: 18,20,22,24 (x2 each), n = 8
/// mean_h = (18+20+22+24) / 4          = 21.0
/// var    = mean((h - 21)^2)           = (9+1+1+9)*2 / 8 = 5.0
/// sd_h   = sqrt(var)                  ~= 2.236
/// ```
#[must_use]
pub fn ordinary_scale(pool: &[(i32, i32, i32, i32)]) -> Option<OrdinaryScale> {
    let n = pool.len();
    if n < MIN_POOL {
        return None;
    }
    let heights: Vec<f32> = pool.iter().map(|&(_, b, _, t)| (t - b) as f32).collect();
    let mean_h = heights.iter().sum::<f32>() / n as f32;
    let variance = heights
        .iter()
        .map(|h| (h - mean_h) * (h - mean_h))
        .sum::<f32>()
        / n as f32;
    let sd_h = variance.sqrt();
    Some(OrdinaryScale { mean_h, sd_h, n })
}

/// Shape-qualified drop-cap detector over `large` (typically
/// [`crate::blob_filter::FilteredBlobs::large`]), scaled by `pool`
/// (typically [`crate::blob_filter::FilteredBlobs::blobs`]).
///
/// A candidate is admitted iff **all** of:
/// - `height_sd >= `[`MIN_HEIGHT_SD`] (see the "Degenerate-sd rule" below
///   for what happens when the pool's own `sd_h` is ~0).
/// - `height_ratio` (`height / mean_h`) lies in
///   `[`[`MIN_HEIGHT_RATIO`]`, `[`MAX_HEIGHT_RATIO`]`]` — the floor guards
///   uniform-type pages where `sd_h` is tiny (this crate's generated
///   corpus is one font per page); the ceiling excludes figures, which
///   measure well past 2x a cap's own ratio headroom on the reference
///   page.
/// - `aspect` (`width / height`) lies in
///   `[`[`MIN_ASPECT`]`, `[`MAX_ASPECT`]`]` — excludes rules/hairlines
///   (very wide, short) and tall narrow marks.
///
/// There is deliberately **no position test** in this function — a cap
/// opening a column mid-page must classify identically to one opening the
/// page. Position (row adjacency) is tested once, downstream, at
/// [`seam_left_extension`]'s vertical-overlap check.
///
/// ## Degenerate-sd rule
///
/// When `pool`'s `sd_h <= f32::EPSILON` (an exactly- or near-uniform
/// pool), `height_sd` is set to `f32::INFINITY` and the SD test is
/// unconditionally skipped — the ratio window alone decides. This is
/// deliberate, not an oversight: computing `(height - mean_h) / sd_h`
/// directly against a near-zero `sd_h` would produce `NaN` whenever
/// `height == mean_h` too, and every comparison against a `NaN` in Rust is
/// `false` — so an un-special-cased degenerate pool would **silently
/// decline every candidate**, exactly backwards from the intended
/// behaviour (a uniform pool should make the ratio window the sole judge,
/// not veto everything). Forcing `f32::INFINITY` instead guarantees the SD
/// comparison always evaluates to `true` and never reaches a `NaN`.
#[must_use]
pub fn detect_drop_caps(
    large: &[(i32, i32, i32, i32)],
    pool: &[(i32, i32, i32, i32)],
) -> Vec<DropCap> {
    let Some(scale) = ordinary_scale(pool) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for &(l, b, r, t) in large {
        let height = (t - b) as f32;
        let width = (r - l) as f32;
        if height <= 0.0 || width <= 0.0 {
            continue;
        }

        let height_sd = if scale.sd_h <= f32::EPSILON {
            f32::INFINITY
        } else {
            (height - scale.mean_h) / scale.sd_h
        };
        let height_ratio = height / scale.mean_h;
        let aspect = width / height;

        let sd_ok = height_sd >= MIN_HEIGHT_SD;
        let ratio_ok = (MIN_HEIGHT_RATIO..=MAX_HEIGHT_RATIO).contains(&height_ratio);
        let aspect_ok = (MIN_ASPECT..=MAX_ASPECT).contains(&aspect);

        if sd_ok && ratio_ok && aspect_ok {
            out.push(DropCap {
                bbox: (l, b, r, t),
                height_sd,
                height_ratio,
                aspect,
            });
        }
    }
    out
}

/// Median of `values`, sorting a private copy in place. Standard
/// odd/even-length definition (middle value; average of the two middle
/// values). Callers are responsible for ensuring `values` is non-empty —
/// [`seam_left_extension`] only calls this after its own `row_spans.len()
/// >= 2` guard.
fn median_i32(values: &mut [i32]) -> f32 {
    values.sort_unstable();
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2] as f32
    } else {
        (values[n / 2 - 1] as f32 + values[n / 2] as f32) / 2.0
    }
}

/// The x-only row-crop seam extension: how far a row's `left` edge should
/// move to recover a neighbour merged into an adjacent drop cap's blob.
///
/// `row_spans` are the row's own `(left, right)` glyph spans (any order,
/// at least 2 required — mirrors
/// [`crate::lstm_recognizer::noise_readmit_reach`]'s "too few blobs to
/// judge anything by" decline). `reach` is a population-relative distance
/// yardstick supplied by the caller (see the module docs) — this function
/// never invents one internally.
///
/// Returns `None` unless **all** of:
/// - `row_spans.len() >= 2` and `reach` is finite and positive.
/// - some cap in `caps` vertically overlaps the row's `[row_bottom,
///   row_top]` band by at least [`MIN_BAND_OVERLAP_FRAC`] of the band's
///   own height.
/// - that cap is horizontally adjacent: `|row_left - cap.bbox.2| <=
///   reach` (`cap.bbox.2` is the cap's right edge).
///
/// When multiple caps qualify, the one with the **largest** vertical
/// overlap wins; ties break on the **smallest** `cap.bbox.0` (left edge).
///
/// The extension itself is bounded by **two independent neighbourhood
/// yardsticks, the smaller of which binds** — never an absolute pixel
/// count:
///
/// ```text
/// med_w    = median(row_spans widths)
/// ext      = round(min(reach, med_w))
/// new_left = max(cap.bbox.0, min(row_left, cap.bbox.2) - ext).max(0)
/// ```
///
/// `new_left` is returned as `Some` only when it is strictly less than
/// `row_left` (a genuine leftward move); it can never fall below the
/// cap's own `bbox.0` (the seam never reaches into the cap's body) nor
/// move by more than `med_w` (one glyph width) even under an oversized
/// `reach`.
#[must_use]
pub fn seam_left_extension(
    row_left: i32,
    row_bottom: f32,
    row_top: f32,
    row_spans: &[(i32, i32)],
    caps: &[DropCap],
    reach: f32,
) -> Option<i32> {
    if row_spans.len() < 2 || !reach.is_finite() || reach <= 0.0 {
        return None;
    }
    let band_h = row_top - row_bottom;
    if band_h <= 0.0 {
        return None;
    }
    let min_overlap = MIN_BAND_OVERLAP_FRAC * band_h;

    let mut best: Option<(&DropCap, f32)> = None;
    for cap in caps {
        let (cap_left, cap_bottom, cap_right, cap_top) = cap.bbox;
        let overlap = (cap_top as f32).min(row_top) - (cap_bottom as f32).max(row_bottom);
        if overlap < min_overlap {
            continue;
        }
        let horiz_gap = (row_left - cap_right).abs() as f32;
        if horiz_gap > reach {
            continue;
        }
        let is_better = match best {
            None => true,
            Some((prev, prev_overlap)) => {
                overlap > prev_overlap || (overlap == prev_overlap && cap_left < prev.bbox.0)
            }
        };
        if is_better {
            best = Some((cap, overlap));
        }
    }
    let (cap, _) = best?;
    let (cap_left, _, cap_right, _) = cap.bbox;

    let mut widths: Vec<i32> = row_spans.iter().map(|&(l, r)| r - l).collect();
    let med_w = median_i32(&mut widths);

    let ext = reach.min(med_w).round() as i32;
    let inner = row_left.min(cap_right) - ext;
    let new_left = inner.max(cap_left).max(0);

    if new_left < row_left {
        Some(new_left)
    } else {
        None
    }
}

/// Page-level drop-cap count: `binary` (foreground = `byte == 0`, same
/// convention as [`crate::conncomp::conn_comp_areas`]) → connected
/// components → **y-UP page-space flip** → [`crate::blob_filter::filter_blobs`]
/// → [`detect_drop_caps`] over the resulting `large`/`blobs` buckets.
///
/// This is the "make the loss loud" half of this module (see the module
/// docs' "Text splice" section for what is deliberately *not* done about
/// the loss itself): a caller wires this into `doc.v1`'s quality signal so
/// a page carrying an unrecovered drop cap is flagged rather than silently
/// truncated.
///
/// The flip mirrors [`crate::blob_filter`]'s own documented "not the real
/// pipeline's page-coordinate flip, just the minimal relabeling that
/// satisfies the `top > bottom` invariant" — `conn_comp_areas` reports
/// boxes in raster space (`y` increasing downward), and
/// `c.bb.y = h - (c.bb.y + c.bb.h)` maps that onto y-UP page space
/// before `filter_blobs`'
/// `box_tuple` relabels `(x, y, x+w, y+h)` as `(left, bottom, right,
/// top)`.
#[must_use]
pub fn count_page_drop_caps(binary: &[u8], w: usize, h: usize) -> usize {
    let mut comps = crate::conncomp::conn_comp_areas(binary, w, h, 8);
    let page_h = h as i32;
    for c in &mut comps {
        c.bb.y = page_h - (c.bb.y + c.bb.h);
    }
    let filtered = crate::blob_filter::filter_blobs(&comps);
    detect_drop_caps(&filtered.large, &filtered.blobs).len()
}

// ---------------------------------------------------------------------
// Falsifiers — disable table (ORCHESTRATOR EXECUTES)
//
// This worker cannot run `cargo test` (repo policy: Sonnet workers edit,
// the orchestrator compiles/tests centrally). Each entry below names the
// exact code change that disables the guard the paired test exists to
// prove, and the failure the orchestrator should observe when running that
// single test with the change applied. None of these edits should be
// landed — they are a checklist for the orchestrator's gate, not a patch.
//
// 1. detector_admits_a_cap_and_rejects_a_rule_and_a_figure
//    - Remove the `aspect_ok` clause from `detect_drop_caps`'s admission
//      test (or widen MAX_ASPECT past 10.0): the 600x60 "rule" fixture is
//      now also admitted -> `detected.len()` becomes 2, not 1.
//    - Widen MAX_HEIGHT_RATIO past ~9.6 (or delete the `height_ratio <=
//      MAX_HEIGHT_RATIO` half of `ratio_ok`): the 260x200 "figure" fixture
//      is now also admitted -> `detected.len()` becomes 2 (or 3 if both
//      edits are applied), not 1.
// 2. sd_and_ratio_guards_are_each_independently_load_bearing
//    - Delete the `sd_ok` clause (or set MIN_HEIGHT_SD to 0.0): the
//      high-sd-pool candidate (height_sd ~2.0, in-window ratio 2.0) is now
//      admitted -> the first `assert!(out.is_empty())` fails.
//    - Delete the `ratio_ok` lower bound (or set MIN_HEIGHT_RATIO to
//      0.0): the uniform-pool candidate (ratio 1.5, high sd) is now
//      admitted -> the second `assert!(out2.is_empty())` fails.
// 3. seam_fires_on_an_adjacent_cap_and_stays_silent_on_a_distant_row
//    - Delete the `horiz_gap > reach` continue (always accept): the
//      "distant" case now returns `Some(..)` instead of `None`.
//    - Replace `ext = reach.min(med_w).round()` with a literal (e.g. `8`):
//      the `Some(100 - ext)` equality assertion fails for any pool whose
//      real `min(reach, med_w)` is not exactly that literal.
// 4. seam_never_reaches_into_the_cap_body
//    - Delete the `.max(cap_left)` clamp: `new_left` can fall below
//      `cap.bbox.0` under an oversized `reach` -> the `new_left >=
//      cap.bbox.0` assertion fails.
//    - Replace `ext = reach.min(med_w)` with `ext = reach` (drop the
//      `med_w` bound): `row_left - new_left` grows to ~1000 (the
//      oversized reach) -> the `<= med_w` assertion fails.
// 5. seam_requires_the_cap_to_span_the_row
//    - Delete the `overlap < min_overlap` continue (accept any cap
//      regardless of vertical position): the "wholly-above" case now
//      returns `Some(..)` instead of `None`.
// 6. seam_scales_exactly_2x_with_a_2x_layout
//    - Replace `ext = reach.min(med_w).round()` with a hardcoded literal:
//      `scaled_ext` no longer equals `base_ext * 2` (a literal is
//      scale-blind by construction).
// 7. synthetic_page_end_to_end
//    - Any of the above disables applied to the *real* page pipeline
//      (`conn_comp_areas` -> flip -> `filter_blobs` -> `detect_drop_caps`
//      -> `seam_left_extension`) breaks the corresponding precondition or
//      per-row assertion in this end-to-end test the same way it breaks
//      the isolated unit test above.
// 8. uniform_pool_sd_zero_still_classifies_by_ratio
//    - Delete the `scale.sd_h <= f32::EPSILON` special case (compute
//      `height_sd` unconditionally as `(height - mean_h) / sd_h`): with
//      `sd_h == 0.0` the 60-height candidate's `height_sd` becomes `+inf`
//      by IEEE 754 division-by-zero rules (still admits, so this
//      particular disable is silent on the admit half) but the
//      `24`-height "silence" candidate's `height_sd` becomes `NaN` (`(24 -
//      20) / 0.0` -> `+inf`, still not NaN here since numerator is
//      nonzero — the real trap is a pool where a *different* candidate's
//      height exactly equals `mean_h`, giving `0.0 / 0.0 = NaN`; either
//      way, `admitted[0].height_sd.is_infinite()` stops being reliably
//      `true` once the explicit branch is removed, and the module doc's
//      own claim ("never NaN") stops holding).
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_admits_a_cap_and_rejects_a_rule_and_a_figure() {
        // pool: 8 blobs, heights 18/20/22/24 (x2 each) -> mean 21.0.
        let pool: Vec<(i32, i32, i32, i32)> = [18, 20, 22, 24, 18, 20, 22, 24]
            .iter()
            .map(|&hgt| (0, 0, hgt, hgt))
            .collect();
        assert_eq!(pool.len(), MIN_POOL);
        let scale = ordinary_scale(&pool).expect("pool meets MIN_POOL");
        assert_eq!(scale.mean_h, 21.0);

        // A genuine drop cap: near-square, ~3.4x mean height.
        let cap = (0, 0, 72, 72);
        // A horizontal rule: moderate height (ratio/sd both pass) but far
        // too wide to be a glyph -- excluded only by the aspect window.
        let rule = (0, 0, 600, 60);
        // A figure: aspect passes (near-square-ish) but height ratio is
        // far past MAX_HEIGHT_RATIO -- excluded only by the ratio ceiling.
        let figure = (0, 0, 260, 200);
        let large = [cap, rule, figure];

        let detected = detect_drop_caps(&large, &pool);
        assert_eq!(detected.len(), 1, "exactly the cap must be admitted");
        assert_eq!(detected[0].bbox, cap);
    }

    #[test]
    fn sd_and_ratio_guards_are_each_independently_load_bearing() {
        // (i) High-sd pool: a candidate whose ratio is comfortably
        // in-window (2.0) must still be declined once its height_sd falls
        // below MIN_HEIGHT_SD.
        let pool_high_sd: Vec<(i32, i32, i32, i32)> = [10, 10, 10, 10, 30, 30, 30, 30]
            .iter()
            .map(|&hgt| (0, 0, hgt, hgt))
            .collect();
        let scale = ordinary_scale(&pool_high_sd).expect("pool meets MIN_POOL");
        assert_eq!(scale.mean_h, 20.0);
        assert_eq!(scale.sd_h, 10.0);
        let candidate = (0, 0, 40, 40); // height 40, ratio 2.0, height_sd 2.0
        let out = detect_drop_caps(&[candidate], &pool_high_sd);
        assert!(
            out.is_empty(),
            "ratio-in-window candidate must still be declined when SD is below MIN_HEIGHT_SD"
        );

        // (ii) Tight (low-sd) pool: a candidate whose height_sd is
        // comfortably above MIN_HEIGHT_SD must still be declined once its
        // ratio falls below MIN_HEIGHT_RATIO.
        let pool_tight: Vec<(i32, i32, i32, i32)> = [18, 18, 18, 18, 22, 22, 22, 22]
            .iter()
            .map(|&hgt| (0, 0, hgt, hgt))
            .collect();
        let scale2 = ordinary_scale(&pool_tight).expect("pool meets MIN_POOL");
        assert_eq!(scale2.mean_h, 20.0);
        assert_eq!(scale2.sd_h, 2.0);
        let candidate2 = (0, 0, 30, 30); // height 30, ratio 1.5, height_sd 5.0
        let out2 = detect_drop_caps(&[candidate2], &pool_tight);
        assert!(
            out2.is_empty(),
            "high-sd candidate must still be declined when ratio is below MIN_HEIGHT_RATIO"
        );
    }

    #[test]
    fn seam_fires_on_an_adjacent_cap_and_stays_silent_on_a_distant_row() {
        let cap = DropCap {
            bbox: (0, 0, 100, 40),
            height_sd: 10.0,
            height_ratio: 3.0,
            aspect: 1.0,
        };
        let row_bottom = 0.0_f32;
        let row_top = 20.0_f32;
        let reach = 25.0_f32;
        let med_w = 40.0_f32; // (30 + 50) / 2, both scenarios below

        // Adjacent: row_left == cap.bbox.2 (the cap's right edge) -> Some.
        let row_spans_adj = [(100, 130), (150, 200)]; // widths 30, 50
        let some = seam_left_extension(100, row_bottom, row_top, &row_spans_adj, &[cap], reach);
        let expected_ext = reach.min(med_w).round() as i32;
        assert_eq!(some, Some(100 - expected_ext));
        assert_eq!(100 - some.unwrap(), expected_ext);

        // Distant: row_left == cap.bbox.2 + 4*reach -> None.
        let far_left = 100 + (4.0 * reach) as i32;
        let row_spans_far = [(far_left, far_left + 30), (far_left + 40, far_left + 90)];
        let none =
            seam_left_extension(far_left, row_bottom, row_top, &row_spans_far, &[cap], reach);
        assert_eq!(none, None);
    }

    #[test]
    fn seam_never_reaches_into_the_cap_body() {
        // A wide cap, touching the row's left edge exactly (gap == 0), and
        // a deliberately oversized reach -- the extension must still be
        // bounded by the row's own median glyph width, not by the reach.
        let cap = DropCap {
            bbox: (0, 0, 500, 40),
            height_sd: 10.0,
            height_ratio: 3.0,
            aspect: 1.0,
        };
        let row_left = 500; // == cap.bbox.2
        let row_bottom = 0.0_f32;
        let row_top = 20.0_f32;
        let row_spans = [(500, 530), (550, 600)]; // widths 30, 50 -> med_w 40.0
        let med_w = 40.0_f32;
        let reach = 1000.0_f32; // far larger than the cap or med_w

        let new_left =
            seam_left_extension(row_left, row_bottom, row_top, &row_spans, &[cap], reach)
                .expect("adjacent, fully-spanning cap must fire");

        assert!(
            new_left >= cap.bbox.0,
            "seam must never cross into the cap's own body"
        );
        assert!(
            (row_left - new_left) as f32 <= med_w,
            "advance must be bounded by the row's own median glyph width, not the oversized reach"
        );
    }

    #[test]
    fn seam_requires_the_cap_to_span_the_row() {
        let row_bottom = 0.0_f32;
        let row_top = 20.0_f32;
        let row_spans = [(100, 130), (150, 190)];
        let reach = 15.0_f32;

        // Spanning cap: vertically overlaps the row's [0, 20] band by its
        // full height (well above MIN_BAND_OVERLAP_FRAC).
        let spanning = DropCap {
            bbox: (80, -10, 100, 30),
            height_sd: 10.0,
            height_ratio: 3.0,
            aspect: 1.0,
        };
        let some = seam_left_extension(100, row_bottom, row_top, &row_spans, &[spanning], reach);
        assert!(some.is_some());

        // Wholly-above cap: its bottom (25) sits above the row's top
        // (20) -- zero vertical overlap, same horizontal position.
        let above = DropCap {
            bbox: (80, 25, 100, 65),
            height_sd: 10.0,
            height_ratio: 3.0,
            aspect: 1.0,
        };
        let none = seam_left_extension(100, row_bottom, row_top, &row_spans, &[above], reach);
        assert_eq!(none, None);
    }

    #[test]
    fn seam_scales_exactly_2x_with_a_2x_layout() {
        let cap = DropCap {
            bbox: (50, 0, 100, 40),
            height_sd: 10.0,
            height_ratio: 3.0,
            aspect: 1.0,
        };
        let row_spans = [(100, 140), (150, 190), (200, 240)]; // widths all 40
        let reach = 15.0_f32;

        let base = seam_left_extension(100, 0.0, 20.0, &row_spans, &[cap], reach)
            .expect("baseline seam must fire");
        let base_ext = 100 - base;

        let cap2 = DropCap {
            bbox: (
                cap.bbox.0 * 2,
                cap.bbox.1 * 2,
                cap.bbox.2 * 2,
                cap.bbox.3 * 2,
            ),
            height_sd: cap.height_sd,
            height_ratio: cap.height_ratio,
            aspect: cap.aspect,
        };
        let row_spans2: Vec<(i32, i32)> = row_spans.iter().map(|&(l, r)| (l * 2, r * 2)).collect();
        let reach2 = reach * 2.0;

        let scaled = seam_left_extension(200, 0.0, 40.0, &row_spans2, &[cap2], reach2)
            .expect("scaled seam must fire");
        let scaled_ext = 200 - scaled;

        assert_eq!(
            scaled_ext,
            base_ext * 2,
            "extension must scale exactly with a uniform 2x layout"
        );
    }

    #[test]
    fn uniform_pool_sd_zero_still_classifies_by_ratio() {
        let pool: Vec<(i32, i32, i32, i32)> = (0..8).map(|_| (0, 0, 20, 20)).collect();
        let scale = ordinary_scale(&pool).expect("pool meets MIN_POOL");
        assert_eq!(scale.mean_h, 20.0);
        assert_eq!(
            scale.sd_h, 0.0,
            "an identical-height pool must have exactly zero population sd"
        );

        // Admits: 3x mean height (ratio 3.0, in-window). The sd test is
        // bypassed via the degenerate-sd rule (height_sd forced to
        // +INFINITY, never NaN).
        let cap = (0, 0, 60, 60);
        let admitted = detect_drop_caps(&[cap], &pool);
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].bbox, cap);
        assert!(
            admitted[0].height_sd.is_infinite() && admitted[0].height_sd > 0.0,
            "degenerate sd must reach +infinity, never NaN"
        );

        // Silence: 1.2x mean height -- ratio 1.2 is below MIN_HEIGHT_RATIO,
        // so it must be declined even though the (bypassed) sd test would
        // have passed trivially.
        let too_small_ratio = (0, 0, 24, 24);
        let declined = detect_drop_caps(&[too_small_ratio], &pool);
        assert!(declined.is_empty());
    }

    /// Mirrors `rectify.rs`'s own `draw_hollow_rect` test helper
    /// (`crates/tesseract-ocr/src/rectify.rs:485-504`): a hollow (2px
    /// border, white interior) rectangle. `filter_blobs`' density
    /// heuristic (`pixel_count >= height*width*0.7` -> "too dense to be
    /// text") rejects a SOLID rectangle outright (a real bug `rectify.rs`
    /// hit once — see its own doc comment); a hollow border keeps density
    /// around 30%, comfortably under the threshold, while remaining one
    /// 8-connected component spanning the requested extent.
    fn draw_hollow_rect(
        buf: &mut [u8],
        w: usize,
        h: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
    ) {
        let border = 2usize;
        for y in y0..y1.min(h) {
            for x in x0..x1.min(w) {
                let on_border =
                    y < y0 + border || y + border >= y1 || x < x0 + border || x + border >= x1;
                if on_border {
                    buf[y * w + x] = 0; // ink
                }
            }
        }
    }

    #[test]
    fn synthetic_page_end_to_end() {
        let w = 900usize;
        let h = 400usize;
        let mut page = vec![255u8; w * h];

        // A 72x72 hollow "cap", raster x=[178,250) y=[30,102) -- clear of
        // every row rect below (row rects start at x0=260, a 10px gap past
        // the cap's raster right edge, so the two never touch under
        // 8-connectivity).
        draw_hollow_rect(&mut page, w, h, 178, 30, 250, 102);

        // 4 rows x 8 rects, heights 18/20/22/24, raster y0 spaced 70px
        // apart (well clear of the 72px-tall cap for rows 1-3).
        let row_heights = [18usize, 20, 22, 24];
        let mut row_y0s = [0usize; 4];
        for (i, &rh) in row_heights.iter().enumerate() {
            let y0 = 40 + i * 70;
            row_y0s[i] = y0;
            for k in 0..8usize {
                let x0 = 260 + k * 70;
                let x1 = x0 + 40;
                draw_hollow_rect(&mut page, w, h, x0, y0, x1, y0 + rh);
            }
        }

        // ---- Preconditions, asserted first. ----
        let mut comps = crate::conncomp::conn_comp_areas(&page, w, h, 8);
        let page_h = h as i32;
        for c in &mut comps {
            c.bb.y = page_h - (c.bb.y + c.bb.h);
        }
        let filtered = crate::blob_filter::filter_blobs(&comps);
        assert_eq!(
            filtered.large.len(),
            1,
            "exactly the 72x72 cap must exceed the pool's max_y and land in .large \
             (structurally proves cap height 72 > max_y, per CLAUDE.md's ~58 measurement \
             on the real page -- FilteredBlobs exposes no max_y field directly)"
        );
        assert!(
            filtered.blobs.len() >= MIN_POOL,
            "all 32 ordinary row rects must survive filter_blobs into the pool"
        );

        // ---- detect + count. ----
        let detected = detect_drop_caps(&filtered.large, &filtered.blobs);
        assert_eq!(detected.len(), 1);
        let cap = detected[0];
        // Cross-check against the hand-computed flipped bbox: raster
        // x=[178,250) y=[30,102) -> page-space (178, h-102, 250, h-30).
        assert_eq!(cap.bbox, (178, h as i32 - 102, 250, h as i32 - 30));

        let counted = count_page_drop_caps(&page, w, h);
        assert_eq!(counted, 1);

        // ---- seam: Some for row 0 (spans the cap), None for rows 1-3
        // (silence twin -- same page, same cap, no vertical overlap). ----
        let reach = 15.0_f32;
        for (i, &rh) in row_heights.iter().enumerate() {
            let y0 = row_y0s[i];
            let row_bottom = (page_h - (y0 as i32 + rh as i32)) as f32;
            let row_top = (page_h - y0 as i32) as f32;
            let row_left = 260;
            let row_spans: Vec<(i32, i32)> = (0..8usize)
                .map(|k| {
                    let x0 = 260 + (k as i32) * 70;
                    (x0, x0 + 40)
                })
                .collect();
            let result =
                seam_left_extension(row_left, row_bottom, row_top, &row_spans, &[cap], reach);
            if i == 0 {
                let new_left = result.expect("row 0 sits under the cap and must recover a seam");
                assert!(new_left < row_left);
            } else {
                assert_eq!(
                    result, None,
                    "row {i} does not vertically span the cap -- must stay silent"
                );
            }
        }
    }
}
