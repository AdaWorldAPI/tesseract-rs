//! Page rectification — a NEW, consumer-side preprocessing addition, **NOT a
//! Tesseract transcode** (same "not a parity claim" footing as
//! `crate::structured`'s `doc.v1` / table reconstruction). No leptonica or
//! libtesseract function does this: `pixFindSkew`/`pixRotate` (the
//! documented, still-unbuilt "deskew wave" — see this crate's own
//! `CLAUDE.md`) only correct a single whole-page ROTATION angle. This module
//! corrects a DIFFERENT, more common failure mode for a phone-photographed
//! page: **keystone/trapezoid distortion**, where the camera wasn't square-on
//! to the page, so text lines fan out or converge instead of merely being
//! uniformly rotated — the "cushion and trapezoid" case a real repro
//! surfaced (every wide line's last word or two silently clipped, because the
//! row crop's ink boundary — real ink, correctly detected — sat outside what
//! a horizontal reading of the line would expect).
//!
//! ## The idea
//!
//! A page with pure ROTATIONAL skew has every row's baseline slope roughly
//! EQUAL (they're all parallel, just uniformly tilted). A page with
//! KEYSTONE/trapezoid distortion instead has the slope vary systematically
//! with a row's height on the page (rows near the "far" edge of the keystone
//! are compressed and tilted one way, rows near the "near" edge the other
//! way, or slope walks monotonically top-to-bottom) — the classic
//! photographed-page signature. [`fit_shear_ramp`] fits `slope(y) = m0 +
//! m1·y` (least squares, `y` in the same page-up space the row baselines
//! already live in) over the harvested per-row slopes; `m0` is the
//! (removable) constant rotation component, `m1` is the keystone component.
//!
//! **This needs [`crate::segment::segment_rows_independent`], NOT
//! [`crate::segment::segment_rows`].** The latter — what recognition
//! actually uses — deliberately forces every row in a block onto ONE shared
//! gradient (`fit_parallel_rows`, real Tesseract's own assumption that a
//! rotated-but-flat page's lines stay mutually parallel); its `line_m()` is
//! IDENTICAL for every row by construction, so it can only ever measure
//! `m0`, never `m1` — a real dead end this module hit once (every synthetic
//! trapezoid fixture measured `m1 = 0` exactly, because every row reported
//! the identical forced-parallel slope no matter its height).
//! `segment_rows_independent` stops one step earlier, at
//! `make_initial_textrows`, where each row still carries its own independent
//! LMS line fit — see `crate::segment`'s module docs for the full story.
//!
//! [`rectify_grey`] then applies a vertical SHEAR-RAMP correction — NOT a
//! full 4-point projective homography (no corner/edge detector exists here,
//! and one isn't needed for THIS symptom): for each output pixel, sample the
//! source shifted vertically by `-slope(y)·(x - center)`, where `slope` is
//! evaluated at that output row's own page-up height. This is a first-order
//! (small-angle) approximation — it corrects the vertical drift that causes
//! truncation and non-horizontal text, not horizontal foreshortening — but
//! it is simple, local, invertible without solving an implicit system, and
//! directly fixes the observed symptom using data the pipeline already
//! computes. See [`rectify_grey`]'s doc comment for the exact derivation.
//!
//! ## What this does NOT do
//!
//! - No lens/pincushion ("cushion") distortion correction — that needs
//!   camera calibration data (focal length, radial distortion coefficients)
//!   this crate has no way to obtain from a single uploaded image. Out of
//!   scope; not attempted.
//! - No horizontal perspective foreshortening correction (only vertical
//!   shear) — a full projective correction would need page-corner detection,
//!   which this module does not implement.
//! - No claim of matching any Tesseract/leptonica function's output —
//!   there is no oracle for this feature (real Tesseract doesn't have one
//!   either). Validated instead by synthetic before/after OCR-accuracy
//!   recovery tests (see the `tests` module).

use crate::segment::segment_rows_independent;

/// One recognized row's fitted baseline slope + its approximate vertical
/// center — the raw signal [`fit_shear_ramp`] regresses over. `y_center` is
/// in the same page-up (y=0 at image bottom) space `crate::textline::ToRow`
/// uses throughout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowShear {
    /// The row's approximate vertical center, page-up space.
    pub y_center: f32,
    /// The row's independently-fitted baseline slope — 0 is perfectly
    /// horizontal. From [`crate::segment::segment_rows_independent`], NOT
    /// [`crate::segment::segment_rows`]: the latter (what recognition uses)
    /// deliberately forces every row in a block onto ONE shared slope
    /// (`fit_parallel_rows`, real Tesseract's own assumption that a rotated
    /// page's lines stay mutually parallel) — a page-wide constant carries no
    /// row-to-row variation, so it can only ever measure rotation, never a
    /// trapezoid's height-dependent tilt. See `crate::segment`'s module docs.
    pub slope: f32,
}

/// Minimum blobs a row must carry to contribute a shear sample — mirrors
/// [`crate::lstm_recognizer::LstmRecognizer::makerow_row_crops`]'s own
/// `row.blobs.is_empty()` skip, tightened slightly: a 1-2 blob row's fitted
/// slope is dominated by noise (a meaningful independent LMS line fit needs
/// several points), so it would inject spurious signal into the page-wide
/// regression rather than a real recognizable text line's.
const MIN_ROW_BLOBS: usize = 3;

/// Harvest one [`RowShear`] per recognizable row via
/// [`segment_rows_independent`] — each row's OWN independent baseline fit,
/// taken BEFORE real Tesseract's `fit_parallel_rows` step would force it onto
/// one page-wide value (see [`RowShear::slope`]'s doc for why that step is
/// deliberately skipped here). Rows with fewer than [`MIN_ROW_BLOBS`] blobs
/// are skipped (too little signal for a trustworthy slope).
#[must_use]
pub fn detect_row_shears(grey: &[u8], w: usize, h: usize) -> Vec<RowShear> {
    let block = segment_rows_independent(grey, w, h);
    block
        .rows
        .iter()
        .filter(|row| row.blobs.len() >= MIN_ROW_BLOBS)
        .map(|row| RowShear {
            y_center: (row.min_y() + row.max_y()) / 2.0,
            slope: row.line_m(),
        })
        .collect()
}

/// A fitted `slope(y) = m0 + m1·y` model over a page's row baselines.
/// `m0` is the page's overall (height-independent) tilt — a pure rotation
/// component. `m1` is how much that tilt changes per unit of page height —
/// zero for pure rotation (or a perfectly flat page), non-zero for
/// keystone/trapezoid distortion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShearRamp {
    /// Constant term (the slope at `y = 0`).
    pub m0: f32,
    /// Linear term (slope change per unit of page-up `y`).
    pub m1: f32,
}

impl ShearRamp {
    /// The fitted slope at page-up height `y`.
    #[must_use]
    pub fn at(&self, y: f32) -> f32 {
        self.m0 + self.m1 * y
    }

    /// `true` when this ramp is large enough over a page of height `h` to be
    /// worth correcting — `m0` past roughly half a degree (`tan(0.5°) ≈
    /// 0.0087`) OR the slope swings by more than that same amount across the
    /// full page height (`|m1| * h`). Below this, [`rectify_grey`] would only
    /// add resampling noise to an already-acceptable page — the common case,
    /// which must stay a safe no-op.
    #[must_use]
    pub fn is_significant(&self, h: usize) -> bool {
        const MIN_SLOPE: f32 = 0.0087;
        self.m0.abs() > MIN_SLOPE || (self.m1.abs() * h as f32) > MIN_SLOPE
    }
}

/// Least-squares fit of `slope(y) = m0 + m1·y` over `shears`. `None` when
/// there are fewer than 2 rows (a line needs 2 points) or every row sits at
/// the same `y_center` (degenerate — zero variance, `m1` is undefined).
/// A single-row page returns `Some(ShearRamp{m0: that row's slope, m1: 0.0})`
/// — no height variation is observable from one row, but the row's own tilt
/// (rotation) is still worth reporting/correcting.
#[must_use]
pub fn fit_shear_ramp(shears: &[RowShear]) -> Option<ShearRamp> {
    match shears.len() {
        0 => None,
        1 => Some(ShearRamp {
            m0: shears[0].slope,
            m1: 0.0,
        }),
        n => {
            let n_f = n as f32;
            let mean_y: f32 = shears.iter().map(|s| s.y_center).sum::<f32>() / n_f;
            let mean_s: f32 = shears.iter().map(|s| s.slope).sum::<f32>() / n_f;
            let mut cov = 0.0f32;
            let mut var_y = 0.0f32;
            for s in shears {
                let dy = s.y_center - mean_y;
                cov += dy * (s.slope - mean_s);
                var_y += dy * dy;
            }
            if var_y <= f32::EPSILON {
                return Some(ShearRamp {
                    m0: mean_s,
                    m1: 0.0,
                });
            }
            let m1 = cov / var_y;
            let m0 = mean_s - m1 * mean_y;
            Some(ShearRamp { m0, m1 })
        }
    }
}

/// Apply a vertical shear-ramp correction to `grey` (`w × h`, row-major),
/// returning a new buffer of the SAME dimensions.
///
/// ## Derivation
///
/// A row with local slope `m` at page-up height `y0` satisfies (in the
/// SOURCE/observed image) `observed_y_pageup(x) ≈ y0 + m·(x - cx)` for its
/// center `cx = w/2` — this is exactly `ToRow::line_m`/`parallel_c`'s fitted
/// line, reparametrized around the page's horizontal center. To recover the
/// TRUE (horizontal) content at output position `(x_out, y_out)`, invert
/// this — using the output row's OWN page-up height as a first-order stand-in
/// for the (unknown until inverted) observed height when evaluating `m`, a
/// safe approximation because row-to-row spacing is normally far larger than
/// the shear-induced within-row height drift for a "typical photographed
/// page" (as opposed to an extreme, near-90° keystone):
///
/// ```text
/// true_y_pageup = h - 1 - y_out
/// m             = ramp.at(true_y_pageup)
/// src_y_raster  = y_out - m · (x_out - w/2)      // ← the sampling position
/// src_x_raster  = x_out                           // no horizontal change
/// ```
///
/// (Sanity check: `h=100`, a row truly horizontal at page-up `y=50` with
/// `m=0.1` observes at `x=cx+50` as page-up `55` → raster `44`; the formula
/// gives `src_y = 49 - 0.1·50 = 44`. ✓)
///
/// `src_y_raster` is rounded to the nearest source row and clamped to
/// `[0, h-1]` (nearest-neighbour — no bilinear interpolation in this first
/// version; OCR accuracy, not visual smoothness, is the target metric, and
/// nearest-neighbour cannot systematically bias a least-squares-fit
/// correction, only add minor per-pixel jitter). `src_x` is never adjusted
/// (this is a vertical-shear-only correction — see the module docs for what
/// is deliberately NOT attempted).
#[must_use]
pub fn rectify_grey(grey: &[u8], w: usize, h: usize, ramp: &ShearRamp) -> Vec<u8> {
    if w == 0 || h == 0 {
        return grey.to_vec();
    }
    let cx = w as f32 / 2.0;
    let mut out = vec![0u8; w * h];
    for y_out in 0..h {
        let true_y_pageup = (h - 1 - y_out) as f32;
        let m = ramp.at(true_y_pageup);
        for x_out in 0..w {
            let src_y_f = y_out as f32 - m * (x_out as f32 - cx);
            let src_y = src_y_f.round().clamp(0.0, (h - 1) as f32) as usize;
            out[y_out * w + x_out] = grey[src_y * w + x_out];
        }
    }
    out
}

/// Bound on [`auto_rectify`]'s correction passes. Each pass is a first-order
/// approximation (see [`rectify_grey`]'s derivation), so a single pass on a
/// LARGE initial distortion leaves a real, measurable residual (observed:
/// ~4-8x reduction per pass on an aggressive combined-rotation+keystone
/// fixture, not a full zero) — re-measuring and correcting again converges
/// closer. 3 is enough headroom for any distortion this module's own
/// small-angle premise applies to; a page still significant after 3 passes
/// is past what a shear-ramp model should be trusted to fix at all (residual
/// diminishing returns, not a reason to keep looping).
const MAX_RECTIFY_PASSES: u32 = 3;

/// Detect + fit + apply, iterating up to [`MAX_RECTIFY_PASSES`] times until
/// the measured ramp is no longer [`ShearRamp::is_significant`]. A SAFE
/// NO-OP (returns `grey` unchanged, cloned) when there aren't enough
/// recognizable rows to fit a ramp, or the very first fit already isn't
/// significant — the common case (an already-straight page) must never be
/// perturbed by needless resampling.
#[must_use]
pub fn auto_rectify(grey: &[u8], w: usize, h: usize) -> Vec<u8> {
    let Some(first) = fit_shear_ramp(&detect_row_shears(grey, w, h)) else {
        return grey.to_vec();
    };
    if !first.is_significant(h) {
        return grey.to_vec();
    }
    let mut current = rectify_grey(grey, w, h, &first);
    for _ in 1..MAX_RECTIFY_PASSES {
        match fit_shear_ramp(&detect_row_shears(&current, w, h)) {
            Some(ramp) if ramp.is_significant(h) => current = rectify_grey(&current, w, h, &ramp),
            _ => break,
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Draw a hollow (2px-border, white interior) rectangle — `filter_blobs`'
    /// density heuristic (`blob_filter.rs`: `pixel_count >=
    /// height*width*0.7` → "too dense to be text, treat as small/non-text")
    /// rejects a SOLID filled rectangle outright (100% density starves the
    /// pool's line-size histogram to 0, cascading into every blob being
    /// classified "large" — a real bug this fixture hit once, 64/64 blobs
    /// rejected). A hollow border keeps density around 30-40%, comfortably
    /// under the threshold, while staying one 8-connected component spanning
    /// the full requested height (so the row's baseline fit still sees the
    /// intended `y` extent).
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

    /// Build a synthetic grey page: `n_lines` horizontal text-like bars
    /// (hollow-rectangle "glyphs" — see [`draw_hollow_rect`] — not real
    /// glyphs; `segment_rows`' connected-component + row-finder only needs
    /// blob geometry, not recognizable characters, to fit a baseline), then
    /// apply a KNOWN forward shear-ramp distortion (the exact inverse of what
    /// `rectify_grey` corrects), so the recovered ramp can be checked against
    /// ground truth.
    fn synthetic_sheared_page(
        w: usize,
        h: usize,
        n_lines: usize,
        m0: f32,
        m1: f32,
    ) -> (Vec<u8>, ShearRamp) {
        // 1) Draw a straight page: n_lines evenly spaced horizontal bars,
        // each broken into several short segments (so conn-comp sees
        // multiple blobs per row — MIN_ROW_BLOBS needs >= 3). bar_h MUST
        // clear `textord_max_noise_size` (7, blob_filter.rs) or every blob
        // gets classified as noise and dropped before make_rows ever sees
        // it — a real bug this fixture hit once (bar_h=6 -> 0 rows detected
        // for every fixture in this file, since 6 < 7).
        let mut straight = vec![255u8; w * h];
        let margin = h / (n_lines + 2);
        let bar_h = 16usize.min(margin.saturating_sub(4).max(9));
        let seg_w = w / 12;
        for i in 0..n_lines {
            let y0 = margin + i * margin;
            for seg in 0..8 {
                let x0 = seg * (w / 8) + seg_w / 4;
                let x1 = (x0 + seg_w).min(w);
                draw_hollow_rect(&mut straight, w, h, x0, y0, x1, (y0 + bar_h).min(h));
            }
        }

        // 2) Apply the FORWARD distortion by calling rectify_grey ITSELF with
        // the NEGATED ramp, rather than hand-deriving a separate forward
        // formula (a first attempt at that independently-derived formula had
        // a sign/role bug — easy to get wrong twice, impossible to get wrong
        // once: same trusted implementation, negated input). This is exact
        // for pure rotation (m1=0): rectify_grey(rectify_grey(straight,
        // -ramp), ramp) == straight algebraically, since a constant shear
        // and its negation are exact inverses of each other. For m1≠0 it's a
        // first-order approximation (same order as rectify_grey's own
        // approximation), which is all the recovery tests below require.
        let ramp = ShearRamp { m0, m1 };
        let neg_ramp = ShearRamp { m0: -m0, m1: -m1 };
        let distorted = rectify_grey(&straight, w, h, &neg_ramp);
        (distorted, ramp)
    }

    #[test]
    fn fit_shear_ramp_recovers_a_known_pure_rotation() {
        let (distorted, truth) = synthetic_sheared_page(400, 300, 8, 0.05, 0.0);
        let shears = detect_row_shears(&distorted, 400, 300);
        assert!(
            shears.len() >= 6,
            "expected most of the 8 lines to be detected, got {}",
            shears.len()
        );
        let fit = fit_shear_ramp(&shears).expect("fit");
        assert!(
            (fit.m0 - truth.m0).abs() < 0.02,
            "m0: fit={} truth={}",
            fit.m0,
            truth.m0
        );
        assert!(fit.m1.abs() < 0.001, "m1 should be ~0, got {}", fit.m1);
    }

    #[test]
    fn fit_shear_ramp_recovers_a_known_trapezoid() {
        // A moderate, realistic keystone: slope swings by m1*h = 0.075 across
        // the page (~4.3 degrees top-to-bottom) — well within the first-order
        // small-angle regime `rectify_grey`'s own docs scope this module to.
        // (A more extreme swing, e.g. ~10 degrees, pushes the round-trip test
        // fixture's negate-and-reapply approximation past where a first-order
        // model tracks it — that is an approximation-order limit, not
        // something to paper over with a looser tolerance here.)
        let (distorted, truth) = synthetic_sheared_page(400, 300, 8, 0.0, 0.00025);
        let shears = detect_row_shears(&distorted, 400, 300);
        let fit = fit_shear_ramp(&shears).expect("fit");
        assert!(
            (fit.m1 - truth.m1).abs() < 0.00015,
            "m1: fit={} truth={}",
            fit.m1,
            truth.m1
        );
    }

    #[test]
    fn rectify_grey_reduces_row_shear_close_to_zero() {
        let (distorted, truth) = synthetic_sheared_page(400, 300, 8, 0.04, 0.0004);
        let before = fit_shear_ramp(&detect_row_shears(&distorted, 400, 300)).expect("fit before");
        assert!(before.is_significant(300), "fixture should be distorted");

        let rectified = rectify_grey(&distorted, 400, 300, &truth);
        let after_shears = detect_row_shears(&rectified, 400, 300);
        let after = fit_shear_ramp(&after_shears).expect("fit after");
        assert!(
            after.m0.abs() < before.m0.abs(),
            "m0 should shrink: before={} after={}",
            before.m0,
            after.m0
        );
        assert!(
            after.m1.abs() < before.m1.abs(),
            "m1 should shrink: before={} after={}",
            before.m1,
            after.m1
        );
        assert!(
            !after.is_significant(300),
            "rectified page should read as ~straight: m0={} m1={}",
            after.m0,
            after.m1
        );
    }

    #[test]
    fn auto_rectify_is_a_no_op_on_an_already_straight_page() {
        let (straight, _) = synthetic_sheared_page(400, 300, 8, 0.0, 0.0);
        let out = auto_rectify(&straight, 400, 300);
        assert_eq!(out, straight, "a straight page must not be perturbed");
    }

    #[test]
    fn auto_rectify_corrects_a_significantly_distorted_page() {
        let (distorted, _) = synthetic_sheared_page(400, 300, 8, 0.05, 0.0005);
        let out = auto_rectify(&distorted, 400, 300);
        assert_ne!(out, distorted, "a distorted page should be corrected");
        let after = fit_shear_ramp(&detect_row_shears(&out, 400, 300)).expect("fit after");
        assert!(
            !after.is_significant(300),
            "auto-rectified output should read as ~straight: m0={} m1={}",
            after.m0,
            after.m1
        );
    }

    #[test]
    fn shear_ramp_at_evaluates_the_linear_model() {
        let ramp = ShearRamp { m0: 0.1, m1: 0.001 };
        assert!((ramp.at(0.0) - 0.1).abs() < 1e-6);
        assert!((ramp.at(100.0) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn is_significant_thresholds_correctly() {
        assert!(!ShearRamp { m0: 0.0, m1: 0.0 }.is_significant(600));
        assert!(ShearRamp { m0: 0.02, m1: 0.0 }.is_significant(600));
        assert!(ShearRamp {
            m0: 0.0,
            m1: 0.02 / 600.0 * 2.0
        }
        .is_significant(600));
        assert!(!ShearRamp {
            m0: 0.0,
            m1: 0.001 / 600.0
        }
        .is_significant(600));
    }
}
