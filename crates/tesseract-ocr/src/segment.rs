//! The shared row-segmentation prefix: binarize the page and find connected
//! components — extracted verbatim (no behaviour change) from
//! `lstm_recognizer.rs`'s `makerow_row_crops`, so every consumer of the line
//! finder shares ONE binarize+conn-comp pass instead of drifting.
//!
//! This is orchestration, not a single Tesseract leaf: it composes already
//! byte-parity-proven pieces (`threshold`, `conncomp`, `blob_filter`,
//! `textline`) the same way `Tesseract::FindLines`'s early stages do, but is
//! not itself claimed as a byte-for-byte transcode of any one function.
//!
//! Two entry points, because `makerow.cpp`'s real pipeline computes TWO
//! different things from the same blobs, and only one of them survives to
//! the end:
//!
//! - [`segment_rows_with_mode`] runs the FULL line finder (`make_rows`, wave 1-3 +
//!   x-height) — the one [`crate::lstm_recognizer::LstmRecognizer`]'s row
//!   crops use. Its final `ToRow::line_m()` is the SAME value for every row
//!   in the block: `make_rows` → `cleanup_rows_making` →
//!   `fit_parallel_rows(block, page_m)` deliberately forces every row onto
//!   one shared page-wide gradient (real Tesseract's actual assumption: a
//!   rotated-but-flat page has all its lines parallel). This is correct and
//!   intentional — do not "fix" it.
//! - [`segment_rows_independent`] stops ONE STEP EARLIER, at
//!   `make_initial_textrows` (`makerow.cpp:254-289`) — before the parallel
//!   constraint is applied, each row still carries its OWN independent LMS
//!   line fit (`fit_lms_line` per row). [`crate::rectify`] needs exactly
//!   this: a page-wide-constant slope can only ever measure ROTATION; a
//!   trapezoid/keystone page's row-to-row slope VARIATION only exists in
//!   this pre-parallel-fit signal, because `fit_parallel_rows` erases it a
//!   few lines later in the real pipeline. The row/blob grouping here is the
//!   cruder pass-0 assignment (`assign_blobs_to_rows(block, None, ...)`, not
//!   yet refined by the three gradient-aware passes `cleanup_rows_making`
//!   runs) — adequate for "how much does this row tilt", not claimed
//!   adequate for anything needing exact blob membership.
//!
//! Both entry points take an explicit [`BinarizeMode`] for the shared
//! binarization step this module performs (`seed_block`). That closes the gap
//! `.claude/harvest/sauvola-vs-otsu-probe.md` measured:
//! `LstmRecognizer::recognize_document_with_mode`'s `binarize_mode` used to
//! reach only the layout `xy_cut` + region/table classification pass, never
//! this module's own independent binarization — so recognized TEXT was
//! mode-blind **by construction**, no matter what mode the caller asked for.
//! The probe saw this as CER being byte-identical between modes on every
//! fixture while ink classification differed ~17×.
//!
//! **The default is unchanged and that is the load-bearing property:**
//! [`BinarizeMode::default()`] is [`BinarizeMode::Otsu`], and every
//! pre-`BinarizeMode` caller's behaviour is preserved byte-for-byte —
//! including [`crate::rectify::detect_row_shears`], which still calls the
//! plain [`segment_rows_independent`]. The golden anchors and the 8+7+0 CER
//! fence both run through this module, so a silent default change would
//! re-pin all of them; `segment_rows_default_mode_is_otsu` guards it
//! directly.

use crate::binarize::{sauvola_binarize, singh_binarize, wolf_binarize, LocalBinarization};
use crate::blob_filter::filter_blobs;
use crate::conncomp::conn_comp_areas;
use crate::textline::{compute_block_xheight, make_initial_textrows, make_rows, ToBlockCtx};
use crate::threshold::{otsu_threshold_gray, threshold_rect_to_binary};
use crate::xy_cut::BinarizeMode;

/// Binarize the page for segmentation under an explicit [`BinarizeMode`] —
/// the shared first step of [`seed_block`].
///
/// **Deliberately NOT a call to [`crate::xy_cut::binarize_page_with`], even
/// though the two do almost the same thing.** `binarize_page_with`'s Otsu
/// arm falls back to a fixed `pixel < 128` split when
/// [`otsu_threshold_gray`] declines to threshold (`hi_value == -1`, only
/// reachable on a degenerate page); THIS module's Otsu binarization —
/// unconditional [`threshold_rect_to_binary`] over whatever
/// [`otsu_threshold_gray`] returns — never had that fallback. That is a
/// real, if narrow (degenerate-page-only), divergence between the two Otsu
/// call sites. Reproducing this module's own exact (lack of) fallback here,
/// rather than delegating to `binarize_page_with`, is what keeps
/// `BinarizeMode::Otsu` byte-identical to this module's pre-`BinarizeMode`
/// behaviour on EVERY input, including that edge case — the hard constraint
/// this whole change is built around. The three local-adaptive modes mirror
/// `binarize_page_with`'s corresponding arms (same too-small-for-window
/// guard), falling back to THIS module's own Otsu arm — not `xy_cut`'s — for
/// the same reason.
fn binarize_page_for_segmentation(grey: &[u8], w: usize, h: usize, mode: BinarizeMode) -> Vec<u8> {
    match mode {
        BinarizeMode::Otsu => {
            let otsu = otsu_threshold_gray(grey, w, 0, 0, w, h);
            threshold_rect_to_binary(grey, w, 0, 0, w, h, otsu)
        }
        BinarizeMode::Sauvola { whsize, k } => {
            local_adaptive_for_segmentation(grey, w, h, whsize, k, sauvola_binarize)
        }
        BinarizeMode::Wolf { whsize, k } => {
            local_adaptive_for_segmentation(grey, w, h, whsize, k, wolf_binarize)
        }
        BinarizeMode::Singh { whsize, k } => {
            local_adaptive_for_segmentation(grey, w, h, whsize, k, singh_binarize)
        }
    }
}

/// Shared body of this module's local-adaptive arms — the segmentation-side
/// twin of `xy_cut::local_adaptive`, differing ONLY in which Otsu arm it
/// falls back to when the page is too small for the requested window (this
/// module's, which has no `hi_value == -1` fallback; see
/// [`binarize_page_for_segmentation`]'s doc comment for why that difference
/// is load-bearing and must not be collapsed).
fn local_adaptive_for_segmentation(
    grey: &[u8],
    w: usize,
    h: usize,
    whsize: usize,
    k: f32,
    method: fn(&[u8], usize, usize, usize, f32) -> LocalBinarization,
) -> Vec<u8> {
    let window_ok = whsize >= 2 && w >= 2 * whsize + 3 && h >= 2 * whsize + 3;
    if !window_ok {
        return binarize_page_for_segmentation(grey, w, h, BinarizeMode::Otsu);
    }
    method(grey, w, h, whsize, k)
        .binary
        .iter()
        .map(|&fg| if fg == 1 { 0 } else { 255 })
        .collect()
}

/// Binarize (mode-selectable, see [`binarize_page_for_segmentation`]) +
/// 8-connected components + the noise/size partition — shared by both
/// [`segment_rows_with_mode`] and [`segment_rows_independent_with_mode`].
/// Returns a `ToBlockCtx` seeded with the surviving blob pool and
/// `filter_blobs`' line-size estimate, ready for either `make_rows` or
/// `make_initial_textrows`.
fn seed_block(grey: &[u8], w: usize, h: usize, mode: BinarizeMode) -> ToBlockCtx {
    // P2: binarize the whole page under the selected mode (foreground = 0
    // per the crate's grey-image convention).
    let binary = binarize_page_for_segmentation(grey, w, h, mode);

    // 3B + 3F₂ leaf 1: components with ink pixel counts (8-connectivity,
    // matching the real pipeline's blob granularity most closely).
    let mut components = conn_comp_areas(&binary, w, h, 8);
    // Raster space → Tesseract y-UP page space for the makerow math.
    for c in &mut components {
        c.bb.y = h as i32 - (c.bb.y + c.bb.h);
    }

    // 3F₂ leaf 2: noise partition + the line-size seed.
    let filtered = filter_blobs(&components);

    ToBlockCtx {
        blobs: filtered.blobs,
        // Retained, NOT dropped — see ToBlockCtx::noise. `make_rows` still
        // only ever sees `blobs`, so segmentation itself is unchanged.
        noise: filtered.noise,
        // Likewise retained — this is the one line where `.large` used to die.
        large: filtered.large,
        block_left: 0,
        line_spacing: filtered.line_spacing,
        line_size: filtered.line_size,
        max_blob_size: filtered.max_blob_size,
        ..Default::default()
    }
}

/// The same full-line-finder composition, with the shared binarization step
/// selectable via an explicit [`BinarizeMode`]. This is the fix for the gap
/// `.claude/harvest/sauvola-vs-otsu-probe.md` measured: it is what lets
/// [`crate::lstm_recognizer::LstmRecognizer::recognize_document_with_mode`]'s
/// mode actually govern word/line recognition, not just region/table
/// classification.
#[must_use]
pub(crate) fn segment_rows_with_mode(
    grey: &[u8],
    w: usize,
    h: usize,
    mode: BinarizeMode,
) -> ToBlockCtx {
    let mut blocks = [seed_block(grey, w, h, mode)];
    let page_m = make_rows(&mut blocks);
    let [mut block] = blocks;
    compute_block_xheight(&mut block, page_m, 0.0);
    block
}

/// Stops at `make_initial_textrows` — each row keeps its OWN independent LMS
/// line fit (`ToRow::line_m()`/`parallel_c()` genuinely vary row-to-row), not
/// yet forced parallel. See the module docs for why [`crate::rectify`] needs
/// this instead of [`segment_rows_with_mode`].
///
/// Exactly [`segment_rows_independent_with_mode`] called with
/// [`BinarizeMode::default()`] (i.e. [`BinarizeMode::Otsu`]) —
/// byte-identical to this function's behaviour before `BinarizeMode`
/// existed. This pass does not touch [`crate::rectify`]'s call site, so
/// preserving this exact signature and default behaviour matters here just
/// as much as for [`segment_rows_with_mode`].
#[must_use]
pub(crate) fn segment_rows_independent(grey: &[u8], w: usize, h: usize) -> ToBlockCtx {
    segment_rows_independent_with_mode(grey, w, h, BinarizeMode::default())
}

/// The same [`segment_rows_independent`] composition, with the shared
/// binarization step selectable via an explicit [`BinarizeMode`]. Added for
/// symmetry with [`segment_rows_with_mode`]; no current caller selects a
/// non-default mode here — `crate::rectify::detect_row_shears` calls
/// [`segment_rows_independent`] directly and is unaffected by this addition.
#[must_use]
pub(crate) fn segment_rows_independent_with_mode(
    grey: &[u8],
    w: usize,
    h: usize,
    mode: BinarizeMode,
) -> ToBlockCtx {
    let mut block = seed_block(grey, w, h, mode);
    make_initial_textrows(&mut block);
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A white page with hollow-rectangle "glyph" blobs drawn at the given
    /// `(x0, y0, x1, y1)` rects. Solid fills trip `filter_blobs`'s density
    /// heuristic (`textord_noise_area_ratio = 0.7`: a blob whose ink covers
    /// ≥ 70% of its own bounding box reads as "too dense to be text" — see
    /// `crate::rectify`'s test module for the same lesson learned there);
    /// hollow borders keep density around 25-40%. A rect's height must also
    /// clear `textord_max_noise_size` (7 px, `blob_filter.rs`) or it is
    /// dropped as noise before `make_rows`/`make_initial_textrows` ever sees
    /// it. No corpus fixture or model file is needed — this is a pure
    /// synthetic geometry test of the mode-threading wiring itself.
    fn hollow_rect_page(w: usize, h: usize, rects: &[(usize, usize, usize, usize)]) -> Vec<u8> {
        let mut buf = vec![255u8; w * h];
        let border = 2usize;
        for &(x0, y0, x1, y1) in rects {
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
        buf
    }

    /// Two rows of three hollow-rectangle blobs each — enough signal for
    /// `make_rows`/`make_initial_textrows` to find multiple rows with
    /// multiple blobs, without depending on any corpus fixture.
    fn two_row_page() -> (Vec<u8>, usize, usize) {
        let (w, h) = (240usize, 160usize);
        let rects = [
            (20usize, 20usize, 60usize, 40usize),
            (80, 20, 120, 40),
            (140, 20, 180, 40),
            (20, 100, 60, 120),
            (80, 100, 120, 120),
            (140, 100, 180, 120),
        ];
        (hollow_rect_page(w, h, &rects), w, h)
    }

    /// Sanity check that the fixture actually exercises the pipeline (a
    /// meaningfully non-trivial base for the equality tests below, not an
    /// accidental empty-vs-empty comparison).
    #[test]
    fn two_row_page_fixture_yields_rows() {
        let (grey, w, h) = two_row_page();
        let block = segment_rows_with_mode(&grey, w, h, BinarizeMode::default());
        assert!(
            !block.rows.is_empty(),
            "fixture must produce at least one row"
        );
    }

    /// **The default-preservation invariant.** `BinarizeMode::default()` must
    /// be exactly `Otsu` *at this entry point* — that equality is what makes
    /// every pre-`BinarizeMode` caller's behaviour unchanged, and it is the
    /// property the golden anchors and the 8+7+0 fence silently depend on.
    ///
    /// Compared explicitly field by field because `ToBlockCtx`/`ToRow` do not
    /// derive `PartialEq`. `blobs` (a `Vec<(i32,i32,i32,i32)>`) does support
    /// `==` and is the earliest, most binarization-sensitive value in the
    /// chain — if it matches, everything the deterministic
    /// `make_rows`/`compute_block_xheight` code derives from it matches too.
    ///
    /// (This previously compared a plain `segment_rows` wrapper against its
    /// `_with_mode` sibling. Once every caller migrated to the sibling that
    /// wrapper became dead code, so it was removed — see the Sauvola pass,
    /// which deleted its redundant `binarize_page` wrappers for the same
    /// reason. Asserting `default() == Otsu` directly keeps the invariant
    /// non-vacuous without keeping an unused function alive to test.)
    #[test]
    fn segment_rows_default_mode_is_otsu() {
        let (grey, w, h) = two_row_page();
        let plain = segment_rows_with_mode(&grey, w, h, BinarizeMode::Otsu);
        let with_mode = segment_rows_with_mode(&grey, w, h, BinarizeMode::default());
        assert_eq!(plain.blobs, with_mode.blobs, "blob pool must match");
        assert_eq!(
            plain.rows.len(),
            with_mode.rows.len(),
            "row count must match"
        );
        assert_eq!(plain.line_spacing, with_mode.line_spacing);
        assert_eq!(plain.line_size, with_mode.line_size);
        assert_eq!(plain.max_blob_size, with_mode.max_blob_size);
        assert_eq!(plain.xheight, with_mode.xheight);
        assert_eq!(plain.block_left, with_mode.block_left);
        assert_eq!(plain.baseline_offset, with_mode.baseline_offset);
        assert_eq!(plain.key_row, with_mode.key_row);
    }

    /// The same default-preservation invariant for
    /// [`segment_rows_independent`] — the sibling `crate::rectify` depends
    /// on and this pass does not otherwise exercise.
    #[test]
    fn segment_rows_independent_matches_with_mode_default() {
        let (grey, w, h) = two_row_page();
        let plain = segment_rows_independent(&grey, w, h);
        let with_mode = segment_rows_independent_with_mode(&grey, w, h, BinarizeMode::default());
        assert_eq!(plain.blobs, with_mode.blobs, "blob pool must match");
        assert_eq!(
            plain.rows.len(),
            with_mode.rows.len(),
            "row count must match"
        );
        assert_eq!(plain.line_spacing, with_mode.line_spacing);
        assert_eq!(plain.line_size, with_mode.line_size);
        assert_eq!(plain.max_blob_size, with_mode.max_blob_size);
        assert_eq!(plain.xheight, with_mode.xheight);
        assert_eq!(plain.block_left, with_mode.block_left);
        assert_eq!(plain.baseline_offset, with_mode.baseline_offset);
        assert_eq!(plain.key_row, with_mode.key_row);
    }
}
