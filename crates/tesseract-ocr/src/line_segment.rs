//! D3.0 — projection-profile line segmenter (Batch 3-alt,
//! `.claude/plans/pdf-to-text-ocr-v1.md`).
//!
//! **APPROXIMATION — not a Tesseract transcode; replaced by the textord
//! batches (plan §P3).** Tesseract's real line finder is the textord
//! connected-component + row/block layout pipeline (`ccstruct`/`textord`,
//! Batches 3A-3F). This module is a deliberately simple, deterministic
//! stand-in: a horizontal ink-profile over an Otsu-binarized page, split into
//! contiguous inked row-runs. It exists to unblock an end-to-end
//! `page image → text` path NOW, before the parity textord work lands; every
//! item here is expected to be replaced, not extended toward parity.
//!
//! Gated behind the `seg-approx` feature so it can never be mistaken for a
//! proven leaf on the crate's default surface.

use crate::threshold::{otsu_threshold_gray, threshold_rect_to_binary};

/// A candidate text-line row band: `top` inclusive, `bottom` exclusive
/// (`grey[top*w..bottom*w]` is the cropped line strip).
///
/// **APPROXIMATION — not a Tesseract transcode; replaced by the textord
/// batches (plan §P3).**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineBand {
    /// First row of the band (inclusive).
    pub top: usize,
    /// One past the last row of the band (exclusive).
    pub bottom: usize,
}

impl LineBand {
    /// Band height in rows (`bottom - top`).
    #[must_use]
    pub fn height(&self) -> usize {
        self.bottom.saturating_sub(self.top)
    }
}

/// Minimum inked-run height (in rows) to be considered a candidate line
/// rather than noise (a stray dark pixel/scan artifact).
///
/// **APPROXIMATION — not a Tesseract transcode; replaced by the textord
/// batches (plan §P3).**
const MIN_BAND_HEIGHT: usize = 3;

/// Rows of padding added on each side of a detected ink-run, before clamping
/// to the page bounds and de-overlapping with neighboring bands.
///
/// **APPROXIMATION — not a Tesseract transcode; replaced by the textord
/// batches (plan §P3).**
const BAND_PADDING: usize = 2;

/// Per-row ink pixel counts for a grey page, using the page's own Otsu
/// decision (`otsu_threshold_gray` over the full rect) when it resolves a
/// binarization, or a fixed `pixel < 128` predicate when it doesn't
/// (`hi_value == -1`, "no opinion").
///
/// **APPROXIMATION — not a Tesseract transcode; replaced by the textord
/// batches (plan §P3).** Split out from [`find_text_lines`] so the
/// `hi_value == -1` fallback branch is directly testable: in practice a
/// single-channel [`otsu_threshold_gray`] on any non-degenerate
/// (`width*height > 0`) page always resolves a `hi_value` of `0` or `1` (the
/// "keep the best of the bad lot" branch always wins when there is only one
/// channel to compare), so this fallback is defensive rather than reachable
/// from [`find_text_lines`] on real input today — documented here rather than
/// silently dead.
fn row_ink_counts(
    grey: &[u8],
    w: usize,
    h: usize,
    otsu: crate::threshold::OtsuChannel,
) -> Vec<usize> {
    let mut profile = vec![0usize; h];
    if otsu.hi_value == -1 {
        // No opinion from Otsu: fall back to a fixed mid-grey split. Ink =
        // "darker than middle grey".
        for (y, row_count) in profile.iter_mut().enumerate() {
            let row = &grey[y * w..(y + 1) * w];
            *row_count = row.iter().filter(|&&p| p < 128).count();
        }
    } else {
        // 0 = foreground/black (see threshold.rs module docs' output convention).
        let binary = threshold_rect_to_binary(grey, w, 0, 0, w, h, otsu);
        for (y, row_count) in profile.iter_mut().enumerate() {
            let row = &binary[y * w..(y + 1) * w];
            *row_count = row.iter().filter(|&&p| p == 0).count();
        }
    }
    profile
}

/// Ink-profile line finder over a GREY page (white background, dark ink).
///
/// **APPROXIMATION — not a Tesseract transcode; replaced by the textord
/// batches (plan §P3).** Algorithm:
///
/// 1. Binarize the whole page with [`otsu_threshold_gray`] +
///    [`threshold_rect_to_binary`] (falling back to a fixed `pixel < 128`
///    predicate if Otsu returns `hi_value == -1`, "don't threshold" — see
///    [`row_ink_counts`]).
/// 2. `profile[y]` = count of foreground (ink) pixels in row `y`. A row is
///    "inked" if `profile[y] > 0`.
/// 3. Contiguous inked-row runs are candidate bands; runs shorter than
///    [`MIN_BAND_HEIGHT`] rows are dropped as noise.
/// 4. Each surviving band is padded by [`BAND_PADDING`] rows on each side,
///    clamped to `[0, h)`; if padding two neighboring bands would make them
///    overlap, the overlap is split at the midpoint of the original
///    (unpadded) gap between them.
///
/// Returns bands top-to-bottom. Empty for a zero-sized page.
#[must_use]
pub fn find_text_lines(grey: &[u8], w: usize, h: usize) -> Vec<LineBand> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let otsu = otsu_threshold_gray(grey, w, 0, 0, w, h);
    let profile = row_ink_counts(grey, w, h, otsu);

    // Contiguous inked-row runs (raw, unpadded): (start inclusive, end exclusive).
    let mut raw_bands: Vec<(usize, usize)> = Vec::new();
    let mut y = 0;
    while y < h {
        if profile[y] > 0 {
            let start = y;
            while y < h && profile[y] > 0 {
                y += 1;
            }
            let end = y;
            if end - start >= MIN_BAND_HEIGHT {
                raw_bands.push((start, end));
            }
        } else {
            y += 1;
        }
    }

    // Pad, clamp, then de-overlap neighbors by splitting the original gap.
    let mut bands: Vec<LineBand> = raw_bands
        .iter()
        .map(|&(start, end)| LineBand {
            top: start.saturating_sub(BAND_PADDING),
            bottom: (end + BAND_PADDING).min(h),
        })
        .collect();

    for i in 1..bands.len() {
        if bands[i].top < bands[i - 1].bottom {
            let gap_start = raw_bands[i - 1].1; // end of previous raw run
            let gap_end = raw_bands[i].0; // start of this raw run
            let mid = (gap_start + gap_end) / 2;
            bands[i - 1].bottom = mid;
            bands[i].top = mid;
        }
    }

    bands
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two dark stripes close enough together that naive ±2 padding would
    /// make the bands overlap: the overlap must be split at the midpoint of
    /// the original (unpadded) gap, not merged or left overlapping.
    #[test]
    fn two_bands_padded_and_split_when_padding_would_overlap() {
        let w = 4;
        let h = 20;
        let mut grey = vec![240u8; w * h]; // white background
                                           // Raw band 1: rows 5,6,7 (height 3, meets MIN_BAND_HEIGHT).
        for y in 5..8 {
            for x in 0..w {
                grey[y * w + x] = 10;
            }
        }
        // Raw band 2: rows 9,10,11 (height 3), only 1 light row (row 8) apart.
        for y in 9..12 {
            for x in 0..w {
                grey[y * w + x] = 10;
            }
        }

        let bands = find_text_lines(&grey, w, h);
        assert_eq!(bands.len(), 2, "two distinct bands: {bands:?}");

        // Unpadded: raw1=(5,8), raw2=(9,12). Padding ±2 would give
        // (3,10) and (7,14) which overlap [7,10). Split at
        // mid((5..8).end=8, (9..12).start=9) = (8+9)/2 = 8.
        assert_eq!(bands[0], LineBand { top: 3, bottom: 8 });
        assert_eq!(bands[1], LineBand { top: 8, bottom: 14 });
        assert!(
            bands[0].bottom <= bands[1].top,
            "bands must not overlap: {bands:?}"
        );
    }

    /// A single 1-px-tall dark row is noise (shorter than `MIN_BAND_HEIGHT`)
    /// and must be dropped entirely, not returned as a degenerate band.
    #[test]
    fn single_pixel_row_is_rejected_as_noise() {
        let w = 5;
        let h = 10;
        let mut grey = vec![240u8; w * h];
        for x in 0..w {
            grey[4 * w + x] = 10; // one dark row, height 1
        }
        let bands = find_text_lines(&grey, w, h);
        assert!(
            bands.is_empty(),
            "single-row run must be dropped: {bands:?}"
        );
    }

    /// The `hi_value == -1` ("no opinion") fallback predicate directly: a
    /// dark-grey row (< 128) counts as fully inked, a light-grey row (>= 128)
    /// counts as not inked at all, regardless of any Otsu threshold value.
    /// (See [`row_ink_counts`] docs: this path is defensive — a real
    /// single-channel [`otsu_threshold_gray`] on non-degenerate input never
    /// actually returns `hi_value == -1` — so it is exercised directly here
    /// rather than through [`find_text_lines`].)
    #[test]
    fn hi_value_minus_one_falls_back_to_fixed_mid_grey_split() {
        let w = 3;
        let h = 2;
        // Row 0: uniformly below 128 (ink under the fallback rule).
        // Row 1: uniformly at/above 128 (background under the fallback rule).
        let grey: Vec<u8> = vec![50, 50, 50, 200, 200, 200];
        let otsu = crate::threshold::OtsuChannel {
            threshold: 100,
            hi_value: -1,
        };
        let profile = row_ink_counts(&grey, w, h, otsu);
        assert_eq!(profile, vec![3, 0]);
    }
}
