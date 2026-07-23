//! Recursive XY-cut page segmentation over a grey line/page image.
//!
//! **Consumer-side layer — NOT a Tesseract transcode.** Tesseract's own layout
//! analysis is the `textord`/`tabfind` connected-component + tab-stop pipeline
//! (`ccstruct`/`textord`), not an XY-cut; none of that is ported here, and the
//! deimposition use-case this module targets (splitting a scanned *imposed*
//! sheet — e.g. a 3-up or 2×2 ganged page — back into its constituent pages) is
//! outside Tesseract's scope entirely. This is a classic Nagy/Seth recursive
//! projection-profile cut, built ON TOP of this crate's proven Otsu binarizer
//! ([`crate::threshold`]); nothing here feeds recognition, so no byte-parity
//! claim applies or is made. The banner mirrors `structured.rs` /
//! `line_segment.rs`.
//!
//! ## What it does
//!
//! Given a grey buffer (white background, dark ink), [`xy_cut`] returns a set of
//! leaf [`PageRect`]s — the maximal Manhattan-layout regions the page decomposes
//! into — in **reading order**. It binarizes ONCE, then recurses: at each level
//! it projects the current rect's ink onto both axes, finds the widest valid
//! whitespace valley across BOTH axes, cuts (k-way) on that axis, and recurses.
//!
//! ## Algorithm & the decisions behind it
//!
//! 1. **Binarize once** ([`binarize_page`]). The whole page is thresholded a
//!    single time with [`otsu_threshold_gray`] + [`threshold_rect_to_binary`],
//!    yielding a buffer in this crate's bitonal convention **`0` = foreground /
//!    ink, `255` = background** (confirmed against the `threshold.rs` module
//!    docs' "Output convention"). Recursion reads slices of this one buffer —
//!    the profile at every level is a cheap re-scan, never a re-threshold, so a
//!    region's binarization can never drift from its parent's. If Otsu returns
//!    `hi_value == -1` ("no opinion", only reachable on a degenerate page) we
//!    fall back to a fixed `pixel < 128` split, same as `line_segment.rs`.
//!    [`binarize_page_with`] exposes this same step under an explicit
//!    [`BinarizeMode`] — `Otsu` (the default, byte-identical to
//!    [`binarize_page`]) or the opt-in adaptive `Sauvola` — but [`xy_cut`]
//!    itself always calls [`binarize_page`], i.e. always `Otsu`.
//!
//! 2. **Both profiles, every level.** For the current rect we compute BOTH the
//!    vertical (per-column) and horizontal (per-row) ink profiles. A profile bin
//!    counts as *inked* when the inked fraction of its perpendicular extent
//!    exceeds [`XyCutParams::ink_threshold_frac`] (default `0.0` ⇒ *any* ink).
//!    We then find **valleys** = maximal runs of empty bins. Only *interior*
//!    valleys count — an empty run touching the rect's first/last inked bin is a
//!    border margin, not a separator, so it is excluded (we scan only between the
//!    first and last inked bin). A valley is a valid separator iff its thickness
//!    ≥ `min_gap_frac × (cut-axis extent)`; the fraction is of the CURRENT
//!    rect's extent, so the same page recurses scale-relative.
//!
//! 3. **Thickest-valley axis choice — not alternate-axis dogma.** Textbook
//!    XY-cut alternates X,Y,X,Y. That is wrong for a `3×1` imposed strip, which
//!    must be cut vertically *repeatedly*. Instead we pick the axis carrying the
//!    single **thickest** valid valley and split at **every** valid valley on
//!    that axis in one pass (k-way, not binary), then recurse into each part with
//!    `depth − 1`. A tie (equal thickest valley on both axes) breaks toward the
//!    **vertical** cut (column-major, the Manhattan default); tests avoid ties by
//!    construction so the returned order is always derivable from the rule.
//!
//! 4. **Termination → a leaf, tightened to ink.** Recursion stops when no valid
//!    valley exists on either axis, `depth` is exhausted, or the rect is already
//!    smaller than [`XyCutParams::min_region_px`] on either side. The surviving
//!    rect is then **tightened to its ink bounding box** and emitted; a rect with
//!    **no ink at all is dropped** (never emitted as an empty region).
//!    `min_region_px` also guards *slivers*: a valley is confirmed only when the
//!    inked runs it would carve off on both sides are each ≥ `min_region_px`, so
//!    a stray marginal noise column is never split off as its own region.
//!
//! 5. **Reading order.** Leaves come back depth-first with vertical cuts ordered
//!    left→right and horizontal cuts top→bottom. So a two-column page yields
//!    `[entire left column, then entire right column]`, and a horizontally-cut
//!    2×2 grid yields `[top-left, top-right, bottom-left, bottom-right]`. This is
//!    **Manhattan-layout reading order**: it is correct exactly when the layout
//!    is recursively cuttable by axis-aligned whitespace. A figure or heading
//!    that *spans* a would-be gutter defeats the cut (the gutter is never empty),
//!    so the spanning region and everything it bridges come back as one larger
//!    leaf — the documented limitation, not a bug.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "raster coordinates and pixel counts stay well within f32/usize range; the gap-threshold arithmetic is deliberately low-precision"
)]

use crate::binarize::sauvola_binarize;
use crate::threshold::{otsu_threshold_gray, threshold_rect_to_binary};

/// An axis-aligned region in raster space: **top-down** image coordinates with
/// `left`/`top` **inclusive** and `right`/`bottom` **exclusive** (so
/// `left..right` × `top..bottom` are the covered pixel columns/rows, matching
/// the half-open convention used by `LineBand` and the renderers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageRect {
    /// Left edge, inclusive.
    pub left: usize,
    /// Top edge, inclusive.
    pub top: usize,
    /// Right edge, exclusive.
    pub right: usize,
    /// Bottom edge, exclusive.
    pub bottom: usize,
}

impl PageRect {
    /// Width in pixels (`right - left`), saturating at 0 for an inverted rect.
    #[must_use]
    pub fn width(&self) -> usize {
        self.right.saturating_sub(self.left)
    }

    /// Height in pixels (`bottom - top`), saturating at 0 for an inverted rect.
    #[must_use]
    pub fn height(&self) -> usize {
        self.bottom.saturating_sub(self.top)
    }
}

/// Tuning for [`xy_cut`]. Every default is documented at its field; the defaults
/// target a normal-DPI scanned page and are deliberately conservative (cut only
/// on generous whitespace, never shave noise slivers).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XyCutParams {
    /// Minimum valley thickness, as a fraction of the CURRENT rect's cut-axis
    /// extent, for an empty run to count as a separator. Default `0.015` ≈ a
    /// gutter at least 1.5 % of the page dimension — wide enough to reject
    /// inter-line and inter-word gaps while catching real column/page gutters.
    /// Scale-relative so a half-size page cuts on a half-size gutter. The
    /// absolute threshold is `ceil(min_gap_frac × extent)`, floored at 1 px.
    pub min_gap_frac: f32,
    /// Regions smaller than this (in px) on either side of a candidate cut are
    /// not split off, and a rect already smaller than this on either side is a
    /// leaf. Default `24` — below a plausible glyph/column width, so this
    /// suppresses splitting off marginal noise while never blocking a real
    /// column split.
    pub min_region_px: usize,
    /// Maximum recursion depth. Default `6` — a Manhattan page is typically
    /// fully decomposed in 2–3 levels; 6 leaves headroom for pathological nests
    /// while bounding worst-case work.
    pub max_depth: usize,
    /// A profile bin counts as inked when the inked fraction of its
    /// perpendicular extent is **strictly greater** than this. Default `0.0` ⇒
    /// any ink at all marks the bin inked (the most sensitive setting, which
    /// keeps thin rules and descenders from opening spurious valleys). Raise it
    /// to ignore light speckle when projecting.
    pub ink_threshold_frac: f32,
}

impl Default for XyCutParams {
    fn default() -> Self {
        Self {
            min_gap_frac: 0.015,
            min_region_px: 24,
            max_depth: 6,
            ink_threshold_frac: 0.0,
        }
    }
}

/// Segment a grey page into Manhattan-layout leaf regions in reading order.
///
/// `grey` is a `w × h` row-major 8-bit grey buffer (white background ≈ 255,
/// dark ink ≈ 0). Returns the leaf [`PageRect`]s — each tightened to its ink
/// bounding box, empty regions dropped — depth-first in reading order (vertical
/// cuts left→right, horizontal cuts top→bottom). An empty or zero-sized page
/// returns `[]`. See the module docs for the full algorithm and its limits.
#[must_use]
pub fn xy_cut(grey: &[u8], w: usize, h: usize, params: &XyCutParams) -> Vec<PageRect> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let binary = binarize_page(grey, w, h);
    let root = PageRect {
        left: 0,
        top: 0,
        right: w,
        bottom: h,
    };
    let mut out = Vec::new();
    split_rect(&binary, w, root, params.max_depth, params, &mut out);
    out
}

/// Segmentation binarization mode for [`binarize_page_with`]. The crate-wide
/// default is [`BinarizeMode::Otsu`] — a single global threshold — and
/// [`binarize_page`] (the helper [`xy_cut`] itself calls) is pinned to
/// exactly that default, byte-for-byte, so this opt-in surface never changes
/// existing behaviour on its own. [`BinarizeMode::Sauvola`] is the adaptive
/// alternative (see [`sauvola_binarize`]): a per-pixel threshold from the
/// local mean/stddev that survives unevenly-lit or aged scans a single
/// global Otsu split washes out. Nothing in this crate selects `Sauvola`
/// implicitly — a caller must construct it explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BinarizeMode {
    /// Global Otsu threshold ([`otsu_threshold_gray`]), with the fixed
    /// `pixel < 128` fallback when Otsu declines to threshold
    /// (`hi_value == -1`; only reachable on a degenerate page). This is the
    /// crate-wide segmentation default.
    #[default]
    Otsu,
    /// Adaptive Sauvola threshold ([`sauvola_binarize`]): a per-pixel
    /// threshold from the local mean and standard deviation over a
    /// `(2·whsize+1)²` window, robust against uneven lighting that defeats a
    /// single global Otsu split. Falls back to [`BinarizeMode::Otsu`] when
    /// the page is too small for the requested window (mirroring the guard
    /// [`sauvola_binarize`] itself asserts on: `whsize >= 2` and
    /// `w, h >= 2·whsize + 3`), so this mode never panics regardless of page
    /// size.
    Sauvola {
        /// Window half-size. Must be `>= 2`, and the page must be at least
        /// `2·whsize + 3` on each side, or this mode falls back to `Otsu`.
        whsize: usize,
        /// Sauvola sensitivity factor `k` (typically `0.34`).
        k: f32,
    },
}

/// Binarize the full page once into this crate's `0` = ink / `255` = background
/// convention, using the crate-wide default ([`BinarizeMode::default`], i.e.
/// [`BinarizeMode::Otsu`]) — byte-identical to this function's behaviour
/// before [`BinarizeMode`] existed. Uses the page's own Otsu decision,
/// falling back to a fixed `pixel < 128` split when Otsu declines to
/// threshold (`hi_value == -1`; only reachable on a degenerate page — see
/// `line_segment::row_ink_counts`).
fn binarize_page(grey: &[u8], w: usize, h: usize) -> Vec<u8> {
    binarize_page_with(grey, w, h, BinarizeMode::default())
}

/// Binarize the full page once into this crate's `0` = ink / `255` = background
/// convention, under an explicit, opt-in [`BinarizeMode`].
///
/// [`BinarizeMode::Otsu`] is exactly [`binarize_page`]'s pre-existing body —
/// global [`otsu_threshold_gray`] + [`threshold_rect_to_binary`], with the
/// same `hi_value == -1` fixed-128 fallback — so selecting it, or the
/// default, reproduces [`binarize_page`] byte-for-byte.
///
/// [`BinarizeMode::Sauvola`] runs [`sauvola_binarize`] and converts its own
/// per-pixel foreground convention (`1` = black text, `0` = background) into
/// this crate's `{0, 255}` convention (`0` = ink, `255` = background) — the
/// inverse byte value, same semantic split. When the page is too small for
/// the requested window, this falls back to [`BinarizeMode::Otsu`] instead
/// of panicking (mirroring the guard [`sauvola_binarize`] itself asserts
/// on).
///
/// This is the opt-in segmentation entry point: [`xy_cut`] and
/// [`binarize_page`] are untouched and keep calling the `Otsu` path
/// unconditionally. Threading this mode up to a document-level API (e.g.
/// `LstmRecognizer::recognize_document`) is a deliberate follow-up, not done
/// here.
#[must_use]
pub fn binarize_page_with(grey: &[u8], w: usize, h: usize, mode: BinarizeMode) -> Vec<u8> {
    match mode {
        BinarizeMode::Otsu => {
            let otsu = otsu_threshold_gray(grey, w, 0, 0, w, h);
            if otsu.hi_value == -1 {
                grey.iter()
                    .map(|&p| if p < 128 { 0 } else { 255 })
                    .collect()
            } else {
                threshold_rect_to_binary(grey, w, 0, 0, w, h, otsu)
            }
        }
        BinarizeMode::Sauvola { whsize, k: factor } => {
            let window_ok = whsize >= 2 && w >= 2 * whsize + 3 && h >= 2 * whsize + 3;
            if !window_ok {
                // Too small for the requested window — fall back to Otsu
                // rather than reaching sauvola_binarize's own panic guard.
                return binarize_page_with(grey, w, h, BinarizeMode::Otsu);
            }
            let sauvola = sauvola_binarize(grey, w, h, whsize, factor);
            sauvola
                .binary
                .iter()
                .map(|&fg| if fg == 1 { 0 } else { 255 })
                .collect()
        }
    }
}

/// Per-column ink profile of `rect` (used to find VERTICAL cuts / left→right
/// splits): `out[xi]` is `true` when column `rect.left + xi` is inked, i.e. the
/// ink fraction of the rect's height exceeds `ink_threshold_frac`.
fn column_ink_profile(binary: &[u8], w: usize, rect: PageRect, params: &XyCutParams) -> Vec<bool> {
    let ext = rect.height();
    let mut profile = vec![false; rect.width()];
    if ext == 0 {
        return profile;
    }
    for (xi, cell) in profile.iter_mut().enumerate() {
        let x = rect.left + xi;
        let mut count = 0usize;
        for y in rect.top..rect.bottom {
            if binary[y * w + x] == 0 {
                count += 1;
            }
        }
        *cell = (count as f32 / ext as f32) > params.ink_threshold_frac;
    }
    profile
}

/// Per-row ink profile of `rect` (used to find HORIZONTAL cuts / top→bottom
/// splits): `out[yi]` is `true` when row `rect.top + yi` is inked, i.e. the ink
/// fraction of the rect's width exceeds `ink_threshold_frac`.
fn row_ink_profile(binary: &[u8], w: usize, rect: PageRect, params: &XyCutParams) -> Vec<bool> {
    let ext = rect.width();
    let mut profile = vec![false; rect.height()];
    if ext == 0 {
        return profile;
    }
    for (yi, cell) in profile.iter_mut().enumerate() {
        let y = rect.top + yi;
        let row = &binary[y * w + rect.left..y * w + rect.right];
        let count = row.iter().filter(|&&p| p == 0).count();
        *cell = (count as f32 / ext as f32) > params.ink_threshold_frac;
    }
    profile
}

/// Given a 1-D ink profile along a cut axis, return the valid cut positions.
///
/// Returns `None` when there is no ink or no confirmed cut; otherwise
/// `Some((max_valley_thickness, cut_positions))` where `cut_positions` are the
/// valley midpoints (in bin coordinates relative to the rect) at which to split,
/// ascending. A valley qualifies when it is (a) *interior* — a maximal empty run
/// strictly between the first and last inked bin, (b) at least
/// `ceil(min_gap_frac × extent)` (≥ 1) bins thick, and (c) not a sliver-maker:
/// the inked runs immediately on both sides are each ≥ `min_region_px`.
fn axis_cuts(
    inked: &[bool],
    min_gap_frac: f32,
    min_region_px: usize,
) -> Option<(usize, Vec<usize>)> {
    let extent = inked.len();
    let first = inked.iter().position(|&b| b)?;
    let last = inked.iter().rposition(|&b| b)?;
    let gap_min = (min_gap_frac * extent as f32).ceil().max(1.0) as usize;

    // (a)+(b): interior empty runs of sufficient thickness. Scanning only
    // `first..=last` makes every run found interior by construction.
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    let mut i = first;
    while i <= last {
        if inked[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i <= last && !inked[i] {
            i += 1;
        }
        let end = i; // exclusive
        if end - start >= gap_min {
            candidates.push((start, end));
        }
    }
    if candidates.is_empty() {
        return None;
    }

    // (c): confirm only valleys whose adjacent inked runs both clear
    // `min_region_px`. Adjacency is measured against the candidate boundaries
    // (the nearest thickness-valid valleys), so a confirmed cut always carves
    // real regions, never slivers.
    let mut confirmed: Vec<(usize, usize)> = Vec::new();
    for (k, &(start, end)) in candidates.iter().enumerate() {
        let left_bound = if k > 0 { candidates[k - 1].1 } else { first };
        let right_bound = if k + 1 < candidates.len() {
            candidates[k + 1].0
        } else {
            last + 1
        };
        let left_span = start - left_bound;
        let right_span = right_bound - end;
        if left_span >= min_region_px && right_span >= min_region_px {
            confirmed.push((start, end));
        }
    }
    if confirmed.is_empty() {
        return None;
    }

    let max_thickness = confirmed
        .iter()
        .map(|&(start, end)| end - start)
        .max()
        .unwrap_or(0);
    let cuts = confirmed
        .iter()
        .map(|&(start, end)| (start + end) / 2)
        .collect();
    Some((max_thickness, cuts))
}

/// Tighten `rect` to the bounding box of the ink it contains. Returns `None`
/// when the rect holds no ink at all (the region is then dropped, never
/// emitted).
fn ink_bbox(binary: &[u8], w: usize, rect: PageRect) -> Option<PageRect> {
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let mut found = false;
    for y in rect.top..rect.bottom {
        for x in rect.left..rect.right {
            if binary[y * w + x] == 0 {
                found = true;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }
    if !found {
        return None;
    }
    Some(PageRect {
        left: min_x,
        top: min_y,
        right: max_x + 1,
        bottom: max_y + 1,
    })
}

/// The recursion core: cut `rect` at the thickest-valley axis, or emit it as a
/// tightened leaf. Children are recursed in ascending cut-boundary order, giving
/// the depth-first reading-order guarantee.
fn split_rect(
    binary: &[u8],
    w: usize,
    rect: PageRect,
    depth: usize,
    params: &XyCutParams,
    out: &mut Vec<PageRect>,
) {
    // Termination guards: exhausted depth or a rect already too small to split.
    if depth == 0 || rect.width() < params.min_region_px || rect.height() < params.min_region_px {
        if let Some(bb) = ink_bbox(binary, w, rect) {
            out.push(bb);
        }
        return;
    }

    let col = column_ink_profile(binary, w, rect, params);
    let row = row_ink_profile(binary, w, rect, params);
    let vcut = axis_cuts(&col, params.min_gap_frac, params.min_region_px);
    let hcut = axis_cuts(&row, params.min_gap_frac, params.min_region_px);

    // Choose the axis with the thickest valid valley; tie → vertical.
    let choose_vertical = match (
        vcut.as_ref().map(|(t, _)| *t),
        hcut.as_ref().map(|(t, _)| *t),
    ) {
        (None, None) => {
            if let Some(bb) = ink_bbox(binary, w, rect) {
                out.push(bb);
            }
            return;
        }
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (Some(v), Some(h)) => v >= h,
    };

    if choose_vertical {
        // Split the x-axis at each valley midpoint; recurse left→right.
        let cuts = vcut.expect("vertical branch implies Some").1;
        let mut bounds = Vec::with_capacity(cuts.len() + 2);
        bounds.push(rect.left);
        bounds.extend(cuts.iter().map(|c| rect.left + c));
        bounds.push(rect.right);
        for pair in bounds.windows(2) {
            let child = PageRect {
                left: pair[0],
                top: rect.top,
                right: pair[1],
                bottom: rect.bottom,
            };
            split_rect(binary, w, child, depth - 1, params, out);
        }
    } else {
        // Split the y-axis at each valley midpoint; recurse top→bottom.
        let cuts = hcut.expect("horizontal branch implies Some").1;
        let mut bounds = Vec::with_capacity(cuts.len() + 2);
        bounds.push(rect.top);
        bounds.extend(cuts.iter().map(|c| rect.top + c));
        bounds.push(rect.bottom);
        for pair in bounds.windows(2) {
            let child = PageRect {
                left: rect.left,
                top: pair[0],
                right: rect.right,
                bottom: pair[1],
            };
            split_rect(binary, w, child, depth - 1, params, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// White page with a solid ink rectangle drawn in `[x0,x1) × [y0,y1)`.
    fn page(w: usize, h: usize) -> Vec<u8> {
        vec![240u8; w * h]
    }
    fn fill(buf: &mut [u8], w: usize, x0: usize, x1: usize, y0: usize, y1: usize) {
        for y in y0..y1 {
            for x in x0..x1 {
                buf[y * w + x] = 10;
            }
        }
    }

    #[test]
    fn single_block_is_one_leaf_tight_to_bbox() {
        let (w, h) = (100, 60);
        let mut g = page(w, h);
        fill(&mut g, w, 30, 70, 20, 40);
        let leaves = xy_cut(&g, w, h, &XyCutParams::default());
        assert_eq!(leaves.len(), 1, "one block → one leaf: {leaves:?}");
        // Solid block ⇒ tightened bbox equals the drawn rect exactly.
        assert_eq!(
            leaves[0],
            PageRect {
                left: 30,
                top: 20,
                right: 70,
                bottom: 40
            }
        );
    }

    #[test]
    fn empty_page_yields_no_leaves() {
        let (w, h) = (40, 40);
        let g = page(w, h); // all background, no ink
        let leaves = xy_cut(&g, w, h, &XyCutParams::default());
        assert!(leaves.is_empty(), "no ink → no leaves: {leaves:?}");
    }

    #[test]
    fn two_columns_wide_gutter_left_then_right() {
        // 200×200 page. Left column x[20,80), right x[120,180), a 40 px white
        // gutter x[80,120). Each column has 4 fake text rows separated by 2 px
        // gaps. gap_min on both axes = ceil(0.015·200) = 3, so the 2 px inter-
        // row gaps are BELOW threshold (never split) while the 40 px gutter is
        // valid → exactly one vertical cut → 2 leaves, left first, each tight.
        let (w, h) = (200, 200);
        let mut g = page(w, h);
        let rows = [(40usize, 50usize), (52, 62), (64, 74), (76, 86)];
        for &(y0, y1) in &rows {
            fill(&mut g, w, 20, 80, y0, y1); // left column text row (full width)
            fill(&mut g, w, 120, 180, y0, y1); // right column text row
        }
        let leaves = xy_cut(&g, w, h, &XyCutParams::default());
        assert_eq!(leaves.len(), 2, "two columns → two leaves: {leaves:?}");
        // Left column first (reading order), tight to its ink extent y[40,86).
        assert_eq!(
            leaves[0],
            PageRect {
                left: 20,
                top: 40,
                right: 80,
                bottom: 86
            }
        );
        assert_eq!(
            leaves[1],
            PageRect {
                left: 120,
                top: 40,
                right: 180,
                bottom: 86
            }
        );
    }

    #[test]
    fn two_by_two_grid_reading_order_horizontal_gutter_thicker() {
        // 200×200 page, four solid blocks. Vertical gutter x[80,100) is 20 px;
        // horizontal gutter y[70,110) is 40 px (thicker). Per the thickest-
        // valley rule the FIRST cut is HORIZONTAL (40 > 20) → [top band, bottom
        // band]; each band then cuts VERTICALLY → [left, right]. Depth-first
        // reading order is therefore [top-left, top-right, bottom-left,
        // bottom-right].
        let (w, h) = (200, 200);
        let mut g = page(w, h);
        // top band y[20,70), bottom band y[110,160)
        fill(&mut g, w, 20, 80, 20, 70); // TL
        fill(&mut g, w, 100, 160, 20, 70); // TR
        fill(&mut g, w, 20, 80, 110, 160); // BL
        fill(&mut g, w, 100, 160, 110, 160); // BR
        let leaves = xy_cut(&g, w, h, &XyCutParams::default());
        assert_eq!(leaves.len(), 4, "2×2 grid → four leaves: {leaves:?}");
        // Expected order derived from the rule above: TL, TR, BL, BR.
        assert_eq!(
            leaves[0],
            PageRect {
                left: 20,
                top: 20,
                right: 80,
                bottom: 70
            }
        ); // TL
        assert_eq!(
            leaves[1],
            PageRect {
                left: 100,
                top: 20,
                right: 160,
                bottom: 70
            }
        ); // TR
        assert_eq!(
            leaves[2],
            PageRect {
                left: 20,
                top: 110,
                right: 80,
                bottom: 160
            }
        ); // BL
        assert_eq!(
            leaves[3],
            PageRect {
                left: 100,
                top: 110,
                right: 160,
                bottom: 160
            }
        ); // BR
    }

    #[test]
    fn thin_gutter_does_not_over_split() {
        // 200×80 page, two solid blocks separated by only a 2 px gutter
        // x[99,101). gap_min = ceil(0.015·200) = 3 > 2, so the gutter is not a
        // valid separator and no cut is made → a single leaf spanning both
        // blocks (tightened across the thin white gap).
        let (w, h) = (200, 80);
        let mut g = page(w, h);
        fill(&mut g, w, 20, 99, 20, 60);
        fill(&mut g, w, 101, 180, 20, 60);
        let leaves = xy_cut(&g, w, h, &XyCutParams::default());
        assert_eq!(
            leaves.len(),
            1,
            "sub-threshold gutter → one leaf: {leaves:?}"
        );
        assert_eq!(
            leaves[0],
            PageRect {
                left: 20,
                top: 20,
                right: 180,
                bottom: 60
            }
        );
    }

    /// Leptonica parity pin for the profile-count convention (operator
    /// directive 2026-07-09: feature parity for the relevant leptonica parts).
    ///
    /// The projection profiles this module cuts on are, in leptonica, the
    /// `pixCountPixelsByRow` / `pixCountPixelsByColumn` counts (`pix3.c:2143/
    /// 2177`, v1.82.0 == the installed liblept). The banked oracle
    /// (`.claude/harvest/oracles/counts_oracle.cpp`, output alongside) builds
    /// the SAME deterministic fixture — `w=97, h=61, grey(x,y) = (7x+13y) %
    /// 251`, ink iff `grey < 128` via `pixThresholdToBinary(pixs, 128)` — and
    /// dumps leptonica's per-row/per-column ON-pixel counts. This test
    /// recomputes the counts through THIS crate's conventions (fixed `p < 128`
    /// split into the `0 = ink` binary, then row/column ink counting exactly as
    /// the profile loops count) and asserts byte-for-byte equality with the
    /// oracle output, pinning: leptonica 1bpp ON-count == our ink count, row
    /// for row, column for column. (Cross-checked independently in Python at
    /// harvest time: True/True.)
    #[test]
    fn profile_counts_match_banked_leptonica_oracle() {
        let oracle = include_str!("../../../.claude/harvest/oracles/counts_oracle_out.txt");
        let mut lines = oracle.lines();
        let header: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
        assert_eq!(header, ["w", "97", "h", "61"], "oracle header changed");
        let parse = |line: &str, tag: &str| -> Vec<usize> {
            let mut it = line.split_whitespace();
            assert_eq!(it.next(), Some(tag));
            it.map(|t| t.parse().expect("count")).collect()
        };
        let oracle_rows = parse(lines.next().expect("rows"), "rows");
        let oracle_cols = parse(lines.next().expect("cols"), "cols");

        let (w, h) = (97usize, 61usize);
        let mut grey = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                grey[y * w + x] = ((x * 7 + y * 13) % 251) as u8;
            }
        }
        // The crate's binary convention (0 = ink) via the same fixed split the
        // oracle thresholds with (pixThresholdToBinary(…, 128): < 128 → ON).
        let binary: Vec<u8> = grey
            .iter()
            .map(|&p| if p < 128 { 0 } else { 255 })
            .collect();

        let rows: Vec<usize> = (0..h)
            .map(|y| {
                binary[y * w..(y + 1) * w]
                    .iter()
                    .filter(|&&p| p == 0)
                    .count()
            })
            .collect();
        let cols: Vec<usize> = (0..w)
            .map(|x| (0..h).filter(|&y| binary[y * w + x] == 0).count())
            .collect();

        assert_eq!(
            rows, oracle_rows,
            "per-row ink counts != pixCountPixelsByRow"
        );
        assert_eq!(
            cols, oracle_cols,
            "per-column ink counts != pixCountPixelsByColumn"
        );
    }

    #[test]
    fn three_across_strip_splits_k_way_at_depth_one() {
        // 300×80 strip (≈ three uprights side by side). Two 20 px gutters
        // x[90,110) and x[190,210). A k-way single-pass split cuts BOTH valleys
        // at once, so even with max_depth = 1 we get 3 leaves, left→right —
        // proving cuts are k-way, not binary-recursive.
        let (w, h) = (300, 80);
        let mut g = page(w, h);
        fill(&mut g, w, 10, 90, 20, 60);
        fill(&mut g, w, 110, 190, 20, 60);
        fill(&mut g, w, 210, 290, 20, 60);
        let params = XyCutParams {
            max_depth: 1,
            ..XyCutParams::default()
        };
        let leaves = xy_cut(&g, w, h, &params);
        assert_eq!(
            leaves.len(),
            3,
            "three panels → three leaves at depth 1: {leaves:?}"
        );
        assert_eq!(
            leaves[0],
            PageRect {
                left: 10,
                top: 20,
                right: 90,
                bottom: 60
            }
        );
        assert_eq!(
            leaves[1],
            PageRect {
                left: 110,
                top: 20,
                right: 190,
                bottom: 60
            }
        );
        assert_eq!(
            leaves[2],
            PageRect {
                left: 210,
                top: 20,
                right: 290,
                bottom: 60
            }
        );
    }

    #[test]
    fn binarize_page_matches_binarize_page_with_otsu_default() {
        // binarize_page (the private helper `xy_cut` itself calls) must stay
        // byte-identical to binarize_page_with's Otsu arm, and Otsu must stay
        // BinarizeMode's default — the whole point of this opt-in wiring is
        // that the pre-existing default path is untouched.
        let (w, h) = (100, 60);
        let mut g = page(w, h);
        fill(&mut g, w, 30, 70, 20, 40);
        let via_private = binarize_page(&g, w, h);
        let via_otsu = binarize_page_with(&g, w, h, BinarizeMode::Otsu);
        let via_default = binarize_page_with(&g, w, h, BinarizeMode::default());
        assert_eq!(
            via_private, via_otsu,
            "binarize_page must equal binarize_page_with(.., BinarizeMode::Otsu)"
        );
        assert_eq!(
            via_otsu, via_default,
            "BinarizeMode::default() must be Otsu"
        );
    }

    #[test]
    fn binarize_page_with_sauvola_differs_from_otsu_and_matches_direct_call() {
        // Two horizontal bands at very different overall brightness (an
        // unevenly-lit page): a dim top band (bg=60) and a bright bottom band
        // (bg=200), each with a small "text" patch ~20 levels darker than its
        // OWN local background. A single global Otsu threshold can only split
        // BETWEEN the two bands, not WITHIN either — so it misclassifies the
        // bulk of the bright band as foreground (both text patches land on
        // the wrong side too). Sauvola's local mean/std tracks each band's
        // own level, so it must disagree with Otsu on, at minimum, the bulk
        // bottom-band background pixels.
        let (w, h) = (40usize, 40usize);
        let mut grey = vec![60u8; w * h];
        for y in 20..40 {
            for x in 0..40 {
                grey[y * w + x] = 200;
            }
        }
        // Top-band text patch: darker than the local 60 background.
        for y in 5..11 {
            for x in 5..11 {
                grey[y * w + x] = 40;
            }
        }
        // Bottom-band text patch: darker than the local 200 background, same
        // 20-level relative dip as the top patch.
        for y in 25..31 {
            for x in 25..31 {
                grey[y * w + x] = 180;
            }
        }

        let otsu = binarize_page_with(&grey, w, h, BinarizeMode::Otsu);
        let sauvola_mode =
            binarize_page_with(&grey, w, h, BinarizeMode::Sauvola { whsize: 8, k: 0.34 });

        assert_ne!(
            otsu, sauvola_mode,
            "Sauvola must disagree with a single global Otsu threshold on an unevenly-lit page"
        );

        // Independently reproduce the {0,255} conversion binarize_page_with
        // applies to Sauvola's own {0,1} foreground convention, and confirm
        // it matches exactly.
        let direct = sauvola_binarize(&grey, w, h, 8, 0.34);
        let expected: Vec<u8> = direct
            .binary
            .iter()
            .map(|&fg| if fg == 1 { 0 } else { 255 })
            .collect();
        assert_eq!(
            sauvola_mode, expected,
            "binarize_page_with(Sauvola) must match a direct sauvola_binarize call under the same {{0,255}} conversion"
        );
    }

    #[test]
    fn binarize_page_with_sauvola_falls_back_to_otsu_on_tiny_image() {
        // whsize=8 requires w,h >= 19 (sauvola_binarize's own guard); a 10x10
        // image is too small, so the Sauvola arm must fall back to Otsu
        // rather than calling sauvola_binarize (which would otherwise panic).
        let (w, h) = (10usize, 10usize);
        let mut g = page(w, h);
        fill(&mut g, w, 2, 8, 2, 8);
        let otsu = binarize_page_with(&g, w, h, BinarizeMode::Otsu);
        let sauvola_fallback =
            binarize_page_with(&g, w, h, BinarizeMode::Sauvola { whsize: 8, k: 0.34 });
        assert_eq!(
            otsu, sauvola_fallback,
            "a too-small-for-window Sauvola request must fall back to Otsu byte-for-byte"
        );
    }
}
