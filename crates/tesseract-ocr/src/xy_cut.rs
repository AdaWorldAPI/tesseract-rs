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
//! 1. **Binarize once** ([`binarize_page_with`]). The whole page is
//!    thresholded a single time — by default with [`otsu_threshold_gray`] +
//!    [`threshold_rect_to_binary`] ([`BinarizeMode::Otsu`]) — yielding a
//!    buffer in this crate's bitonal convention **`0` = foreground / ink,
//!    `255` = background** (confirmed against the `threshold.rs` module docs'
//!    "Output convention"). Recursion reads slices of this one buffer — the
//!    profile at every level is a cheap re-scan, never a re-threshold, so a
//!    region's binarization can never drift from its parent's. If Otsu
//!    returns `hi_value == -1` ("no opinion", only reachable on a degenerate
//!    page) we fall back to a fixed `pixel < 128` split, same as
//!    `line_segment.rs`. [`xy_cut`] selects the binarization mode via
//!    [`XyCutParams::binarize_mode`] — [`BinarizeMode::Otsu`] is the default,
//!    byte-identical to this crate's behaviour before that field existed; the
//!    opt-in adaptive [`BinarizeMode::Sauvola`] is also selectable, for
//!    source pages a single global threshold under-serves (uneven
//!    illumination, aged scans).
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

use crate::binarize::{sauvola_binarize, singh_binarize, wolf_binarize, LocalBinarization};
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
    /// Segmentation binarization mode (see [`binarize_page_with`]). Default
    /// [`BinarizeMode::Otsu`] — a single global threshold, byte-identical to
    /// this crate's behaviour before this field existed. Set
    /// [`BinarizeMode::Sauvola`] to opt into the adaptive per-pixel threshold
    /// for source pages a single global split under-serves (uneven
    /// illumination, aged scans).
    pub binarize_mode: BinarizeMode,
}

impl Default for XyCutParams {
    fn default() -> Self {
        Self {
            min_gap_frac: 0.015,
            min_region_px: 24,
            max_depth: 6,
            ink_threshold_frac: 0.0,
            binarize_mode: BinarizeMode::default(),
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
    let binary = binarize_page_with(grey, w, h, params.binarize_mode);
    let root = PageRect {
        left: 0,
        top: 0,
        right: w,
        bottom: h,
    };
    let mut out = Vec::new();
    split_rect_inner(&binary, w, root, params.max_depth, params, None, &mut out);
    out
}

/// [`xy_cut`], with ONE additional rule: a candidate rect that
/// [`crate::pageseg::region_is_table`] classifies a table (evaluated against
/// `table_binary`, NOT `grey`'s own binarization — see below) is emitted as
/// ONE leaf, tightened to its ink bbox, WITHOUT further splitting — even
/// though the table's own internal column gutters would otherwise trigger
/// further vertical cuts exactly like any other whitespace-separated layout.
///
/// # Why this exists — the defect it closes
///
/// Plain [`xy_cut`] has no concept of "this whitespace gutter is a table
/// column, not a paragraph break": once a table's borders are stripped (or
/// even on the bare page — a table's internal corridors ARE genuine
/// whitespace), recursive splitting treats each column as its own top-level
/// leaf. `crate::lstm_recognizer::LstmRecognizer::recognize_page_blocks_words_with_mode`
/// then recognizes each column-leaf INDEPENDENTLY, producing one text line
/// per printed ROW **but only within that one column** — so
/// `crate::structured::extract_table_grid`'s "rows ARE the recognized lines"
/// assumption breaks: a real 7×4 table becomes ~28 single-column lines
/// (column-major reading), because the table was fragmented before
/// recognition ever ran, not because the grid-reconstruction logic is wrong.
///
/// Stopping the recursion at the table's own boundary keeps it as ONE full
/// bbox, so the line-finder recognizes each printed row as ONE line spanning
/// every column — exactly the shape `extract_table_grid`'s OWN
/// whitespace-gap column splitter already expects and was built for.
///
/// # Why `table_binary` is a SEPARATE parameter from `grey`
///
/// The table DECISION (`region_is_table`, ultimately `decide_if_table`)
/// counts printed rules (`nhb`/`nvb`) — real information that exists on the
/// page AS PRINTED but is INTENTIONALLY erased from a border-stripped page
/// (`crate::pageseg::strip_borders_grey`, used to keep those rules from
/// being recognized as glyphs). If this function binarized `grey` itself for
/// the table check, calling it on a stripped page would see zero rules and
/// never detect a table at all — reproducing the exact coupling
/// `DocumentOptions::strip_borders`'s own docs describe ("the borders are
/// simultaneously what ruins the columns and what proves it is a table").
/// `table_binary` therefore should almost always be the ORIGINAL page's
/// binarization (via [`binarize_page_with`]), computed once by the caller,
/// even when `grey` here is a stripped or otherwise-modified buffer used for
/// the actual cut/segmentation decisions.
///
/// # Panics
/// Panics if `table_binary.len() != w * h`.
#[must_use]
pub fn xy_cut_table_aware(
    grey: &[u8],
    w: usize,
    h: usize,
    params: &XyCutParams,
    table_binary: &[u8],
) -> Vec<PageRect> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    assert_eq!(
        table_binary.len(),
        w * h,
        "table_binary must be the SAME w*h as grey — it is the ORIGINAL \
         page's binarization, not a crop"
    );
    let binary = binarize_page_with(grey, w, h, params.binarize_mode);
    let root = PageRect {
        left: 0,
        top: 0,
        right: w,
        bottom: h,
    };
    let mut out = Vec::new();
    split_rect_inner(
        &binary,
        w,
        root,
        params.max_depth,
        params,
        Some(table_binary),
        &mut out,
    );
    out
}

/// Segmentation binarization mode for [`binarize_page_with`]. The crate-wide
/// default is [`BinarizeMode::Otsu`] — a single global threshold, and the
/// value both this enum's [`Default`] and [`XyCutParams::binarize_mode`]
/// resolve to, so this opt-in surface never changes existing behaviour on
/// its own. [`BinarizeMode::Sauvola`] is the adaptive alternative (see
/// [`sauvola_binarize`]): a per-pixel threshold from the local mean/stddev
/// that survives unevenly-lit or aged scans a single global Otsu split
/// washes out. Nothing in this crate selects `Sauvola` implicitly — a
/// caller must construct it explicitly, e.g.
/// `XyCutParams { binarize_mode: BinarizeMode::Sauvola { whsize: 16, k: 0.34 }, ..Default::default() }`.
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
    /// Adaptive Wolf-Jolion threshold ([`wolf_binarize`]) — Sauvola with its
    /// *fixed* `R = 128` replaced by the image's **actual** maximum local
    /// standard deviation, and the correction anchored to the global minimum
    /// grey. Aimed at faded / low-contrast source, where Sauvola's fixed `R`
    /// collapses and it under-thresholds. Same small-page fallback to
    /// [`BinarizeMode::Otsu`] as `Sauvola`.
    ///
    /// **Quality-fence footing, NOT byte-parity** — leptonica does not
    /// implement this method, so no oracle exists; see
    /// [`crate::binarize`]'s module docs.
    Wolf {
        /// Window half-size. Must be `>= 2`, and the page must be at least
        /// `2·whsize + 3` on each side, or this mode falls back to `Otsu`.
        whsize: usize,
        /// Wolf sensitivity `k` — reference default `0.5`. **Not** Sauvola's
        /// `0.34`: `k` is not transferable between methods.
        k: f32,
    },
    /// Adaptive Singh et al. threshold ([`singh_binarize`]) — a **per-pixel**
    /// mean deviation `∂ = I - m` in place of the windowed standard
    /// deviation, so the second-moment accumulation disappears entirely and
    /// the cost is flat in window size. Same small-page fallback to
    /// [`BinarizeMode::Otsu`] as `Sauvola`.
    ///
    /// **Quality-fence footing, NOT byte-parity** — no reference
    /// implementation identified; see [`crate::binarize`]'s module docs.
    Singh {
        /// Window half-size. Must be `>= 2`, and the page must be at least
        /// `2·whsize + 3` on each side, or this mode falls back to `Otsu`.
        whsize: usize,
        /// Singh sensitivity `k`, paper range `[0, 1]` (its document figure
        /// uses `0.06`). Sweep empirically — see [`singh_binarize`] for why a
        /// figure's printed value cannot be taken at face value.
        k: f32,
    },
}

/// Binarize the full page once into this crate's `0` = ink / `255` = background
/// convention, under an explicit [`BinarizeMode`].
///
/// [`BinarizeMode::Otsu`] — global [`otsu_threshold_gray`] +
/// [`threshold_rect_to_binary`], with the `hi_value == -1` fixed-128
/// fallback when Otsu declines to threshold (only reachable on a degenerate
/// page — see `line_segment::row_ink_counts`) — is the crate-wide default
/// and reproduces this crate's segmentation binarization exactly as it
/// behaved before [`BinarizeMode`] existed.
///
/// [`BinarizeMode::Sauvola`] runs [`sauvola_binarize`] and converts its own
/// per-pixel foreground convention (`1` = black text, `0` = background) into
/// this crate's `{0, 255}` convention (`0` = ink, `255` = background) — the
/// inverse byte value, same semantic split. When the page is too small for
/// the requested window, this falls back to [`BinarizeMode::Otsu`] instead
/// of panicking (mirroring the guard [`sauvola_binarize`] itself asserts
/// on).
///
/// This is the segmentation entry point [`xy_cut`] calls with
/// [`XyCutParams::binarize_mode`] — `Otsu` by default, so every existing
/// caller that never touches the new field keeps its exact prior behaviour.
/// [`LstmRecognizer::recognize_document`](crate::LstmRecognizer::recognize_document)'s
/// own region/table-classification binarizer is a SEPARATE call graph (see
/// [`LstmRecognizer::recognize_document_with_mode`](crate::LstmRecognizer::recognize_document_with_mode))
/// that also forwards its `binarize_mode` here, but independently of any
/// particular `xy_cut` caller.
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
        BinarizeMode::Sauvola { whsize, k } => {
            local_adaptive(grey, w, h, whsize, k, sauvola_binarize)
        }
        BinarizeMode::Wolf { whsize, k } => local_adaptive(grey, w, h, whsize, k, wolf_binarize),
        BinarizeMode::Singh { whsize, k } => local_adaptive(grey, w, h, whsize, k, singh_binarize),
    }
}

/// Shared body of the three local-adaptive arms of [`binarize_page_with`]:
/// check the window fits, run `method`, and convert its own foreground
/// convention (`1` = black text, `0` = background) into this crate's
/// `{0, 255}` convention (`0` = ink, `255` = background) — the inverse byte
/// value, same semantic split.
///
/// The window check mirrors the guard each method asserts on
/// (`whsize >= 2` and `w, h >= 2·whsize + 3`); too small falls back to
/// [`BinarizeMode::Otsu`] rather than reaching that panic. Every adaptive
/// mode shares this so a new rung cannot forget the fallback and turn a
/// small page into a panic.
fn local_adaptive(
    grey: &[u8],
    w: usize,
    h: usize,
    whsize: usize,
    k: f32,
    method: fn(&[u8], usize, usize, usize, f32) -> LocalBinarization,
) -> Vec<u8> {
    let window_ok = whsize >= 2 && w >= 2 * whsize + 3 && h >= 2 * whsize + 3;
    if !window_ok {
        return binarize_page_with(grey, w, h, BinarizeMode::Otsu);
    }
    method(grey, w, h, whsize, k)
        .binary
        .iter()
        .map(|&fg| if fg == 1 { 0 } else { 255 })
        .collect()
}

// ---------------------------------------------------------------------------
// Escalation ladder — see [`binarize_page_escalating`].
// ---------------------------------------------------------------------------

/// Which rung of [`binarize_page_escalating`]'s ladder a page's binarization
/// actually came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalatedMode {
    /// Rung 1 — the zero-cost default; used whenever it produces a plausible
    /// ink fraction (the overwhelming common case).
    Otsu,
    /// Rung 2 — Otsu's ink fraction was implausible; Sauvola recovered.
    Sauvola,
    /// Rung 3 — Otsu's AND Sauvola's ink fractions were both implausible;
    /// Wolf recovered.
    Wolf,
    /// Rung 4 — the unconditional last resort. Reached when Otsu, Sauvola
    /// AND Wolf all read implausible; returned regardless of its own ink
    /// fraction, since there is nowhere further to escalate.
    Singh,
}

/// The result of [`binarize_page_escalating`]: the binarized page, which rung
/// produced it, and that rung's own ink fraction (for callers that want to
/// report or log the escalation, e.g. a debug view — see
/// [`EscalatedMode`]'s doc for why a silent single default is the wrong
/// architecture here).
#[derive(Debug, Clone)]
pub struct EscalatedBinarization {
    /// The binarized page in this crate's `{0, 255}` convention.
    pub binary: Vec<u8>,
    /// Which rung produced [`Self::binary`].
    pub mode: EscalatedMode,
    /// That rung's own ink fraction (fraction of pixels classified as ink).
    pub ink_frac: f32,
}

/// Lower edge of the plausible-ink-fraction band a rung must clear to be
/// accepted. Below this, a method has classified the page as functionally
/// BLANK — Sauvola's measured catastrophic mode on severe faded contrast
/// (`ink_frac = 0.0000` exactly, `.claude/harvest/sauvola-vs-otsu-probe.md`
/// / `CLAUDE.md` § "faded-contrast corpus arm").
///
/// `0.005` sits strictly between the two real numbers it must separate: the
/// genuine collapse (`0.0000`) and the LEAST healthy real-corpus reading that
/// must still pass (`faded_060.pgm`'s Sauvola result, `0.0246`).
const PLAUSIBLE_INK_FRAC_LO: f32 = 0.005;

/// Upper edge of the plausible-ink-fraction band. Above this, a method has
/// classified roughly HALF the page as ink — Otsu's measured catastrophic
/// mode under uneven illumination (`0.45`–`0.50` across all four
/// `uneven_*.pgm` fixtures, vs. a healthy reading around `0.026`–`0.030`).
///
/// `0.15` sits strictly between the two real numbers it must separate: the
/// highest healthy reading measured across every fixture in this file's own
/// test corpus (`0.0303`) and the lowest flood reading measured
/// (`uneven_vignette_060.pgm`'s Otsu result, `0.4566`) — five times the
/// healthy ceiling, nowhere near the flood floor.
const PLAUSIBLE_INK_FRAC_HI: f32 = 0.15;

/// Fraction of pixels classified as ink (`0`) in a binarized `{0, 255}`
/// buffer. `0.0` for an empty buffer (never divides by zero).
#[must_use]
fn ink_fraction(binary: &[u8]) -> f32 {
    if binary.is_empty() {
        return 0.0;
    }
    let ink = binary.iter().filter(|&&p| p == 0).count();
    ink as f32 / binary.len() as f32
}

/// Whether `frac` sits inside [`PLAUSIBLE_INK_FRAC_LO`]..=[`PLAUSIBLE_INK_FRAC_HI`].
#[must_use]
fn is_plausible_ink_frac(frac: f32) -> bool {
    (PLAUSIBLE_INK_FRAC_LO..=PLAUSIBLE_INK_FRAC_HI).contains(&frac)
}

/// Binarize `grey` via a **lazy escalation ladder** — Otsu → Sauvola → Wolf →
/// Singh — instead of committing to one method as a blind default.
///
/// # Why an escalation ladder, not a single chosen default
///
/// This crate spent real measurement effort (`CLAUDE.md` §§ "Sauvola default-
/// flip", "faded-contrast corpus arm", "Wolf-Jolion") establishing that
/// **neither Otsu nor Sauvola is individually safe as an unconditional
/// default**: Otsu floods to `~0.48` ink fraction under uneven illumination
/// (a real document defect any photographed page can carry), and Sauvola
/// collapses to `0.0000` — the WHOLE PAGE silently blank — under compressed
/// contrast (faded print, worn toner). Each failure is catastrophic and
/// **cheaply detectable from the binarized output alone**: a healthy page's
/// ink fraction clusters tightly around `0.026`–`0.030` across every method
/// this crate implements (measured on the full `corpus/quality/{uneven,
/// faded}_*.pgm` set), so a reading far outside that band is not a subtle
/// hint, it is the method openly failing.
///
/// Given that, committing to one method — Otsu forever, or flipping to
/// Sauvola forever — is throwing away information that is already sitting in
/// the very output being produced. This function reads that information: run
/// the cheap rung, check whether its own ink fraction is plausible, and only
/// pay for the next rung when it is not. On the overwhelming common case
/// (Otsu succeeds) this costs exactly what the existing default already
/// costs — one Otsu pass, nothing more.
///
/// # The rung order is measured, not arbitrary
///
/// 1. **Otsu** — zero-cost, and its own worst measured failure (illumination
///    flood) is loud (`~0.48`), never silent.
/// 2. **Sauvola** — the byte-parity leaf; fixes Otsu's illumination flood
///    (measured `~17×` ink-fraction reduction, `CLAUDE.md` § "Sauvola does
///    what it claims") but has its OWN silent collapse on faded contrast.
/// 3. **Wolf** — reached only when BOTH Otsu and Sauvola read implausible.
///    Measured (this function's own construction, real
///    `uneven_linear_085.pgm` pixels with a faded-contrast compression
///    layered on top): a combined illumination+faded-contrast page defeats
///    Otsu (`0.4893`) AND Sauvola (`0.0000`) simultaneously, while Wolf
///    stays plausible (`0.0275`) — the genuinely NEW information Wolf
///    contributes over Sauvola (no fixed `R = 128`, see
///    [`crate::binarize`]'s module docs).
/// 4. **Singh** — the unconditional last resort. **Honesty pin:** no
///    combined degradation this function's own construction could reach
///    ever showed Singh recovering where Wolf had already failed — at the
///    most extreme combination measured (illumination + `b=0.01` contrast
///    compression + additive noise), Wolf and Singh collapsed TOGETHER
///    (`0.0000` each). Singh's presence here is a defensive floor (some
///    output beats none, and it is a genuinely different formula with no
///    windowed second moment — see [`singh_binarize`]), not a proven fourth
///    recovery path. Do not cite rung 4 as measured; it is CONJECTURE.
///
/// # Cost
///
/// One binarization pass in the common case; up to four on a page bad enough
/// to defeat the first three. `whsize`/`k` for the three adaptive rungs
/// mirror this crate's own established defaults (Sauvola `k=0.34`, Wolf
/// `k=0.5`, Singh `k=0.06` — see [`BinarizeMode`]'s variants); only `whsize`
/// is a caller parameter, since it is the one knob tied to source DPI rather
/// than to the method itself.
#[must_use]
pub fn binarize_page_escalating(
    grey: &[u8],
    w: usize,
    h: usize,
    whsize: usize,
) -> EscalatedBinarization {
    let otsu = binarize_page_with(grey, w, h, BinarizeMode::Otsu);
    let otsu_frac = ink_fraction(&otsu);
    if is_plausible_ink_frac(otsu_frac) {
        return EscalatedBinarization {
            binary: otsu,
            mode: EscalatedMode::Otsu,
            ink_frac: otsu_frac,
        };
    }

    let sauvola = binarize_page_with(grey, w, h, BinarizeMode::Sauvola { whsize, k: 0.34 });
    let sauvola_frac = ink_fraction(&sauvola);
    if is_plausible_ink_frac(sauvola_frac) {
        return EscalatedBinarization {
            binary: sauvola,
            mode: EscalatedMode::Sauvola,
            ink_frac: sauvola_frac,
        };
    }

    let wolf = binarize_page_with(grey, w, h, BinarizeMode::Wolf { whsize, k: 0.5 });
    let wolf_frac = ink_fraction(&wolf);
    if is_plausible_ink_frac(wolf_frac) {
        return EscalatedBinarization {
            binary: wolf,
            mode: EscalatedMode::Wolf,
            ink_frac: wolf_frac,
        };
    }

    let singh = binarize_page_with(grey, w, h, BinarizeMode::Singh { whsize, k: 0.06 });
    let singh_frac = ink_fraction(&singh);
    EscalatedBinarization {
        binary: singh,
        mode: EscalatedMode::Singh,
        ink_frac: singh_frac,
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

/// A fallback-pass valley must be at least this fraction of the WIDEST
/// interior valley to join the accepted cluster — the "these all look like the
/// same kind of separator" test. Real column gutters on one page are cut from
/// the same typographic measure and come out near-identical; the 1-2 px
/// accidental alignments that any ragged text block contains are an order of
/// magnitude thinner (measured on `corpus/quality/resgrid.pgm`: real gutters
/// 69-70 px against noise of 1-2 px, a 35× separation).
const GUTTER_CLUSTER_FRAC: f32 = 0.6;

/// A fallback-pass gutter must be at least this fraction of the MEAN BAND it
/// separates. This is the principled form of the rule the page-relative
/// threshold gets wrong: a separator earns its status by being wide relative to
/// **the things it separates**, not relative to the whole page.
const GUTTER_MIN_COLUMN_FRAC: f32 = 0.05;

/// Given a 1-D ink profile along a cut axis, return the valid cut positions.
///
/// Returns `None` when there is no ink or no confirmed cut; otherwise
/// `Some((max_valley_thickness, cut_positions))` where `cut_positions` are the
/// valley midpoints (in bin coordinates relative to the rect) at which to split,
/// ascending. A valley qualifies when it is (a) *interior* — a maximal empty run
/// strictly between the first and last inked bin, (b) at least
/// `ceil(min_gap_frac × extent)` (≥ 1) bins thick **or** admitted by the
/// column-gutter fallback described below, and (c) not a sliver-maker:
/// the inked runs immediately on both sides are each ≥ `min_region_px`.
///
/// # The column-gutter fallback (`allow_gutter_fallback`)
///
/// Rule (b)'s threshold is `ceil(min_gap_frac × extent)` where `extent` is the
/// CURRENT RECT's cut-axis extent — the full page width for a top-level column
/// cut. That makes the absolute gutter requirement independent of how many
/// columns the page has, while real gutters scale with COLUMN width (≈ `W/n`).
/// Expressed against a column the requirement therefore grows linearly in `n`:
/// at `min_gap_frac = 0.015` a 2-column page needs a 3 %-of-column gutter, but
/// an 8-column page needs ~12 %. Past that point NO vertical cut is found, the
/// page splits only horizontally, and every emitted strip spans all `n` columns
/// — so the line-finder reads straight across the gutters and concatenates one
/// line from every column into a single line of output.
///
/// When rule (b) admits NOTHING and `allow_gutter_fallback` is set, a second
/// pass judges the valleys against each other instead of against the page: take
/// the widest interior valley, keep the cluster within [`GUTTER_CLUSTER_FRAC`]
/// of it, and accept that cluster only if it has ≥ 2 members (a genuine
/// multi-column grid, not one ambiguous gap) AND the widest valley clears
/// [`GUTTER_MIN_COLUMN_FRAC`] of the mean band it separates. The pass is
/// strictly additive — it cannot run at all unless the existing rule found
/// nothing — so every page that splits today splits identically.
///
/// # Why only the vertical axis passes `true`
///
/// The same page-relative scaling is "wrong" on the Y axis too, but the
/// permissive form MUST NOT be applied there: inter-line leading is typically
/// 20-40 % of a line's own band height, a HIGHER ratio than a column gutter is
/// of a column's width, so no width-ratio rule can separate "line gap" from
/// "column gutter" — a Y-axis fallback would shred ordinary body text into
/// one region per line. On the Y axis, suppressing line-splitting is the
/// correct, load-bearing behaviour of the page-relative threshold, so the
/// horizontal call site passes `false`.
fn axis_cuts(
    inked: &[bool],
    min_gap_frac: f32,
    min_region_px: usize,
    allow_gutter_fallback: bool,
) -> Option<(usize, Vec<usize>)> {
    let extent = inked.len();
    let first = inked.iter().position(|&b| b)?;
    let last = inked.iter().rposition(|&b| b)?;
    let gap_min = (min_gap_frac * extent as f32).ceil().max(1.0) as usize;

    // (a): every interior empty run. Scanning only `first..=last` makes every
    // run found interior by construction.
    let mut valleys: Vec<(usize, usize)> = Vec::new();
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
        valleys.push((start, i)); // end exclusive
    }

    // (b): runs of sufficient thickness, measured against the rect extent.
    let mut candidates: Vec<(usize, usize)> = valleys
        .iter()
        .copied()
        .filter(|&(s, e)| e - s >= gap_min)
        .collect();

    // (b'): the column-gutter fallback — see the doc comment. Strictly
    // additive: gated on the page-relative pass having admitted nothing.
    if candidates.is_empty() && allow_gutter_fallback {
        if let Some(wmax) = valleys.iter().map(|&(s, e)| e - s).max() {
            let cluster: Vec<(usize, usize)> = valleys
                .iter()
                .copied()
                .filter(|&(s, e)| (e - s) as f32 >= GUTTER_CLUSTER_FRAC * wmax as f32)
                .collect();
            let mean_band = extent as f32 / (cluster.len() + 1) as f32;
            if cluster.len() >= 2 && wmax as f32 >= GUTTER_MIN_COLUMN_FRAC * mean_band {
                candidates = cluster;
            }
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
/// The recursive core shared by [`xy_cut`] and [`xy_cut_table_aware`].
///
/// `table_binary: None` reproduces [`xy_cut`]'s exact pre-existing recursion
/// byte-for-byte — the table check below is skipped entirely, so this is a
/// pure refactor from that caller's perspective (verified by the existing
/// `xy_cut` test suite, unchanged). `Some(tb)` adds exactly one rule: before
/// computing any cut on the CURRENT candidate `rect`, check whether `tb`
/// (the caller's fixed reference — see [`xy_cut_table_aware`]'s docs for why
/// it is not `binary`) classifies `rect` a table; if so, stop here and emit
/// `rect` as one leaf, same as the depth-exhausted/too-small terminal cases.
fn split_rect_inner(
    binary: &[u8],
    w: usize,
    rect: PageRect,
    depth: usize,
    params: &XyCutParams,
    table_binary: Option<&[u8]>,
    out: &mut Vec<PageRect>,
) {
    // Termination guards: exhausted depth or a rect already too small to split.
    if depth == 0 || rect.width() < params.min_region_px || rect.height() < params.min_region_px {
        if let Some(bb) = ink_bbox(binary, w, rect) {
            out.push(bb);
        }
        return;
    }

    // Table veto: a candidate rect already classified as a table stops
    // recursing HERE, before any whitespace-gap cut decision is even made —
    // its internal column gutters must not be treated as ordinary layout
    // breaks. See `xy_cut_table_aware`'s docs for the full rationale.
    //
    // `h = tb.len() / w` is exact and safe: `w` is the recursion's ORIGINAL
    // page width (invariant across every recursive call — only `rect`
    // shrinks, `w` never does), never zero here (the `w == 0` page is
    // rejected before recursion starts), and `table_binary`'s length is
    // asserted `== w * h` by `xy_cut_table_aware` before this function is
    // ever called.
    if let Some(tb) = table_binary {
        let h = tb.len() / w;
        let region = (
            rect.left as i32,
            rect.top as i32,
            rect.right as i32,
            rect.bottom as i32,
        );
        if crate::pageseg::region_is_table(tb, w, h, region) {
            if let Some(bb) = ink_bbox(binary, w, rect) {
                out.push(bb);
            }
            return;
        }
    }

    let col = column_ink_profile(binary, w, rect, params);
    let row = row_ink_profile(binary, w, rect, params);
    // VERTICAL (column) cuts opt into the column-gutter fallback; HORIZONTAL
    // (line) cuts must NOT — see `axis_cuts`'s doc comment, "Why only the
    // vertical axis passes `true`".
    let vcut = axis_cuts(&col, params.min_gap_frac, params.min_region_px, true);
    let hcut = axis_cuts(&row, params.min_gap_frac, params.min_region_px, false);

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
            split_rect_inner(binary, w, child, depth - 1, params, table_binary, out);
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
            split_rect_inner(binary, w, child, depth - 1, params, table_binary, out);
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

    /// A ruled grid — same shape as `pageseg.rs`'s own `table_fixture()`
    /// (4 horizontal + 4 vertical rules, `o100`-survivable lengths, clears
    /// [`crate::pageseg::TABLE_SCORE_THRESHOLD`]). Used ONLY as the
    /// `table_binary` classification reference — see
    /// `xy_cut_table_aware_keeps_a_ruled_table_as_one_leaf`'s own doc for why
    /// the SEGMENTED buffer must be a different, rule-free fixture.
    fn ruled_grid(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![255u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let hline = [20usize, 90, 160, 230]
                    .iter()
                    .any(|&r| y >= r && y < r + 2 && (20..220).contains(&x));
                let vline = [20usize, 90, 160, 220]
                    .iter()
                    .any(|&c| x >= c && x < c + 2 && (20..232).contains(&y));
                if hline || vline {
                    buf[y * w + x] = 0;
                }
            }
        }
        buf
    }

    /// Ink blocks in the SAME cell positions `ruled_grid` lays out (three
    /// columns × two rows), but with NO rules at all — what a table looks
    /// like to the RECOGNIZER after `strip_borders` paints the rules to
    /// background: real per-cell content, genuinely empty corridors between
    /// columns, nothing bridging them.
    fn ruled_grid_content_only(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![255u8; w * h];
        for &(x0, x1) in &[(40usize, 70usize), (110, 140), (180, 210)] {
            for &(y0, y1) in &[(40usize, 70usize), (110, 140)] {
                for y in y0..y1 {
                    for x in x0..x1 {
                        buf[y * w + x] = 0;
                    }
                }
            }
        }
        buf
    }

    #[test]
    fn xy_cut_table_aware_keeps_a_ruled_table_as_one_leaf() {
        // The measured real-world shape (`tests/lab_table_columns.rs`):
        // `strip_borders` removes a table's printed rules from the buffer
        // the RECOGNIZER segments, while the TABLE DECISION still reads the
        // original, rule-intact page. A ruled grid with NO cell content is
        // topologically CONNECTED (every corridor is bridged by a crossing
        // rule), so it can never fragment in the first place — this is
        // exactly what this test's own anti-vacuity check caught when first
        // tried with rules present in BOTH buffers (`plain.len() == 1`
        // already, before any veto). The two-buffer split below reproduces
        // the real scenario instead: `content` (no rules, real per-cell
        // ink) is what gets segmented; `table_binary` (rules, no cell
        // content needed — `region_is_table` only counts structure) is what
        // gets classified.
        let (w, h) = (240, 280);
        let table_binary = ruled_grid(w, h);
        let content = ruled_grid_content_only(w, h);

        // Anti-vacuity: PLAIN xy_cut on the CONTENT buffer must actually
        // fragment into multiple leaves, or the veto below proves nothing.
        let plain = xy_cut(&content, w, h, &XyCutParams::default());
        assert!(
            plain.len() > 1,
            "fixture precondition failed: plain xy_cut must fragment the \
             rule-free cell content into multiple leaves, or this test \
             cannot show the veto doing anything (got {} leaf/leaves: \
             {plain:?})",
            plain.len()
        );

        let aware = xy_cut_table_aware(&content, w, h, &XyCutParams::default(), &table_binary);
        assert_eq!(
            aware.len(),
            1,
            "a table region must survive as ONE leaf, not fragment into \
             per-column strips: {aware:?}"
        );
        // The single leaf must span (tightened to ink) the WHOLE content
        // area, including the internal column corridors that have no ink
        // of their own — a bounding box, not a trimmed union of the cells.
        let bb = aware[0];
        assert!(
            bb.left <= 40 && bb.right >= 210,
            "must span full width: {bb:?}"
        );
        assert!(
            bb.top <= 40 && bb.bottom >= 140,
            "must span full height: {bb:?}"
        );
    }

    #[test]
    fn xy_cut_table_aware_does_not_veto_ordinary_multi_column_prose() {
        // The EXACT fixture `two_columns_wide_gutter_left_then_right` above
        // uses — no rules, just two paragraph-shaped ink blocks. Passing
        // this SAME buffer as `table_binary` must not change the result:
        // ordinary prose must still split into its two columns. This is the
        // anti-false-positive half — a veto that fires on everything would
        // "pass" the table test above for the wrong reason.
        let (w, h) = (200, 200);
        let mut g = page(w, h);
        let rows = [(40usize, 50usize), (52, 62), (64, 74), (76, 86)];
        for &(y0, y1) in &rows {
            fill(&mut g, w, 20, 80, y0, y1);
            fill(&mut g, w, 120, 180, y0, y1);
        }
        let aware = xy_cut_table_aware(&g, w, h, &XyCutParams::default(), &g);
        assert_eq!(
            aware.len(),
            2,
            "ordinary two-column prose (no rules) must NOT be vetoed into \
             one leaf: {aware:?}"
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
    fn xy_cut_params_default_binarize_mode_is_otsu() {
        // The whole point of this opt-in wiring is that a caller who never
        // touches XyCutParams::binarize_mode gets the exact pre-existing
        // (Otsu) behaviour: default() must resolve to Otsu, and binarizing
        // through XyCutParams::default()'s mode must match an explicit
        // BinarizeMode::Otsu call byte-for-byte.
        assert_eq!(BinarizeMode::default(), BinarizeMode::Otsu);
        assert_eq!(XyCutParams::default().binarize_mode, BinarizeMode::Otsu);

        let (w, h) = (100, 60);
        let mut g = page(w, h);
        fill(&mut g, w, 30, 70, 20, 40);
        let via_default_params = binarize_page_with(&g, w, h, XyCutParams::default().binarize_mode);
        let via_otsu = binarize_page_with(&g, w, h, BinarizeMode::Otsu);
        assert_eq!(
            via_default_params, via_otsu,
            "XyCutParams::default()'s mode must binarize identically to an explicit BinarizeMode::Otsu"
        );
    }

    #[test]
    fn xy_cut_reacts_to_binarize_mode_selection() {
        // Reuses the exact two-brightness-band fixture from
        // `binarize_page_with_sauvola_differs_from_otsu_and_matches_direct_call`
        // below (proven there to make Otsu and Sauvola disagree pixel-for-
        // pixel): a dim top band (bg=60) and a bright bottom band (bg=200),
        // each with its own small text patch ~20 levels darker than its OWN
        // local background. A single global Otsu threshold can only split
        // BETWEEN the two bands — erasing one band's internal
        // patch-vs-background contrast entirely — while Sauvola's local
        // mean/std preserves both patches. This is therefore a direct proof
        // that `XyCutParams::binarize_mode` actually reaches `xy_cut`'s
        // segmentation decision, not merely accepted-and-ignored.
        let (w, h) = (40usize, 40usize);
        let mut grey = vec![60u8; w * h];
        for y in 20..40 {
            for x in 0..40 {
                grey[y * w + x] = 200;
            }
        }
        for y in 5..11 {
            for x in 5..11 {
                grey[y * w + x] = 40;
            }
        }
        for y in 25..31 {
            for x in 25..31 {
                grey[y * w + x] = 180;
            }
        }

        let otsu_params = XyCutParams {
            binarize_mode: BinarizeMode::Otsu,
            ..XyCutParams::default()
        };
        let sauvola_params = XyCutParams {
            binarize_mode: BinarizeMode::Sauvola { whsize: 8, k: 0.34 },
            ..XyCutParams::default()
        };
        let otsu_leaves = xy_cut(&grey, w, h, &otsu_params);
        let sauvola_leaves = xy_cut(&grey, w, h, &sauvola_params);
        assert_ne!(
            otsu_leaves, sauvola_leaves,
            "xy_cut must actually use XyCutParams::binarize_mode: Otsu erases one \
             band's internal contrast (a single global split), Sauvola preserves \
             both patches, so the two modes must segment this page differently. \
             otsu={otsu_leaves:?} sauvola={sauvola_leaves:?}"
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

    /// Draw an `n_cols`-column grid of text-like lines: each column is a stack
    /// of `n_lines` horizontal bars, columns separated by `gutter` px of
    /// whitespace. Bars are drawn with a small random-ish right-edge ragged
    /// margin so the column profile is not a perfect rectangle (real text is
    /// ragged), but every gutter column stays ink-free at EVERY row — which is
    /// what makes it a genuine column gutter rather than an accidental gap.
    fn column_grid(
        w: usize,
        h: usize,
        n_cols: usize,
        gutter: usize,
        n_lines: usize,
    ) -> (Vec<u8>, usize, usize) {
        let mut g = page(w, h);
        let col_w = (w - gutter * (n_cols - 1)) / n_cols;
        let line_h = 6usize;
        let line_gap = 4usize;
        let top = 8usize;
        for c in 0..n_cols {
            let x0 = c * (col_w + gutter);
            for l in 0..n_lines {
                let y0 = top + l * (line_h + line_gap);
                let y1 = y0 + line_h;
                if y1 + 4 > h {
                    break;
                }
                // Ragged right edge: cycle the line's width so the block is
                // not a perfect rectangle (mirrors real justified/ragged text).
                let shrink = [0usize, 7, 3, 11, 5][l % 5];
                let x1 = (x0 + col_w).saturating_sub(shrink);
                fill(&mut g, w, x0, x1, y0, y1);
            }
        }
        (g, w, h)
    }

    /// **The multi-column gutter falsifier.** A wide page of `n` columns whose
    /// gutters are generous RELATIVE TO A COLUMN (the only scale that matters
    /// typographically) but small relative to the whole PAGE must still split
    /// into `n` column leaves.
    ///
    /// This is the defect measured on a real 8-column upload: `axis_cuts`
    /// derived its gutter threshold as `ceil(min_gap_frac × extent)` where
    /// `extent` is the CURRENT RECT's cut-axis extent — the full page width for
    /// the top-level column cut. That makes the absolute gutter requirement
    /// independent of how many columns the page has, while real gutters scale
    /// with COLUMN width (≈ `W/n`). Expressed against a column, the requirement
    /// therefore grows linearly in `n`: at `min_gap_frac = 0.015` a 2-column
    /// page needs a 3 %-of-column gutter, but an 8-column page needs ~12 %.
    /// Past that point NO vertical cut is found at all, the page splits only
    /// horizontally, and every emitted strip spans all `n` columns — so the
    /// line-finder reads straight across the gutters and concatenates one line
    /// from every column into a single line of output.
    ///
    /// Measured on the committed `corpus/quality/resgrid.pgm` (3208 px wide,
    /// 8 columns): `gap_min` = 49 px against real gutters of 69-70 px — it
    /// clears the bar by only 1.43×, which is why the existing 8+7+0 CER fence
    /// never caught this. The fixture below is the same layout class with the
    /// gutter proportion tightened to a still-perfectly-legible 16 px on a
    /// 1600 px page (8 % of a column, ~1.9 % of a page vs the 1.5 % nominal).
    #[test]
    fn tight_gutters_on_a_wide_multi_column_page_still_split_into_columns() {
        let (w, h, n_cols, gutter) = (1600usize, 260usize, 8usize, 16usize);
        let (g, w, h) = column_grid(w, h, n_cols, gutter, 13);
        let params = XyCutParams::default();

        // Anti-vacuity: the gutter must genuinely be BELOW the page-relative
        // threshold, or this fixture would pass for the trivial reason and
        // prove nothing about the defect.
        let page_relative = (params.min_gap_frac * w as f32).ceil().max(1.0) as usize;
        assert!(
            gutter < page_relative,
            "fixture must exercise the defect: gutter {gutter} px must be BELOW \
             the page-relative threshold {page_relative} px, else the old rule \
             would have accepted it anyway"
        );

        let leaves = xy_cut(&g, w, h, &params);
        assert_eq!(
            leaves.len(),
            n_cols,
            "a {n_cols}-column page with {gutter} px gutters must yield {n_cols} \
             column leaves, not {} — merged leaves mean the line-finder will \
             read ACROSS the gutters and concatenate all columns into one line",
            leaves.len()
        );
        // Every leaf must be a COLUMN (tall and narrow), never a full-width
        // strip: a full-width strip is the exact shape that produces the
        // read-across-the-gutter output.
        for lf in &leaves {
            assert!(
                lf.width() < w / 2,
                "leaf {lf:?} spans {} px of a {w} px page — that is a full-width \
                 strip, not a column",
                lf.width()
            );
        }
    }

    /// The can-it-stay-silent twin. A DENSE SINGLE-COLUMN page of ragged text
    /// must NOT gain spurious vertical splits: the fix must widen the gutter
    /// rule only where a real regular column structure exists, never split a
    /// paragraph on the accidental 1-2 px whitespace alignments that any ragged
    /// text block contains. Without this, a rule permissive enough to fix the
    /// 8-column case would shred ordinary body text into vertical slivers.
    #[test]
    fn a_dense_single_column_page_gains_no_spurious_vertical_splits() {
        let (w, h) = (1600usize, 260usize);
        let (g, w, h) = column_grid(w, h, 1, 0, 13);
        let leaves = xy_cut(&g, w, h, &XyCutParams::default());
        for lf in &leaves {
            assert!(
                lf.width() > w / 2,
                "single-column page split vertically into a {} px sliver ({lf:?}) \
                 — the gutter rule fired on accidental ragged-edge whitespace",
                lf.width()
            );
        }
    }

    // -----------------------------------------------------------------
    // Escalation ladder — pure boundary tests. Corpus-fixture tests (which
    // rung a real degraded page lands on) live in
    // `tests/binarize_escalation.rs`, matching this crate's convention that
    // tests needing `corpus/` PGMs live in `tests/`, not the lib module.
    // -----------------------------------------------------------------

    /// The three real numbers [`PLAUSIBLE_INK_FRAC_LO`] must separate,
    /// measured on the committed `corpus/quality/*.pgm` set (see
    /// [`binarize_page_escalating`]'s doc): Sauvola's genuine collapse
    /// (`0.0000`, must be REJECTED), the least healthy real reading that
    /// must still pass (`faded_060.pgm`'s Sauvola result, `0.0246`,
    /// ACCEPTED), and a normal healthy reading (`0.028`, ACCEPTED).
    #[test]
    fn plausible_lo_separates_collapse_from_a_real_faded_reading() {
        assert!(
            !is_plausible_ink_frac(0.0000),
            "exact zero (Sauvola's measured collapse) must be rejected"
        );
        assert!(
            is_plausible_ink_frac(0.0246),
            "faded_060.pgm's measured Sauvola reading must be accepted"
        );
        assert!(
            is_plausible_ink_frac(0.028),
            "a healthy reading must be accepted"
        );
    }

    /// The two real numbers [`PLAUSIBLE_INK_FRAC_HI`] must separate: the
    /// highest healthy reading measured across every fixture in this
    /// crate's quality corpus (`0.0303`, must be ACCEPTED) and the lowest
    /// flood reading measured (`uneven_vignette_060.pgm`'s Otsu result,
    /// `0.4566`, must be REJECTED).
    #[test]
    fn plausible_hi_separates_a_real_healthy_ceiling_from_the_lowest_flood() {
        assert!(
            is_plausible_ink_frac(0.0303),
            "the highest measured healthy reading must be accepted"
        );
        assert!(
            !is_plausible_ink_frac(0.4566),
            "the lowest measured flood reading must be rejected"
        );
    }

    #[test]
    fn ink_fraction_of_empty_buffer_is_zero_not_a_panic() {
        assert_eq!(ink_fraction(&[]), 0.0);
    }

    #[test]
    fn ink_fraction_counts_zero_bytes_as_ink() {
        // {0,255} convention: 0 = ink. 3 ink px of 4 = 0.75.
        assert_eq!(ink_fraction(&[0, 0, 0, 255]), 0.75);
    }
}
