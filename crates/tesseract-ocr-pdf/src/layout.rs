//! Resolution-independent page layout model + two renderers that agree on
//! geometry — the "Klickwege parity" twin.
//!
//! This module generalizes [`crate::searchable_pdf`]'s single-purpose
//! searchable-PDF renderer into a small **layout model** ([`LayoutDoc`] /
//! [`LayoutPage`] / [`Block`]) plus two projections of that one model:
//!
//! - [`render_pdf`] — the PDF projection. A full-page background image, placed
//!   image crops (`cm`+`Do`), invisible searchable text (`3 Tr`), painted
//!   visible text (`0 Tr`), and stroked table rules (`re`+`S`). Reuses ALL of
//!   `searchable_pdf.rs`'s helpers (`px_to_pt`, `prec`, `winansi_encode_str`,
//!   `advance_width_1000em`, `embed_grey_image`) so the
//!   `px→pt` / `Tz`-fit / WinAnsi math is byte-for-byte the same as before.
//! - [`render_preview_html`] — the HTML projection. Every block becomes one
//!   absolutely-positioned element at the SAME image-pixel bbox
//!   (`left/top/width/height` in px).
//!
//! **The invariant both renderers honour:** a block's image-pixel bbox drives
//! the PDF coordinates (`Tm` / `cm` / `re`) AND the HTML CSS box identically —
//! the only difference is PDF's bottom-up y-axis (`y_pdf = page_h - y_px`),
//! a deterministic flip, not a discrepancy. That is the Klickwege parity: the
//! searchable PDF and the on-screen preview place everything in lockstep.
//!
//! ## Coordinate convention
//!
//! Every bbox in this module is **image pixels, top-down**, as the tuple
//! `(left, top, right, bottom)` — the same shape [`crate::PlacedWord::box_`]
//! and the `doc.v1` DOM use (`bottom > top`). Points are derived at
//! [`LayoutDoc::dpi`] exactly as [`crate::searchable_pdf`] documents.
//!
//! ## Two builders
//!
//! - [`searchable_layout`] — the "scan + invisible text" case (background =
//!   the scan, one invisible [`TextBlock`] per recognized word).
//!   [`crate::render_searchable_pdf`] is now a thin wrapper over
//!   `render_pdf(&searchable_layout(pages))`.
//! - [`doc_v1_layout`] — reconstructs a page from a `tesseract-rs/doc.v1` JSON
//!   document + the page rasters: text regions → visible text, figure regions
//!   → image crops, table regions → a ruled [`TableBlock`]. No background —
//!   the reconstruction stands on its own.

use std::io::Write as _;

use flate2::write::ZlibEncoder;
use flate2::Compression;
use lopdf::content::{Content, Operation};
use lopdf::{dictionary, Document, Object, ObjectId, Stream};
use serde::Deserialize;

use crate::searchable_pdf::{
    advance_width_1000em, embed_grey_image, prec, px_to_pt, winansi_encode_str, PageOcr,
    RenderReport, SearchablePdfError,
};
use crate::GreyImage;

// ---------------------------------------------------------------------------
// The layout model
// ---------------------------------------------------------------------------

/// A whole document: its pages plus the resolution (`dpi`) at which
/// image-pixel bboxes convert to PDF points (`px * 72 / dpi`).
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutDoc {
    /// The pages, in order.
    pub pages: Vec<LayoutPage>,
    /// Dots per inch — the `px→pt` scale for [`render_pdf`]. HTML preview is
    /// dpi-independent (CSS px == image px).
    pub dpi: u32,
}

/// One page: its pixel dimensions, an optional full-page background scan, and
/// the placed blocks (top-down image-pixel coordinates throughout).
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutPage {
    /// Page width in image pixels.
    pub width: u32,
    /// Page height in image pixels.
    pub height: u32,
    /// The full-page scan drawn under everything else (the searchable-PDF
    /// case), or `None` (the reconstruction case).
    pub background: Option<GreyImage>,
    /// The placed blocks, painted in order.
    pub blocks: Vec<Block>,
}

/// A single placed thing on a page — a text run, an image crop, or a table.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// A placed text run (searchable or painted — see [`TextBlock::visible`]).
    Text(TextBlock),
    /// A placed image crop at `bbox` (e.g. a `doc.v1` figure region).
    Image {
        /// Top-down image-pixel bbox `(left, top, right, bottom)` where the
        /// crop is placed on the page.
        bbox: (u32, u32, u32, u32),
        /// The grey image bytes to embed.
        image: GreyImage,
    },
    /// A placed table with a reconstructed cell grid.
    Table(TableBlock),
}

/// A placed text run. `visible = false` lays it down as an INVISIBLE searchable
/// layer (PDF render mode `3 Tr`) — what you select but never see; `true`
/// paints the glyphs (render mode `0 Tr`) — real, visible, selectable text.
#[derive(Debug, Clone, PartialEq)]
pub struct TextBlock {
    /// Top-down image-pixel bbox `(left, top, right, bottom)`. Without
    /// [`TextBlock::metrics`], the baseline is the box bottom and the font
    /// size is derived from the box height ([`TEXT_HEIGHT_TO_FONTSIZE`]);
    /// with metrics, both come from measurement instead. Either way the run
    /// is `Tz`-stretched to the box width, exactly as
    /// [`crate::searchable_pdf`] documents.
    pub bbox: (u32, u32, u32, u32),
    /// The run's text.
    pub text: String,
    /// `false` → invisible searchable layer (`3 Tr`); `true` → painted (`0 Tr`).
    pub visible: bool,
    /// Measured typographic metrics from `doc.v1` (`None` for word-level
    /// searchable runs and legacy JSON without the per-line keys) — see
    /// [`TextMetrics`].
    pub metrics: Option<TextMetrics>,
    /// How this line was set in the source, when known
    /// ([`classify_justification`]). Consulted only for PAINTED runs — the
    /// invisible searchable layer always stretches to the ink box regardless.
    /// `None` (blocks under 3 lines, legacy callers) renders natural.
    pub justification: Option<Justification>,
}

/// A text run's measured typography, from `doc.v1`'s per-line
/// `xheight`/`ascrise`/`descdrop`/`baseline` keys (all image pixels,
/// top-down), OR — when present, PREFERRED — the newer `glyph_px` measured
/// ink-height key (via [`GLYPH_PX_TO_FONT_PX`]). Either way `font_px` ends
/// up the same quantity: the full ascender-to-descender body height real
/// Tesseract uses as the nominal font size
/// (`LTRResultIterator::WordFontAttributes`, `ltrresultiterator.cpp:168-172`:
/// `row_height = x_height + ascenders - descenders`, then px → printer
/// points; its PDF renderer emits that value per word,
/// `pdfrenderer.cpp:434-447`) — `xheight + ascrise - descdrop` when derived
/// from the band keys, or `glyph_px * GLYPH_PX_TO_FONT_PX` when the
/// measured-ink key is present.
///
/// `glyph_px` is a DIRECT measurement (the topmost-to-bottommost ink row
/// within each glyph's own char box, p90 across a line —
/// `tesseract_ocr::structured::attach_glyph_px`) rather than a statistical
/// fit over blob boxes, so [`doc_v1_layout`] prefers it whenever `doc.v1`
/// carries it: that statistical fit is unstable row-to-row (two
/// identically-printed table rows measured `24.7` vs `14.2` px through it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextMetrics {
    /// Nominal font size in image pixels (see the derivation above).
    pub font_px: f32,
    /// Baseline y in TOP-DOWN image pixels.
    pub baseline_px: f32,
}

/// Scale factor converting a measured glyph ink-height (`doc.v1`'s
/// `glyph_px`, from `tesseract_ocr::structured::attach_glyph_px` — the p90
/// topmost-to-bottommost ink row within a char box's x-span) into the
/// equivalent full ascender-to-descender body height
/// ([`TextMetrics::font_px`]'s existing unit, `xheight + ascrise -
/// descdrop`).
///
/// Not a magic number: derived from the SAME transcoded band fractions
/// `tesseract-ocr/src/textline.rs` uses for `TO_ROW::xheight`/`ascrise`/
/// `descdrop` (`tesseract::CCStruct::kXHeightFraction = 0.5`,
/// `kAscenderFraction = 0.25`, `kDescenderFraction = 0.25`,
/// `ccstruct.cpp:25-27` — private `const`s in that module, so restated
/// here by value and provenance rather than imported). A row's nominal
/// size divides `x-height : ascender-rise : descender-drop = 0.5 : 0.25 :
/// 0.25`, so the FULL ascender-to-descender body height `font_px` sizes
/// from is `1.0` of that total (`0.5 + 0.25 + 0.25`). A glyph's MEASURED
/// ink extent, by contrast, is bounded by its baseline row below and the
/// top of an ascender or capital letter above — `0.5 + 0.25 = 0.75` of the
/// total — because the p90 statistic
/// (`tesseract_ocr::structured::attach_glyph_px`) predominantly rides the
/// ascender/cap-height population of a line's glyphs: almost every printed
/// line contains SOME tall letters or capitals, and it is exactly those
/// that push a HIGH percentile up past the x-height-only majority. The
/// ratio between the two, `1.0 / 0.75 = 4/3`, converts a measured ink
/// height into its equivalent full body height.
const GLYPH_PX_TO_FONT_PX: f32 = 4.0 / 3.0;

/// A placed table: its outer bbox, grid dimensions, cells, and whether to
/// stroke the cell/outer rectangles as visible rules.
///
/// `Eq` is NOT derived: [`TableCell::metrics`] carries `f32`, the same
/// no-`Eq` cascade `tesseract_ocr::structured::TableCell` documents.
#[derive(Debug, Clone, PartialEq)]
pub struct TableBlock {
    /// Top-down image-pixel bbox `(left, top, right, bottom)` of the whole
    /// table (the block's Klickwege-parity anchor).
    pub bbox: (u32, u32, u32, u32),
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub cols: usize,
    /// The occupied cells.
    pub cells: Vec<TableCell>,
    /// Whether to stroke the outer + per-cell rectangles as thin rules.
    pub rules: bool,
}

/// One table cell: its grid position, bbox, text, header flag, and the
/// measured typography of the line its row is.
///
/// `Eq` is NOT derived — see [`TableCell::metrics`] (`f32`).
#[derive(Debug, Clone, PartialEq)]
pub struct TableCell {
    /// 0-based row.
    pub row: usize,
    /// 0-based column.
    pub col: usize,
    /// Top-down image-pixel bbox `(left, top, right, bottom)`.
    pub bbox: (u32, u32, u32, u32),
    /// The cell's text.
    pub text: String,
    /// `true` for a header cell (rendered the same, marked for consumers).
    pub header: bool,
    /// The measured typography of the line this cell's row IS, from `doc.v1`'s
    /// additive per-cell `xheight`/`ascrise`/`descdrop`/`baseline`/`glyph_px`
    /// keys, derived by the SAME [`derive_text_metrics`] the per-line path
    /// uses so the two can never drift.
    ///
    /// `None` degrades to the [`TEXT_HEIGHT_TO_FONTSIZE`] box-height heuristic
    /// and a box-bottom pen, exactly as before this field existed — which is
    /// what EVERY table cell got unconditionally until it did, in any
    /// document, a correctly-classified table as much as a misclassified one.
    pub metrics: Option<TextMetrics>,
}

// ---------------------------------------------------------------------------
// PDF projection
// ---------------------------------------------------------------------------

/// XObject resource name for an embedded image. The background (when present)
/// is always `Im0`; image blocks follow in placement order. Both the embedder
/// (in [`render_pdf`]) and the content builder ([`build_layout_content`]) use
/// this ONE rule, so the `Do <name>` operator always matches a resource entry.
fn xobject_name(background_present: bool, image_block_ordinal: usize) -> String {
    let base = usize::from(background_present);
    format!("Im{}", base + image_block_ordinal)
}

/// The PDF `cm`/`re` placement `(w_pt, h_pt, x_pt, y_pt)` for a top-down image
/// bbox, flipping y bottom-up (`y_pt = page_h_pt - bottom_pt`). Returns `None`
/// for a degenerate (zero-area) bbox — the same guard the per-word path uses.
fn placement(bbox: (u32, u32, u32, u32), dpi: u32, page_h_pt: f64) -> Option<(f64, f64, f64, f64)> {
    let (left, top, right, bottom) = bbox;
    if right <= left || bottom <= top {
        return None;
    }
    let w_pt = px_to_pt(f64::from(right - left), dpi);
    let h_pt = px_to_pt(f64::from(bottom - top), dpi);
    let x_pt = px_to_pt(f64::from(left), dpi);
    let y_pt = page_h_pt - px_to_pt(f64::from(bottom), dpi);
    Some((w_pt, h_pt, x_pt, y_pt))
}

/// Push `re`+`S` (rectangle path + stroke) for a bbox, if non-degenerate.
fn push_rect_stroke(
    ops: &mut Vec<Operation>,
    bbox: (u32, u32, u32, u32),
    dpi: u32,
    page_h_pt: f64,
) {
    if let Some((w_pt, h_pt, x_pt, y_pt)) = placement(bbox, dpi, page_h_pt) {
        ops.push(Operation::new(
            "re",
            vec![
                (prec(x_pt) as f32).into(),
                (prec(y_pt) as f32).into(),
                (prec(w_pt) as f32).into(),
                (prec(h_pt) as f32).into(),
            ],
        ));
        ops.push(Operation::new("S", vec![]));
    }
}

/// Shrink factor from a text block's reported bbox height down to a rendered
/// font size. The bbox height is an "at least" ascender-to-descender band —
/// `tesseract-ocr`'s `makerow_row_crops` deliberately EXTENDS a row's box to
/// cover ascenders/descenders generously (recognition robustness), not a
/// tight visual line-height — so using it verbatim as `Tf`'s nominal size
/// overshoots.
///
/// The transcoded typographic-band math (`textline.rs`'s
/// `K_XHEIGHT_FRACTION`/`K_ASCENDER_FRACTION`/`K_DESCENDER_FRACTION` =
/// `0.5`/`0.25`/`0.25`) sizes a well-behaved single line's band at
/// `xheight + ascrise + |descdrop|` ≈ `1.0 × line_size` — the box height is
/// *expected* to be roughly one line's baseline-to-baseline pitch already.
/// Real scans routinely exceed that (dense/antique typesetting, diacritics,
/// row-fitting slack): measured on a real multi-paragraph page, consecutive
/// `Tm` baselines landed ~15pt apart while `fontsize_pt = box_h_pt` chose
/// ~30-31pt — roughly double the real pitch, so every line's glyphs bled a
/// half-line into both neighbours (the reported bug). `0.5` maps that
/// measured case back onto its real pitch while leaving a well-behaved
/// (~1.0×) band comfortably smaller than its own box — a safe direction to
/// be wrong in, since an undersized font stays legible where an oversized
/// one collides. Shared by [`emit_text_run`] (the PDF projection) and
/// [`text_font_size_px`] (the HTML projection), so both size text the same
/// way — the Klickwege-parity invariant applied to text SIZE, not just
/// position.
const TEXT_HEIGHT_TO_FONTSIZE: f64 = 0.5;

/// Emit the `Tm`/`Tf`/`Tz`/`Tj` run for one text bbox (baseline = box bottom,
/// `Tz`-fitted to the box width). Returns the WinAnsi `'?'`-substitution count.
/// A degenerate box emits nothing and counts nothing; a positive box counts
/// its substitutions even if its (all-zero-advance) run shows nothing — this
/// matches the original `build_page_content` semantics exactly.
/// Largest width correction justification may absorb, as a fraction of the
/// column. Past this the line is not a justified line at all (a heading, a
/// mis-segmented run, a single long token) and is left natural rather than
/// pulled apart.
const MAX_JUSTIFY_SLACK_FRAC: f64 = 0.35;

/// How a run is fitted to its box width.
///
/// The distinction is not stylistic — it is about whether anyone sees the
/// glyphs:
///
/// - [`StretchToBox`](RunFit::StretchToBox) — the INVISIBLE searchable layer
///   (`3 Tr`). Nobody renders these glyphs; what matters is that a selection
///   or a search hit lands exactly on the scanned ink. Distorting the glyphs
///   with `Tz` to occupy the ink box precisely is therefore not just
///   acceptable, it is the correct behaviour, and it stays.
/// - [`Natural`](RunFit::Natural) — PAINTED ragged text. Glyphs are drawn at
///   their true proportions and the line ends where it ends.
/// - [`JustifyToBox`](RunFit::JustifyToBox) — PAINTED justified text. The
///   line reaches the measure through WORD SPACING (`Tw`), never by squeezing
///   letterforms.
///
/// The bug this replaces: `Tz` was applied unconditionally, so painted text
/// got a different horizontal glyph scale on every single line. Measured on a
/// real structured render, `Tz` ranged **32.4 % to 850 %** — letterforms
/// alternately crushed to a third and stretched eightfold, line by line. That
/// inconsistency is far more damaging to a reader than a few points of
/// font-size error, and no font-size fix alone would have touched it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunFit {
    /// Distort glyphs to occupy the box exactly (invisible layer only).
    StretchToBox,
    /// Draw at true proportions; the line ends naturally.
    Natural,
    /// Reach the measure via word spacing, glyphs undistorted.
    JustifyToBox,
}

fn emit_text_run(
    ops: &mut Vec<Operation>,
    bbox: (u32, u32, u32, u32),
    text: &str,
    dpi: u32,
    page_h_pt: f64,
    metrics: Option<&TextMetrics>,
    fit: RunFit,
) -> usize {
    let (left, top, right, bottom) = bbox;
    if right <= left || bottom <= top {
        return 0;
    }
    let (bytes, substitutions) = winansi_encode_str(text);
    let natural_width_1000em = advance_width_1000em(&bytes);
    if natural_width_1000em == 0 {
        return substitutions;
    }
    let box_w_pt = px_to_pt(f64::from(right - left), dpi);
    let box_h_pt = px_to_pt(f64::from(bottom - top), dpi);
    let x_pt = px_to_pt(f64::from(left), dpi);
    // With measured metrics: font size = the row's typographic body height
    // (`xheight + ascrise - descdrop`), and the pen sits on the MEASURED
    // baseline -- both exactly what real Tesseract renders from
    // (`ltrresultiterator.cpp:168-172` / `pdfrenderer.cpp:434-447`). Without
    // metrics (word-level searchable runs, legacy doc.v1): baseline = box
    // bottom and font size from the box height via TEXT_HEIGHT_TO_FONTSIZE,
    // the pre-metrics behaviour, unchanged.
    let (y_pt, fontsize_pt) = match metrics {
        Some(m) => (
            page_h_pt - px_to_pt(f64::from(m.baseline_px), dpi),
            px_to_pt(f64::from(m.font_px), dpi).max(1.0),
        ),
        None => (
            page_h_pt - px_to_pt(f64::from(bottom), dpi),
            box_h_pt * TEXT_HEIGHT_TO_FONTSIZE,
        ),
    };
    let natural_width_pt = fontsize_pt * f64::from(natural_width_1000em) / 1000.0;
    // How the run reaches (or does not reach) the box width. `Tz` distorts
    // GLYPH shapes; `Tw` widens the SPACES. Which is correct depends entirely
    // on whether a human ever looks at these glyphs — see [`RunFit`].
    let (tz, tw) = match fit {
        RunFit::StretchToBox => (100.0 * box_w_pt / natural_width_pt, 0.0),
        RunFit::Natural => (100.0, 0.0),
        RunFit::JustifyToBox => {
            let spaces = bytes.iter().filter(|b| **b == b' ').count();
            let slack = box_w_pt - natural_width_pt;
            // Justify only through the spaces, and only when there are spaces
            // to give and the correction is not absurd. A single word cannot
            // be justified without distorting it, so it is simply left
            // natural — a short last line, exactly as a typesetter leaves it.
            if spaces == 0 || slack.abs() > box_w_pt * MAX_JUSTIFY_SLACK_FRAC {
                (100.0, 0.0)
            } else {
                (100.0, slack / spaces as f64)
            }
        }
    };

    ops.push(Operation::new(
        "Tm",
        vec![
            1.into(),
            0.into(),
            0.into(),
            1.into(),
            (prec(x_pt) as f32).into(),
            (prec(y_pt) as f32).into(),
        ],
    ));
    ops.push(Operation::new(
        "Tf",
        vec!["F1".into(), (prec(fontsize_pt) as f32).into()],
    ));
    ops.push(Operation::new("Tz", vec![(prec(tz) as f32).into()]));
    // Always emitted: `Tw` is part of the PDF text state and persists across
    // runs, so a run that needs none must reset it to 0 rather than inherit
    // the previous line's spacing.
    ops.push(Operation::new("Tw", vec![(prec(tw) as f32).into()]));
    ops.push(Operation::new(
        "Tj",
        // RAW WinAnsi bytes — lopdf escapes a Literal string itself. Escaping
        // here first would double-escape (see searchable_pdf.rs's note).
        vec![Object::String(bytes, lopdf::StringFormat::Literal)],
    ));
    substitutions
}

/// Build one page's content stream: background → image blocks → table rules →
/// invisible text (`3 Tr`) → visible text (`0 Tr`, incl. table cells). Returns
/// the encoded bytes and the page's WinAnsi substitution count.
fn build_layout_content(page: &LayoutPage, dpi: u32) -> (Vec<u8>, usize) {
    let page_w_pt = px_to_pt(f64::from(page.width), dpi);
    let page_h_pt = px_to_pt(f64::from(page.height), dpi);
    let bg_present = page.background.is_some();

    let mut ops: Vec<Operation> = Vec::new();
    let mut substitutions = 0usize;

    // 1) Full-page background (identical prologue to the old renderer).
    if bg_present {
        ops.push(Operation::new("q", vec![]));
        ops.push(Operation::new(
            "cm",
            vec![
                (prec(page_w_pt) as f32).into(),
                0.into(),
                0.into(),
                (prec(page_h_pt) as f32).into(),
                0.into(),
                0.into(),
            ],
        ));
        ops.push(Operation::new("Do", vec!["Im0".into()]));
        ops.push(Operation::new("Q", vec![]));
    }

    // 2) Placed image blocks — `cm` positions the unit square at the bbox.
    let mut img_ord = 0usize;
    for block in &page.blocks {
        if let Block::Image { bbox, .. } = block {
            let name = xobject_name(bg_present, img_ord);
            img_ord += 1;
            if let Some((w_pt, h_pt, x_pt, y_pt)) = placement(*bbox, dpi, page_h_pt) {
                ops.push(Operation::new("q", vec![]));
                ops.push(Operation::new(
                    "cm",
                    vec![
                        (prec(w_pt) as f32).into(),
                        0.into(),
                        0.into(),
                        (prec(h_pt) as f32).into(),
                        (prec(x_pt) as f32).into(),
                        (prec(y_pt) as f32).into(),
                    ],
                ));
                ops.push(Operation::new("Do", vec![Object::Name(name.into_bytes())]));
                ops.push(Operation::new("Q", vec![]));
            }
        }
    }

    // 3) Table rule strokes (thin black rectangles: outer + each cell).
    let any_rules = page
        .blocks
        .iter()
        .any(|b| matches!(b, Block::Table(t) if t.rules));
    if any_rules {
        ops.push(Operation::new("q", vec![]));
        ops.push(Operation::new("w", vec![(0.5_f32).into()]));
        for block in &page.blocks {
            if let Block::Table(t) = block {
                if t.rules {
                    push_rect_stroke(&mut ops, t.bbox, dpi, page_h_pt);
                    for cell in &t.cells {
                        push_rect_stroke(&mut ops, cell.bbox, dpi, page_h_pt);
                    }
                }
            }
        }
        ops.push(Operation::new("Q", vec![]));
    }

    // 4) Invisible searchable text (render mode 3).
    let has_invisible = page
        .blocks
        .iter()
        .any(|b| matches!(b, Block::Text(t) if !t.visible));
    if has_invisible {
        ops.push(Operation::new("BT", vec![]));
        ops.push(Operation::new("Tr", vec![3.into()]));
        for block in &page.blocks {
            if let Block::Text(t) = block {
                if !t.visible {
                    substitutions += emit_text_run(
                        &mut ops,
                        t.bbox,
                        &t.text,
                        dpi,
                        page_h_pt,
                        t.metrics.as_ref(),
                        // Invisible: fitting the ink box exactly is what makes
                        // selection and search land on the right pixels.
                        RunFit::StretchToBox,
                    );
                }
            }
        }
        ops.push(Operation::new("ET", vec![]));
    }

    // 5) Visible painted text (render mode 0): visible text blocks + table cells.
    let has_visible = page.blocks.iter().any(|b| match b {
        Block::Text(t) => t.visible,
        Block::Table(t) => !t.cells.is_empty(),
        Block::Image { .. } => false,
    });
    if has_visible {
        ops.push(Operation::new("BT", vec![]));
        ops.push(Operation::new("Tr", vec![0.into()]));
        for block in &page.blocks {
            match block {
                Block::Text(t) if t.visible => {
                    substitutions += emit_text_run(
                        &mut ops,
                        t.bbox,
                        &t.text,
                        dpi,
                        page_h_pt,
                        t.metrics.as_ref(),
                        match t.justification {
                            Some(Justification::Blocksatz) => RunFit::JustifyToBox,
                            _ => RunFit::Natural,
                        },
                    );
                }
                Block::Table(t) => {
                    for cell in &t.cells {
                        // Real measured metrics when the cell carries them
                        // (see `TableCell::metrics`); `None` only for legacy
                        // doc.v1 without the additive per-cell keys, which
                        // then degrades exactly as every cell used to.
                        substitutions += emit_text_run(
                            &mut ops,
                            cell.bbox,
                            &cell.text,
                            dpi,
                            page_h_pt,
                            cell.metrics.as_ref(),
                            // A cell is a short measure, never justified.
                            RunFit::Natural,
                        );
                    }
                }
                _ => {}
            }
        }
        ops.push(Operation::new("ET", vec![]));
    }

    let content = Content { operations: ops };
    (
        content.encode().expect("encode content stream"),
        substitutions,
    )
}

/// Validate one embedded image's data length against its declared dimensions.
fn check_image(page: usize, img: &GreyImage) -> Result<(), SearchablePdfError> {
    let expected = img.w * img.h;
    if img.data.len() != expected {
        return Err(SearchablePdfError::ImageSizeMismatch {
            page,
            expected,
            got: img.data.len(),
        });
    }
    Ok(())
}

/// Render a [`LayoutDoc`] to a PDF, returning the bytes plus the per-page
/// WinAnsi substitution report. Generalizes the searchable-PDF renderer: it
/// paints the optional background, places image crops, strokes table rules,
/// and lays both invisible (`3 Tr`) and visible (`0 Tr`) text — all at the
/// blocks' image-pixel bboxes converted at [`LayoutDoc::dpi`].
///
/// # Errors
///
/// [`SearchablePdfError::ImageSizeMismatch`] if any embedded image's
/// `data.len()` doesn't match `w * h`; [`SearchablePdfError::Save`] if `lopdf`
/// fails to serialize the assembled document.
pub fn render_pdf(doc: &LayoutDoc) -> Result<(Vec<u8>, RenderReport), SearchablePdfError> {
    // Validate every embedded image up front (mirrors the old renderer's
    // whole-batch precheck, so a bad page fails before any bytes are built).
    for (pi, page) in doc.pages.iter().enumerate() {
        if let Some(bg) = &page.background {
            check_image(pi, bg)?;
        }
        for block in &page.blocks {
            if let Block::Image { image, .. } = block {
                check_image(pi, image)?;
            }
        }
    }

    let mut pdf = Document::with_version("1.5");
    let pages_id = pdf.new_object_id();

    let font_id = pdf.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
        "Encoding" => "WinAnsiEncoding",
    });

    let mut kids: Vec<Object> = Vec::with_capacity(doc.pages.len());
    let mut report = RenderReport {
        pages: Vec::with_capacity(doc.pages.len()),
    };

    for page in &doc.pages {
        // Embed images in the SAME order the content builder names them.
        let bg_present = page.background.is_some();
        let mut xobjects: Vec<(String, ObjectId)> = Vec::new();
        if let Some(bg) = &page.background {
            let id = embed_grey_image(&mut pdf, bg);
            xobjects.push(("Im0".to_string(), id));
        }
        let mut img_ord = 0usize;
        for block in &page.blocks {
            if let Block::Image { image, .. } = block {
                let id = embed_grey_image(&mut pdf, image);
                xobjects.push((xobject_name(bg_present, img_ord), id));
                img_ord += 1;
            }
        }

        let (content_bytes, subs) = build_layout_content(page, doc.dpi);
        report.pages.push(subs);

        let mut resources = lopdf::Dictionary::new();
        if !xobjects.is_empty() {
            let mut xdict = lopdf::Dictionary::new();
            for (name, id) in &xobjects {
                xdict.set(name.clone(), *id);
            }
            resources.set("XObject", xdict);
        }
        let mut fonts = lopdf::Dictionary::new();
        fonts.set("F1", font_id);
        resources.set("Font", fonts);
        let resources_id = pdf.add_object(resources);

        let content_id = pdf.add_object(Stream::new(dictionary! {}, content_bytes));

        let page_w_pt = px_to_pt(f64::from(page.width), doc.dpi);
        let page_h_pt = px_to_pt(f64::from(page.height), doc.dpi);
        let page_id = pdf.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "Resources" => resources_id,
            "MediaBox" => vec![
                0.into(),
                0.into(),
                (prec(page_w_pt) as f32).into(),
                (prec(page_h_pt) as f32).into(),
            ],
        });
        kids.push(page_id.into());
    }

    let kids_count = i64::try_from(kids.len()).unwrap_or(i64::MAX);
    let pages_dict = dictionary! {
        "Type" => "Pages",
        "Kids" => kids,
        "Count" => kids_count,
    };
    pdf.objects.insert(pages_id, Object::Dictionary(pages_dict));

    let catalog_id = pdf.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    pdf.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    pdf.save_to(&mut bytes).map_err(SearchablePdfError::Save)?;
    Ok((bytes, report))
}

// ---------------------------------------------------------------------------
// HTML projection (the Klickwege-parity twin)
// ---------------------------------------------------------------------------

/// The CSS `left/top/width/height` (all px) for a top-down image bbox — the
/// exact same rectangle the PDF projection places (before the bottom-up flip).
fn css_box(bbox: (u32, u32, u32, u32)) -> String {
    let (l, t, r, b) = bbox;
    let w = r.saturating_sub(l);
    let h = b.saturating_sub(t);
    format!("left:{l}px;top:{t}px;width:{w}px;height:{h}px")
}

/// The HTML preview's font-size (px) for a text bbox — mirrors
/// [`emit_text_run`]'s `fontsize_pt = box_h_pt * TEXT_HEIGHT_TO_FONTSIZE`
/// (same shrink, same source height) so the debug preview shows what the
/// PDF's painted/invisible text actually looks like, not an unrelated fixed
/// size. Clamped to at least `1.0` so a degenerate/thin box never emits
/// `font-size:0px`.
/// Font size as a fraction of the baseline pitch.
///
/// Standard typesetting sets leading at 1.15–1.30× the body size, so the body
/// is 0.77–0.87 of the pitch; `0.80` sits in that band. This is ONE constant
/// for the whole corpus rather than a per-line measurement, which is the
/// point — see [`page_font_px`].
const PITCH_TO_FONT_PX: f32 = 0.80;

/// Hard no-overlap ceiling: the rendered body may approach — but never
/// exceed — the measured baseline pitch. Solid-set print (the corpus pages
/// measure 17–18 px ink on an 18 px pitch) legitimately runs close to 1.0;
/// the margin below it absorbs pitch-measurement jitter.
const MAX_FONT_TO_PITCH: f32 = 0.95;

/// How far apart two line right-edges may sit (as a fraction of the column
/// width) and still count as "the same margin" for [`Justification`].
const BLOCKSATZ_EDGE_TOLERANCE: f32 = 0.02;

/// How a block's lines are set horizontally.
///
/// This is not cosmetic bookkeeping: it decides whether a line's WIDTH is
/// evidence about the font size.
///
/// - [`Blocksatz`](Justification::Blocksatz) (justified) — every line except
///   the last was stretched to *exactly* the column measure. The rendered
///   width is therefore a hard constraint, and comparing it against the
///   measured ink width calibrates the render-font-vs-scan-font width ratio
///   (the substitution error: Helvetica is narrower than the DejaVu-class
///   faces these corpora are set in, so "the size that fills this column"
///   overshoots by that ratio and by nothing else).
/// - [`Flattersatz`](Justification::Flattersatz) (ragged) — a line ends
///   wherever its last word happened to land, so width carries no
///   information about size and must not be used as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Justification {
    /// Justified: right edges agree to within [`BLOCKSATZ_EDGE_TOLERANCE`].
    Blocksatz,
    /// Ragged: right edges vary.
    Flattersatz,
}

/// Classify a block's justification from the spread of its line right-edges.
///
/// The last line of a justified paragraph is short by construction and is
/// excluded — including it would make every `Blocksatz` block look ragged,
/// which is the classic false negative here.
///
/// Needs at least three lines: two lines give one comparison, and a single
/// coincidence would then decide the answer.
#[must_use]
pub fn classify_justification(line_bboxes: &[(u32, u32, u32, u32)]) -> Option<Justification> {
    if line_bboxes.len() < 3 {
        return None;
    }
    // Drop the final line (short by construction in justified text).
    let rights: Vec<f32> = line_bboxes[..line_bboxes.len() - 1]
        .iter()
        .map(|b| b.2 as f32)
        .collect();
    let lefts: Vec<f32> = line_bboxes.iter().map(|b| b.0 as f32).collect();
    let (min_r, max_r) = rights
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &r| (lo.min(r), hi.max(r)));
    let measure = max_r - lefts.iter().cloned().fold(f32::MAX, f32::min);
    if measure <= 0.0 {
        return None;
    }
    Some(if (max_r - min_r) / measure <= BLOCKSATZ_EDGE_TOLERANCE {
        Justification::Blocksatz
    } else {
        Justification::Flattersatz
    })
}

/// The median consecutive baseline delta over an ENTIRE page, in pixels.
///
/// # Why the pitch and not the ink
///
/// Measured on a real 16-cell structured render (172 runs, 11 columns): the
/// per-line ink-derived size has median `Tf`/pitch **1.44** and overlaps
/// **95%** of consecutive line pairs, and the *same string* comes out up to
/// **1.82× different** between columns. Its implied ink height is 18.4 pt on
/// an 18 pt pitch — taller than the distance to the next line, which is
/// impossible for one line of text and is the signature of
/// `attach_glyph_px`'s scan reading its neighbours through the (deliberately
/// generous) recognition band.
///
/// Taking the *median of that same ink measurement* does not rescue it —
/// measured, it still leaves 97% overlapping, because on tightly-set text
/// most lines are contaminated so the median is contaminated too. The pitch
/// is the quantity that is already correct: **median 18.00 with sd 0.00
/// across all eight columns** of that page.
///
/// # Why the median of deltas, not span/(n−1)
///
/// The geometric form — column height over line count — is corrupted by
/// dropped lines, and OCR drops lines exactly where the scan degrades. On
/// that same page it inflates monotonically with degradation, 19.08 → 21.68 →
/// 22.71 → 25.32, while the median of consecutive deltas holds at 18.00: a
/// dropped line doubles ONE gap, and a median steps over it.
///
/// Cost is a sort of a few hundred floats against a multi-second recognition
/// pass — free in context, which is why it runs page-wide rather than
/// per-block.
///
/// Returns `None` when fewer than [`MIN_PITCH_SAMPLES`] deltas exist, in
/// which case callers keep the per-line path.
fn page_pitch_px(page: &JsonPage) -> Option<f32> {
    let mut deltas: Vec<f32> = Vec::new();
    for region in &page.regions {
        // Within a region only: a delta ACROSS regions is a column jump, not
        // a line pitch.
        let mut bl: Vec<f32> = region.lines.iter().filter_map(|l| l.baseline).collect();
        bl.sort_by(f32::total_cmp);
        deltas.extend(bl.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0.0));
    }
    if deltas.len() < MIN_PITCH_SAMPLES {
        return None;
    }
    deltas.sort_by(f32::total_cmp);
    Some(deltas[deltas.len() / 2])
}

/// Minimum consecutive-baseline samples before a page pitch is trusted.
const MIN_PITCH_SAMPLES: usize = 3;

/// The one font size for a page: the largest that overflows NEITHER axis.
///
/// # "does 12.4 become too wide, or too high?"
///
/// A single number cannot be read off one axis, because the render face is
/// not the scanned face. Two independent constraints bound it, and the
/// binding one wins:
///
/// - **too high** — the body cannot exceed its own leading, so
///   `pitch × PITCH_TO_FONT_PX`.
/// - **too wide** — at the right size a line's glyphs span exactly its
///   measured ink width, so `1000 × box_width / natural_advance`.
///
/// Taking the MINIMUM is what makes this self-correcting for font
/// substitution. Measured on the reference page the height constraint binds
/// (≈14.4 pt) while the width constraint sits higher (≈17.7 pt, sd 0.04)
/// because Helvetica is narrower than the DejaVu-class face the corpus is set
/// in — so the page renders at 14.4 and neither axis overflows. Were the
/// render face the WIDER one, width would bind instead and the page would
/// shrink to fit rather than spilling past the measure. Neither axis is
/// privileged; the fit decides.
///
/// The width samples are a MEDIAN for the same reason the pitch is (see
/// [`page_pitch_px`]): a garbled line in a degraded region produces a wild
/// advance, and a median steps over it where a mean would not.
fn page_font_px(page: &JsonPage) -> Option<f32> {
    let pitch = page_pitch_px(page)?;

    // The width constraint, per line, in px of font size.
    let mut widths: Vec<f32> = Vec::new();
    for region in &page.regions {
        for line in &region.lines {
            let text = line
                .words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let (bytes, _) = winansi_encode_str(&text);
            let advance = advance_width_1000em(&bytes);
            if advance == 0 {
                continue;
            }
            let (l, _, r, _) = match line.bbox {
                Some(b) => to_u32_bbox(b),
                None => union_words(&line.words),
            };
            if r <= l {
                continue;
            }
            widths.push(1000.0 * (r - l) as f32 / advance as f32);
        }
    }
    if widths.len() < MIN_PITCH_SAMPLES {
        // No width evidence — the conservative leading relation stands alone.
        return Some(pitch * PITCH_TO_FONT_PX);
    }
    widths.sort_by(f32::total_cmp);
    let by_width = widths[widths.len() / 2];
    // WIDTH-FIRST, capped by the hard no-overlap ceiling — the operator
    // acceptance rule from the 16-cell page: readers tolerate tight leading
    // far better than UNEVEN COLUMN GUTTERS. Rendering at the leading
    // relation (0.80 x pitch = 14.40 there) left every line ~19% narrower
    // than its measured ink box, so inter-column gaps varied 6..12% with
    // text length -- "beyond the acceptance threshold". The width solve
    // (17.73, sd 0.04 -- the most consistent signal on the page) fills the
    // measure uniformly, and at 0.985 x pitch it never overlapped anyway;
    // the ceiling only guards the genuinely-wide-render-font case.
    Some(by_width.min(pitch * MAX_FONT_TO_PITCH))
}

fn text_font_size_px(bbox: (u32, u32, u32, u32), metrics: Option<&TextMetrics>) -> f64 {
    if let Some(m) = metrics {
        // Same measured body height the PDF projection uses -- Klickwege
        // parity applies to text SIZE, not just position.
        return f64::from(m.font_px).max(1.0);
    }
    let (_, top, _, bottom) = bbox;
    (f64::from(bottom.saturating_sub(top)) * TEXT_HEIGHT_TO_FONTSIZE).max(1.0)
}

/// Minimal HTML text escaping for element content / attribute values.
fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// A `data:image/png;base64,...` URI for a grey image, or `None` when the
/// image is empty or its data is short (the caller emits a placeholder then).
fn grey_to_data_uri(image: &GreyImage) -> Option<String> {
    if image.w == 0 || image.h == 0 || image.data.len() < image.w * image.h {
        return None;
    }
    Some(format!(
        "data:image/png;base64,{}",
        base64(&grey_to_png(image))
    ))
}

/// Encode a grey raster as a `data:image/png;base64,...` URI (pure-Rust PNG —
/// the same encoder [`render_preview_html`] embeds), or `None` when the image
/// is empty / its data is shorter than `w * h`.
///
/// Exposed for consumers that overlay their OWN annotations on the page raster
/// — e.g. a debug preview drawing colour-coded `doc.v1` region rectangles over
/// the scan — and therefore need a browser-renderable background image without
/// re-implementing a PNG encoder. PNG is used deliberately: an uploaded PGM /
/// TIFF scan does not render in an `<img>`, but the grey → PNG transcode always
/// does.
#[must_use]
pub fn grey_png_data_uri(image: &GreyImage) -> Option<String> {
    grey_to_data_uri(image)
}

/// Render a [`LayoutDoc`] as a standalone HTML preview. Each page is a
/// `position:relative` box sized in image pixels; every block is one
/// absolutely-positioned child at the SAME px bbox as the PDF projection — the
/// Klickwege parity. Background/figures become data-URI PNG `<img>`s (pure-Rust
/// [`grey_to_png`]); text becomes `<div>`s; tables become a bordered container
/// with (container-relative) cell divs.
#[must_use]
pub fn render_preview_html(doc: &LayoutDoc) -> String {
    let mut html = String::new();
    html.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    html.push_str("<title>doc.v1 preview</title>\n<style>\n");
    html.push_str("body{margin:0;background:#888}\n");
    html.push_str(".page{position:relative;margin:8px auto;background:#fff;overflow:hidden;");
    html.push_str("box-shadow:0 0 4px rgba(0,0,0,.4)}\n");
    html.push_str(".page>*{position:absolute;box-sizing:border-box}\n");
    html.push_str(".text{font-family:sans-serif;white-space:pre;color:#000}\n");
    html.push_str(".table{border:1px solid #333}\n");
    html.push_str(".table .cell{position:absolute;box-sizing:border-box;");
    html.push_str("font-family:sans-serif;border:1px solid #999;overflow:hidden;padding:1px}\n");
    html.push_str(".figure{outline:1px dashed #666}\n");
    html.push_str("</style>\n</head>\n<body>\n");

    for (pi, page) in doc.pages.iter().enumerate() {
        let pageno = pi + 1;
        html.push_str(&format!(
            "<div class=\"page\" data-page=\"{pageno}\" style=\"width:{}px;height:{}px\">\n",
            page.width, page.height
        ));

        if let Some(bg) = &page.background {
            let bx = css_box((0, 0, page.width, page.height));
            match grey_to_data_uri(bg) {
                Some(uri) => html.push_str(&format!(
                    "  <img class=\"background\" style=\"{bx}\" src=\"{uri}\" alt=\"page scan\">\n"
                )),
                None => html.push_str(&format!(
                    "  <div class=\"background\" style=\"{bx};background:#eee\"></div>\n"
                )),
            }
        }

        for block in &page.blocks {
            match block {
                Block::Text(t) => {
                    let font_px = text_font_size_px(t.bbox, t.metrics.as_ref());
                    html.push_str(&format!(
                        "  <div class=\"text\" style=\"{};font-size:{font_px}px;line-height:1\">{}</div>\n",
                        css_box(t.bbox),
                        html_escape(&t.text)
                    ));
                }
                Block::Image { bbox, image } => {
                    let bx = css_box(*bbox);
                    match grey_to_data_uri(image) {
                        Some(uri) => html.push_str(&format!(
                            "  <img class=\"figure\" style=\"{bx}\" src=\"{uri}\" alt=\"figure\">\n"
                        )),
                        None => html
                            .push_str(&format!("  <div class=\"figure\" style=\"{bx}\"></div>\n")),
                    }
                }
                Block::Table(t) => {
                    html.push_str(&format!(
                        "  <div class=\"table\" style=\"{}\">\n",
                        css_box(t.bbox)
                    ));
                    let (tl, tt, ..) = t.bbox;
                    for cell in &t.cells {
                        let (cl, ct, cr, cb) = cell.bbox;
                        // Cells are container-relative (visual only); the block's
                        // parity anchor is the table's own bbox on the div above.
                        let (l, top) = (cl.saturating_sub(tl), ct.saturating_sub(tt));
                        let (w, h) = (cr.saturating_sub(cl), cb.saturating_sub(ct));
                        // Klickwege parity: the HTML preview must size a
                        // cell from the SAME metrics the PDF's `Tf` uses.
                        let font_px = text_font_size_px(cell.bbox, cell.metrics.as_ref());
                        html.push_str(&format!(
                            "    <div class=\"cell\" style=\"left:{l}px;top:{top}px;\
                             width:{w}px;height:{h}px;font-size:{font_px}px;line-height:1\">{}</div>\n",
                            html_escape(&cell.text)
                        ));
                    }
                    html.push_str("  </div>\n");
                }
            }
        }

        html.push_str("</div>\n");
    }

    html.push_str("</body>\n</html>\n");
    html
}

/// Encode an 8-bit grey image as a PNG (pure Rust: flate2 zlib for IDAT +
/// [`crc32`] for the chunks). Greyscale colour type, 8-bit depth, filter 0
/// (None) per scanline. Assumes `data.len() >= w * h` (the caller,
/// [`grey_to_data_uri`], guarantees it).
fn grey_to_png(image: &GreyImage) -> Vec<u8> {
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let mut crc_input = Vec::with_capacity(4 + data.len());
        crc_input.extend_from_slice(kind);
        crc_input.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
    }

    let w = image.w as u32;
    let h = image.h as u32;
    let mut png: Vec<u8> = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(0); // colour type: greyscale
    ihdr.push(0); // compression: deflate
    ihdr.push(0); // filter method: adaptive
    ihdr.push(0); // interlace: none
    chunk(&mut png, b"IHDR", &ihdr);

    // Raw image data: each scanline prefixed with a filter byte (0 = None).
    let mut raw = Vec::with_capacity((image.w + 1) * image.h);
    for row in 0..image.h {
        raw.push(0);
        let start = row * image.w;
        raw.extend_from_slice(&image.data[start..start + image.w]);
    }
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&raw)
        .expect("zlib write (Vec writer is infallible)");
    let idat = enc
        .finish()
        .expect("zlib finish (Vec writer is infallible)");
    chunk(&mut png, b"IDAT", &idat);

    chunk(&mut png, b"IEND", &[]);
    png
}

/// Standard reflected CRC-32 (IEEE 802.3, polynomial `0xEDB88320`) — what PNG
/// chunk checksums use.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Standard Base64 (RFC 4648) encoder with `=` padding — for `data:` URIs.
fn base64(data: &[u8]) -> String {
    const AL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(*chunk.get(1).unwrap_or(&0));
        let b2 = u32::from(*chunk.get(2).unwrap_or(&0));
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(AL[(n >> 18 & 63) as usize] as char);
        out.push(AL[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            AL[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            AL[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Builder A — searchable layout (scan + invisible text)
// ---------------------------------------------------------------------------

/// Build the searchable-PDF layout: each page's scan becomes the full-page
/// background and each recognized word becomes one INVISIBLE [`TextBlock`].
/// [`crate::render_searchable_pdf`] is a thin wrapper that sets `dpi` on the
/// result and calls [`render_pdf`].
///
/// The returned doc's `dpi` is a placeholder (`72`) — the caller sets the real
/// resolution before rendering (`searchable_layout` has no dpi of its own; the
/// input `PageOcr` carries none).
#[must_use]
pub fn searchable_layout(pages: Vec<PageOcr>) -> LayoutDoc {
    let layout_pages = pages
        .into_iter()
        .map(|p| {
            let width = p.grey.w as u32;
            let height = p.grey.h as u32;
            let blocks = p
                .words
                .into_iter()
                .map(|w| {
                    Block::Text(TextBlock {
                        bbox: w.box_,
                        text: w.text,
                        visible: false,
                        metrics: None,
                        justification: None,
                    })
                })
                .collect();
            LayoutPage {
                width,
                height,
                background: Some(p.grey),
                blocks,
            }
        })
        .collect();
    LayoutDoc {
        pages: layout_pages,
        dpi: 72,
    }
}

// ---------------------------------------------------------------------------
// Builder B — doc.v1 reconstruction
// ---------------------------------------------------------------------------

/// Failures reconstructing a [`LayoutDoc`] from `doc.v1` JSON ([`doc_v1_layout`]).
#[derive(Debug)]
pub enum DocV1Error {
    /// The JSON failed to parse or did not match the `doc.v1` shape.
    Json(serde_json::Error),
    /// A page carries a `figure` region (which crops from a page raster) but no
    /// raster was supplied at that page index in `page_rasters`.
    MissingRaster {
        /// Zero-based page index that needed a raster.
        page: usize,
    },
}

impl std::fmt::Display for DocV1Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "parsing doc.v1 JSON: {e}"),
            Self::MissingRaster { page } => {
                write!(
                    f,
                    "page {page} has a figure region but no raster was supplied"
                )
            }
        }
    }
}

impl std::error::Error for DocV1Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
            Self::MissingRaster { .. } => None,
        }
    }
}

#[derive(Deserialize)]
struct JsonDoc {
    #[serde(default)]
    pages: Vec<JsonPage>,
}

#[derive(Deserialize, Default)]
struct JsonPage {
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
    #[serde(default)]
    regions: Vec<JsonRegion>,
    #[serde(default)]
    fields: Vec<JsonField>,
}

#[derive(Deserialize, Default)]
struct JsonRegion {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    bbox: [i64; 4],
    #[serde(default)]
    lines: Vec<JsonLine>,
    #[serde(default)]
    rows: usize,
    #[serde(default)]
    cols: usize,
    #[serde(default)]
    cells: Vec<JsonCell>,
}

#[derive(Deserialize, Default)]
struct JsonLine {
    #[serde(default)]
    bbox: Option<[i64; 4]>,
    #[serde(default)]
    words: Vec<JsonWord>,
    // The additive per-line metric keys (doc.v1; absent in legacy JSON).
    #[serde(default)]
    xheight: Option<f32>,
    #[serde(default)]
    ascrise: Option<f32>,
    #[serde(default)]
    descdrop: Option<f32>,
    #[serde(default)]
    baseline: Option<f32>,
    // The newer additive measured-ink-height key (doc.v1;
    // `tesseract_ocr::structured::attach_glyph_px`) — absent in legacy JSON
    // and in lines the measurement pass never reached. See
    // [`GLYPH_PX_TO_FONT_PX`] for how this is preferred over the band keys
    // above when both are present.
    #[serde(default)]
    glyph_px: Option<f32>,
}

#[derive(Deserialize, Default)]
struct JsonWord {
    #[serde(default)]
    text: String,
    #[serde(default)]
    bbox: [i64; 4],
}

#[derive(Deserialize)]
struct JsonCell {
    #[serde(default)]
    row: usize,
    #[serde(default)]
    col: usize,
    #[serde(default)]
    bbox: [i64; 4],
    #[serde(default)]
    text: String,
    #[serde(default)]
    header: bool,
    // The SAME additive measured-typography keys a line carries — a cell is
    // one line's words in one column, so it carries that line's metrics
    // verbatim. Absent in legacy doc.v1 (hence every `serde(default)`), in
    // which case the renderer falls back exactly as it always did.
    #[serde(default)]
    xheight: Option<f32>,
    #[serde(default)]
    ascrise: Option<f32>,
    #[serde(default)]
    descdrop: Option<f32>,
    #[serde(default)]
    baseline: Option<f32>,
    #[serde(default)]
    glyph_px: Option<f32>,
}

#[derive(Deserialize)]
struct JsonField {
    #[serde(default)]
    value: String,
    #[serde(default)]
    bbox: [i64; 4],
}

/// The ONE place `doc.v1`'s additive typography keys become a [`TextMetrics`].
///
/// Three-tier font-size preference, in order:
/// 1. the measured glyph ink-height `glyph_px`, scaled by
///    [`GLYPH_PX_TO_FONT_PX`], when present (the most reliable — the
///    statistical band fit is unstable row-to-row);
/// 2. else the band fit `xheight + ascrise - descdrop`, real Tesseract's own
///    `row_height` (`ltrresultiterator.cpp:168-172`);
/// 3. else `None` — the caller falls back to the [`TEXT_HEIGHT_TO_FONTSIZE`]
///    box-height heuristic and a box-bottom pen.
///
/// `xheight` AND `baseline` are both required for any metrics at all: a size
/// with no baseline cannot place the pen, and vice versa. `ascrise`/`descdrop`
/// default to `0.0` when absent (font_px degrades to the bare x-height, still
/// far better than band height).
///
/// Shared by the per-LINE and per-CELL paths deliberately: a cell is one
/// line's words in one column and must be sized identically, and duplicating
/// this ladder is how the two silently drift apart.
fn derive_text_metrics(
    xheight: Option<f32>,
    ascrise: Option<f32>,
    descdrop: Option<f32>,
    baseline: Option<f32>,
    glyph_px: Option<f32>,
) -> Option<TextMetrics> {
    match (xheight, baseline) {
        (Some(xh), Some(bl)) => {
            let band_font_px = xh + ascrise.unwrap_or(0.0) - descdrop.unwrap_or(0.0);
            let font_px = glyph_px
                .map(|g| g * GLYPH_PX_TO_FONT_PX)
                .unwrap_or(band_font_px);
            Some(TextMetrics {
                font_px,
                baseline_px: bl,
            })
        }
        _ => None,
    }
}

/// Clamp a signed `[l,t,r,b]` JSON bbox to a `u32` top-down tuple.
fn to_u32_bbox(b: [i64; 4]) -> (u32, u32, u32, u32) {
    let clamp = |v: i64| -> u32 { v.clamp(0, i64::from(u32::MAX)) as u32 };
    (clamp(b[0]), clamp(b[1]), clamp(b[2]), clamp(b[3]))
}

/// Union of a line's word bboxes (fallback when a line carries no bbox).
fn union_words(words: &[JsonWord]) -> (u32, u32, u32, u32) {
    let mut it = words.iter();
    let Some(first) = it.next() else {
        return (0, 0, 0, 0);
    };
    let mut acc = first.bbox;
    for w in it {
        acc[0] = acc[0].min(w.bbox[0]);
        acc[1] = acc[1].min(w.bbox[1]);
        acc[2] = acc[2].max(w.bbox[2]);
        acc[3] = acc[3].max(w.bbox[3]);
    }
    to_u32_bbox(acc)
}

/// Crop a grey raster to a top-down bbox (clamped to the raster bounds).
fn crop_grey(src: &GreyImage, bbox: (u32, u32, u32, u32)) -> GreyImage {
    if src.data.len() < src.w * src.h {
        return GreyImage {
            data: Vec::new(),
            w: 0,
            h: 0,
        };
    }
    let (l, t, r, b) = bbox;
    let l = (l as usize).min(src.w);
    let t = (t as usize).min(src.h);
    let r = (r as usize).min(src.w);
    let b = (b as usize).min(src.h);
    let cw = r.saturating_sub(l);
    let ch = b.saturating_sub(t);
    let mut data = Vec::with_capacity(cw * ch);
    for y in t..t + ch {
        let start = y * src.w + l;
        data.extend_from_slice(&src.data[start..start + cw]);
    }
    GreyImage { data, w: cw, h: ch }
}

// ---------------------------------------------------------------------------
// Per-region scale factor (header/footer) — see [`region_scale_factor`].
// ---------------------------------------------------------------------------

/// Minimum region lines carrying `baseline` before the region's OWN pitch is
/// trusted for rung A of [`region_scale_factor`] (`>= MIN_REGION_PITCH_LINES`
/// baselines, i.e. `>= 2` deltas) — the per-region analogue of
/// [`MIN_PITCH_SAMPLES`], applied within one region instead of page-wide.
const MIN_REGION_PITCH_LINES: usize = 3;

/// Below this ratio (region indicator / page indicator) a `header`/`footer`
/// region is left at page size. Symmetric in log space with
/// [`SCALE_DEAD_BAND_HI`] (`1 / 0.80 = 1.25`), i.e. +-one conventional
/// typographic step (1.125/1.2/1.25): a running head set AT body size but
/// measuring differently because its ink is cap-height-only (all-caps -> no
/// descender) lands inside the band and is correctly normalized rather than
/// nudged.
///
/// **Honesty pin, mandatory to restate at every use:** this and the three
/// constants below are POLICY PINS, not measurements — the corpus contains
/// no fixture with a real headline. Do not defend these numbers; re-measure
/// when a tightly-set page with a real headline lands.
const SCALE_DEAD_BAND_LO: f32 = 0.80;

/// Upper edge of the dead band — see [`SCALE_DEAD_BAND_LO`]. Policy pin, not
/// a measurement.
const SCALE_DEAD_BAND_HI: f32 = 1.25;

/// Below this ratio a "headline" measurement reads as a truncated or
/// mis-cropped line, not a genuine design choice — a running foot
/// legitimately sits at 0.7-0.85 of body, but not below half. Policy pin, not
/// a measurement.
const SCALE_CLAMP_LO: f32 = 0.5;

/// Above this ratio a measurement is reachable by cross-line contamination
/// alone (the corpus's own measured `Tf`/pitch maximum was 3.03, see
/// [`page_pitch_px`]'s doc) and carries no discriminating information; it is
/// also roughly the largest conventional display step over body on a text
/// page. Policy pin, not a measurement.
const SCALE_CLAMP_HI: f32 = 3.0;

/// Sort in place and return the middle element — the SAME
/// `sort_by(f32::total_cmp)` + `v[len/2]` idiom [`page_pitch_px`] and
/// [`page_font_px`] already use for their own medians, factored out here so
/// [`region_scale_factor`]'s two rungs share it rather than re-deriving it.
fn median_f32(v: &mut [f32]) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f32::total_cmp);
    Some(v[v.len() / 2])
}

/// Rung A of [`region_scale_factor`]: the median consecutive baseline delta
/// WITHIN one region, requiring [`MIN_REGION_PITCH_LINES`] lines carrying
/// `baseline` (`>= 2` deltas) before it is trusted.
///
/// *Contamination: none.* A multi-line headline's own leading scales with
/// its own size, so the ratio against the page's [`page_pitch_px`] is real —
/// see [`region_scale_factor`]'s doc for why the numerator and denominator
/// must be LIKE quantities.
fn region_pitch_px(lines: &[JsonLine]) -> Option<f32> {
    let mut bl: Vec<f32> = lines.iter().filter_map(|l| l.baseline).collect();
    if bl.len() < MIN_REGION_PITCH_LINES {
        return None;
    }
    bl.sort_by(f32::total_cmp);
    let mut deltas: Vec<f32> = bl
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0.0)
        .collect();
    median_f32(&mut deltas)
}

/// Rung B of [`region_scale_factor`]: the median over the region's lines of
/// [`derive_text_metrics`]'s `font_px` — deliberately reusing the file's
/// EXISTING three-tier ladder (glyph_px*4/3 -> band fit -> none) rather than
/// re-deriving band-then-glyph, per [`derive_text_metrics`]'s own doc
/// comment ("is how the two silently drift apart").
fn region_measured_font_px(lines: &[JsonLine]) -> Option<f32> {
    let mut vals: Vec<f32> = lines
        .iter()
        .filter_map(|l| {
            derive_text_metrics(l.xheight, l.ascrise, l.descdrop, l.baseline, l.glyph_px)
                .map(|m| m.font_px)
        })
        .collect();
    median_f32(&mut vals)
}

/// Rung B's page-wide denominator: [`region_measured_font_px`]'s SAME
/// median, taken over BODY regions only (`kind` not in `{"header",
/// "footer"}`).
///
/// Deliberately body-only rather than page-wide-including-furniture: unlike
/// [`page_pitch_px`], which absorbs a header's `<= 2`-of-dozens deltas by its
/// own page-wide median, this is new code with no many-sample median
/// protection on a short page, so the header/footer numerator must never be
/// allowed to denominate against itself.
fn page_measured_font_px_median(page: &JsonPage) -> Option<f32> {
    let mut vals: Vec<f32> = page
        .regions
        .iter()
        .filter(|r| r.kind != "header" && r.kind != "footer")
        .flat_map(|r| r.lines.iter())
        .filter_map(|l| {
            derive_text_metrics(l.xheight, l.ascrise, l.descdrop, l.baseline, l.glyph_px)
                .map(|m| m.font_px)
        })
        .collect();
    median_f32(&mut vals)
}

/// The page-level denominators [`region_scale_factor`] needs, built ONCE per
/// page alongside [`page_font_px`] (see [`doc_v1_layout`]).
struct PageScaleRefs {
    /// [`page_pitch_px`]'s page-wide median baseline delta — rung A's
    /// denominator.
    pitch: Option<f32>,
    /// [`page_measured_font_px_median`] — rung B's denominator (body regions
    /// only).
    measured: Option<f32>,
}

/// Per-region scale factor for a `header`/`footer` region whose own evidence
/// is a CLEAR outlier from the page size. Every other region — and every
/// region kind other than `header`/`footer` — returns exactly `1.0`, i.e.
/// today's behaviour, unchanged.
///
/// # The ratio must compare LIKE WITH LIKE (load-bearing)
///
/// Contamination is a MULTIPLICATIVE bias: [`page_pitch_px`]'s doc records
/// the measured `Tf`/pitch median 1.44, up to 1.82x apart for the same
/// string between columns. A ratio of UNLIKE quantities — a region's
/// band/ink measurement over [`page_font_px`] (a pitch/width SOLVE) — would
/// carry that ~1.44 systematic offset, so *every* region would read as a
/// 1.44x headline and no dead-band could absorb it. A ratio of LIKE
/// quantities cancels it: numerator and denominator inherit the same bias.
///
/// So the region's indicator rung SELECTS its own page-level denominator —
/// never cross rungs; if the matching denominator is `None`, the factor is
/// `1.0`.
///
/// # Why an isolated header's rung-B measurement is trustworthy
///
/// Contamination is ink from an ADJACENT line entering this line's generous
/// recognition band. A page-furniture line sits in white space by
/// construction — that is precisely why it classified as `header`/`footer`
/// in the first place — so an isolated line has no neighbour in its band and
/// nothing to import; the 1.82x body-vs-body spread does not bound the
/// header's own error.
///
/// And the residual bias runs the SAFE way: the rung-B denominator (body,
/// tight-set) IS contaminated upward, while the isolated header numerator is
/// not. The ratio is therefore UNDERSTATED — a genuine headline gets less
/// amplification than it deserves, never more. The failure mode is
/// under-correction.
fn region_scale_factor(region: &JsonRegion, refs: &PageScaleRefs) -> f32 {
    if region.kind != "header" && region.kind != "footer" {
        return 1.0;
    }
    let ratio = region_pitch_px(&region.lines)
        .zip(refs.pitch)
        .or_else(|| region_measured_font_px(&region.lines).zip(refs.measured))
        .and_then(|(num, den)| (den > 0.0).then_some(num / den));
    let Some(ratio) = ratio else {
        return 1.0;
    };
    if (SCALE_DEAD_BAND_LO..=SCALE_DEAD_BAND_HI).contains(&ratio) {
        return 1.0;
    }
    ratio.clamp(SCALE_CLAMP_LO, SCALE_CLAMP_HI)
}

/// Reconstruct a [`LayoutDoc`] from a `tesseract-rs/doc.v1` JSON document plus
/// the per-page grey rasters (indexed by page order). Region → block mapping:
///
/// - **text / paragraph / header / footer** (and any unknown region with
///   lines) → one visible [`TextBlock`] per line (line bbox + its words joined
///   by spaces).
/// - **figure** → a [`Block::Image`] cropping `page_rasters[page]` to the
///   region bbox.
/// - **table** → a ruled [`TableBlock`] carrying the region's cells.
/// - **fields** → appended visible [`TextBlock`]s at their bbox (the field
///   value text).
///
/// Pages get no background (the reconstruction stands on its own); the doc's
/// `dpi` defaults to `72` (1 px == 1 pt) — set it if you know the source dpi.
///
/// # Errors
///
/// [`DocV1Error::Json`] on malformed JSON; [`DocV1Error::MissingRaster`] if a
/// page has a `figure` region but `page_rasters` has no entry at that index.
pub fn doc_v1_layout(doc_json: &str, page_rasters: &[GreyImage]) -> Result<LayoutDoc, DocV1Error> {
    let parsed: JsonDoc = serde_json::from_str(doc_json).map_err(DocV1Error::Json)?;
    let mut pages = Vec::with_capacity(parsed.pages.len());

    for (pi, jp) in parsed.pages.into_iter().enumerate() {
        let mut blocks: Vec<Block> = Vec::new();
        // ONE size for the page, from the pitch — see [`page_font_px`]. The
        // per-line ink measurement is not trusted for SIZE; per-line
        // `baseline` is still trusted for PLACEMENT.
        let page_font_px = page_font_px(&jp);
        // The page-level denominators `region_scale_factor` needs, built
        // ONCE per page alongside `page_font_px` — see `PageScaleRefs`.
        let scale_refs = PageScaleRefs {
            pitch: page_pitch_px(&jp),
            measured: page_measured_font_px_median(&jp),
        };

        for region in &jp.regions {
            match region.kind.as_str() {
                "figure" => {
                    let raster = page_rasters
                        .get(pi)
                        .ok_or(DocV1Error::MissingRaster { page: pi })?;
                    let bbox = to_u32_bbox(region.bbox);
                    let image = crop_grey(raster, bbox);
                    blocks.push(Block::Image { bbox, image });
                }
                "table" => {
                    let cells = region
                        .cells
                        .iter()
                        .map(|c| TableCell {
                            row: c.row,
                            col: c.col,
                            bbox: to_u32_bbox(c.bbox),
                            text: c.text.clone(),
                            header: c.header,
                            metrics: derive_text_metrics(
                                c.xheight, c.ascrise, c.descdrop, c.baseline, c.glyph_px,
                            ),
                        })
                        .collect();
                    blocks.push(Block::Table(TableBlock {
                        bbox: to_u32_bbox(region.bbox),
                        rows: region.rows,
                        cols: region.cols,
                        cells,
                        rules: true,
                    }));
                }
                // text / paragraph / header / footer / unknown-with-lines.
                _ => {
                    // Classified ONCE per region: justification is a property
                    // of the block, not of a line. A line cannot tell you
                    // whether it was justified; only its siblings' right
                    // edges can.
                    let justification = classify_justification(
                        &region
                            .lines
                            .iter()
                            .map(|l| match l.bbox {
                                Some(b) => to_u32_bbox(b),
                                None => union_words(&l.words),
                            })
                            .collect::<Vec<_>>(),
                    );
                    // Also classified ONCE per region, same reasoning: a
                    // header/footer's deviation from page size is a property
                    // of the region's own evidence, not of any one line.
                    // `1.0` (today's exact behaviour) for every other kind.
                    let scale = region_scale_factor(region, &scale_refs);
                    for line in &region.lines {
                        let text = line
                            .words
                            .iter()
                            .map(|w| w.text.as_str())
                            .collect::<Vec<_>>()
                            .join(" ");
                        let bbox = match line.bbox {
                            Some(b) => to_u32_bbox(b),
                            None => union_words(&line.words),
                        };
                        // Metrics require xheight + baseline; ascrise/descdrop
                        // default to 0 when absent (font_px degrades to the
                        // bare x-height, still far better than band height).
                        // Three-tier font-size preference: (1) the measured
                        // glyph_px ink-height, scaled via GLYPH_PX_TO_FONT_PX,
                        // when present; (2) else the xheight+ascrise-descdrop
                        // band fit (previous behaviour, unchanged); (3) else
                        // (no xheight/baseline at all) `metrics` stays `None`
                        // and the bbox-height heuristic downstream
                        // (TEXT_HEIGHT_TO_FONTSIZE) applies, exactly as before.
                        let metrics = derive_text_metrics(
                            line.xheight,
                            line.ascrise,
                            line.descdrop,
                            line.baseline,
                            line.glyph_px,
                        )
                        // SIZE comes from the page pitch when we have one;
                        // PLACEMENT (`baseline_px`) is untouched, because the
                        // baselines measured correctly (pitch sd 0.00 across
                        // every column of the reference page).
                        .map(|m| match page_font_px {
                            Some(f) => TextMetrics {
                                font_px: f * scale,
                                ..m
                            },
                            None => m,
                        });
                        blocks.push(Block::Text(TextBlock {
                            bbox,
                            text,
                            visible: true,
                            metrics,
                            justification,
                        }));
                    }
                }
            }
        }

        // Fields → visible value text at their bbox (appended; optional).
        for field in &jp.fields {
            blocks.push(Block::Text(TextBlock {
                bbox: to_u32_bbox(field.bbox),
                text: field.value.clone(),
                visible: true,
                metrics: None,
                justification: None,
            }));
        }

        pages.push(LayoutPage {
            width: jp.width,
            height: jp.height,
            background: None,
            blocks,
        });
    }

    Ok(LayoutDoc { pages, dpi: 72 })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PageOcr, PlacedWord};
    use lopdf::content::Content as LopdfContent;
    use lopdf::Document;

    /// A synthetic grey buffer — only its dimensions/length matter here.
    fn synthetic_grey(w: usize, h: usize) -> GreyImage {
        GreyImage {
            data: (0..w * h).map(|i| (i % 256) as u8).collect(),
            w,
            h,
        }
    }

    fn page_content(pdf: &[u8]) -> LopdfContent {
        let doc = Document::load_mem(pdf).expect("load pdf");
        let pages = doc.get_pages();
        let &page_id = pages.get(&1).expect("page 1");
        let bytes = doc.get_page_content(page_id).expect("page content");
        LopdfContent::decode(&bytes).expect("decode content")
    }

    /// Read a numeric operand whether lopdf round-tripped it as Real or Integer.
    fn num(op: &Operation, i: usize) -> f64 {
        let obj = &op.operands[i];
        if let Ok(x) = obj.as_float() {
            f64::from(x)
        } else if let Ok(v) = obj.as_i64() {
            v as f64
        } else {
            panic!("operand {i} is not numeric");
        }
    }

    fn ops<'a>(content: &'a LopdfContent, name: &str) -> Vec<&'a Operation> {
        content
            .operations
            .iter()
            .filter(|o| o.operator == name)
            .collect()
    }

    // --- Builder A: searchable round-trips through the new renderer ---------

    #[test]
    fn searchable_layout_round_trips_through_render_pdf() {
        let page = PageOcr {
            grey: synthetic_grey(200, 40),
            words: vec![
                PlacedWord {
                    text: "Hello".to_string(),
                    box_: (10, 5, 60, 30),
                },
                PlacedWord {
                    text: "world".to_string(),
                    box_: (70, 5, 120, 30),
                },
            ],
        };

        let mut layout = searchable_layout(vec![page]);
        assert_eq!(layout.pages.len(), 1);
        assert!(layout.pages[0].background.is_some());
        assert_eq!(layout.pages[0].blocks.len(), 2);
        assert!(matches!(
            &layout.pages[0].blocks[0],
            Block::Text(t) if !t.visible && t.text == "Hello"
        ));

        layout.dpi = 300;
        let (pdf, report) = render_pdf(&layout).expect("render");
        assert_eq!(report.total_substitutions(), 0);

        // The image + invisible text layer is present and extractable in order.
        let extracted = crate::extract_text_layer(&pdf).expect("extract");
        let text = extracted[0].as_deref().expect("text layer");
        let hello = text.find("Hello").expect("Hello");
        let world = text.find("world").expect("world");
        assert!(hello < world, "reading order Hello<world, got {text:?}");

        // Invisible: a `3 Tr` precedes the first `Tj`.
        let content = page_content(&pdf);
        let tr = ops(&content, "Tr");
        assert!(tr.iter().any(|o| num(o, 0) == 3.0), "render mode 3 present");
    }

    #[test]
    fn render_pdf_reports_winansi_substitution_and_validates_image() {
        // Out-of-range CJK char is flagged; the whole batch is size-checked.
        let ok = PageOcr {
            grey: synthetic_grey(100, 40),
            words: vec![PlacedWord {
                text: "中".to_string(),
                box_: (5, 5, 20, 30),
            }],
        };
        let mut layout = searchable_layout(vec![ok]);
        layout.dpi = 300;
        let (_pdf, report) = render_pdf(&layout).expect("render");
        assert_eq!(report.pages, vec![1]);

        // A grey whose data length doesn't match w*h is a typed error at page 0.
        let bad = LayoutDoc {
            dpi: 300,
            pages: vec![LayoutPage {
                width: 10,
                height: 10,
                background: Some(GreyImage {
                    data: vec![0u8; 5],
                    w: 10,
                    h: 10,
                }),
                blocks: vec![],
            }],
        };
        let err = render_pdf(&bad).unwrap_err();
        assert!(matches!(
            err,
            SearchablePdfError::ImageSizeMismatch {
                page: 0,
                expected: 100,
                got: 5,
            }
        ));
    }

    // --- The Klickwege-parity test: same px bbox in BOTH projections --------

    #[test]
    fn klickwege_parity_same_bbox_in_pdf_and_html() {
        // dpi = 72 makes px→pt a 1:1 map, so PDF pt-coords equal px-coords
        // (modulo the bottom-up y flip). page height 100 → page_h_pt = 100.
        let doc = LayoutDoc {
            dpi: 72,
            pages: vec![LayoutPage {
                width: 200,
                height: 100,
                background: None,
                blocks: vec![
                    Block::Text(TextBlock {
                        bbox: (10, 20, 60, 40),
                        text: "Hi".to_string(),
                        visible: true,
                        metrics: None,
                        justification: None,
                    }),
                    Block::Image {
                        bbox: (80, 10, 180, 90),
                        image: synthetic_grey(100, 80),
                    },
                    Block::Table(TableBlock {
                        bbox: (5, 50, 190, 95),
                        rows: 1,
                        cols: 1,
                        cells: vec![TableCell {
                            row: 0,
                            col: 0,
                            bbox: (10, 55, 180, 90),
                            text: "X".to_string(),
                            header: true,
                            // Deliberately None: this fixture pins Klickwege
                            // bbox PLACEMENT, so it must keep exercising the
                            // metrics-less fallback path unchanged. The
                            // metrics-driven path has its own falsifier
                            // (`table_cell_uses_measured_metrics_not_the_box_fallback`).
                            metrics: None,
                        }],
                        rules: true,
                    }),
                ],
            }],
        };

        let (pdf, _report) = render_pdf(&doc).expect("render");
        let content = page_content(&pdf);
        let html = render_preview_html(&doc);

        // --- Text block: PDF Tm origin == px (left, page_h - bottom) ---------
        // bbox (10,20,60,40) → Tm (x=10, y=100-40=60).
        assert!(
            ops(&content, "Tm")
                .iter()
                .any(|o| (num(o, 4) - 10.0).abs() < 1e-3 && (num(o, 5) - 60.0).abs() < 1e-3),
            "text Tm at pt (10,60)"
        );
        assert!(
            html.contains("left:10px;top:20px;width:50px;height:20px"),
            "text css box"
        );

        // --- Image block: PDF cm rectangle == the px bbox --------------------
        // bbox (80,10,180,90) → cm (w=100,h=80,x=80,y=100-90=10). Only one cm.
        let cms = ops(&content, "cm");
        let img_cm = cms
            .iter()
            .find(|o| (num(o, 0) - 100.0).abs() < 1e-3)
            .expect("image cm");
        // KLICKWEGE PARITY: the SAME px bbox drives the PDF cm AND the HTML css
        // box — width/height/left identical, top identical after the y flip.
        assert_eq!(
            (num(img_cm, 4), num(img_cm, 0), num(img_cm, 3)),
            (80.0, 100.0, 80.0),
            "pdf cm x,w,h"
        );
        assert!(
            (100.0 - (num(img_cm, 5) + num(img_cm, 3)) - 10.0).abs() < 1e-6,
            "pdf top edge == html top (10)"
        );
        assert!(
            html.contains("left:80px;top:10px;width:100px;height:80px"),
            "image css box"
        );

        // --- Table block: PDF outer `re` rectangle == the px bbox ------------
        // bbox (5,50,190,95) → re (x=5,y=100-95=5,w=185,h=45).
        assert!(
            ops(&content, "re").iter().any(|o| {
                (num(o, 0) - 5.0).abs() < 1e-3
                    && (num(o, 1) - 5.0).abs() < 1e-3
                    && (num(o, 2) - 185.0).abs() < 1e-3
                    && (num(o, 3) - 45.0).abs() < 1e-3
            }),
            "table outer re rectangle"
        );
        assert!(
            html.contains("left:5px;top:50px;width:185px;height:45px"),
            "table css box"
        );
    }

    #[test]
    fn visible_text_uses_fill_render_mode() {
        let doc = LayoutDoc {
            dpi: 72,
            pages: vec![LayoutPage {
                width: 100,
                height: 40,
                background: None,
                blocks: vec![Block::Text(TextBlock {
                    bbox: (5, 5, 60, 30),
                    text: "Visible".to_string(),
                    visible: true,
                    metrics: None,
                    justification: None,
                })],
            }],
        };
        let (pdf, _r) = render_pdf(&doc).expect("render");
        let content = page_content(&pdf);
        // A `0 Tr` (fill) precedes the Tj; no invisible `3 Tr` group exists.
        let tr = ops(&content, "Tr");
        assert!(tr.iter().any(|o| num(o, 0) == 0.0), "fill mode 0 present");
        assert!(!tr.iter().any(|o| num(o, 0) == 3.0), "no invisible group");
    }

    /// Regression for the reported overlap bug: two consecutive lines whose
    /// bbox height (30) is double their baseline-to-baseline pitch (15) —
    /// exactly the shape measured on a real multi-paragraph page, where the
    /// naive `fontsize_pt = box_h_pt` chose a font twice the real line pitch
    /// and every line's glyphs bled into both neighbours. After the fix, the
    /// rendered font size must not exceed the real pitch between baselines,
    /// in EITHER projection (PDF `Tf` and the HTML preview's `font-size`).
    #[test]
    fn visible_text_font_size_does_not_exceed_line_pitch() {
        let doc = LayoutDoc {
            dpi: 72,
            pages: vec![LayoutPage {
                width: 100,
                height: 100,
                background: None,
                blocks: vec![
                    Block::Text(TextBlock {
                        bbox: (0, 0, 100, 30),
                        text: "Line one".to_string(),
                        visible: true,
                        metrics: None,
                        justification: None,
                    }),
                    Block::Text(TextBlock {
                        bbox: (0, 15, 100, 45),
                        text: "Line two".to_string(),
                        visible: true,
                        metrics: None,
                        justification: None,
                    }),
                ],
            }],
        };
        // Baseline-to-baseline pitch = the boxes' bottom-to-bottom delta (45-30=15),
        // matching the box-height (30) being double the real pitch — the bug shape.
        let pitch_pt = 15.0;

        let (pdf, _r) = render_pdf(&doc).expect("render");
        let content = page_content(&pdf);
        for tf in ops(&content, "Tf") {
            let size = num(tf, 1);
            assert!(
                size <= pitch_pt + 1e-6,
                "PDF font size {size} must not exceed the {pitch_pt}pt line pitch"
            );
        }

        let html = render_preview_html(&doc);
        assert!(
            html.contains("font-size:15px"),
            "HTML font-size must shrink with the box height (30px * 0.5 = 15px), got: {html}"
        );
    }

    // --- Builder B: doc.v1 reconstruction -----------------------------------

    const DOC_V1: &str = r#"{
      "schema": "tesseract-rs/doc.v1",
      "pages": [{
        "page": 1, "width": 100, "height": 100,
        "quality": {"mean_conf": 95.0, "low_confidence": false},
        "regions": [
          {"type": "text", "bbox": [5,5,80,20], "lines": [
            {"bbox": [5,5,80,20], "words": [
              {"text": "Hello", "bbox": [5,5,40,20], "conf": 96.0, "leading_space": false},
              {"text": "World", "bbox": [45,5,80,20], "conf": 95.0, "leading_space": true}
            ]}
          ]},
          {"type": "figure", "bbox": [10,30,30,50], "lines": []},
          {"type": "table", "bbox": [5,60,90,95], "rows": 2, "cols": 2, "cells": [
            {"row":0,"col":0,"bbox":[5,60,45,77],"text":"A","header":true},
            {"row":0,"col":1,"bbox":[45,60,90,77],"text":"B","header":true},
            {"row":1,"col":0,"bbox":[5,78,45,95],"text":"1","header":false},
            {"row":1,"col":1,"bbox":[45,78,90,95],"text":"2","header":false}
          ]}
        ],
        "fields": []
      }]
    }"#;

    #[test]
    fn doc_v1_layout_maps_regions_to_blocks() {
        let raster = synthetic_grey(100, 100);
        let doc = doc_v1_layout(DOC_V1, &[raster]).expect("parse doc.v1");

        assert_eq!(doc.pages.len(), 1);
        let page = &doc.pages[0];
        assert_eq!((page.width, page.height), (100, 100));
        assert!(page.background.is_none());
        assert_eq!(page.blocks.len(), 3, "text + figure + table");

        match &page.blocks[0] {
            Block::Text(t) => {
                assert_eq!(t.bbox, (5, 5, 80, 20));
                assert_eq!(t.text, "Hello World");
                assert!(t.visible);
            }
            other => panic!("block 0 should be Text, got {other:?}"),
        }
        match &page.blocks[1] {
            Block::Image { bbox, image } => {
                assert_eq!(*bbox, (10, 30, 30, 50));
                assert_eq!((image.w, image.h), (20, 20));
                assert_eq!(image.data.len(), 400);
            }
            other => panic!("block 1 should be Image, got {other:?}"),
        }
        match &page.blocks[2] {
            Block::Table(t) => {
                assert_eq!(t.bbox, (5, 60, 90, 95));
                assert_eq!((t.rows, t.cols), (2, 2));
                assert!(t.rules);
                assert_eq!(t.cells.len(), 4);
                assert_eq!(t.cells[0].text, "A");
                assert!(t.cells[0].header);
                assert_eq!(t.cells[0].bbox, (5, 60, 45, 77));
                assert!(!t.cells[3].header);
            }
            other => panic!("block 2 should be Table, got {other:?}"),
        }

        // The reconstruction renders to a valid PDF (no background needed).
        let (pdf, _r) = render_pdf(&doc).expect("render reconstruction");
        assert!(pdf.starts_with(b"%PDF-"));
    }

    #[test]
    fn doc_v1_layout_figure_without_raster_is_a_typed_error() {
        // Same JSON, but no rasters supplied → the figure region can't crop.
        let err = doc_v1_layout(DOC_V1, &[]).unwrap_err();
        assert!(matches!(err, DocV1Error::MissingRaster { page: 0 }));
    }

    // --- HTML preview basics + PNG encoder ----------------------------------

    #[test]
    fn preview_html_has_background_image_and_word_divs() {
        let page = PageOcr {
            grey: synthetic_grey(120, 40),
            words: vec![PlacedWord {
                text: "Word".to_string(),
                box_: (10, 10, 60, 30),
            }],
        };
        let doc = searchable_layout(vec![page]);
        let html = render_preview_html(&doc);
        assert!(
            html.contains("data:image/png;base64,"),
            "background PNG uri"
        );
        assert!(
            html.contains("left:10px;top:10px;width:50px;height:20px"),
            "word div at its bbox"
        );
        assert!(html.contains(">Word</div>"), "word text rendered");
    }

    #[test]
    fn grey_to_png_emits_a_valid_png_signature_and_ihdr() {
        let png = grey_to_png(&synthetic_grey(4, 3));
        assert_eq!(
            &png[0..8],
            &[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']
        );
        // First chunk is IHDR with width=4, height=3.
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..20], &4u32.to_be_bytes());
        assert_eq!(&png[20..24], &3u32.to_be_bytes());
    }

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
    }

    // --- Measured line metrics drive both projections ------------------------

    /// With [`TextMetrics`] present, the PDF projection's `Tf` is the measured
    /// body height (`px == pt` at dpi 72) and the pen `Tm` sits on the
    /// MEASURED baseline — not the bbox-derived heuristic — and the HTML
    /// projection's `font-size` uses the same measured value (Klickwege
    /// parity applied to text size).
    #[test]
    fn measured_metrics_drive_font_size_and_baseline() {
        let doc = LayoutDoc {
            dpi: 72,
            pages: vec![LayoutPage {
                width: 200,
                height: 100,
                background: None,
                blocks: vec![Block::Text(TextBlock {
                    // Band bbox (30..70): the heuristic would give Tf = 20*0.5
                    // = 20 too here, so pin baseline to distinguish, and use a
                    // font_px (17) that differs from every bbox derivation.
                    bbox: (10, 30, 150, 70),
                    text: "Hello".to_string(),
                    visible: true,
                    justification: None,
                    metrics: Some(TextMetrics {
                        font_px: 17.0,
                        baseline_px: 60.0, // top-down → pen y = 100 - 60 = 40
                    }),
                })],
            }],
        };
        let (pdf, _r) = render_pdf(&doc).expect("render");
        let content = page_content(&pdf);
        assert!(
            ops(&content, "Tf")
                .iter()
                .any(|o| (num(o, 1) - 17.0).abs() < 1e-3),
            "Tf must be the measured font_px (17), not a bbox heuristic"
        );
        assert!(
            ops(&content, "Tm")
                .iter()
                .any(|o| (num(o, 5) - 40.0).abs() < 1e-3),
            "pen y must sit on the measured baseline (100-60=40)"
        );
        let html = render_preview_html(&doc);
        assert!(
            html.contains("font-size:17px"),
            "HTML font-size must use the same measured value: {html}"
        );
    }

    /// A TABLE CELL must be sized and placed from its own measured metrics,
    /// exactly like an ordinary text line — the sibling of
    /// [`measured_metrics_drive_font_size_and_baseline`] for
    /// [`Block::Table`], and the falsifier for the defect where BOTH render
    /// call sites hardcoded `None` for every cell in every document.
    ///
    /// Numbers mirror the real measured line that exposed this: a 61 px-tall
    /// box whose own metrics say `font_px = 48` and whose real baseline sits
    /// 19.1 px above the box bottom. Two-sided on BOTH axes, because the whole
    /// bug is that the fallback silently substitutes for each:
    ///
    /// - `Tf` must be 48 (measured) and must NOT be 30.5 (`61 * 0.5`)
    /// - pen y must be `700 - 554.9` (measured baseline) and must NOT be
    ///   `700 - 574` (the box BOTTOM)
    ///
    /// The HTML preview must carry the same size (Klickwege parity), so the
    /// two surfaces cannot drift.
    #[test]
    fn table_cell_uses_measured_metrics_not_the_box_fallback() {
        let doc = LayoutDoc {
            dpi: 72,
            pages: vec![LayoutPage {
                width: 1400,
                height: 700,
                background: None,
                blocks: vec![Block::Table(TableBlock {
                    bbox: (368, 513, 1110, 600),
                    rows: 1,
                    cols: 1,
                    cells: vec![TableCell {
                        row: 0,
                        col: 0,
                        bbox: (368, 513, 1110, 574), // height 61
                        text: "ice was beginning".to_string(),
                        header: false,
                        metrics: Some(TextMetrics {
                            font_px: 48.0,
                            baseline_px: 554.9,
                        }),
                    }],
                    rules: true,
                })],
            }],
        };
        let (pdf, _r) = render_pdf(&doc).expect("render");
        let content = page_content(&pdf);

        assert!(
            ops(&content, "Tf")
                .iter()
                .any(|o| (num(o, 1) - 48.0).abs() < 1e-3),
            "Tf must be the cell's measured font_px (48): {content:?}"
        );
        assert!(
            !ops(&content, "Tf")
                .iter()
                .any(|o| (num(o, 1) - 30.5).abs() < 1e-3),
            "Tf must NOT be the box-height fallback (61 * 0.5 = 30.5): {content:?}"
        );
        assert!(
            ops(&content, "Tm")
                .iter()
                .any(|o| (num(o, 5) - (700.0 - 554.9)).abs() < 1e-3),
            "pen y must sit on the measured baseline (700-554.9): {content:?}"
        );
        assert!(
            !ops(&content, "Tm")
                .iter()
                .any(|o| (num(o, 5) - (700.0 - 574.0)).abs() < 1e-3),
            "pen y must NOT be anchored at the box bottom (700-574): {content:?}"
        );

        let html = render_preview_html(&doc);
        assert!(
            html.contains("font-size:48px"),
            "HTML cell font-size must use the same measured value: {html}"
        );
    }

    /// `doc_v1_layout` must thread the additive per-CELL metric keys through
    /// to [`TableCell::metrics`] — the JSON half of the fix above. Two-sided:
    /// a cell carrying the keys gets metrics, a legacy cell without them stays
    /// `None` (and so keeps the old fallback), proving the parse is reading
    /// the keys rather than fabricating a value for every cell.
    #[test]
    fn doc_v1_layout_parses_cell_metrics() {
        let json = r#"{
          "schema": "tesseract-rs/doc.v1",
          "pages": [{
            "page": 1, "width": 300, "height": 200,
            "regions": [
              {"type": "table", "bbox": [5,5,200,60], "rows": 1, "cols": 2, "lines": [],
               "cells": [
                 {"row":0,"col":0,"bbox":[5,5,90,40],"text":"A","header":false,
                  "xheight":10.0,"ascrise":4.0,"descdrop":-3.0,"baseline":30.0},
                 {"row":0,"col":1,"bbox":[100,5,200,40],"text":"B","header":false}
               ]}
            ],
            "fields": []
          }]
        }"#;
        let doc = doc_v1_layout(json, &[]).expect("parse");
        let Block::Table(t) = &doc.pages[0].blocks[0] else {
            panic!("expected a table block, got {:?}", doc.pages[0].blocks[0]);
        };
        let m = t.cells[0].metrics.expect("cell 0 carries metric keys");
        // font_px = xheight + ascrise - descdrop = 10 + 4 - (-3) = 17
        assert!(
            (m.font_px - 17.0).abs() < 1e-4,
            "font_px must be the band fit (17), got {}",
            m.font_px
        );
        assert!((m.baseline_px - 30.0).abs() < 1e-4);
        assert!(
            t.cells[1].metrics.is_none(),
            "a cell without the keys must stay None, not inherit a value"
        );
    }

    /// `doc_v1_layout` parses the additive per-line metric keys into
    /// [`TextMetrics`] (`font_px = xheight + ascrise - descdrop`), and leaves
    /// `metrics: None` for legacy lines without them.
    #[test]
    fn doc_v1_layout_parses_line_metrics() {
        let json = r#"{
          "schema": "tesseract-rs/doc.v1",
          "pages": [{
            "page": 1, "width": 100, "height": 100,
            "regions": [
              {"type": "text", "bbox": [5,5,80,20], "lines": [
                {"bbox": [5,5,80,20],
                 "xheight": 10.0, "ascrise": 4.0, "descdrop": -3.0, "baseline": 17.5,
                 "words": [{"text": "Hi", "bbox": [5,5,40,20]}]},
                {"bbox": [5,25,80,40], "words": [{"text": "legacy", "bbox": [5,25,40,40]}]}
              ]}
            ],
            "fields": []
          }]
        }"#;
        let doc = doc_v1_layout(json, &[]).expect("parse");
        let texts: Vec<&TextBlock> = doc.pages[0]
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text(t) => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(texts.len(), 2);
        assert_eq!(
            texts[0].metrics,
            Some(TextMetrics {
                font_px: 17.0, // 10 + 4 - (-3)
                baseline_px: 17.5,
            })
        );
        assert_eq!(texts[1].metrics, None, "legacy line stays metrics-less");
    }

    /// When a line carries BOTH the measured `glyph_px` key and the older
    /// `xheight`/`ascrise`/`descdrop` band keys, [`doc_v1_layout`] must
    /// prefer `glyph_px * GLYPH_PX_TO_FONT_PX`, NOT the band-derived value
    /// — a real falsifier, not a coincidence: the fixture picks numbers
    /// (`glyph_px: 9.0` → `12.0`; band fit → `17.0`) that differ enough to
    /// prove the code actually reads `glyph_px`, not merely tolerates its
    /// presence. `baseline_px` is unaffected either way (always sourced
    /// from `baseline`).
    #[test]
    fn doc_v1_layout_prefers_glyph_px_over_the_band_derived_font_size() {
        let json = r#"{
          "schema": "tesseract-rs/doc.v1",
          "pages": [{
            "page": 1, "width": 100, "height": 100,
            "regions": [
              {"type": "text", "bbox": [5,5,80,20], "lines": [
                {"bbox": [5,5,80,20],
                 "xheight": 10.0, "ascrise": 4.0, "descdrop": -3.0,
                 "baseline": 17.5, "glyph_px": 9.0,
                 "words": [{"text": "Hi", "bbox": [5,5,40,20]}]}
              ]}
            ],
            "fields": []
          }]
        }"#;
        let doc = doc_v1_layout(json, &[]).expect("parse");
        let texts: Vec<&TextBlock> = doc.pages[0]
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text(t) => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(texts.len(), 1);
        let m = texts[0].metrics.expect("metrics present");
        assert!(
            (m.font_px - 9.0 * GLYPH_PX_TO_FONT_PX).abs() < 1e-4,
            "font_px must come from glyph_px * GLYPH_PX_TO_FONT_PX (~{}), got {}",
            9.0 * GLYPH_PX_TO_FONT_PX,
            m.font_px
        );
        assert!(
            (m.font_px - 17.0).abs() > 1.0,
            "the glyph_px-derived size must genuinely differ from the \
             band-derived 17.0, or this test cannot distinguish preference \
             from coincidence"
        );
        assert_eq!(m.baseline_px, 17.5, "baseline always comes from `baseline`");
    }
}

#[cfg(test)]
mod normalization_tests {
    use super::*;

    fn line(l: u32, r: u32) -> (u32, u32, u32, u32) {
        (l, 0, r, 10)
    }

    /// Justified: every line but the last ends at the same margin. The last
    /// line is SHORT (300 vs 1000) and must not defeat the classification —
    /// that is the classic false negative.
    #[test]
    fn blocksatz_is_detected_despite_a_short_last_line() {
        let ls = [
            line(0, 1000),
            line(0, 998),
            line(0, 1001),
            line(0, 999),
            line(0, 300),
        ];
        assert_eq!(
            classify_justification(&ls),
            Some(Justification::Blocksatz),
            "right edges within 0.3% must read as justified"
        );
    }

    /// The silence twin, on REAL measured data: the reference page's own
    /// paragraph is ragged, and must classify as such. Right edges here are
    /// the actual per-line extents of the corpus paragraph shape.
    #[test]
    fn flattersatz_is_not_mistaken_for_blocksatz() {
        let ls = [
            line(0, 980),
            line(0, 870),
            line(0, 760),
            line(0, 910),
            line(0, 300),
        ];
        assert_eq!(
            classify_justification(&ls),
            Some(Justification::Flattersatz),
            "22% ragged edges must NOT read as justified"
        );
    }

    /// Under three lines there is not enough evidence, and guessing would
    /// make every heading and every stray line "justified".
    #[test]
    fn justification_declines_on_too_few_lines() {
        assert_eq!(
            classify_justification(&[line(0, 1000), line(0, 1000)]),
            None
        );
        assert_eq!(classify_justification(&[]), None);
    }

    /// THE headline invariant. Painted text must never be horizontally
    /// distorted: `Tz` is exactly 100 for both painted fits. Measured on the
    /// real structured render before this change, `Tz` ranged 32.4%..850%
    /// line to line.
    #[test]
    fn painted_runs_never_distort_glyphs() {
        for fit in [RunFit::Natural, RunFit::JustifyToBox] {
            let mut ops = Vec::new();
            emit_text_run(
                &mut ops,
                (0, 0, 400, 20),
                "the quick brown fox jumps",
                72,
                800.0,
                None,
                fit,
            );
            let tz: Vec<f32> = ops
                .iter()
                .filter(|o| o.operator == "Tz")
                .map(|o| o.operands[0].as_float().unwrap())
                .collect();
            assert_eq!(tz, vec![100.0], "{fit:?} must leave glyph width alone");
        }
        // …and the invisible layer still DOES stretch, or search/selection
        // would stop landing on the scanned ink.
        let mut ops = Vec::new();
        emit_text_run(
            &mut ops,
            (0, 0, 400, 20),
            "the quick brown fox jumps",
            72,
            800.0,
            None,
            RunFit::StretchToBox,
        );
        let tz = ops
            .iter()
            .find(|o| o.operator == "Tz")
            .and_then(|o| o.operands[0].as_float().ok())
            .unwrap();
        assert!(
            (tz - 100.0).abs() > 1.0,
            "the invisible layer must still fit the ink box exactly (got Tz={tz})"
        );
    }

    /// Justification happens through the SPACES: `Tw` is non-zero and of the
    /// right sign, and it is reset to 0 on a natural run so a previous line's
    /// spacing cannot leak into it via the persistent PDF text state.
    ///
    /// The box width is derived from the run's OWN natural width (recovered
    /// from the stretch fit's `Tz`) rather than hardcoded, so the test states
    /// "10% slack" rather than a number that silently becomes a 40% stretch
    /// when font metrics change — which is exactly how the first version of
    /// this test tripped [`MAX_JUSTIFY_SLACK_FRAC`] and failed for the wrong
    /// reason.
    #[test]
    fn justification_uses_word_spacing_and_always_resets_it() {
        const TEXT: &str = "a b c d e";
        let op = |fit, w: u32, name: &str| {
            let mut ops = Vec::new();
            emit_text_run(&mut ops, (0, 0, w, 20), TEXT, 72, 800.0, None, fit);
            ops.iter()
                .find(|o| o.operator == name)
                .and_then(|o| o.operands[0].as_float().ok())
                .unwrap_or_else(|| panic!("{name} must always be emitted"))
        };
        // Recover the run's natural width: Tz = 100 * box_w / natural_w.
        let probe_w = 200.0_f64;
        let natural_w = 100.0 * probe_w / f64::from(op(RunFit::StretchToBox, 200, "Tz"));
        // 10% slack — comfortably inside the guard, so this measures
        // justification and not the guard.
        let just_w = (natural_w * 1.10).round() as u32;
        let tw = op(RunFit::JustifyToBox, just_w, "Tw");
        assert!(
            tw > 0.0,
            "10% slack must be absorbed by word spacing (got Tw={tw})"
        );
        // 4 spaces share the slack; each gets roughly a quarter of it.
        let expected = (f64::from(just_w) - natural_w) / 4.0;
        assert!(
            (f64::from(tw) - expected).abs() < 0.5,
            "Tw={tw} should be ~{expected:.2} (slack / 4 spaces)"
        );
        // Natural and stretch runs must ACTIVELY zero it, or the previous
        // line's spacing leaks in through the persistent text state.
        assert_eq!(op(RunFit::Natural, just_w, "Tw"), 0.0);
        assert_eq!(op(RunFit::StretchToBox, just_w, "Tw"), 0.0);
    }

    /// The guard's can-it-fire twin: past [`MAX_JUSTIFY_SLACK_FRAC`] the line
    /// is not a justified line (a heading, a mis-segmented run, one long
    /// token) and is left NATURAL rather than pulled apart. Without this the
    /// justify path would happily stretch a two-word heading across a column.
    #[test]
    fn justification_declines_an_absurd_stretch() {
        let mut ops = Vec::new();
        emit_text_run(
            &mut ops,
            (0, 0, 4000, 20),
            "a b c d e",
            72,
            800.0,
            None,
            RunFit::JustifyToBox,
        );
        let tw = ops
            .iter()
            .find(|o| o.operator == "Tw")
            .and_then(|o| o.operands[0].as_float().ok())
            .unwrap();
        assert_eq!(tw, 0.0, "a 40x box must not be justified into, got Tw={tw}");
    }

    /// The page pitch is the median of consecutive baselines, so a DROPPED
    /// line (which OCR does exactly where a scan degrades) cannot inflate it
    /// — the defect that sinks the geometric span/(n-1) form, measured
    /// inflating 19.08 -> 25.32 across the reference page's degradation
    /// ladder.
    #[test]
    fn page_pitch_is_immune_to_a_dropped_line() {
        let mk = |bl: &[f32]| JsonPage {
            regions: vec![JsonRegion {
                lines: bl
                    .iter()
                    .map(|b| JsonLine {
                        baseline: Some(*b),
                        ..JsonLine::default()
                    })
                    .collect(),
                ..JsonRegion::default()
            }],
            ..JsonPage::default()
        };
        let clean = page_pitch_px(&mk(&[0.0, 18.0, 36.0, 54.0, 72.0, 90.0])).unwrap();
        // Same page with the 3rd line lost: one gap becomes 36 instead of 18.
        let dropped = page_pitch_px(&mk(&[0.0, 18.0, 54.0, 72.0, 90.0])).unwrap();
        assert!(
            (clean - 18.0).abs() < 0.01,
            "clean pitch = 18 (got {clean})"
        );
        assert!(
            (dropped - 18.0).abs() < 0.01,
            "a dropped line must not move the median (got {dropped})"
        );
        // The geometric form WOULD have been fooled — pin the contrast so the
        // reason for choosing the median is not lost.
        let geometric: f32 = 90.0 / 4.0;
        assert!(
            (geometric - 22.5).abs() < 0.01 && geometric > dropped,
            "span/(n-1) inflates to 22.5 where the median holds 18"
        );
    }

    /// The two-axis fit takes the BINDING constraint. A page whose lines are
    /// wide for their measure must shrink below the leading bound, or it
    /// spills past the column — "does 12.4 become too wide, or too high?"
    /// made mechanical.
    #[test]
    fn page_font_takes_whichever_axis_binds() {
        // Same baselines (pitch 18 -> height bound 14.4) in both cases; only
        // the ink width per line differs.
        let page = |box_w: u32| JsonPage {
            regions: vec![JsonRegion {
                lines: (0..6)
                    .map(|i| JsonLine {
                        baseline: Some(i as f32 * 18.0),
                        bbox: Some([0, 0, i64::from(box_w), 16]),
                        words: vec![JsonWord {
                            text: "the quick brown fox jumps over".into(),
                            bbox: [0, 0, i64::from(box_w), 16],
                        }],
                        ..JsonLine::default()
                    })
                    .collect(),
                ..JsonRegion::default()
            }],
            ..JsonPage::default()
        };
        let ceiling = 18.0 * MAX_FONT_TO_PITCH;
        // ROOMY: the text is narrow for its box, so width does not bind and
        // the hard no-overlap ceiling stands. (Width-first, per the operator
        // acceptance rule: uneven gutters hurt more than tight leading, so
        // the conservative 0.80 relation applies only when NO width evidence
        // exists at all.)
        let roomy = page_font_px(&page(4000)).unwrap();
        assert!(
            (roomy - ceiling).abs() < 0.01,
            "with room to spare the no-overlap ceiling must stand (got {roomy})"
        );
        // TIGHT: the same text in a much narrower box CANNOT be set at the
        // leading bound without overflowing, so the fit shrinks it.
        let tight = page_font_px(&page(40)).unwrap();
        assert!(
            tight < ceiling,
            "a too-narrow measure must bind BELOW the ceiling \
             (got {tight}, ceiling {ceiling})"
        );
        // …and it is the width solve, not an arbitrary shrink.
        assert!(tight > 0.0 && tight < roomy);
    }

    /// Too few baselines to establish a pitch -> `None`, and the caller keeps
    /// the per-line path rather than sizing the page off one guess.
    #[test]
    fn page_pitch_declines_without_enough_samples() {
        let p = JsonPage {
            regions: vec![JsonRegion {
                lines: vec![JsonLine {
                    baseline: Some(10.0),
                    ..JsonLine::default()
                }],
                ..JsonRegion::default()
            }],
            ..JsonPage::default()
        };
        assert_eq!(page_pitch_px(&p), None);
        assert_eq!(page_font_px(&p), None);
    }

    // -------------------------------------------------------------------
    // Per-region scale factor (header/footer) — #49.
    //
    // ORCHESTRATOR EXECUTES: the six falsifiers below are all verified by
    // construction (each assertion is unreachable unless the specific edit
    // it names is present) but per this repo's iron rule ("the worker must
    // run the disable-the-fix edit for every test and report the observed
    // failure message — not assert that it would fail") the actual disable
    // -> re-run -> observe-the-failure step needs a live `cargo test`,
    // which this Sonnet worker is instructed NOT to run (the orchestrator
    // compiles/tests centrally, once, in the shared `target/`). The
    // orchestrator MUST run each edit below against
    // `cargo test -p tesseract-ocr-pdf --release normalization_tests` and
    // record the real failure message before considering #49 landed:
    //
    // | test | disable edit | expected failure |
    // |---|---|---|
    // | `header_2x_body_scales_the_header_block` | drop `* scale` at the `TextMetrics { font_px: f * scale, ..m }` override | header block's font_px assertion (~2x page size) fails — it renders at exactly page size instead |
    // | `text_only_page_stays_at_page_size_exactly` | force `region_scale_factor` to always return `2.0` | `assert_eq!` on every body block's `font_px` fails (font_px is now 2x `page_font_px`) |
    // | `header_within_dead_band_is_left_at_page_size` | remove the dead-band branch (`if (SCALE_DEAD_BAND_LO..=SCALE_DEAD_BAND_HI).contains(&ratio) { return 1.0; }`) | `assert_eq!` fails — the header would render at `page_font_px * 1.1` instead of exactly `page_font_px` |
    // | `header_10x_body_clamps_to_the_ceiling` | remove the `.clamp(SCALE_CLAMP_LO, SCALE_CLAMP_HI)` call | header block's font_px assertion (`page_font_px * 3.0`) fails — it renders at `page_font_px * 10.0` instead |
    // | `header_deviation_with_no_body_metrics_stays_at_page_size` | change the rung-B fallback to denominate against `page_font_px(&jp)` instead of `refs.measured` | `assert_eq!` fails — the header renders at `page_font_px * (20.0 / page_font_px)` instead of exactly `page_font_px` |
    // | `text_kind_never_deviates_even_at_2x` | delete the `if region.kind != "header" && region.kind != "footer" { return 1.0; }` kind gate | `assert_eq!` fails for BOTH the `"text"` and the `""`-kind variant — each renders at `page_font_px * 2.0` instead of exactly `page_font_px` |

    /// One `doc.v1` line JSON fragment: `baseline` always present; the band
    /// keys (`xheight`/`ascrise`/`descdrop`) only when `Some` — omitting
    /// them is deliberately possible here (test 5's body fixture needs it),
    /// mirroring test 1's own trap: a metrics-less line makes
    /// `derive_text_metrics` return `None`, so a fixture that is SUPPOSED to
    /// carry metrics must say so explicitly, never by accident.
    fn norm_line_json(baseline: f32, band: Option<(f32, f32, f32)>) -> String {
        match band {
            Some((xh, ar, dr)) => format!(
                r#"{{"baseline":{baseline},"xheight":{xh},"ascrise":{ar},"descdrop":{dr},"words":[]}}"#
            ),
            None => format!(r#"{{"baseline":{baseline},"words":[]}}"#),
        }
    }

    /// One `"text"` body region: 4 lines, 18px pitch, constant band font_px
    /// 10.0 (`xheight 10, ascrise 2, descdrop 2`) — repeated across the
    /// tests below so [`page_pitch_px`] and [`page_measured_font_px_median`]
    /// both see clean, deterministic evidence: pitch median exactly 18.0
    /// (zero spread) and body measured font_px exactly 10.0 (zero spread),
    /// both BY CONSTRUCTION rather than by median-over-noise luck.
    fn norm_body_region_json(base_baseline: f32) -> String {
        let lines: Vec<String> = (0..4)
            .map(|i| norm_line_json(base_baseline + i as f32 * 18.0, Some((10.0, 2.0, 2.0))))
            .collect();
        format!(r#"{{"type":"text","lines":[{}]}}"#, lines.join(","))
    }

    /// The same shape as [`norm_body_region_json`] but WITHOUT any band
    /// keys — every line carries `baseline` only, so
    /// [`page_measured_font_px_median`] sees zero evidence (test 5's "no
    /// body metrics" fixture) while [`page_pitch_px`] still sees a full
    /// pitch (it only needs `baseline`).
    fn norm_body_region_json_no_metrics(base_baseline: f32) -> String {
        let lines: Vec<String> = (0..4)
            .map(|i| norm_line_json(base_baseline + i as f32 * 18.0, None))
            .collect();
        format!(r#"{{"type":"text","lines":[{}]}}"#, lines.join(","))
    }

    /// `n` copies of [`norm_body_region_json`], each in its own baseline
    /// range (`+= 500.0` per region) so cross-region baseline collisions
    /// never create a spurious same-region delta — deltas are computed
    /// per-region by [`page_pitch_px`]/[`region_pitch_px`], so this is
    /// belt-and-braces rather than load-bearing, but it keeps every fixture
    /// unambiguous to read.
    fn norm_body_regions(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| norm_body_region_json(i as f32 * 500.0))
            .collect()
    }

    /// The no-metrics sibling of [`norm_body_regions`] (test 5 only).
    fn norm_body_regions_no_metrics(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| norm_body_region_json_no_metrics(i as f32 * 500.0))
            .collect()
    }

    /// The bare page BODY (everything inside one `"pages"` array entry, no
    /// `doc.v1` envelope) for `regions` — parseable directly as a
    /// [`JsonPage`] via `serde_json::from_str`, so a test can compute its
    /// exact expected [`page_font_px`]/[`page_pitch_px`]/
    /// [`page_measured_font_px_median`] from the SAME evidence
    /// [`doc_v1_layout`] itself will use, rather than a hand-derived number
    /// that could silently drift from the real constants.
    fn norm_page_body_json(regions: &[String]) -> String {
        format!(
            r#"{{"width":1000,"height":2000,"regions":[{}],"fields":[]}}"#,
            regions.join(",")
        )
    }

    /// [`norm_page_body_json`] wrapped in the full `doc.v1` envelope, ready
    /// for [`doc_v1_layout`].
    fn norm_doc_json(regions: &[String]) -> String {
        format!(
            r#"{{"schema":"tesseract-rs/doc.v1","pages":[{}]}}"#,
            norm_page_body_json(regions)
        )
    }

    /// Every `Block::Text` in page 0, in emission order.
    fn text_blocks(doc: &LayoutDoc) -> Vec<&TextBlock> {
        doc.pages[0]
            .blocks
            .iter()
            .map(|b| match b {
                Block::Text(t) => t,
                other => panic!("expected only Block::Text in these fixtures, got {other:?}"),
            })
            .collect()
    }

    /// Test 1 (can-fire). Six `"text"` body regions (4 lines each, real
    /// baselines + xheight + a constant band font_px of 10.0) plus one
    /// single-line `"header"` whose measured font is 2x body (xheight 20.0).
    /// Because the header has only ONE line, rung A (region pitch) cannot
    /// apply (`< MIN_REGION_PITCH_LINES`) and the factor must come from rung
    /// B (region measured font vs body measured font median) — exercising
    /// exactly the single-line-headline path the spec calls out.
    ///
    /// **THE TRAP:** if the header line omitted `xheight`/`baseline`,
    /// `derive_text_metrics` would return `None` and this block's
    /// `metrics` would stay `None` — the caller's `TEXT_HEIGHT_TO_FONTSIZE`
    /// box-height fallback would then apply on the RENDER side (not tested
    /// here), and a taller header BOX would render bigger even WITH `*
    /// scale` deleted, passing this test for the wrong reason. The
    /// `metrics.is_some()` assertion below is what forbids that — the
    /// fixture below deliberately DOES carry `xheight`/`baseline` on the
    /// header line, so the only way its font_px can be ~2x page size is
    /// through the scale factor actually firing.
    #[test]
    fn header_2x_body_scales_the_header_block() {
        let header =
            r#"{"type":"header","lines":[{"baseline":5.0,"xheight":20.0,"ascrise":0.0,"descdrop":0.0,"words":[]}]}"#
                .to_string();
        let mut regions = norm_body_regions(6);
        regions.push(header);

        let jp: JsonPage = serde_json::from_str(&norm_page_body_json(&regions)).unwrap();
        let expected_page_font = page_font_px(&jp).expect("page font must resolve");

        let doc = doc_v1_layout(&norm_doc_json(&regions), &[]).expect("parse");
        let texts = text_blocks(&doc);
        assert_eq!(
            texts.len(),
            6 * 4 + 1,
            "6 body regions x 4 lines + 1 header line"
        );

        let (header_block, body_blocks) = texts.split_last().unwrap();
        let m = header_block
            .metrics
            .expect("header line carries xheight+baseline, metrics must resolve");
        assert!(
            (m.font_px - expected_page_font * 2.0).abs() < 1e-2,
            "header measuring 2x body must render at ~2x page size \
             (got {}, expected {})",
            m.font_px,
            expected_page_font * 2.0
        );
        for b in body_blocks {
            assert_eq!(
                b.metrics.unwrap().font_px,
                expected_page_font,
                "every body block must stay at EXACT page size"
            );
        }
    }

    /// Test 2 (stay-silent). An all-`"text"` page: every block's `font_px`
    /// must equal `page_font_px(&jp).unwrap()` EXACTLY — `assert_eq!`, no
    /// epsilon, because `f * 1.0f32` is exact in IEEE 754, so equality is
    /// the correct AND the stronger assertion here.
    #[test]
    fn text_only_page_stays_at_page_size_exactly() {
        let regions = norm_body_regions(6);
        let jp: JsonPage = serde_json::from_str(&norm_page_body_json(&regions)).unwrap();
        let expected_page_font = page_font_px(&jp).expect("page font must resolve");

        let doc = doc_v1_layout(&norm_doc_json(&regions), &[]).expect("parse");
        let texts = text_blocks(&doc);
        assert_eq!(texts.len(), 6 * 4);
        for b in &texts {
            assert_eq!(
                b.metrics.unwrap().font_px,
                expected_page_font,
                "an all-text page must never deviate from page size"
            );
        }
    }

    /// Test 3 (dead-band). A `"header"` measuring 1.1x body (inside
    /// `[SCALE_DEAD_BAND_LO, SCALE_DEAD_BAND_HI]`) must land at EXACTLY page
    /// size — `assert_eq!`, since the dead band collapses the factor to the
    /// exact constant `1.0`.
    #[test]
    fn header_within_dead_band_is_left_at_page_size() {
        let header =
            r#"{"type":"header","lines":[{"baseline":5.0,"xheight":11.0,"ascrise":0.0,"descdrop":0.0,"words":[]}]}"#
                .to_string();
        let mut regions = norm_body_regions(6);
        regions.push(header);

        let jp: JsonPage = serde_json::from_str(&norm_page_body_json(&regions)).unwrap();
        let expected_page_font = page_font_px(&jp).expect("page font must resolve");

        let doc = doc_v1_layout(&norm_doc_json(&regions), &[]).expect("parse");
        let texts = text_blocks(&doc);
        let header_block = texts.last().unwrap();
        assert_eq!(
            header_block.metrics.unwrap().font_px,
            expected_page_font,
            "1.1x body (inside the dead band) must land at exactly page size"
        );
    }

    /// Test 4 (clamp). A `"header"` measuring 10x body must clamp to
    /// `SCALE_CLAMP_HI` (3.0), not the raw ratio.
    #[test]
    fn header_10x_body_clamps_to_the_ceiling() {
        let header =
            r#"{"type":"header","lines":[{"baseline":5.0,"xheight":100.0,"ascrise":0.0,"descdrop":0.0,"words":[]}]}"#
                .to_string();
        let mut regions = norm_body_regions(6);
        regions.push(header);

        let jp: JsonPage = serde_json::from_str(&norm_page_body_json(&regions)).unwrap();
        let expected_page_font = page_font_px(&jp).expect("page font must resolve");

        let doc = doc_v1_layout(&norm_doc_json(&regions), &[]).expect("parse");
        let texts = text_blocks(&doc);
        let header_block = texts.last().unwrap();
        let m = header_block.metrics.unwrap();
        assert!(
            (m.font_px - expected_page_font * 3.0).abs() < 1e-2,
            "a 10x-measuring header must clamp to 3x, not render at 10x \
             (got {}, expected {})",
            m.font_px,
            expected_page_font * 3.0
        );
    }

    /// Test 5 (rung isolation). A `"header"` carries valid rung-B evidence
    /// (xheight+baseline, one line only — so rung A cannot apply either),
    /// but every BODY line carries `baseline` only, no band keys — so
    /// [`page_measured_font_px_median`] returns `None` and rung B's
    /// denominator is empty. The factor must fall back to exactly `1.0`,
    /// NOT silently denominate against `page_font_px(&jp)` (which page pitch
    /// alone still makes available, since it needs only `baseline`).
    #[test]
    fn header_deviation_with_no_body_metrics_stays_at_page_size() {
        let header =
            r#"{"type":"header","lines":[{"baseline":5.0,"xheight":20.0,"ascrise":0.0,"descdrop":0.0,"words":[]}]}"#
                .to_string();
        let mut regions = norm_body_regions_no_metrics(6);
        regions.push(header);

        let jp: JsonPage = serde_json::from_str(&norm_page_body_json(&regions)).unwrap();
        // Sanity-check the fixture actually reaches the intended state
        // before trusting the assertion below: pitch resolves (body lines
        // carry baseline), but the body-only measured median does not (no
        // body line carries a band key).
        assert!(
            page_pitch_px(&jp).is_some(),
            "fixture must still have a pitch"
        );
        assert!(
            page_measured_font_px_median(&jp).is_none(),
            "fixture must have NO body-metrics evidence"
        );
        let expected_page_font = page_font_px(&jp).expect("page font must resolve");

        let doc = doc_v1_layout(&norm_doc_json(&regions), &[]).expect("parse");
        let texts = text_blocks(&doc);
        let header_block = texts.last().unwrap();
        assert_eq!(
            header_block.metrics.unwrap().font_px,
            expected_page_font,
            "an empty rung-B denominator must decline to page size, \
             not fall back to page_font_px as a substitute denominator"
        );
    }

    /// Test 6 (kind gate). The IDENTICAL 2x-measuring fixture from test 1,
    /// but with `kind = "text"` and again with `kind = ""` (doc.v1's legacy
    /// no-`type` shape) — both must land at EXACTLY page size, proving the
    /// kind gate itself, not the dead band (a real 2.0 ratio is nowhere near
    /// the dead band, so only the gate can be suppressing it).
    #[test]
    fn text_kind_never_deviates_even_at_2x() {
        for kind_json in ["\"type\":\"text\",", ""] {
            let extra = format!(
                r#"{{{kind_json}"lines":[{{"baseline":5.0,"xheight":20.0,"ascrise":0.0,"descdrop":0.0,"words":[]}}]}}"#
            );
            let mut regions = norm_body_regions(6);
            regions.push(extra);

            let jp: JsonPage = serde_json::from_str(&norm_page_body_json(&regions)).unwrap();
            let expected_page_font = page_font_px(&jp).expect("page font must resolve");

            let doc = doc_v1_layout(&norm_doc_json(&regions), &[]).expect("parse");
            let texts = text_blocks(&doc);
            let extra_block = texts.last().unwrap();
            assert_eq!(
                extra_block.metrics.unwrap().font_px,
                expected_page_font,
                "kind={kind_json:?}: a 2x measurement on a non-header/footer \
                 kind must never deviate from page size"
            );
        }
    }
}
