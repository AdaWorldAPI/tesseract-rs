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
//!   `advance_width_1000em`, `escape_pdf_literal`, `embed_grey_image`) so the
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
    advance_width_1000em, embed_grey_image, escape_pdf_literal, prec, px_to_pt, winansi_encode_str,
    PageOcr, RenderReport, SearchablePdfError,
};
use crate::GreyImage;

// ---------------------------------------------------------------------------
// The layout model
// ---------------------------------------------------------------------------

/// A whole document: its pages plus the resolution (`dpi`) at which
/// image-pixel bboxes convert to PDF points (`px * 72 / dpi`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutDoc {
    /// The pages, in order.
    pub pages: Vec<LayoutPage>,
    /// Dots per inch — the `px→pt` scale for [`render_pdf`]. HTML preview is
    /// dpi-independent (CSS px == image px).
    pub dpi: u32,
}

/// One page: its pixel dimensions, an optional full-page background scan, and
/// the placed blocks (top-down image-pixel coordinates throughout).
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextBlock {
    /// Top-down image-pixel bbox `(left, top, right, bottom)`. The baseline is
    /// the box bottom and the run is `Tz`-stretched to the box width, exactly
    /// as [`crate::searchable_pdf`] documents.
    pub bbox: (u32, u32, u32, u32),
    /// The run's text.
    pub text: String,
    /// `false` → invisible searchable layer (`3 Tr`); `true` → painted (`0 Tr`).
    pub visible: bool,
}

/// A placed table: its outer bbox, grid dimensions, cells, and whether to
/// stroke the cell/outer rectangles as visible rules.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// One table cell: its grid position, bbox, text, and header flag.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Emit the `Tm`/`Tf`/`Tz`/`Tj` run for one text bbox (baseline = box bottom,
/// `Tz`-fitted to the box width). Returns the WinAnsi `'?'`-substitution count.
/// A degenerate box emits nothing and counts nothing; a positive box counts
/// its substitutions even if its (all-zero-advance) run shows nothing — this
/// matches the original `build_page_content` semantics exactly.
fn emit_text_run(
    ops: &mut Vec<Operation>,
    bbox: (u32, u32, u32, u32),
    text: &str,
    dpi: u32,
    page_h_pt: f64,
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
    // Baseline = box bottom (APPROX); PDF y is bottom-up.
    let y_pt = page_h_pt - px_to_pt(f64::from(bottom), dpi);
    let fontsize_pt = box_h_pt;
    let natural_width_pt = fontsize_pt * f64::from(natural_width_1000em) / 1000.0;
    let tz = 100.0 * box_w_pt / natural_width_pt;

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
    ops.push(Operation::new(
        "Tj",
        vec![Object::String(
            escape_pdf_literal(&bytes),
            lopdf::StringFormat::Literal,
        )],
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
                    substitutions += emit_text_run(&mut ops, t.bbox, &t.text, dpi, page_h_pt);
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
                    substitutions += emit_text_run(&mut ops, t.bbox, &t.text, dpi, page_h_pt);
                }
                Block::Table(t) => {
                    for cell in &t.cells {
                        substitutions +=
                            emit_text_run(&mut ops, cell.bbox, &cell.text, dpi, page_h_pt);
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
    html.push_str(".text{font:12px/1 sans-serif;white-space:pre;color:#000}\n");
    html.push_str(".table{border:1px solid #333}\n");
    html.push_str(".table .cell{position:absolute;box-sizing:border-box;");
    html.push_str("font:11px/1 sans-serif;border:1px solid #999;overflow:hidden;padding:1px}\n");
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
                    html.push_str(&format!(
                        "  <div class=\"text\" style=\"{}\">{}</div>\n",
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
                        html.push_str(&format!(
                            "    <div class=\"cell\" style=\"left:{l}px;top:{top}px;\
                             width:{w}px;height:{h}px\">{}</div>\n",
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

#[derive(Deserialize)]
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

#[derive(Deserialize)]
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

#[derive(Deserialize)]
struct JsonLine {
    #[serde(default)]
    bbox: Option<[i64; 4]>,
    #[serde(default)]
    words: Vec<JsonWord>,
}

#[derive(Deserialize)]
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
}

#[derive(Deserialize)]
struct JsonField {
    #[serde(default)]
    value: String,
    #[serde(default)]
    bbox: [i64; 4],
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
                        blocks.push(Block::Text(TextBlock {
                            bbox,
                            text,
                            visible: true,
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
}
