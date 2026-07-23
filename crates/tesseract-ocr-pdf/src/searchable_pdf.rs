//! D4.5 — searchable-PDF renderer.
//!
//! Given a scanned page image plus recognized words with boxes, emits a PDF
//! page that draws the ORIGINAL scanned image and lays an INVISIBLE text
//! layer over it (PDF render mode 3, `Tr`) positioned per word box — the
//! classic "searchable PDF" shape: what you see is the scan, what you can
//! select/search/copy is the recognized text.
//!
//! ## What is transcoded from `pdfrenderer.cpp`, and what is APPROX
//!
//! The overall *layout model* — draw the image full-page via `cm`+`Do`, then
//! an invisible `BT 3 Tr ... ET` text object with one `Tm`-positioned,
//! `Tz`-stretched run per word — is exactly libtesseract's
//! `TessPDFRenderer::GetPDFTextObjects` (`api/pdfrenderer.cpp:331-501`):
//! - **The image-paint prologue** (`pdf_str << "q " << width << " 0 0 " <<
//!   height << " 0 0 cm"` + `" /Im1 Do"` + `" Q\n"`, lines 354-358): this
//!   crate reproduces the identical `cm` matrix shape (`w_pt 0 0 h_pt 0 0
//!   cm`, scaling the unit square to the full page in points) and `Do`
//!   operator, differing only in the XObject name (`/Im0`, matching this
//!   crate's own `make_scanned_pdf` convention) and in embedding
//!   `DeviceGray`/`FlateDecode` directly rather than JPEG.
//! - **Invisible text** (`"BT\n3 Tr"`, line 374): reproduced verbatim —
//!   render mode 3 ("neither fill nor stroke text, i.e. invisible", PDF
//!   32000-1:2008 §9.3.3 Table 106) is what makes the text layer
//!   selectable/searchable but never painted, so the visible page is exactly
//!   the underlying scan.
//! - **Per-word `Tf`+`Tz`+text-show** (lines 453-488): reproduced in spirit —
//!   declare a font size, set a horizontal stretch (`Tz`) so the glyph run's
//!   *natural* width matches the *measured* word-box width, then show the
//!   text. libtesseract's stretch formula is `h_stretch = kCharWidth *
//!   (100 * word_length_pts / (fontsize * pdf_word_len))` (`kCharWidth = 2`,
//!   line 184) — a formula tuned for its own glyphless-font's fixed
//!   half-em advance width per character (see below). This crate computes
//!   `Tz` from a REAL font's AFM advance widths (see
//!   [`advance_width_1000em`]), so the constant-`kCharWidth` shortcut does
//!   not apply here; the *goal* (natural width → measured box width via
//!   `Tz`) is the same, the *formula* differs because the font differs.
//!
//! **Deliberately NOT transcoded (v1 scope, per the task):**
//! - **The glyphless CID font machinery** (`AppendFont`/`AppendCIDToGIDMap`
//!   /`AppendCIDFontType2`, lines 519-649): a `.notdef`-only TrueType font
//!   with an all-zero `CIDToGIDMap`, so every codepoint paints nothing even
//!   if the render mode were visible, and Unicode text survives copy-paste
//!   via UTF-16BE CID codes (§ header comment, lines 1-178). This crate uses
//!   a **standard, non-embedded Type1 font (`Helvetica`, `WinAnsiEncoding`)**
//!   instead — v1 is text-searchable exactly like the real renderer (render
//!   mode 3 hides it either way), but its glyphs are placeholder Latin glyph
//!   shapes with real AFM advance widths, not a customized notdef-only font.
//!   Geometrically this is IDENTICAL to libtesseract's approach (an
//!   arbitrary/placeholder glyph shape stretched via `Tz` to the measured
//!   word width) — the two differ only in font internals, exactly the
//!   documented scope boundary. See `AppendFont`'s own doc comment
//!   (lines 43-178) for libtesseract's reasoning about why *its* font must
//!   be glyphless (copy/paste fidelity across arbitrary Unicode) — a reason
//!   that is orthogonal to searchability itself.
//! - **Baseline geometry** (`GetWordBaseline`/`AffineMatrix`/`ClipBaseline`,
//!   lines 227-313): this crate's [`PageOcr`]/[`PlacedWord`] carry only an
//!   axis-aligned box (no textord baseline fit exists yet in this transcode
//!   — see `tesseract-ocr/src/renderer.rs`'s own placeholder list), so the
//!   baseline is fixed at the **bottom of the word's box** and the text is
//!   always horizontal (`a=1 b=0 c=0 d=1`, i.e. libtesseract's own
//!   `WRITING_DIRECTION_LEFT_TO_RIGHT` identity case). Documented APPROX
//!   until a real baseline exists.
//!
//! ## Coordinate + `Tz` math (this crate's model)
//!
//! Image pixels (top-down, origin top-left) → PDF points (bottom-up, origin
//! bottom-left) at the given `dpi`:
//!
//! ```text
//! px_to_pt(px) = px * 72.0 / dpi
//! page_w_pt    = px_to_pt(image_width_px)
//! page_h_pt    = px_to_pt(image_height_px)
//! word.x_pt    = px_to_pt(box.left)
//! word.y_pt    = page_h_pt - px_to_pt(box.bottom)   // baseline = box bottom (APPROX)
//! word.size_pt = px_to_pt(box.bottom - box.top)     // font size = box height (APPROX)
//! ```
//!
//! `Tm` is then `1 0 0 1 word.x_pt word.y_pt` (pure translation, horizontal
//! text only — see baseline-APPROX above). `Tz` (horizontal scaling,
//! percent) is chosen so the font's *natural* advance-width sum for the
//! word's text, scaled by `Tz/100`, equals the box's measured width in
//! points:
//!
//! ```text
//! natural_width_pt = size_pt * (sum of AFM advance widths, /1000 em units)
//! Tz = 100 * box_width_pt / natural_width_pt
//! ```
//!
//! An empty `natural_width_pt` (e.g. a word with only zero-width — i.e.
//! absent-from-the-table — characters, which cannot occur given the
//! all-256-entries table below, but kept total) skips the `Tz`/`Tj` pair for
//! that word entirely rather than dividing by zero.
//!
//! ## WinAnsi encoding policy (v1)
//!
//! The built-in `Helvetica`/`WinAnsiEncoding` font only has glyphs for
//! `u+0000..=u+00FF` under the WinAnsi mapping, and this crate does not
//! implement the (non-trivial, remapped) CP1252-vs-Latin1 code points
//! `0x80..=0x9F`. The v1 policy, applied per `char`:
//! - `'\u{20}'..='\u{7E}'` (ASCII) and `'\u{A0}'..='\u{FF}'` (Latin-1
//!   supplement, identical to WinAnsi in this range) map directly to that
//!   byte value.
//! - Everything else (control characters, `0x80..=0x9F`, and any character
//!   above `u+00FF`, e.g. CJK/Cyrillic/emoji) is **lossily substituted with
//!   `'?'` (`0x3F`)** — flagged by [`render_searchable_pdf`] returning the
//!   count of substituted characters per word in its report (see
//!   [`RenderReport`]).

use std::io::Write as _;

use flate2::write::ZlibEncoder;
use flate2::Compression;
use lopdf::{dictionary, Document, ObjectId, Stream};

use crate::GreyImage;

/// Failures building or serializing a searchable PDF ([`render_searchable_pdf`]).
#[derive(Debug)]
pub enum SearchablePdfError {
    /// `lopdf` failed to serialize the assembled document (`Document::save_to`
    /// returns `std::io::Result` — its own writer plumbing, not
    /// [`lopdf::Error`] — even though every other fallible `lopdf` call in
    /// this crate uses the latter).
    Save(std::io::Error),
    /// A page's image dimensions don't match `grey.data.len()` (`w * h`).
    ImageSizeMismatch {
        /// Zero-based page index.
        page: usize,
        /// `grey.w * grey.h`.
        expected: usize,
        /// `grey.data.len()`.
        got: usize,
    },
}

impl std::fmt::Display for SearchablePdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Save(e) => write!(f, "serializing searchable PDF: {e}"),
            Self::ImageSizeMismatch {
                page,
                expected,
                got,
            } => write!(
                f,
                "page {page}: grey image data length {got} does not match w*h {expected}"
            ),
        }
    }
}

impl std::error::Error for SearchablePdfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Save(e) => Some(e),
            Self::ImageSizeMismatch { .. } => None,
        }
    }
}

/// One recognized word to place on the invisible text layer, in image pixel
/// coordinates (top-down, origin top-left — the same convention
/// [`GreyImage`] and [`crate::extract_page_image`] use).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedWord {
    /// The word's text (already assembled from unichar ids, e.g. via
    /// `tesseract_ocr::renderer`'s `word_text`/[`tesseract_core::ids_to_text`]).
    pub text: String,
    /// `(left, top, right, bottom)` in image pixels, top-down (so
    /// `bottom > top`). This is the SAME shape
    /// `tesseract_ocr::renderer`'s `to_image_box` produces from a
    /// [`tesseract_core::WordResult`]'s bottom-up `char_boxes` — callers
    /// wiring live OCR output through this renderer are expected to have
    /// already run that conversion (see the `tesseract-ocr-pdf` binary's
    /// `--searchable-pdf` wiring for the concrete call).
    pub box_: (u32, u32, u32, u32),
}

/// One page: its scanned grey image plus the words recognized on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageOcr {
    /// The page's scanned image — drawn, unmodified, as the page's visible
    /// content.
    pub grey: GreyImage,
    /// Recognized words, in any order (reading order is not required — the
    /// text layer's *search* order follows this `Vec`'s order, but PDF
    /// viewers select/extract by position, not by content-stream order, for
    /// invisible text).
    pub words: Vec<PlacedWord>,
}

/// Per-page substitution counts from the WinAnsi lossy-mapping policy (see
/// the module docs' "WinAnsi encoding policy" section) — diagnostic only,
/// never affects the produced PDF's validity.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderReport {
    /// `report.pages[i]` = number of characters substituted with `'?'` on
    /// page `i` because they fell outside the v1 WinAnsi range.
    pub pages: Vec<usize>,
}

impl RenderReport {
    /// Total substitutions across every page.
    #[must_use]
    pub fn total_substitutions(&self) -> usize {
        self.pages.iter().sum()
    }
}

/// Adobe Helvetica AFM advance widths (Core 14 fonts, `Helvetica.afm`,
/// `WinAnsiEncoding`), in `1/1000 em` units, indexed by byte value
/// `0..=255`. ASCII printable (`0x20..=0x7E`) values are the standard,
/// widely-published Helvetica AFM widths (PDF 32000-1:2008 Annex D lists the
/// same Core-14 metrics). Control codes (`0x00..=0x1F`, `0x7F`) are `0`
/// (never shown). The `0x80..=0x9F` CP1252-specific block and the
/// `0xA0..=0xFF` Latin-1-supplement block use the Helvetica AFM's published
/// widths for those glyphs where a glyph exists; this crate does not
/// generate text bytes in `0x80..=0x9F` at all (see the WinAnsi policy in
/// the module docs), so those 32 entries are a documented placeholder
/// (`556`, the digit width) rather than measured — never exercised by
/// [`winansi_encode`]'s output.
#[rustfmt::skip]
const HELVETICA_WINANSI_WIDTHS: [u16; 256] = [
    // 0x00..=0x1F: control, never shown.
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    // 0x20..=0x2F
    278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,
    // 0x30..=0x3F
    556,556,556,556,556,556,556,556,556,556,278,278,584,584,584,556,
    // 0x40..=0x4F
    1015,667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,
    // 0x50..=0x5F
    667,778,722,667,611,722,667,944,667,667,611,278,278,278,469,556,
    // 0x60..=0x6F
    333,556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,
    // 0x70..=0x7F
    556,556,333,500,278,556,500,722,500,500,500,334,260,334,584,0,
    // 0x80..=0x9F: CP1252-specific block. Not produced by winansi_encode
    // (see module docs); placeholder width (digit width, 556).
    556,556,556,556,556,556,556,556,556,556,556,556,556,556,556,556,
    556,556,556,556,556,556,556,556,556,556,556,556,556,556,556,556,
    // 0xA0..=0xAF (Latin-1 supplement, identical to WinAnsi here)
    278,333,556,556,556,556,260,556,333,737,370,556,584,333,556,333,
    // 0xB0..=0xBF
    737,556,400,400,333,556,556,556,556,556,333,333,333,606,556,556,
    // 0xC0..=0xCF
    667,667,667,667,667,667,1000,722,667,667,667,667,278,278,278,278,
    // 0xD0..=0xDF
    722,722,722,722,722,722,722,584,722,722,722,722,667,667,611,556,
    // 0xE0..=0xEF
    556,556,556,556,556,556,556,556,556,556,556,556,556,556,556,556,
    // 0xF0..=0xFF
    556,556,556,556,556,556,556,584,556,556,556,556,556,500,556,500,
];

/// Map one `char` to a WinAnsi byte per the v1 policy documented in the
/// module docs' "WinAnsi encoding policy" section: ASCII and Latin-1
/// supplement pass through directly, everything else lossily maps to `'?'`.
/// Returns `(byte, was_substituted)`.
fn winansi_encode(ch: char) -> (u8, bool) {
    let code = ch as u32;
    if (0x20..=0x7E).contains(&code) || (0xA0..=0xFF).contains(&code) {
        (code as u8, false)
    } else {
        (b'?', true)
    }
}

/// Encode `text` to WinAnsi bytes, returning the bytes and the number of
/// lossy `'?'` substitutions (see [`winansi_encode`]).
pub(crate) fn winansi_encode_str(text: &str) -> (Vec<u8>, usize) {
    let mut bytes = Vec::with_capacity(text.len());
    let mut substitutions = 0usize;
    for ch in text.chars() {
        let (b, subst) = winansi_encode(ch);
        bytes.push(b);
        if subst {
            substitutions += 1;
        }
    }
    (bytes, substitutions)
}

/// Sum of Helvetica AFM advance widths for `bytes`, in `1/1000 em` units.
pub(crate) fn advance_width_1000em(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .map(|&b| u32::from(HELVETICA_WINANSI_WIDTHS[b as usize]))
        .sum()
}

/// Escape a WinAnsi byte string for a PDF literal string (`(...)`) —
/// PDF 32000-1:2008 §7.3.4.2: backslash and both parentheses must be
/// backslash-escaped; every other byte (including all of `0x80..=0xFF`,
/// which is exactly a WinAnsi-encoded byte, not UTF-8) passes through
/// verbatim.
pub(crate) fn escape_pdf_literal(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() + 2);
    for &b in bytes {
        match b {
            b'\\' | b'(' | b')' => {
                out.push(b'\\');
                out.push(b);
            }
            _ => out.push(b),
        }
    }
    out
}

/// `px * 72.0 / dpi` — image pixels to PDF points at the given resolution.
pub(crate) fn px_to_pt(px: f64, dpi: u32) -> f64 {
    px * 72.0 / f64::from(dpi)
}

/// Round to 3 decimal places for PDF numeric output (mirrors
/// `pdfrenderer.cpp`'s own `prec()`, which exists for the same reason: avoid
/// scientific notation and keep the file diffable/small — PDF 32000-1:2008
/// places no precision requirement on real numbers).
pub(crate) fn prec(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Embed one 8-bit `DeviceGray` image as a `FlateDecode` XObject — mirrors
/// `examples/make_scanned_pdf.rs`'s image-embedding shape (same filter,
/// colour space, bit depth), factored out here so both call sites share it.
pub(crate) fn embed_grey_image(doc: &mut Document, grey: &GreyImage) -> ObjectId {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&grey.data)
        .expect("zlib-compress grey bytes (Vec<u8> writer is infallible)");
    let compressed = encoder
        .finish()
        .expect("finish zlib stream (Vec<u8> writer is infallible)");

    let image_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => grey.w as i64,
        "Height" => grey.h as i64,
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 8,
        "Filter" => "FlateDecode",
    };
    doc.add_object(Stream::new(image_dict, compressed))
}

/// Render one or more OCR'd pages into a single searchable PDF: each page
/// shows its original scanned image with an invisible, `Tz`-fitted text
/// layer positioned per word box. See the module docs for the full
/// coordinate/`Tz` model and the WinAnsi lossy-mapping policy.
///
/// This is a thin wrapper over the general [`crate::layout`] renderer:
/// [`crate::layout::searchable_layout`] turns the `(image, words)` pages into a
/// [`crate::layout::LayoutDoc`] (full-page background + one invisible text
/// block per word) and [`crate::layout::render_pdf`] emits the PDF. The exact
/// `px→pt` / `Tz`-fit / WinAnsi / `prec` math — and the per-word ordering — is
/// unchanged (the same shared helpers in this module do the work), so this
/// function's output and behaviour are identical to the pre-generalization
/// renderer.
///
/// # Errors
///
/// [`SearchablePdfError::ImageSizeMismatch`] if a page's `grey.data.len()`
/// doesn't match `grey.w * grey.h`; [`SearchablePdfError::Save`] if `lopdf`
/// fails to serialize the assembled document.
pub fn render_searchable_pdf(
    pages: &[PageOcr],
    dpi: u32,
) -> Result<(Vec<u8>, RenderReport), SearchablePdfError> {
    let mut layout = crate::layout::searchable_layout(pages.to_vec());
    layout.dpi = dpi;
    crate::layout::render_pdf(&layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::content::Content as LopdfContent;

    /// A tiny synthetic grey "image": a 20x10 chequerboard-ish grid, just
    /// bytes with some variation (never decoded/inspected by these tests —
    /// only its dimensions matter for the geometry math).
    fn synthetic_grey(w: usize, h: usize) -> GreyImage {
        let data = (0..w * h).map(|i| (i % 256) as u8).collect();
        GreyImage { data, w, h }
    }

    fn find_page_content(doc: &Document) -> Vec<u8> {
        let pages = doc.get_pages();
        let &page_id = pages.get(&1).expect("page 1 exists");
        doc.get_page_content(page_id).expect("page content stream")
    }

    #[test]
    fn round_trip_two_words_extract_text_layer_recovers_both_in_order() {
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
        let (pdf_bytes, report) = render_searchable_pdf(&[page], 300).expect("render");
        assert_eq!(report.total_substitutions(), 0);

        let extracted = crate::extract_text_layer(&pdf_bytes).expect("extract_text_layer");
        assert_eq!(extracted.len(), 1);
        let text = extracted[0].as_deref().expect("page has a text layer");
        // Order matters: "Hello" must appear before "world" (lopdf's
        // extract_text walks Tj operators in content-stream order, which is
        // this word Vec's order).
        let hello_pos = text.find("Hello").expect("Hello present");
        let world_pos = text.find("world").expect("world present");
        assert!(
            hello_pos < world_pos,
            "expected reading order Hello before world, got: {text:?}"
        );
    }

    #[test]
    fn geometry_tm_matches_hand_computed_box_at_known_dpi() {
        // 300 dpi -> 1 px = 72/300 = 0.24 pt exactly.
        let page = PageOcr {
            grey: synthetic_grey(300, 100),
            words: vec![PlacedWord {
                text: "A".to_string(),
                box_: (100, 20, 150, 70), // left=100, top=20, right=150, bottom=70
            }],
        };
        let (pdf_bytes, _report) = render_searchable_pdf(&[page], 300).expect("render");
        let doc = Document::load_mem(&pdf_bytes).expect("load generated pdf");
        let content_bytes = find_page_content(&doc);
        let content = LopdfContent::decode(&content_bytes).expect("decode content stream");

        let tm = content
            .operations
            .iter()
            .find(|op| op.operator == "Tm")
            .expect("a Tm operation is present");
        let x: f64 = tm.operands[4].as_float().expect("Tm x operand") as f64;
        let y: f64 = tm.operands[5].as_float().expect("Tm y operand") as f64;

        // page_h_pt = 100 * 72/300 = 24.0
        // x_pt = left * 72/300 = 100 * 0.24 = 24.0
        // y_pt = page_h_pt - bottom_pt = 24.0 - 70*0.24 = 24.0 - 16.8 = 7.2
        let expected_x = 100.0 * 72.0 / 300.0;
        let expected_y = 100.0 * 72.0 / 300.0 - 70.0 * 72.0 / 300.0;
        assert!(
            (x - expected_x).abs() < 1e-3,
            "Tm x: expected {expected_x}, got {x}"
        );
        assert!(
            (y - expected_y).abs() < 1e-3,
            "Tm y: expected {expected_y}, got {y}"
        );
    }

    #[test]
    fn invisible_render_mode_precedes_first_tj() {
        let page = PageOcr {
            grey: synthetic_grey(100, 40),
            words: vec![PlacedWord {
                text: "x".to_string(),
                box_: (5, 5, 20, 30),
            }],
        };
        let (pdf_bytes, _report) = render_searchable_pdf(&[page], 300).expect("render");
        let doc = Document::load_mem(&pdf_bytes).expect("load generated pdf");
        let content_bytes = find_page_content(&doc);
        let content = LopdfContent::decode(&content_bytes).expect("decode content stream");

        let tr_pos = content
            .operations
            .iter()
            .position(|op| op.operator == "Tr")
            .expect("a Tr (render mode) operation is present");
        assert_eq!(
            content.operations[tr_pos]
                .operands
                .first()
                .and_then(|o| o.as_i64().ok()),
            Some(3),
            "expected render mode 3 (invisible)"
        );
        let tj_pos = content
            .operations
            .iter()
            .position(|op| op.operator == "Tj")
            .expect("a Tj (show text) operation is present");
        assert!(
            tr_pos < tj_pos,
            "3 Tr must precede the first Tj (tr_pos={tr_pos}, tj_pos={tj_pos})"
        );
    }

    #[test]
    fn lossy_winansi_substitution_is_flagged_in_report() {
        let page = PageOcr {
            grey: synthetic_grey(100, 40),
            // U+4E2D ("middle", CJK) is outside the v1 WinAnsi range.
            words: vec![PlacedWord {
                text: "中".to_string(),
                box_: (5, 5, 20, 30),
            }],
        };
        let (pdf_bytes, report) = render_searchable_pdf(&[page], 300).expect("render");
        assert_eq!(report.pages, vec![1]);
        assert_eq!(report.total_substitutions(), 1);

        // The '?' substitute still round-trips as extractable text (lossy,
        // but not a PDF-validity or extraction failure).
        let extracted = crate::extract_text_layer(&pdf_bytes).expect("extract_text_layer");
        assert_eq!(extracted[0].as_deref().map(str::trim_end), Some("?"));
    }

    #[test]
    fn image_size_mismatch_is_a_typed_error() {
        let page = PageOcr {
            grey: GreyImage {
                data: vec![0u8; 5], // too short for 10x10
                w: 10,
                h: 10,
            },
            words: vec![],
        };
        let err = render_searchable_pdf(&[page], 300).unwrap_err();
        assert!(matches!(
            err,
            SearchablePdfError::ImageSizeMismatch {
                page: 0,
                expected: 100,
                got: 5,
            }
        ));
    }

    #[test]
    fn zero_area_box_is_skipped_without_panicking() {
        let page = PageOcr {
            grey: synthetic_grey(50, 50),
            words: vec![PlacedWord {
                text: "degenerate".to_string(),
                box_: (10, 10, 10, 10), // zero width and height
            }],
        };
        let (pdf_bytes, report) = render_searchable_pdf(&[page], 300).expect("render");
        assert_eq!(report.total_substitutions(), 0);
        let extracted = crate::extract_text_layer(&pdf_bytes).expect("extract_text_layer");
        // No word survived (skipped), so the page has no text layer.
        assert!(extracted[0].is_none());
    }
}
