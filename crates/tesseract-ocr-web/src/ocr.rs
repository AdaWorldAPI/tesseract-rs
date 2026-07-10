//! Decode arbitrary image bytes → grey → recognize → text/JSON + stats.

use std::io::Cursor;
use std::time::Instant;

use image::{ImageReader, Limits};
use tesseract_ocr::german_invoice_fields;

use crate::state::AppState;

/// Hard ceiling on a single decoded dimension (guards a degenerate aspect that
/// slips under the pixel budget, e.g. `1 × 400_000_000`).
const MAX_DIM: u32 = 20_000;
/// Pixel budget (width × height). Bounds the grey buffer + all downstream OCR
/// allocation. 40 MP comfortably covers a 300 dpi A3 scan while a hostile
/// "22000×22000" bomb is rejected before it can allocate ~500 MB.
const MAX_PIXELS: u64 = 40_000_000;
/// Cap the decoder's own intermediate allocation (a compressed bomb can inflate
/// far past its byte size). Applies during `decode()`, before our pixel check.
const MAX_DECODE_ALLOC: u64 = 256 * 1024 * 1024;
/// Smallest dimension the recognizer's proven line path accepts (`E-OCR-FROMPIX-1`
/// floor is 3 px); anything narrower cannot hold a glyph.
const MIN_DIM: usize = 3;

/// The output format the client asked for, from the upload form's `format`
/// multipart field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain recognized text (the default).
    Text,
    /// The structured `tesseract-rs/doc.v1` JSON DOM plus a German-invoice
    /// field harvest (see `tesseract_ocr::structured`).
    Json,
}

impl OutputFormat {
    /// Parse the multipart `format` field value. `"json"` selects
    /// [`OutputFormat::Json`]; anything else — including an absent field, an
    /// empty string, or an unrecognized value — falls back to
    /// [`OutputFormat::Text`]. Never errors: an unknown format is not a user
    /// mistake worth rejecting the upload over.
    #[must_use]
    pub fn from_field(value: Option<&str>) -> Self {
        match value {
            Some("json") => Self::Json,
            _ => Self::Text,
        }
    }
}

/// The result of OCR-ing one uploaded/fetched image in text mode.
pub struct OcrOutcome {
    /// The recognized text (lines joined with `\n`).
    pub text: String,
    /// Decoded image width in pixels.
    pub width: usize,
    /// Decoded image height in pixels.
    pub height: usize,
    /// Number of characters in the recognized text.
    pub char_count: usize,
    /// Number of non-empty recognized lines.
    pub line_count: usize,
    /// Wall-clock recognition time in milliseconds (decode excluded).
    pub elapsed_ms: f64,
}

/// The result of OCR-ing one uploaded/fetched image in JSON mode: the
/// rendered `tesseract-rs/doc.v1` document (structure + harvested fields) plus
/// the same stats shape as [`OcrOutcome`], but word/line counts instead of a
/// (meaningless, for JSON) character count.
pub struct OcrJsonOutcome {
    /// The rendered `doc.v1` JSON document.
    pub json: String,
    /// Decoded image width in pixels.
    pub width: usize,
    /// Decoded image height in pixels.
    pub height: usize,
    /// Total recognized words across all lines.
    pub word_count: usize,
    /// Number of non-empty recognized lines.
    pub line_count: usize,
    /// Wall-clock recognition time in milliseconds (decode excluded).
    pub elapsed_ms: f64,
}

/// Decode `bytes` (PNG / JPEG / WebP / TIFF / GIF / BMP / PNM — via the `image`
/// crate, all pure-Rust decoders) to 8-bit grey, bounded against
/// decompression / pixel bombs: the decoder is capped at [`MAX_DECODE_ALLOC`]
/// and [`MAX_DIM`], and the decoded pixel count is rejected above
/// [`MAX_PIXELS`] before the grey buffer (and the larger OCR working set) is
/// ever materialized. Shared by both [`ocr_image_bytes`] and
/// [`ocr_image_bytes_json`] so the two output modes decode identically.
fn decode_grey(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    // Sniff the format from the bytes, then decode under explicit limits — the
    // `image` defaults set only a 512 MiB alloc cap and NO dimension cap, so a
    // tiny compressed file can still decode to a gigapixel raster.
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("could not read image: {e}"))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_DIM);
    limits.max_image_height = Some(MAX_DIM);
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    reader.limits(limits);

    let dynimg = reader.decode().map_err(|e| {
        format!("could not decode image (PNG, JPEG, WebP, TIFF, GIF, BMP, PNM supported): {e}")
    })?;

    let (w, h) = (dynimg.width() as usize, dynimg.height() as usize);
    if w == 0 || h == 0 {
        return Err("decoded image has zero size".to_string());
    }
    if w < MIN_DIM || h < MIN_DIM {
        return Err(format!("image is too small to contain text ({w}x{h})"));
    }
    // Reject an oversized pixel count BEFORE `to_luma8` allocates a second
    // full-resolution buffer and before the recognizer's larger working set.
    if (w as u64) * (h as u64) > MAX_PIXELS {
        return Err(format!(
            "image too large: {w}x{h} exceeds the {} megapixel limit",
            MAX_PIXELS / 1_000_000
        ));
    }

    let grey = dynimg.to_luma8();
    drop(dynimg); // free the decoded source before the recognizer's working set
    Ok((grey.into_raw(), w, h)) // Vec<u8>, row-major, len == w*h
}

/// Decode `bytes` to grey and run the full page-recognition path
/// ([`LstmRecognizer::recognize_page_makerow`]). Returns text + stats, or a
/// user-safe error string.
///
/// This is heavy synchronous CPU work — callers MUST run it off the async
/// runtime (via `spawn_blocking`); see [`crate::routes`].
pub fn ocr_image_bytes(state: &AppState, bytes: &[u8]) -> Result<OcrOutcome, String> {
    let (raw, w, h) = decode_grey(bytes)?;

    let t0 = Instant::now();
    let text = state
        .recognizer
        .recognize_page_makerow(&raw, w, h, state.dict.as_ref())
        .map_err(|e| format!("recognition failed: {e}"))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let char_count = text.chars().count();
    let line_count = text.lines().filter(|l| !l.trim().is_empty()).count();
    Ok(OcrOutcome {
        text,
        width: w,
        height: h,
        char_count,
        line_count,
        elapsed_ms,
    })
}

/// Decode `bytes` to grey and run the canonical one-shot structured-document
/// path ([`LstmRecognizer::recognize_document`]): word/box recognition →
/// `doc.v1` DOM → numeric hardening → German-invoice field harvest → region
/// classification (page furniture + XY-cut blocks + halftone figures) →
/// `doc.v1` JSON. The composition itself lives in `tesseract-ocr` so this
/// demo and the `tesseract-ogar` executor share ONE source of truth. Same
/// off-runtime contract as [`ocr_image_bytes`] — heavy synchronous CPU work,
/// callers MUST run it via `spawn_blocking`.
pub fn ocr_image_bytes_json(state: &AppState, bytes: &[u8]) -> Result<OcrJsonOutcome, String> {
    let (raw, w, h) = decode_grey(bytes)?;

    let t0 = Instant::now();
    let specs = german_invoice_fields();
    let doc = state
        .recognizer
        .recognize_document(&raw, w, h, state.dict.as_ref(), Some(&specs))
        .map_err(|e| format!("recognition failed: {e}"))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(OcrJsonOutcome {
        json: doc.json,
        width: w,
        height: h,
        word_count: doc.word_count,
        line_count: doc.line_count,
        elapsed_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_from_field_defaults_to_text() {
        assert_eq!(OutputFormat::from_field(None), OutputFormat::Text);
        assert_eq!(OutputFormat::from_field(Some("")), OutputFormat::Text);
        assert_eq!(OutputFormat::from_field(Some("text")), OutputFormat::Text);
        assert_eq!(OutputFormat::from_field(Some("bogus")), OutputFormat::Text);
        assert_eq!(OutputFormat::from_field(Some("JSON")), OutputFormat::Text); // case-sensitive
    }

    #[test]
    fn output_format_from_field_recognizes_json() {
        assert_eq!(OutputFormat::from_field(Some("json")), OutputFormat::Json);
    }
}
