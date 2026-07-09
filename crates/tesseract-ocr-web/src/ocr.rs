//! Decode arbitrary image bytes → grey → recognize → text + stats.

use std::io::Cursor;
use std::time::Instant;

use image::{ImageReader, Limits};

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

/// The result of OCR-ing one uploaded/fetched image.
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

/// Decode `bytes` (PNG / JPEG / WebP / TIFF / GIF / BMP / PNM — via the `image`
/// crate, all pure-Rust decoders) to
/// 8-bit grey and run the full page-recognition path
/// ([`LstmRecognizer::recognize_page_makerow`]). Returns text + stats, or a
/// user-safe error string.
///
/// This is heavy synchronous CPU work — callers MUST run it off the async
/// runtime (via `spawn_blocking`); see [`crate::routes`]. Decode is bounded
/// against decompression / pixel bombs: the decoder is capped at
/// [`MAX_DECODE_ALLOC`] and [`MAX_DIM`], and the decoded pixel count is rejected
/// above [`MAX_PIXELS`] before the grey buffer (and the larger OCR working set)
/// is ever materialized.
pub fn ocr_image_bytes(state: &AppState, bytes: &[u8]) -> Result<OcrOutcome, String> {
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
    let raw = grey.into_raw(); // Vec<u8>, row-major, len == w*h

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
