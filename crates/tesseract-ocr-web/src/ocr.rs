//! Decode arbitrary image bytes → grey → recognize → text + stats.

use std::time::Instant;

use crate::state::AppState;

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

/// Decode `bytes` (PNG / JPEG / PNM — via the `image` crate, all pure Rust) to
/// 8-bit grey and run the full page-recognition path
/// ([`LstmRecognizer::recognize_page_makerow`]). Returns text + stats, or a
/// user-safe error string.
pub fn ocr_image_bytes(state: &AppState, bytes: &[u8]) -> Result<OcrOutcome, String> {
    // `image::load_from_memory` sniffs the format; `to_luma8` gives row-major
    // 8-bit grey — exactly the layout recognize_page_makerow expects.
    let dynimg = image::load_from_memory(bytes)
        .map_err(|e| format!("could not decode image (PNG/JPEG/PNM supported): {e}"))?;
    let grey = dynimg.to_luma8();
    let (w, h) = (grey.width() as usize, grey.height() as usize);
    if w == 0 || h == 0 {
        return Err("decoded image has zero size".to_string());
    }
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
