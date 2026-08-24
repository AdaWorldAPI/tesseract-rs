//! Arbitrary image bytes -> grey raster, bomb-bounded.
//!
//! The same limits `tesseract-ocr-web::ocr::decode_grey` enforces, copied
//! rather than shared (that crate is bin-only, no lib target) — the numbers
//! themselves are policy, not an accident of that crate's layout, so keeping
//! them identical is the point.

use std::io::Cursor;

use image::{ImageReader, Limits};

/// Hard ceiling on a single decoded dimension.
const MAX_DIM: u32 = 20_000;
/// Pixel budget (width x height).
const MAX_PIXELS: u64 = 40_000_000;
/// Cap the decoder's own intermediate allocation.
const MAX_DECODE_ALLOC: u64 = 256 * 1024 * 1024;
/// Smallest dimension the recognizer's line path accepts.
const MIN_DIM: usize = 3;

/// Decode `bytes` (PNG/JPEG/WebP/TIFF/GIF/BMP/PNM) to an 8-bit grey raster.
///
/// # Errors
/// A user-safe message on a format sniff failure, an oversized/undersized
/// image, or a pixel budget breach.
pub fn decode_grey(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
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
    if (w as u64) * (h as u64) > MAX_PIXELS {
        return Err(format!(
            "image too large: {w}x{h} exceeds the {} megapixel limit",
            MAX_PIXELS / 1_000_000
        ));
    }

    let grey = dynimg.to_luma8();
    drop(dynimg);
    Ok((grey.into_raw(), w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1x1 truncated byte stream is not any supported format — the sniff
    /// step itself must fail, not panic.
    #[test]
    fn garbage_bytes_are_rejected_not_panicked_on() {
        let err = decode_grey(b"not an image").unwrap_err();
        assert!(err.contains("could not read image") || err.contains("could not decode"));
    }

    /// A real 1x1 PNG clears every format check but must be rejected by the
    /// `MIN_DIM` floor — proves the floor actually declines something, not
    /// just documents an intention.
    #[test]
    fn a_real_but_too_small_image_is_declined_by_min_dim() {
        // A genuine 1x1 RGB PNG (chunk CRCs computed with Python's zlib.crc32
        // against the exact chunk bytes, not hand-typed) — a wrong CRC makes
        // the decoder reject the file before MIN_DIM is ever consulted, which
        // is precisely the false-negative this test exists to avoid: it must
        // fail on "too small", never on a decode error.
        let png: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x78,
            0xDA, 0x63, 0xF8, 0xFF, 0xFF, 0x3F, 0x00, 0x05, 0xFE, 0x02, 0xFE, 0x33, 0x12, 0x95,
            0x14, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let err = decode_grey(png).unwrap_err();
        assert!(err.contains("too small"), "got: {err}");
    }
}
