//! Recognizer **A6b** — the pure-Rust image front-end: an image FILE on disk →
//! the grey pixel buffer + pre-scale to the network input height. This closes
//! `image file → text` when composed with A6a ([`from_grey_pix`]) + B3-core
//! ([`recognize_grid`]).
//!
//! Per the founding directive (**no leptonica at runtime**), image decode is
//! pure-Rust. This module reads **P5 PGM** (portable grey map) — a standard
//! 8-bit-grey image file that both this parser and leptonica's `pixRead` decode
//! to identical pixels (a lossless raw format, so decode is byte-exact by
//! construction).
//!
//! ## The scale boundary (honest)
//! `ImageData::PreScale` calls leptonica's general `pixScale(src, f, f)` with
//! `f = target_height / input_height`. **At `f == 1.0` (an image already at the
//! model height) `pixScale` is a plain copy**, so [`prescale_grey_to_height`] is
//! identity and the whole `image → text` path is byte-parity-proven (A6a + B1 +
//! beam + B3-core). For other heights, leptonica's `pixScale` is a specific
//! depth/factor-dependent resampler (linear-interp ≥0.7, area-map <0.7) whose
//! byte-exact transcode is a deferred sub-leaf; this module's non-identity path
//! is a **marked bilinear approximation** — functional, but NOT byte-identical
//! to leptonica. Supply a height-`target` image for byte-parity.
//!
//! [`from_grey_pix`]: tesseract_recognizer::from_grey_pix
//! [`recognize_grid`]: crate::LstmRecognizer::recognize_grid

/// An error parsing a PGM image file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PgmError {
    /// Not a `P5` (binary grey) PGM.
    BadMagic,
    /// A header field (width/height/maxval) was missing or malformed.
    BadHeader,
    /// `maxval` exceeds 255 (16-bit samples are out of scope).
    Not8Bit,
    /// The pixel data is shorter than `width × height`.
    Truncated,
}

impl std::fmt::Display for PgmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a P5 PGM"),
            Self::BadHeader => write!(f, "malformed PGM header"),
            Self::Not8Bit => write!(f, "PGM maxval > 255 (16-bit not supported)"),
            Self::Truncated => write!(f, "PGM pixel data truncated"),
        }
    }
}

impl std::error::Error for PgmError {}

/// Parse a **P5** (binary 8-bit grey) PGM: `P5` magic, whitespace-separated
/// `width height maxval` (with `#` comments allowed), a single whitespace byte,
/// then `width × height` raw grey bytes (row-major). Returns
/// `(grey, width, height)`.
///
/// # Errors
///
/// [`PgmError`] on a bad magic/header, a `maxval > 255`, or truncated data.
pub fn parse_pgm(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), PgmError> {
    if bytes.len() < 2 || &bytes[0..2] != b"P5" {
        return Err(PgmError::BadMagic);
    }
    let mut pos = 2;
    // Read three ASCII integers, skipping whitespace and `#`-to-EOL comments.
    let mut fields = [0usize; 3];
    for field in &mut fields {
        // Skip whitespace + comments.
        loop {
            while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos < bytes.len() && bytes[pos] == b'#' {
                while pos < bytes.len() && bytes[pos] != b'\n' {
                    pos += 1;
                }
            } else {
                break;
            }
        }
        // Read digits.
        let start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos == start {
            return Err(PgmError::BadHeader);
        }
        let n: usize = std::str::from_utf8(&bytes[start..pos])
            .ok()
            .and_then(|s| s.parse().ok())
            .ok_or(PgmError::BadHeader)?;
        *field = n;
    }
    let [width, height, maxval] = fields;
    if maxval > 255 {
        return Err(PgmError::Not8Bit);
    }
    // Exactly one whitespace byte separates the header from the raster.
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return Err(PgmError::BadHeader);
    }
    pos += 1;
    let count = width.checked_mul(height).ok_or(PgmError::BadHeader)?;
    let data = bytes.get(pos..pos + count).ok_or(PgmError::Truncated)?;
    Ok((data.to_vec(), width, height))
}

/// Pre-scale a grey image to `target_height` (the network input height, e.g. 36).
///
/// **Identity — byte-exact — when `height == target_height`** (leptonica's
/// `pixScale` at factor 1.0 is a copy): the canonical line-recognition case, for
/// which the full `image → text` path is byte-parity-proven. For other heights
/// this is a **marked bilinear approximation** (NOT byte-identical to leptonica's
/// `pixScale`; see the module docs). Returns `(scaled_grey, scaled_width)`.
#[must_use]
pub fn prescale_grey_to_height(
    grey: &[u8],
    width: usize,
    height: usize,
    target_height: usize,
) -> (Vec<u8>, usize) {
    if height == target_height || height == 0 || width == 0 {
        return (grey.to_vec(), width);
    }
    // APPROXIMATION (marked): bilinear resample. leptonica's pixScale uses
    // linear-interp/area-map with its own fixed-point arithmetic; this is NOT
    // byte-identical to it — supply a height-`target_height` image for parity.
    let im_factor = target_height as f32 / height as f32;
    let target_width = ((im_factor * width as f32) + 0.5) as usize; // IntCastRounded
    let target_width = target_width.max(1);
    let mut out = vec![0u8; target_width * target_height];
    for oy in 0..target_height {
        let sy = ((oy as f32 + 0.5) / im_factor - 0.5).clamp(0.0, (height - 1) as f32);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let fy = sy - y0 as f32;
        for ox in 0..target_width {
            let sx = ((ox as f32 + 0.5) / im_factor - 0.5).clamp(0.0, (width - 1) as f32);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let fx = sx - x0 as f32;
            let p00 = f32::from(grey[y0 * width + x0]);
            let p01 = f32::from(grey[y0 * width + x1]);
            let p10 = f32::from(grey[y1 * width + x0]);
            let p11 = f32::from(grey[y1 * width + x1]);
            let top = p00 + (p01 - p00) * fx;
            let bot = p10 + (p11 - p10) * fx;
            out[oy * target_width + ox] = (top + (bot - top) * fy + 0.5) as u8;
        }
    }
    (out, target_width)
}

/// `pixScaleGrayLI` → `scaleGrayLILow` (leptonica 1.82.0 `scale1.c:2324`): 8-bit
/// grey scaling by **linear interpolation**, via 16×16 sub-pixel bilinear
/// weighting. `src` is `hs × ws` row-major grey; the output is `hd × wd` where
/// the caller sets `wd = round(f·ws)`, `hd = round(f·hs)` (`pixScaleGrayLI`'s own
/// dimension formula). Valid for `f ≥ 0.7` (leptonica's `pixScaleGeneral` routes
/// smaller factors to area-map). This is the **un-sharpened** scale — the full
/// `pixScale` applies `pixUnsharpMasking` on top (a separate sub-leaf).
///
/// Byte-exact vs leptonica: the sub-pixel location is `(int)(scx·j)` truncated,
/// split into `xp = ·>>4` (src pixel) + `xf = ·&0xF` (1/16 fraction); the four
/// neighbours are weighted by `(16−xf)(16−yf) … xf·yf` and combined as
/// `(Σ + 128)/256`. Leptonica's `wpl`-packing is internal, so every logical read
/// is `src[y·ws + x]`.
#[must_use]
pub fn scale_gray_li(src: &[u8], ws: usize, hs: usize, wd: usize, hd: usize) -> Vec<u8> {
    let scx = 16.0 * ws as f32 / wd as f32;
    let scy = 16.0 * hs as f32 / hd as f32;
    let wm2 = ws as i32 - 2;
    let hm2 = hs as i32 - 2;
    let get = |y: i32, x: i32| i32::from(src[(y as usize) * ws + x as usize]);
    let mut dst = vec![0u8; wd * hd];
    for i in 0..hd {
        let ypm = (scy * i as f32) as i32;
        let yp = ypm >> 4;
        let yf = ypm & 0x0f;
        for j in 0..wd {
            let xpm = (scx * j as f32) as i32;
            let xp = xpm >> 4;
            let xf = xpm & 0x0f;
            let v00_val = get(yp, xp);
            let v10_val;
            let v01_val;
            let v11_val;
            if xp > wm2 || yp > hm2 {
                if yp > hm2 && xp <= wm2 {
                    // near bottom
                    v01_val = v00_val;
                    v10_val = get(yp, xp + 1);
                    v11_val = v10_val;
                } else if xp > wm2 && yp <= hm2 {
                    // near right side
                    v01_val = get(yp + 1, xp);
                    v10_val = v00_val;
                    v11_val = v01_val;
                } else {
                    // LR corner
                    v10_val = v00_val;
                    v01_val = v00_val;
                    v11_val = v00_val;
                }
            } else {
                v10_val = get(yp, xp + 1);
                v01_val = get(yp + 1, xp);
                v11_val = get(yp + 1, xp + 1);
            }
            let v00 = (16 - xf) * (16 - yf) * v00_val;
            let v10 = xf * (16 - yf) * v10_val;
            let v01 = (16 - xf) * yf * v01_val;
            let v11 = xf * yf * v11_val;
            dst[i * wd + j] = ((v00 + v01 + v10 + v11 + 128) / 256) as u8;
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_p5_pgm() {
        // "P5\n3 2\n255\n" + 6 bytes.
        let mut b = b"P5\n3 2\n255\n".to_vec();
        b.extend_from_slice(&[10, 20, 30, 40, 50, 60]);
        let (grey, w, h) = parse_pgm(&b).expect("valid pgm");
        assert_eq!((w, h), (3, 2));
        assert_eq!(grey, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn pgm_header_with_comment() {
        let mut b = b"P5\n# a comment\n3 2\n255\n".to_vec();
        b.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let (grey, w, h) = parse_pgm(&b).expect("valid pgm");
        assert_eq!((w, h, grey.len()), (3, 2, 6));
    }

    #[test]
    fn pgm_errors() {
        assert_eq!(parse_pgm(b"P6\n1 1\n255\n\0"), Err(PgmError::BadMagic));
        assert_eq!(parse_pgm(b"P5\n2 2\n255\n\0"), Err(PgmError::Truncated));
    }

    #[test]
    fn scale_gray_li_identity_at_factor_one() {
        // f == 1.0 (wd==ws, hd==hs): scx=scy=16, so xf=yf=0 and
        // val = (256·v00 + 128)/256 = v00 — an exact identity. (Real byte-parity
        // vs leptonica pixScaleGrayLI is the scale_li_dump example, 6/6 factors.)
        let src: Vec<u8> = (0..12).map(|i| (i * 17) as u8).collect();
        let out = scale_gray_li(&src, 4, 3, 4, 3);
        assert_eq!(out, src, "factor-1.0 LI scale is identity");
    }

    #[test]
    fn prescale_identity_at_target_height() {
        let grey: Vec<u8> = (0..12).collect();
        let (out, w) = prescale_grey_to_height(&grey, 4, 3, 3);
        assert_eq!(w, 4);
        assert_eq!(out, grey, "identity when height == target");
    }

    #[test]
    fn prescale_changes_height_otherwise() {
        let grey: Vec<u8> = (0..24).collect();
        let (out, w) = prescale_grey_to_height(&grey, 4, 6, 3);
        assert_eq!(out.len(), w * 3, "scaled to height 3");
    }
}
