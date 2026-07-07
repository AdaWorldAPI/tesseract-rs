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

/// `pixUnsharpMaskingGray2D` (leptonica 1.82.0 `enhance.c`) — the 2-D unsharp
/// mask `pixScale` applies on top of the resample. The ruff harvest of
/// `enhance.c` showed the dispatch `pixUnsharpMasking → pixUnsharpMaskingGray →
/// pixUnsharpMaskingGrayFast → pixUnsharpMaskingGray2D`; `pixScale` always uses
/// `halfwidth ∈ {1, 2}` + `L_BOTH_DIRECTIONS`, which lands exactly here (the
/// general `pixBlockconvGray`/`pixacc` path is never reached).
///
/// Separable box low-pass: horizontal INT-sum of `2·halfwidth+1` bytes into an
/// f32 buffer, then vertical sum × `norm` (`1/9` or `1/25`) = the low-pass `L`;
/// the sharpened pixel is `N = I + fract·(I − L)`, `(int)(N + 0.5)` clamped to
/// `[0,255]`. The `halfwidth`-wide border keeps the source pixels
/// (`pixCopyBorder`), so `out` starts as a copy and only the interior is set.
/// `halfwidth` MUST be 1 or 2 (`pixScale`'s only values).
///
/// # Panics
///
/// Panics if `halfwidth` is not 1 or 2, or `grey.len() < w·h`.
#[must_use]
pub fn unsharp_mask_gray_2d(
    grey: &[u8],
    w: usize,
    h: usize,
    halfwidth: usize,
    fract: f32,
) -> Vec<u8> {
    assert!(halfwidth == 1 || halfwidth == 2, "halfwidth must be 1 or 2");
    assert!(grey.len() >= w * h, "grey buffer too small");
    let mut out = grey.to_vec(); // pixCopyBorder: border = source; interior overwritten below
    if fract <= 0.0 {
        return out;
    }
    let hw = halfwidth;
    if w <= 2 * hw || h <= 2 * hw {
        return out; // no interior to sharpen; the border-copy is the whole image
    }

    // Horizontal low-pass: fpix[i][j] = int-sum of (2·hw+1) source bytes, as f32.
    let mut fpix = vec![0.0_f32; w * h];
    for i in 0..h {
        for j in hw..w - hw {
            let mut s = 0_i32;
            for k in 0..=2 * hw {
                s += i32::from(grey[i * w + j - hw + k]);
            }
            fpix[i * w + j] = s as f32;
        }
    }

    // Vertical low-pass (× norm) → L, then sharpen N = I + fract·(I − L).
    let taps = 2 * hw + 1;
    let norm = (1.0_f64 / (taps * taps) as f64) as f32; // (f32)(1.0/9.0) or (1.0/25.0)
    for i in hw..h - hw {
        for j in hw..w - hw {
            let mut fsum = 0.0_f32;
            for k in 0..=2 * hw {
                fsum += fpix[(i - hw + k) * w + j];
            }
            let l = norm * fsum; // L: low-pass value (f32)
            let sval = f32::from(grey[i * w + j]); // I: source pixel
                                                   // N = I + fract·(I − L) in f32; then +0.5 promotes to f64 (0.5 is a
                                                   // double literal in C), and (int) truncates toward zero.
            let sharpened = sval + fract * (sval - l);
            let ival = (f64::from(sharpened) + 0.5) as i32;
            out[i * w + j] = ival.clamp(0, 255) as u8;
        }
    }
    out
}

/// `pixScale(pixs, f, f)` for 8-bit grey on the **`f ≥ 0.7`** path (leptonica
/// 1.82.0 `scale1.c` `pixScale` → `pixScaleGeneral`) — the dispatch the ruff
/// harvest mapped, composed from the two proven leaves:
///
/// - `f == 1.0` → copy (identity; the model-height case).
/// - `0.7 ≤ f` → `pixScaleGrayLI` ([`scale_gray_li`]); THEN, if `f < 1.4`, the
///   default sharpen `pixUnsharpMasking(·, 2, 0.4)` ([`unsharp_mask_gray_2d`]);
///   `f ≥ 1.4` skips the sharpen. `pixScale` sets `sharpfract = 0.4`,
///   `sharpwidth = 2` for `maxscale ≥ 0.7`.
///
/// Returns `(scaled_grey, wd, hd)`. **`f < 0.7` (area-map) is a separate leaf**
/// (task #31) — this asserts `f ≥ 0.7`.
///
/// # Panics
///
/// Panics if `f < 0.7` (routes to `pixScaleAreaMap`, not yet ported) or the
/// buffer is too small.
#[must_use]
pub fn pix_scale_grey_li(grey: &[u8], w: usize, h: usize, f: f32) -> (Vec<u8>, usize, usize) {
    assert!(grey.len() >= w * h, "grey buffer too small");
    if f == 1.0 {
        return (grey.to_vec(), w, h); // pixScaleGeneral: scalex==scaley==1 → copy
    }
    assert!(
        f >= 0.7,
        "f < 0.7 routes to pixScaleAreaMap (a separate leaf)"
    );
    let wd = ((f * w as f32) + 0.5) as usize; // pixScaleGrayLI's dims
    let hd = ((f * h as f32) + 0.5) as usize;
    let scaled = scale_gray_li(grey, w, h, wd, hd);
    if f < 1.4 {
        // pixScale's default sharpen for maxscale ∈ [0.7,1.4): (f32)0.4, width 2.
        let sharpfract = 0.4_f64 as f32;
        (unsharp_mask_gray_2d(&scaled, wd, hd, 2, sharpfract), wd, hd)
    } else {
        (scaled, wd, hd) // f ≥ 1.4: pixScaleGeneral clones (no sharpen)
    }
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
    fn unsharp_no_op_at_fract_zero() {
        // fract <= 0 → clone (no sharpening). (Real byte-parity vs
        // pixUnsharpMasking is the unsharp_dump example, 2/2 pixScale cases.)
        let grey: Vec<u8> = (0..25).map(|i| (i * 9) as u8).collect();
        assert_eq!(unsharp_mask_gray_2d(&grey, 5, 5, 2, 0.0), grey);
    }

    #[test]
    fn pix_scale_grey_li_identity_at_factor_one() {
        // f == 1.0 → copy. (Real byte-parity vs pixScale is the pixscale_dump
        // example, 6/6 factors f=0.72..1.5.)
        let grey: Vec<u8> = (0..12).collect();
        let (out, wd, hd) = pix_scale_grey_li(&grey, 4, 3, 1.0);
        assert_eq!((wd, hd), (4, 3));
        assert_eq!(out, grey);
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
