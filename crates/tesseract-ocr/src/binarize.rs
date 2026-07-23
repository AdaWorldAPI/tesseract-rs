//! Sauvola adaptive binarization — a byte-parity transcode of leptonica's
//! `pixSauvolaBinarize` chain, from the `AdaWorldAPI/leptonica` fork
//! (`src/binarize.c` + `src/convolve.c` + `src/pix2.c`), proven byte-identical
//! against the installed liblept 1.82.0 (`.claude/harvest/oracles/sauvola_oracle.cpp`).
//!
//! The chain (`addborder = 1`, the document path):
//! ```text
//!   pixAddMirroredBorder(whsize+1)            (pix2.c:2122)
//!     -> pixWindowedMean       (u32 integral, convolve.c:1055 / 499)
//!     -> pixWindowedMeanSquare (f64 integral, convolve.c:1170 / 1353)
//!     -> pixSauvolaGetThreshold  t = m·(1 - k·(1 - s/128)),  s = sqrt(ms - m²)
//!                                                       (binarize.c:711)
//!     -> pixApplyLocalThreshold  grey < t  => ON (black text)  (binarize.c:791)
//! ```
//!
//! Why it matters: the layout-stage binarization today is global Otsu
//! (`threshold.rs`), which annihilates unevenly-lit / aged scans (the
//! ImproveQuality lesson). Sauvola is the adaptive alternative — a per-pixel
//! threshold from the local mean and standard deviation, so a shadowed corner
//! keeps its own threshold instead of going all-black. The recognizer LSTM
//! still consumes grey (`image_input::from_grey_pix`); Sauvola feeds the
//! *segmentation* stage (`xy_cut::binarize_page`), where a bad global threshold
//! fragments the page.
//!
//! Fidelity notes (byte-parity depends on each):
//! - the u32 accumulator is **wrapping** (`l_uint32` overflow); the 4-corner
//!   window difference recovers the true window sum modulo 2^32 because that sum
//!   is `<= 255·(2·whsize+1)² < 2^32`.
//! - the mean-square accumulator is `f64` and holds exact integers (sums of
//!   `u8²` stay well under 2^52).
//! - `mean` casts `(f32 norm)·sum` to `u8` (truncate); `mean_square` casts
//!   `(f64 norm)·sum + 0.5` to `u32` (round); the threshold casts an `f64`
//!   expression to `i32` then stores the low 8 bits — all reproduced exactly.

use std::num::Wrapping;

/// pixAddMirroredBorder (`pix2.c:2122`) — reflect a border of `b` pixels on all
/// four sides. The reflection is edge-duplicated: `bordered[b-1-j] = img[j]`, so
/// the pixel just outside the image equals the edge pixel. Left/right are filled
/// over the centre rows, then top/bottom over the *full* width so the corners
/// mirror correctly (the same order as the leptonica rasterops).
#[must_use]
pub fn add_mirrored_border(src: &[u8], w: usize, h: usize, b: usize) -> (usize, usize, Vec<u8>) {
    let wd = w + 2 * b;
    let hd = h + 2 * b;
    let mut d = vec![0u8; wd * hd];
    // Centre.
    for y in 0..h {
        for x in 0..w {
            d[(y + b) * wd + (x + b)] = src[y * w + x];
        }
    }
    // Left + right, over the centre rows [b, b+h).
    for y in 0..h {
        let row = (y + b) * wd;
        for j in 0..b {
            d[row + (b - 1 - j)] = d[row + (b + j)]; // left:  col(b-1-j) <- centre col j
            d[row + (b + w + j)] = d[row + (b + w - 1 - j)]; // right: <- centre col (w-1-j)
        }
    }
    // Top + bottom, over the full width (includes the just-filled L/R borders).
    for i in 0..b {
        let (dst_top, src_top) = ((b - 1 - i) * wd, (b + i) * wd);
        let (dst_bot, src_bot) = ((b + h + i) * wd, (b + h - 1 - i) * wd);
        for x in 0..wd {
            d[dst_top + x] = d[src_top + x];
            d[dst_bot + x] = d[src_bot + x];
        }
    }
    (wd, hd, d)
}

/// blockconvAccumLow, d==8 (`convolve.c:499`) — the u32 summed-area table.
/// Wrapping to match `l_uint32` semantics; window differences recover the sum.
fn blockconv_accum(src: &[u8], w: usize, h: usize) -> Vec<Wrapping<u32>> {
    let mut a = vec![Wrapping(0u32); w * h];
    for j in 0..w {
        let v = Wrapping(u32::from(src[j]));
        a[j] = if j == 0 { v } else { a[j - 1] + v };
    }
    for i in 1..h {
        let (row, prow) = (i * w, (i - 1) * w);
        for j in 0..w {
            let v = Wrapping(u32::from(src[row + j]));
            a[row + j] = if j == 0 {
                v + a[prow]
            } else {
                v + a[row + j - 1] + a[prow + j] - a[prow + j - 1]
            };
        }
    }
    a
}

/// pixMeanSquareAccum (`convolve.c:1353`) — the f64 summed-area table of squares
/// (exact integer arithmetic: partial sums stay below 2^52).
fn mean_square_accum(src: &[u8], w: usize, h: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; w * h];
    for j in 0..w {
        let v = f64::from(src[j]);
        a[j] = if j == 0 { v * v } else { a[j - 1] + v * v };
    }
    for i in 1..h {
        let (row, prow) = (i * w, (i - 1) * w);
        for j in 0..w {
            let v = f64::from(src[row + j]);
            a[row + j] = if j == 0 {
                a[prow] + v * v
            } else {
                a[row + j - 1] + a[prow + j] - a[prow + j - 1] + v * v
            };
        }
    }
    a
}

/// pixWindowedMean (`convolve.c:1055`), `hasborder=1, normflag=1`. Strips a
/// `(wc+1, hc+1)` border → `(w-2(wc+1)) × (h-2(hc+1))` u8 local-mean map.
fn windowed_mean(bordered: &[u8], w: usize, h: usize, wc: usize, hc: usize) -> Vec<u8> {
    let c = blockconv_accum(bordered, w, h);
    let wd = w - 2 * (wc + 1);
    let hd = h - 2 * (hc + 1);
    let (wincr, hincr) = (2 * wc + 1, 2 * hc + 1);
    let norm = 1.0f32 / ((wincr as f32) * (hincr as f32));
    let mut d = vec![0u8; wd * hd];
    for i in 0..hd {
        let (r1, r2) = (i * w, (i + hincr) * w);
        for j in 0..wd {
            let val = c[r2 + j + wincr] - c[r2 + j] - c[r1 + j + wincr] + c[r1 + j];
            d[i * wd + j] = (norm * val.0 as f32) as u8;
        }
    }
    d
}

/// pixWindowedMeanSquare (`convolve.c:1170`), `hasborder=1`. → u32 mean-square map.
fn windowed_mean_square(bordered: &[u8], w: usize, h: usize, wc: usize, hc: usize) -> Vec<u32> {
    let a = mean_square_accum(bordered, w, h);
    let wd = w - 2 * (wc + 1);
    let hd = h - 2 * (hc + 1);
    let (wincr, hincr) = (2 * wc + 1, 2 * hc + 1);
    // norm: 1.0 (f64) / ((f32)wincr * hincr) — the denominator is an f32 product.
    let denom = (wincr as f32) * (hincr as f32);
    let norm = 1.0f64 / f64::from(denom);
    let mut d = vec![0u32; wd * hd];
    for i in 0..hd {
        let (r1, r2) = (i * w, (i + hincr) * w);
        for j in 0..wd {
            let val = a[r2 + j + wincr] - a[r2 + j] - a[r1 + j + wincr] + a[r1 + j];
            d[i * wd + j] = (norm * val + 0.5) as u32;
        }
    }
    d
}

/// pixSauvolaGetThreshold (`binarize.c:711`): `t = m·(1 - k·(1 - s/128))`,
/// `s = sqrt(ms - m²)`. The `w·h > 100000` sqrt table is numerically identical
/// to `sqrtf` for `var >= 0`, so `sqrtf` is used directly (`binarize.c:768-771`).
fn sauvola_get_threshold(mean: &[u8], ms: &[u32], n: usize, factor: f32) -> Vec<u8> {
    let mut d = vec![0u8; n];
    for idx in 0..n {
        let mv = i32::from(mean[idx]);
        let var = ms[idx] as i32 - mv * mv;
        let sd = (var as f32).sqrt();
        let thresh =
            (f64::from(mv) * (1.0 - f64::from(factor) * (1.0 - f64::from(sd) / 128.0))) as i32;
        d[idx] = thresh as u8; // SET_DATA_BYTE = low 8 bits
    }
    d
}

/// pixApplyLocalThreshold (`binarize.c:791`): `grey < thresh` → ON (1 = black
/// foreground); one byte (0/1) per pixel (the packed 1bpp is `SET_DATA_BIT`).
fn apply_local_threshold(grey: &[u8], thresh: &[u8], n: usize) -> Vec<u8> {
    let mut d = vec![0u8; n];
    for idx in 0..n {
        if grey[idx] < thresh[idx] {
            d[idx] = 1;
        }
    }
    d
}

/// The Sauvola result of the `addborder = 1` path: the per-pixel local threshold
/// map (8bpp) and the binary foreground mask (one 0/1 byte per pixel, ON=black).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sauvola {
    /// Image width.
    pub w: usize,
    /// Image height.
    pub h: usize,
    /// Per-pixel local threshold (`pixth`, 8bpp).
    pub threshold: Vec<u8>,
    /// Per-pixel binary foreground (`pixd`, 1 = black text, 0 = background).
    pub binary: Vec<u8>,
}

/// pixSauvolaBinarize (`binarize.c:602`), the `addborder = 1` path: grey 8bpp →
/// (threshold map, binary mask). `whsize` is the window half-size (`>= 2`);
/// `factor` is `k` (`>= 0`, typically `0.34`).
///
/// # Panics
/// Panics if `grey.len() != w·h`, if `whsize < 2`, or if the image is too small
/// for the window (`w < 2·whsize + 3` or `h < 2·whsize + 3`) — the same guards
/// leptonica returns an error for.
#[must_use]
pub fn sauvola_binarize(grey: &[u8], w: usize, h: usize, whsize: usize, factor: f32) -> Sauvola {
    assert_eq!(grey.len(), w * h, "grey buffer is not w·h");
    assert!(whsize >= 2, "whsize must be >= 2");
    assert!(
        w >= 2 * whsize + 3 && h >= 2 * whsize + 3,
        "whsize too large for image"
    );
    // pixg = mirror-bordered by whsize+1; pixsc = the original grey.
    let (bw, bh, bordered) = add_mirrored_border(grey, w, h, whsize + 1);
    let mean = windowed_mean(&bordered, bw, bh, whsize, whsize);
    let ms = windowed_mean_square(&bordered, bw, bh, whsize, whsize);
    debug_assert_eq!(mean.len(), w * h);
    let threshold = sauvola_get_threshold(&mean, &ms, w * h, factor);
    let binary = apply_local_threshold(grey, &threshold, w * h);
    Sauvola {
        w,
        h,
        threshold,
        binary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_border_reflects_edge_duplicated() {
        // 3x1 row [10,20,30], border 2 -> [30,20,10,20,30,30,20] wait: reflect.
        let (w, h, d) = add_mirrored_border(&[10, 20, 30], 3, 1, 2);
        assert_eq!((w, h), (7, 5));
        // centre row is row 2: cols 2..5 = 10,20,30. left border cols: d[b-1-j]=centre[j]
        // col1 = centre0 = 10, col0 = centre1 = 20; right col5 = centre(w-1)=30, col6=centre1=20
        let row = &d[2 * 7..3 * 7];
        assert_eq!(row, &[20, 10, 10, 20, 30, 30, 20]);
    }

    #[test]
    fn flat_image_threshold_is_fraction_of_value() {
        // A constant grey image has zero variance -> sd=0 -> t = m·(1-k).
        // 40x40 of value 200, whsize 3, k=0.5 -> t = 200·0.5 = 100 everywhere,
        // and grey(200) < 100 is false -> all background (0).
        let g = vec![200u8; 40 * 40];
        let s = sauvola_binarize(&g, 40, 40, 3, 0.5);
        assert!(
            s.threshold.iter().all(|&t| t == 100),
            "flat threshold = m(1-k)"
        );
        assert!(s.binary.iter().all(|&b| b == 0), "200 !< 100 -> background");
    }

    #[test]
    fn dark_pixel_below_local_threshold_is_foreground() {
        // Bright field (240) with a dark 1-px dip (20): local mean stays high,
        // so the dark pixel falls below its threshold -> foreground.
        let mut g = vec![240u8; 40 * 40];
        g[20 * 40 + 20] = 20;
        let s = sauvola_binarize(&g, 40, 40, 3, 0.34);
        assert_eq!(s.binary[20 * 40 + 20], 1, "dark dip is foreground");
    }
}
