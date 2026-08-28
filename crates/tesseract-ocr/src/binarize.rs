//! Local adaptive binarization — the Niblack family: **Sauvola**
//! (byte-parity transcode), **Wolf-Jolion**, and **Singh et al.**
//!
//! # ⚠ The three methods do NOT stand on the same evidential footing
//!
//! Read this before citing any of them:
//!
//! | method | footing | why |
//! |---|---|---|
//! | [`sauvola_binarize`] | **byte-parity** vs liblept 1.82.0 | leptonica implements `pixSauvolaBinarize`, so an oracle exists |
//! | [`wolf_binarize`] | **quality fence, NOT parity** | leptonica does not implement it; transcribed from the reference C++ (see below), never diffed against a running oracle |
//! | [`singh_binarize`] | **quality fence, NOT parity** | leptonica does not implement it; transcribed from the paper's equations, no reference implementation identified |
//!
//! This is the repo's standing rule that a non-parity leaf must never be
//! presented as a parity leaf (`CLAUDE.md`; same footing as `structured.rs`
//! and `rectify.rs`). The two new formulas are **verified transcriptions from
//! primary sources** — not recollection, which had already produced a wrong
//! Wolf formula once — but a verified transcription is not a measured
//! agreement. Full provenance, the source line numbers, and the measured
//! timing table: `.claude/harvest/binarization-roadmap.md`.
//!
//! # Sauvola — the byte-parity leaf
//!
//! A byte-parity transcode of leptonica's
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
//! Why it matters: the layout-stage binarization defaults to global Otsu
//! (`threshold.rs`), which annihilates unevenly-lit / aged scans (the
//! ImproveQuality lesson). Sauvola is the adaptive alternative — a per-pixel
//! threshold from the local mean and standard deviation, so a shadowed corner
//! keeps its own threshold instead of going all-black. The recognizer LSTM
//! still consumes grey (`image_input::from_grey_pix`); Sauvola feeds the
//! *segmentation* stage (`xy_cut::binarize_page_with`, selectable via
//! `XyCutParams::binarize_mode` — see `xy_cut.rs`'s `BinarizeMode` docs) and
//! `LstmRecognizer::recognize_document_with_mode`'s region/table
//! classification pass, where a bad global threshold fragments the page.
//! Otsu remains the default in both places; Sauvola is opt-in only.
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
//!
//! # Wolf-Jolion and Singh — the two fence-footing rungs
//!
//! Both reuse Sauvola's *machinery* unchanged — the mirror border, the two
//! integral images, the windowed mean, the local standard deviation — and
//! replace only the **closing formula**. That is the whole reason they are
//! cheap to add here: the expensive, parity-proven half is shared verbatim
//! ([`windowed_stats`]), so a rung is one function, not one pipeline.
//!
//! ```text
//!   Sauvola     t = m·(1 - k·(1 - s/128))          R = 128, FIXED
//!   Wolf        t = m + k·(s/max_s - 1)·(m - min_I)  max_s, min_I = GLOBAL
//!   Singh       t = m·(1 + k·(∂/(1-∂) - 1)),  ∂ = I - m   PER-PIXEL, on [0,1]
//! ```
//!
//! What each one fixes, and the cost:
//!
//! - **Wolf** replaces Sauvola's *fixed* `R = 128` with the image's **actual**
//!   maximum local standard deviation, and anchors the correction to the
//!   global minimum grey rather than to the local mean. Sauvola's weakness is
//!   faded / low-contrast scans: there every `s ≪ 128`, the term collapses,
//!   and `t → m` — it under-thresholds exactly where contrast is worst. Cost:
//!   `max_s` and `min_I` are **global reductions**, so Wolf is inherently
//!   two-pass and cannot stream. Sauvola can.
//! - **Singh** replaces the *windowed* standard deviation with a **per-pixel**
//!   mean deviation `∂ = I(x,y) - m(x,y)`. Nothing is accumulated for it, so
//!   Singh does not swap the mean-square integral for a different accumulator
//!   — it **deletes that half outright**: no [`windowed_mean_square`], no f64
//!   integral, no `sqrt`. Its measured cost is flat in window size (paper
//!   Table 1: ~0.19-0.25 s across windows 3→35 where Sauvola goes 7.1→13.3 s),
//!   which is the actual claim — not raw speed at a fixed window.
//!
//! Sensitivity `k` is **not transferable between these methods** (the
//! reference implementation warns about this, and Niblack famously wants a
//! *negative* `k`). Sweep it per method; do not carry Sauvola's `0.34` across.

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

/// Local standard deviation `s = sqrt(ms - m²)` from one pixel's windowed
/// mean and mean-square, exactly as `pixSauvolaGetThreshold` computes it
/// (`binarize.c:768-771`) — `i32` variance (so a negative variance from the
/// two independent quantizations yields `NaN` through `sqrt`, matching C's
/// `sqrtf` on a negative float rather than silently clamping).
///
/// Extracted verbatim so [`wolf_binarize`] shares the parity-proven
/// arithmetic rather than restating it. The expression is unchanged from the
/// transcribed original; moving it behind a call cannot alter the result
/// because every operation and every intermediate type is identical.
#[inline]
fn local_sd(mean_v: u8, ms_v: u32) -> f32 {
    let mv = i32::from(mean_v);
    let var = ms_v as i32 - mv * mv;
    (var as f32).sqrt()
}

/// pixSauvolaGetThreshold (`binarize.c:711`): `t = m·(1 - k·(1 - s/128))`,
/// `s = sqrt(ms - m²)`. The `w·h > 100000` sqrt table is numerically identical
/// to `sqrtf` for `var >= 0`, so `sqrtf` is used directly (`binarize.c:768-771`).
fn sauvola_get_threshold(mean: &[u8], ms: &[u32], n: usize, factor: f32) -> Vec<u8> {
    let mut d = vec![0u8; n];
    for idx in 0..n {
        let mv = i32::from(mean[idx]);
        let sd = local_sd(mean[idx], ms[idx]);
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

/// The result of any method in this module: the per-pixel local threshold map
/// (8bpp) and the binary foreground mask (one 0/1 byte per pixel, ON=black).
///
/// **Invariant, and it holds for all three methods:** `binary` is exactly
/// [`apply_local_threshold`] of the source grey against `threshold`, i.e.
/// `binary[i] == (grey[i] < threshold[i]) as u8`. So a caller may re-derive
/// one from the other, and a caller inspecting `threshold` is looking at the
/// numbers that actually made the decision — not a rounded view of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalBinarization {
    /// Image width.
    pub w: usize,
    /// Image height.
    pub h: usize,
    /// Per-pixel local threshold (`pixth`, 8bpp).
    pub threshold: Vec<u8>,
    /// Per-pixel binary foreground (`pixd`, 1 = black text, 0 = background).
    pub binary: Vec<u8>,
}

/// The `addborder = 1` Sauvola result. Retained as the name this type had
/// when Sauvola was the module's only method; identical to
/// [`LocalBinarization`], which is what [`wolf_binarize`] and
/// [`singh_binarize`] also return.
pub type Sauvola = LocalBinarization;

/// The shared front half of every method here: mirror-border by `whsize + 1`,
/// then the windowed mean and windowed mean-square over the
/// `(2·whsize+1)²` window. Byte-parity-proven as part of the Sauvola chain
/// (`pixWindowedMean` / `pixWindowedMeanSquare`); both new rungs reuse it
/// unchanged rather than growing a second, unproven accumulator.
///
/// Returns `(mean, mean_square)`, each `w·h` in the ORIGINAL image's
/// geometry (the border is stripped by the windowing step).
///
/// [`singh_binarize`] takes only the `mean` half — its per-pixel `∂` needs no
/// second-moment accumulation at all — and pays for the unused mean-square
/// pass. That is deliberate: sharing the one proven front half is worth more
/// here than saving a pass in a mode that is off by default. If Singh is ever
/// promoted to a hot path, split this into a mean-only variant then, with the
/// measurement that justifies it.
fn windowed_stats(grey: &[u8], w: usize, h: usize, whsize: usize) -> (Vec<u8>, Vec<u32>) {
    let (bw, bh, bordered) = add_mirrored_border(grey, w, h, whsize + 1);
    let mean = windowed_mean(&bordered, bw, bh, whsize, whsize);
    let ms = windowed_mean_square(&bordered, bw, bh, whsize, whsize);
    debug_assert_eq!(mean.len(), w * h);
    debug_assert_eq!(ms.len(), w * h);
    (mean, ms)
}

/// The guards every method in this module shares, stated once.
///
/// # Panics
/// Panics if `grey.len() != w·h`, if `whsize < 2`, or if the image is too
/// small for the window (`w < 2·whsize + 3` or `h < 2·whsize + 3`) — the same
/// guards leptonica returns an error for.
fn assert_window_fits(grey: &[u8], w: usize, h: usize, whsize: usize) {
    assert_eq!(grey.len(), w * h, "grey buffer is not w·h");
    assert!(whsize >= 2, "whsize must be >= 2");
    assert!(
        w >= 2 * whsize + 3 && h >= 2 * whsize + 3,
        "whsize too large for image"
    );
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
    assert_window_fits(grey, w, h, whsize);
    // pixg = mirror-bordered by whsize+1; pixsc = the original grey.
    let (mean, ms) = windowed_stats(grey, w, h, whsize);
    let threshold = sauvola_get_threshold(&mean, &ms, w * h, factor);
    let binary = apply_local_threshold(grey, &threshold, w * h);
    LocalBinarization {
        w,
        h,
        threshold,
        binary,
    }
}

/// Wolf-Jolion threshold: `t = m + k·(s/max_s - 1)·(m - min_I)`.
///
/// `max_s` is the maximum local standard deviation over the WHOLE image and
/// `min_I` the global minimum grey — both global reductions, which is why
/// this is inherently two-pass (see the module docs).
///
/// Two clamping facts, both checked rather than assumed:
/// - `s/max_s - 1 <= 0` always (by definition of a maximum) and
///   `m - min_I >= 0` always (by definition of a minimum), so for `k >= 0` the
///   correction is non-positive and **`t <= m <= 255`**. The threshold can
///   therefore never exceed the `u8` range from above, and storing it as `u8`
///   loses nothing at the top.
/// - `t < 0` IS reachable (a dark region far above the global minimum).
///   Clamping to `0` is decision-identical, not merely close: the test is
///   `grey < t`, and for any `t <= 0` no `u8` grey satisfies it — the pixel is
///   background either way.
///
/// Degenerate case: a perfectly flat image has `s = 0` everywhere, so
/// `max_s = 0` and `s/max_s` is `0/0`. Defined here as `0` — every pixel then
/// has the minimum relative deviation, giving `t = m - k·(m - min_I)`, which
/// on a flat page is `t = m` (since `m == min_I`) and yields all-background.
/// That is the right answer for a blank page, and it is what the `0/0` limit
/// would give anyway.
/// The global reduction half of [`wolf_get_threshold`], extracted as its own
/// primitive: the maximum local standard deviation anywhere on the page.
/// This is Wolf's own page-relative calibration constant (`s / max_s`
/// replaces Sauvola's fixed `s / 128`) — computed standalone here so a
/// caller can use it as a CHEAP PREDICTOR of Sauvola's fixed-reference
/// failure mode (low `max_s` means the page's own best local contrast never
/// approaches Sauvola's assumed full-contrast constant) without paying for
/// Wolf's second (per-pixel) pass.
///
/// Costs exactly the same one `windowed_stats` pass every rung in this
/// module already shares — no new O(w·h) work beyond what a caller who then
/// runs Sauvola or Wolf anyway was already going to pay.
///
/// # Panics
/// Same guards as [`sauvola_binarize`].
#[must_use]
pub fn global_max_local_stddev(grey: &[u8], w: usize, h: usize, whsize: usize) -> f32 {
    assert_window_fits(grey, w, h, whsize);
    let (mean, ms) = windowed_stats(grey, w, h, whsize);
    let mut max_s = 0.0f32;
    for idx in 0..mean.len() {
        let sd = local_sd(mean[idx], ms[idx]);
        if sd > max_s {
            max_s = sd;
        }
    }
    max_s
}

fn wolf_get_threshold(grey: &[u8], mean: &[u8], ms: &[u32], n: usize, k: f32) -> Vec<u8> {
    // Pass 1 — the two global reductions.
    let mut max_s = 0.0f32;
    for idx in 0..n {
        let sd = local_sd(mean[idx], ms[idx]);
        // NaN-safe: a NaN sd (negative variance from the two independent
        // quantizations) must not become the maximum and poison every pixel.
        if sd > max_s {
            max_s = sd;
        }
    }
    let min_i = f32::from(grey.iter().copied().min().unwrap_or(0));

    // Pass 2 — the closing formula.
    let mut d = vec![0u8; n];
    for idx in 0..n {
        let m = f32::from(mean[idx]);
        let sd = local_sd(mean[idx], ms[idx]);
        let rel = if max_s > 0.0 { sd / max_s } else { 0.0 };
        let t = m + k * (rel - 1.0) * (m - min_i);
        // A NaN here means `local_sd` saw a negative variance — reachable
        // because `mean` and `ms` quantize independently (u8 truncate vs u32
        // round), so a near-flat window can produce `ms < m²` by a hair. Such
        // a window has essentially no local deviation, and `0` (⟹ nothing
        // satisfies `grey < t`) is the background verdict that region wants.
        d[idx] = if t.is_nan() {
            0
        } else {
            t.clamp(0.0, 255.0) as u8
        };
    }
    d
}

/// Wolf-Jolion adaptive binarization — **quality-fence footing, NOT
/// byte-parity** (leptonica does not implement it; see the module docs).
///
/// Same signature and same guards as [`sauvola_binarize`]; `k` is the
/// sensitivity, whose reference default is `0.5` — **not** Sauvola's `0.34`,
/// because `k` is not transferable between methods.
///
/// # Panics
/// Same as [`sauvola_binarize`]: `grey.len() != w·h`, `whsize < 2`, or the
/// image too small for the window.
#[must_use]
pub fn wolf_binarize(grey: &[u8], w: usize, h: usize, whsize: usize, k: f32) -> LocalBinarization {
    assert_window_fits(grey, w, h, whsize);
    let (mean, ms) = windowed_stats(grey, w, h, whsize);
    let threshold = wolf_get_threshold(grey, &mean, &ms, w * h, k);
    let binary = apply_local_threshold(grey, &threshold, w * h);
    LocalBinarization {
        w,
        h,
        threshold,
        binary,
    }
}

/// Singh et al. threshold, eq. (13): `T = m·[1 + k·(∂/(1-∂) - 1)]` with
/// `∂ = I(x,y) - m(x,y)`, on intensities normalized to `[0, 1]`.
///
/// # The algebraic rearrangement is load-bearing, not a style choice
///
/// Written literally, `∂/(1-∂)` has a singularity at `∂ = 1`, reachable when
/// `I = 1` and `m = 0` (a white pixel inside a fully black window). A direct
/// transcription produces `inf`, then `0 · inf = NaN` through the outer `m·`,
/// and a NaN threshold silently misclassifies that pixel.
///
/// But the singularity **cancels**. Expand:
///
/// ```text
///   T = m + k·( m·∂/(1-∂) - m )
/// ```
///
/// and note `1 - ∂ = 1 - I + m`, so the risky product is
/// `m·(I - m) / (1 - I + m)`. Its denominator vanishes only when `I = 1` and
/// `m = 0` together, and there the numerator carries a matching factor of `m`:
/// as `m → 0` with `I = 1`, the quotient is `m·(1-m)/m = 1-m → 1`. Finite.
/// So the guarded value at the singular point is `1`, not `inf`, and `T → k`
/// rather than `NaN`.
///
/// The denominator cannot be negative (`I <= 1` and `m >= 0` give
/// `1 - I + m >= 0`), so a single `> EPSILON` guard covers it.
///
/// # The three behaviours the paper states, all reproduced by this form
///
/// - `k = 0` ⟹ `T = m`, the plain local mean.
/// - A uniform window (`I = m` ⟹ `∂ = 0`) ⟹ `T = m·(1-k) < m` ⟹ the pixel is
///   **background**. This is the blank-region behaviour Niblack gets wrong.
/// - `m = 0` ⟹ `T = 0` ⟹ background.
///
/// All three are asserted in this module's tests, which is the closest thing
/// to an oracle available for a method with no reference implementation.
///
/// # Deviation from the paper, stated rather than hidden
///
/// Eq. (8) applies the threshold as `I <= T` ⟹ foreground; this crate's
/// [`apply_local_threshold`] uses `grey < t` (leptonica's convention, shared
/// with the parity-proven Sauvola path). The two differ at exactly one grey
/// level. Consistency across the three methods and with the rest of the crate
/// is worth more than that one level in a fence-footing rung; a caller who
/// needs the paper's exact boundary can shift the stored threshold by one.
fn singh_get_threshold(grey: &[u8], mean: &[u8], n: usize, k: f32) -> Vec<u8> {
    let mut d = vec![0u8; n];
    for idx in 0..n {
        let i_n = f32::from(grey[idx]) / 255.0;
        let m_n = f32::from(mean[idx]) / 255.0;
        let den = 1.0 - i_n + m_n;
        // `m·∂/(1-∂)`, with the cancelled limit at the singular point.
        let ratio = if den > f32::EPSILON {
            m_n * (i_n - m_n) / den
        } else {
            1.0
        };
        let t = m_n + k * (ratio - m_n);
        // With the pole cancelled above, `t` is finite for every finite
        // input — so this guard covers exactly one case: a caller passing a
        // non-finite `k`. It is NOT a hedge against the formula itself, which
        // the `den` guard already makes total.
        d[idx] = if t.is_nan() {
            0
        } else {
            (t * 255.0).clamp(0.0, 255.0) as u8
        };
    }
    d
}

/// Singh et al. adaptive binarization (arXiv 1201.5227) — **quality-fence
/// footing, NOT byte-parity** (no reference implementation identified; see
/// the module docs).
///
/// Same signature and same guards as [`sauvola_binarize`]. `k` is the
/// sensitivity; **the paper's own range is `[0, 1]`** and its document figure
/// uses `0.06`. The paper elsewhere prints `k = 15`, which cannot be the same
/// parameter (at `k = 15` the threshold goes negative for every pixel and the
/// whole page is foreground) — so sweep `k` empirically in `[0, 1]` rather
/// than taking a figure's value. Details in
/// `.claude/harvest/binarization-roadmap.md`.
///
/// # Panics
/// Same as [`sauvola_binarize`]: `grey.len() != w·h`, `whsize < 2`, or the
/// image too small for the window.
#[must_use]
pub fn singh_binarize(grey: &[u8], w: usize, h: usize, whsize: usize, k: f32) -> LocalBinarization {
    assert_window_fits(grey, w, h, whsize);
    let (mean, _ms) = windowed_stats(grey, w, h, whsize);
    let threshold = singh_get_threshold(grey, &mean, w * h, k);
    let binary = apply_local_threshold(grey, &threshold, w * h);
    LocalBinarization {
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

    // ── Wolf-Jolion ──────────────────────────────────────────────────────

    /// A FADED page: background `200`, a faint vertical stripe at `180` —
    /// 20 grey levels of contrast, the regime Sauvola's fixed `R = 128` is
    /// documented to fail in (`s ≪ 128` ⟹ the term collapses ⟹ `t → m·(1-k)`,
    /// far below the ink).
    ///
    /// **This is the falsifier that makes Wolf worth having.** It asserts
    /// BOTH directions on the SAME fixture: Sauvola misses the stripe, Wolf
    /// catches it. A test that only asserted "Wolf finds the stripe" would
    /// pass just as happily if `wolf_binarize` were a second name for
    /// `sauvola_binarize` — it would prove the code runs, not that the method
    /// differs. Neither may flood the background, or "detects more ink" would
    /// be satisfiable by thresholding everything.
    #[test]
    fn wolf_recovers_faint_ink_that_sauvola_misses() {
        const W: usize = 60;
        const H: usize = 60;
        let mut g = vec![200u8; W * H];
        for y in 0..H {
            for x in 28..32 {
                g[y * W + x] = 180;
            }
        }
        let ink = 30 * W + 30; // stripe centre
        let bg = 30 * W + 5; // far from the stripe

        let s = sauvola_binarize(&g, W, H, 5, 0.34);
        let wo = wolf_binarize(&g, W, H, 5, 0.5);

        eprintln!(
            "faint stripe: sauvola t(ink)={} t(bg)={} | wolf t(ink)={} t(bg)={}",
            s.threshold[ink], s.threshold[bg], wo.threshold[ink], wo.threshold[bg]
        );

        assert_eq!(
            s.binary[ink], 0,
            "Sauvola is EXPECTED to miss 20-level contrast here (fixed R=128); \
             if it now finds it, this fixture no longer isolates the difference \
             Wolf exists to fix — rebuild a fainter one rather than deleting \
             this assertion"
        );
        assert_eq!(
            wo.binary[ink], 1,
            "Wolf must recover the faint stripe — that IS the method's claim"
        );
        assert_eq!(s.binary[bg], 0, "Sauvola must not flood the background");
        assert_eq!(wo.binary[bg], 0, "Wolf must not flood the background");
    }

    /// **Page-scale falsifier, at unit scale.** Grounded in a REAL
    /// measurement, not an invented fixture: `corpus/gen/gen_faded_contrast.py`
    /// applied to a real corpus page (uniform dynamic-range compression
    /// toward mid-grey — a faded/worn-toner scan, not a lighting field) and
    /// `binarize_ab.rs` recognized the result under all four modes. On the
    /// severe fade (`faded_085.pgm`, spread compressed from 255 to 38 grey
    /// levels): Otsu 42/42 words correct, Wolf 42/42, Singh 42/42 — **Sauvola
    /// 0/42, every word lost, CER 1.0**. The page's own histogram (measured):
    /// a spike of 355,298 px (96.4%) at grey 147 (background), a cluster of
    /// 4,624 px (1.25%) near grey 109 (ink), the rest antialiasing in between.
    ///
    /// This is a SHARPER and more useful result than the earlier "Wolf beats
    /// Sauvola" framing suggests, and it does not go the direction a first
    /// guess would predict:
    ///
    /// - **Otsu is fine.** A uniform monotonic compression preserves the
    ///   histogram's bimodal SHAPE (spike at 147, cluster at 109, a real
    ///   valley between), and Otsu's threshold depends only on that shape —
    ///   not on the absolute contrast. Faded contrast alone does not defeat a
    ///   global histogram threshold.
    /// - **Sauvola specifically fails, and for the reason the module docs
    ///   name:** ink is SPARSE (1.25% of pixels), so almost every local
    ///   window is nearly all background — local mean `m` sits near 147, and
    ///   with `s << 128` uniformly, `t = m*(1-k*(1-s/128))` collapses toward
    ///   `m*(1-k)` regardless of the window's real content. At `m≈147`,
    ///   `k=0.34`, that is `≈97` — comfortably ABOVE the actual ink value
    ///   (109), so every ink pixel fails `grey < t` and reads as background.
    ///   The whole page returns empty.
    /// - **Wolf and Singh both recover, by different independent mechanisms**
    ///   (Wolf's `max_s` renormalization; Singh's per-pixel `∂` has no fixed
    ///   `R` to collapse in the first place) — two different fixes converging
    ///   on the same page, which is corroborating rather than redundant.
    ///
    /// This fixture reproduces the same qualitative shape at unit scale:
    /// sparse ink (a small block, not a full-height stripe — density matters
    /// here, not just contrast) on a near-uniform background, at the real
    /// measured cluster centres (147 / 109) rather than round numbers.
    #[test]
    fn sauvola_fails_on_sparse_ink_at_faded_contrast_but_wolf_and_singh_recover() {
        const W: usize = 60;
        const H: usize = 60;
        // Background 147 with a SPARSE 6x6 ink block at 109 -- sparse, not a
        // full-height stripe, because density (how much of a local window is
        // background) is what drives Sauvola's collapse here, not contrast
        // alone (the wolf_recovers_faint_ink test above already covers a
        // dense full-height stripe at a different mean level).
        let mut g = vec![147u8; W * H];
        for y in 27..33 {
            for x in 27..33 {
                g[y * W + x] = 109;
            }
        }
        let ink = 30 * W + 30; // block centre
        let bg = 30 * W + 5; // far from the block

        let s = sauvola_binarize(&g, W, H, 5, 0.34);
        let wo = wolf_binarize(&g, W, H, 5, 0.5);
        let si = singh_binarize(&g, W, H, 5, 0.06);

        eprintln!(
            "sparse faded ink: sauvola t(ink)={} t(bg)={} | wolf t(ink)={} t(bg)={} | \
             singh t(ink)={} t(bg)={}",
            s.threshold[ink],
            s.threshold[bg],
            wo.threshold[ink],
            wo.threshold[bg],
            si.threshold[ink],
            si.threshold[bg]
        );

        assert_eq!(
            s.binary[ink], 0,
            "Sauvola is EXPECTED to miss sparse ink at this contrast (fixed \
             R=128 collapses toward m*(1-k) when nearly the whole local \
             window is background); if it now finds it, this fixture no \
             longer reproduces the measured faded_085.pgm failure -- \
             re-derive the numbers from a fresh binarize_ab run rather than \
             deleting this assertion"
        );
        assert_eq!(
            wo.binary[ink], 1,
            "Wolf must recover the sparse ink -- measured on the real page"
        );
        assert_eq!(
            si.binary[ink], 1,
            "Singh must ALSO recover it, by an independent mechanism (no \
             fixed R to collapse) -- measured on the real page"
        );
        assert_eq!(s.binary[bg], 0, "Sauvola must not flood the background");
        assert_eq!(wo.binary[bg], 0, "Wolf must not flood the background");
        assert_eq!(si.binary[bg], 0, "Singh must not flood the background");
    }

    /// Degenerate `max_s == 0`: a flat page has zero local deviation
    /// everywhere, so `s/max_s` is `0/0`. Guarded to `0`, which lands on
    /// `t = m` (because `m == min_I` on a flat page) and yields all-background
    /// — the right answer for a blank page, and no NaN anywhere.
    #[test]
    fn wolf_flat_image_is_all_background_not_nan() {
        let g = vec![200u8; 40 * 40];
        let wo = wolf_binarize(&g, 40, 40, 3, 0.5);
        assert!(
            wo.threshold.iter().all(|&t| t == 200),
            "flat page: m == min_I ⟹ the correction is zero ⟹ t == m"
        );
        assert!(wo.binary.iter().all(|&b| b == 0), "blank page has no ink");
    }

    // ── Singh et al. ─────────────────────────────────────────────────────

    /// The paper's first stated behaviour: `k = 0` ⟹ `T = m` exactly.
    #[test]
    fn singh_k_zero_is_the_plain_local_mean() {
        let mut g = vec![210u8; 40 * 40];
        g[20 * 40 + 20] = 30;
        let si = singh_binarize(&g, 40, 40, 3, 0.0);
        let (mean, _) = windowed_stats(&g, 40, 40, 3);
        // Both sides quantize through u8, so compare within one level.
        for (i, (&t, &m)) in si.threshold.iter().zip(mean.iter()).enumerate() {
            let d = i32::from(t) - i32::from(m);
            assert!(d.abs() <= 1, "k=0 ⟹ T=m (idx {i}: {d} off)");
        }
    }

    /// The paper's second and third stated behaviours, on one fixture:
    /// a uniform window (`I == m` ⟹ `∂ == 0`) gives `T = m·(1-k) < m` ⟹
    /// background; and a genuinely dark dip still reads as foreground, so the
    /// "everything is background" verdict is not simply the method being inert.
    #[test]
    fn singh_uniform_window_is_background_but_a_dip_is_not() {
        let mut g = vec![210u8; 40 * 40];
        g[20 * 40 + 20] = 30;
        let si = singh_binarize(&g, 40, 40, 3, 0.5);
        let flat = 5 * 40 + 5;
        assert_eq!(
            si.binary[flat], 0,
            "uniform window ⟹ ∂=0 ⟹ T=m(1-k)<m ⟹ background"
        );
        assert_eq!(si.binary[20 * 40 + 20], 1, "a real dip is still ink");
    }

    /// The singularity, reached deliberately: `∂ = 1` needs `I = 1` and
    /// `m = 0` together — a white pixel whose whole window is black AND a
    /// window big enough that `255 / (2·whsize+1)²` truncates to `0`. At
    /// `whsize = 8` the window is `17² = 289`, so `255/289 = 0.88 → 0`.
    ///
    /// A literal transcription of eq. (13) yields `0 · inf = NaN` here. The
    /// cancelled form gives `T = k` exactly, which is what this asserts —
    /// `15 == (0.06 · 255) as u8`. Checking for "not NaN" would be too weak:
    /// the NaN guard maps to `0`, so a broken implementation would still
    /// produce a valid-looking `u8`. Only the exact limit value distinguishes
    /// the correct branch from the guard.
    #[test]
    fn singh_singularity_yields_the_cancelled_limit_not_nan() {
        const W: usize = 40;
        let mut g = vec![0u8; W * W];
        let white = 20 * W + 20;
        g[white] = 255;
        let si = singh_binarize(&g, W, W, 8, 0.06);
        let (mean, _) = windowed_stats(&g, W, W, 8);
        assert_eq!(
            mean[white], 0,
            "fixture precondition: the 17x17 window mean must truncate to 0, \
             or the singular point is never reached and this test proves nothing"
        );
        assert_eq!(
            si.threshold[white], 15,
            "at ∂→1 the m· factor cancels the pole: T → k = 0.06 ⟹ 15/255"
        );
        assert_eq!(si.binary[white], 0, "white on black is not ink");
    }

    // ── shared invariant ─────────────────────────────────────────────────

    /// [`LocalBinarization`]'s documented invariant, checked for all three
    /// methods: `binary` is exactly `grey < threshold`. Without this a caller
    /// inspecting `threshold` could be looking at a rounded view rather than
    /// the numbers that actually decided — the trap a float-decision /
    /// u8-reported split would create.
    #[test]
    fn binary_is_always_derivable_from_the_reported_threshold() {
        const W: usize = 48;
        let mut g = vec![190u8; W * W];
        for y in 10..20 {
            for x in 10..38 {
                g[y * W + x] = 60;
            }
        }
        for (name, r) in [
            ("sauvola", sauvola_binarize(&g, W, W, 4, 0.34)),
            ("wolf", wolf_binarize(&g, W, W, 4, 0.5)),
            ("singh", singh_binarize(&g, W, W, 4, 0.06)),
        ] {
            for (i, ((&gv, &b), &t)) in g
                .iter()
                .zip(r.binary.iter())
                .zip(r.threshold.iter())
                .enumerate()
            {
                assert_eq!(
                    b,
                    u8::from(gv < t),
                    "{name}: binary must equal grey < threshold at idx {i}"
                );
            }
        }
    }
}
