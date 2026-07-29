//! Deskew — page ROTATION leaves, a byte-parity transcode of leptonica's
//! `pixRotate90` (d==8 only), `pixFindDifferentialSquareSum`, and
//! `pixRotateAMGray`/`pixRotateAMGrayCorner` (`src/{rotateorth.c,skew.c,
//! rotateam.c}`, `AdaWorldAPI/leptonica` fork, v1.82.0 == the installed
//! liblept — zero ABI/version skew, the same footing the Sauvola leaf had),
//! proven byte-identical against real liblept via
//! `.claude/harvest/oracles/skew_oracle.cpp`.
//!
//! This is a slice of the full deskew wave (`.claude/plans/deskew-wave-v1.md`,
//! leaves D1-D7 of D1-D8). NOT here: the pipeline wiring (D8 — whether
//! deskew runs by default, and its composition with `rectify.rs`'s keystone
//! correction). See the plan for the full sequencing and why the leaves
//! split this way.
//!
//! ## What's here
//! - [`rotate90_grey`] — `pixRotate90`, d==8 only (`rotateorth.c:222-232` CW /
//!   `:310-320` CCW): a pure index remap, no float math at all, lossless.
//!   Used by [`deskew_both`]'s 90°-round-trip.
//! - [`find_differential_square_sum`] — `pixFindDifferentialSquareSum`
//!   (`skew.c:1111-1155`): the score the D4 sweep maximizes.
//! - [`v_shear_corner`] / [`v_shear_center`] — `pixVShearCorner` /
//!   `pixVShearCenter` (`shear.c:370-382` / `:432-444`, over the
//!   `pixVShear` kernel at `shear.c:237-316`): the rasterop-band-copy
//!   vertical shear the D4 sweep applies to every candidate angle before
//!   scoring it. `pixHShear` (shear.c's OTHER half, used only by
//!   `pixRotate2Shear`/`pixRotate3Shear`, a later/different leaf) is NOT
//!   transcoded here — D4 never calls it.
//! - [`find_skew_sweep_and_search_score_pivot`] — `pixFindSkewSweepAndSearchScorePivot`
//!   (`skew.c:665-973`): the actual detector `pixFindSkew` calls (NOT the
//!   standalone, quadratic-fit `pixFindSkewSweep`, which sits off
//!   `pixFindSkew`'s call path entirely). Coarse sweep (raw max, [`v_shear_corner`]/
//!   [`v_shear_center`] + [`find_differential_square_sum`]) then an
//!   interval-halving binary search.
//! - [`rotate_am_gray`] / [`rotate_am_gray_corner`] — `pixRotateAMGray` /
//!   `pixRotateAMGrayCorner` (`rotateam.c:288-317` + `:391-448` low kernel;
//!   `:584-613` + `:683-736` low kernel): the grey area-map rotation the
//!   recognizer actually wants for post-deskew correction — rotating in grey
//!   preserves the antialiasing the LSTM input step depends on, whereas
//!   rotating only the binary detector buffer would throw it away.
//! - [`deskew_general`] — `pixDeskewGeneral` (`skew.c:289-351`): the D4/D5
//!   composition — binarize (fixed threshold) for DETECTION only, run the
//!   D4 sweep+search, gate on angle/confidence, then (if the gate clears)
//!   rotate the ORIGINAL grey via [`rotate_am_gray`]. GREY input only — see
//!   its own doc for exactly why the C's `depth == 1` branch is a deliberate
//!   non-goal here, not an oversight.
//! - [`deskew_both`] — `pixDeskewBoth` (`skew.c:166-189`): [`deskew_general`]
//!   (with `pixDeskew`'s own all-default parameters) → [`rotate90_grey`]
//!   clockwise → [`deskew_general`] again → [`rotate90_grey`]
//!   counter-clockwise. Two passes because the D4 detector is directional.
//!
//! ## Two crate-convention notes that matter for every leaf here
//! - **Grey is raw, unflipped.** 8bpp grey buffers in this crate are
//!   leptonica-native: white background ≈ 255, dark ink ≈ 0, no inversion
//!   (matches `image_input.rs`, `xy_cut.rs`'s documented convention). So
//!   `L_BRING_IN_WHITE` fills off-canvas pixels with `255` directly — no
//!   conversion needed at the grey boundary.
//! - **Binary is flipped.** THIS CRATE'S binary buffers use `0 = ON`
//!   (ink/foreground), matching `binreduce.rs`'s documented convention — the
//!   OPPOSITE of leptonica's native `1 = ON`. [`find_differential_square_sum`],
//!   [`v_shear_corner`], and [`v_shear_center`] all take a buffer in THIS
//!   crate's convention; convert at the caller boundary (see
//!   [`find_differential_square_sum`]'s own docs), never silently assume
//!   leptonica's polarity.
//!
//! ## Leptonica's area-map rotation CROPS — byte-parity means reproducing it
//! `pixRotateAMGray`/`pixRotateAMGrayCorner` return the SAME `w × h` they
//! were given; pixels that rotate off-canvas are lost, replaced by
//! `grayval`. This is not "fixed" here: matching that crop IS the parity
//! contract. Leptonica's canvas-expansion path (`pixEmbedForRotation`) is a
//! deliberate SKIP for this leaf (a different, `+0.5`-round-half-up rounding
//! convention, plus a dormant `i32` overflow risk at very large dimensions —
//! see the plan's non-goals list) — it is its own future leaf, not folded in
//! here as an "improvement."

use crate::binreduce::reduce_rank_binary_cascade;

/// `pixRotate90`, **d==8 only** (`rotateorth.c:165-377` is the full function;
/// the 8bpp branches are `:222-232` clockwise / `:310-320` counter-clockwise).
/// A pure index remap — no interpolation, no float math at all. `direction`
/// is `1` for clockwise, `-1` for counter-clockwise (matches the C parameter
/// exactly; any other value is a caller error, see `# Panics`). Output
/// dimensions are swapped: `(out_w, out_h) = (h, w)`.
///
/// Derivation: `rotateorth.c`'s own locals are confusingly named for this —
/// it calls `pixGetDimensions(pixs, &hd, &wd, &d)` with the width/height
/// OUTPUT ARGUMENTS reversed (the C comment `/* note: reversed */` flags this
/// itself), so `hd` receives the SOURCE width and `wd` the SOURCE height.
/// Rather than carry that confusing swap into Rust, the loops below were
/// re-derived directly in terms of this function's own `w`/`h`/row/col
/// (cross-checked against a hand-traced 2×3 example before writing this):
/// ```text
///   CW:  out[r][c] = in[h-1-c][r]   for r in 0..w, c in 0..h
///   CCW: out[r][c] = in[c][w-1-r]   for r in 0..w, c in 0..h
/// ```
/// where `out` is row-major with width `h` (so `out[r][c]` means the flat
/// index `r * h + c`) and `in` is row-major with width `w`. `CW` then `CCW`
/// (or four applications of either) is the identity — see the tests.
///
/// # Panics
/// Panics if `grey.len() != w * h`, or if `direction` is not `1` or `-1`.
#[must_use]
pub fn rotate90_grey(grey: &[u8], w: usize, h: usize, direction: i32) -> (Vec<u8>, usize, usize) {
    assert_eq!(grey.len(), w * h, "grey buffer is not w·h");
    assert!(
        direction == 1 || direction == -1,
        "direction must be 1 (clockwise) or -1 (counter-clockwise)"
    );
    let (out_w, out_h) = (h, w);
    let mut out = vec![0u8; out_w * out_h];
    if direction == 1 {
        for r in 0..out_h {
            for c in 0..out_w {
                out[r * out_w + c] = grey[(h - 1 - c) * w + r];
            }
        }
    } else {
        for r in 0..out_h {
            for c in 0..out_w {
                out[r * out_w + c] = grey[c * w + (w - 1 - r)];
            }
        }
    }
    (out, out_w, out_h)
}

/// `pixCountPixelsByRow` (`pix3.c:2142-2167`) — one foreground-pixel count
/// per row. Not a named leaf of its own (the manifest's leaf table lists it,
/// but it is SKIP'd here as a public deliverable — it exists purely as
/// [`find_differential_square_sum`]'s prerequisite). Leptonica's `tab8`
/// lookup table is a speed device over the SAME bit-count this crate
/// computes directly on its own byte-per-pixel convention — behavior, not
/// the bit-trick, is the established standard for every reduce/expand/count
/// leaf in this crate (see `binreduce.rs`).
///
/// `binary` is in THIS CRATE's convention: `0 = ON` (see the module doc).
fn count_pixels_by_row(binary: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut na = Vec::with_capacity(h);
    for y in 0..h {
        let mut count: i32 = 0;
        for x in 0..w {
            if binary[y * w + x] == 0 {
                count += 1;
            }
        }
        na.push(count as f32);
    }
    na
}

/// `pixFindDifferentialSquareSum` (`skew.c:1111-1155`) — the score the D4
/// sweep + binary-search detector maximizes (not yet built; see the module
/// doc): the sum of squared differences between consecutive per-row
/// foreground-pixel counts, over an inner band of rows that skips a margin
/// at the top and bottom (so a spurious signal from an all-black band at
/// either edge doesn't dominate — `skew.c`'s own comment).
///
/// `binary` is in THIS CRATE'S convention (`0` = ON/foreground, matching
/// `binreduce.rs`) — NOT leptonica's native `1 = ON`. Convert at the caller
/// boundary (see the module doc); this function never silently assumes
/// leptonica's polarity.
///
/// # Precision (do not "improve" any of this — see
/// `.claude/harvest/leptonica-skew-callgraph.txt` STEP 4 items 9-10)
/// - `skiph = (l_int32)(0.05 * w)`: `0.05` is an f64 literal in C, `w`
///   promotes to f64, the PRODUCT is computed in **f64**, THEN truncated
///   toward zero — `(0.05_f64 * w as f64) as i32`, NOT an all-f32
///   computation (there is no f32-typed operand here to force a narrower
///   conversion, unlike item 8's `minthresh`).
/// - `skip`/`nskip` are pure integer arithmetic (truncating division),
///   `nskip >= 1` always.
/// - The accumulation (`sum += diff * diff`) is **sequential f32**, in row
///   order (`nskip .. n-nskip`, where `n == h`) — an `f64` accumulator would
///   NOT match the C oracle bit-for-bit on images with enough rows for f32
///   rounding to accumulate differently than f64-then-narrow would.
///
/// # Panics
/// Panics if `binary.len() != w * h`.
#[must_use]
pub fn find_differential_square_sum(binary: &[u8], w: usize, h: usize) -> f32 {
    assert_eq!(binary.len(), w * h, "binary buffer is not w·h");
    let na = count_pixels_by_row(binary, w, h);

    // skiph = (l_int32)(0.05 * w) -- f64 product, THEN truncated toward zero.
    let skiph = (0.05_f64 * w as f64) as i32;
    // skip = L_MIN(h / 10, skiph) -- pure integer, truncating division.
    let skip = (h as i32 / 10).min(skiph);
    // nskip = L_MAX(skip / 2, 1) -- pure integer, truncating division.
    let nskip = (skip / 2).max(1);

    let n = na.len() as i32; // == h: one count per row, unconditionally.
    let mut sum: f32 = 0.0; // Sequential f32 accumulation -- row order matters.
    let mut i = nskip;
    while i < n - nskip {
        let diff = na[i as usize] - na[(i - 1) as usize];
        sum += diff * diff;
        i += 1;
    }
    sum
}

/// `MinDiffFromHalfPi` (`shear.c:64`) — the shear angle must not get closer
/// than this (radians) to `±π/2` (`tan` blows up there); enforced by
/// [`normalize_angle_for_shear`].
const MIN_DIFF_FROM_HALF_PI: f32 = 0.04;

/// `normalizeAngleForShear` (`shear.c:831-854`, `static` in C, hence private
/// here too) — folds `radang` into `[-pi2, pi2]` (`pi2` computed from ITS OWN
/// literal, `3.14159265/2.0`, a DIFFERENT digit count than `skew.c`'s
/// `deg2rad` literal `3.1415926535` — see
/// `.claude/harvest/leptonica-skew-callgraph.txt` STEP 4 item 2; never
/// substitute a shared `PI` constant for either), then clamps within
/// `mindif` of either edge.
///
/// # Precision
/// `pi2 = (3.14159265_f64 / 2.0_f64) as f32` — an f64 division narrowed to
/// f32 ONCE. The fold branch (`radang < -pi2 || radang > pi2`) is
/// realistically dead code on this wave's angle range (always well inside
/// `±pi/2`) but is transcribed faithfully since it is reachable in
/// principle; its own arithmetic is plain f32 throughout (no double literal
/// appears in that branch, unlike `pi2`'s own computation above it).
fn normalize_angle_for_shear(radang: f32, mindif: f32) -> f32 {
    // `shear.c:840`'s literal is `3.14159265` — which is NOT `f64::consts::PI`
    // (`0x400921fb53c8d4f1` vs `0x400921fb54442d18`; audit §2 is right that the
    // two differ in f64). Substituting the constant is nonetheless free HERE,
    // and only here, **because of the `as f32`**: after the narrowing both
    // spellings collapse to the identical f32 (verified by bit comparison, not
    // assumed). Written as the constant because clippy's `approx_constant`
    // rejects the literal and this repo forbids `#[allow]`.
    //
    // If a future change ever removes the `as f32` and keeps this in f64, the
    // substitution silently becomes WRONG and only a parity diff would catch
    // it. See `.claude/plans/deskew-wave-v1.md` § "The pi-literal rule".
    let pi2 = (core::f64::consts::PI / 2.0_f64) as f32;
    let mut radang = radang;
    if radang < -pi2 || radang > pi2 {
        // C integer-truncating division -- realistically dead code on this
        // wave's angle range, transcribed faithfully regardless.
        radang -= ((radang / pi2) as i32) as f32 * pi2;
    }
    if radang > pi2 - mindif {
        radang = pi2 - mindif;
    } else if radang < -pi2 + mindif {
        radang = -pi2 + mindif;
    }
    radang
}

/// One column-band copy from `pixVShear`'s `pixRasterop` calls
/// (`shear.c:288-313`): copies columns `[dx, dx+dw)` of `src` into the SAME
/// columns of `dst`, each row shifted vertically by `dy`
/// (`dst[row][x] = src[row - dy][x]`, only where `row - dy` is a valid
/// source row) — unwritten destination cells keep whatever `dst` already
/// held (the incolor fill, or an earlier band's copy).
///
/// This is `pixRasterop`'s clip-to-intersection behavior
/// (`.claude/harvest/leptonica-rop-callgraph.txt`, the precedent `morph.rs`'s
/// `dilate_rect`/`erode_rect` already follow) specialized to the ONE shape
/// every call site in `pixVShear` actually uses: source column range equal
/// to the destination's (`sx == dx`, never horizontally shifted), a
/// full-height request (`dh == h`), and a zero-based source row (`sy == 0`)
/// — algebraically verified equivalent to the general two-sided clip
/// (`if (dx<0) {...}` / `dhangw` / `shangw`, mirrored for y) for exactly
/// this parameter shape, not re-derived by guesswork.
fn v_shear_copy_band(dst: &mut [u8], src: &[u8], w: i32, h: i32, dx: i32, dw: i32, dy: i32) {
    if dw <= 0 {
        return;
    }
    let x0 = dx.max(0);
    let x1 = (dx + dw).min(w);
    if x0 >= x1 {
        return;
    }
    for row in 0..h {
        let src_row = row - dy;
        if src_row < 0 || src_row >= h {
            continue;
        }
        for x in x0..x1 {
            dst[(row * w + x) as usize] = src[(src_row * w + x) as usize];
        }
    }
}

/// `pixVShear` (`shear.c:237-316`), the `pixd == NULL` case only (D4 never
/// reuses a caller-supplied destination buffer or shears in place; the `IP`
/// variants — `pixVShearIP`, and by extension `pixRotateShearIP` — are an
/// explicit v1 non-goal per `.claude/plans/deskew-wave-v1.md`). Pivots about
/// the vertical line `x == xloc`: columns to the right shift downward for a
/// positive angle, columns to the left shift upward.
///
/// `binary` is in THIS CRATE's convention (`0 = ON`, matching every other
/// leaf in this module); `fill` is the raw byte used for the `incolor`
/// background fill IN THIS CRATE's convention (`255` for `L_BRING_IN_WHITE`
/// — the only value the D4 sweep ever passes; `0` for `L_BRING_IN_BLACK`,
/// never exercised by the sweep), mirroring how [`rotate_am_gray`] takes
/// `grayval` directly rather than an incolor enum.
///
/// # Precision (do not "improve" any of this)
/// - `tanangle = tan(radang)`: `radang` (f32) promotes to f64 for the C
///   `tan()` call, which returns f64, narrowed to f32 ONCE on assignment —
///   `f64::from(radang).tan() as f32`, NOT `radang.tan()`.
/// - `invangle = |1. / tanangle|`: `1.` is a C double literal, so `tanangle`
///   (f32) promotes BACK to f64 for the division, narrowed to f32 ONCE again
///   — `(1.0_f64 / f64::from(tanangle)).abs() as f32`. Two separate
///   promote-compute-narrow steps, not one.
/// - `initxincr = (l_int32)(invangle / 2.)`: dividing by exactly 2 is exact
///   in any binary floating-point precision (halves the exponent, mantissa
///   untouched, no underflow at these magnitudes), so computing this via f64
///   (as C's `2.` literal forces) or plain f32 gives the bit-identical
///   truncated integer either way — written here as f64 to mirror the C
///   promotion exactly, not because the precision matters at this one site.
/// - The per-band `xincr` formulas (`(l_int32)(invangle * (vshift ± 0.5) +
///   0.5) - (...)`) promote `vshift` (i32) to f64 via the `0.5` double
///   literal, multiply by `invangle` (f32, promoted to f64 to match), add
///   `0.5` (f64), THEN truncate toward zero to i32 — computed here entirely
///   in f64 before the final cast, matching C's promotion chain.
/// - `sign = L_SIGN(radang)` (`environ.h:253`, `(x<0)?-1:1`) and every
///   subsequent `sign*vshift` combination are PURE INTEGER arithmetic in the
///   C (`sign`, `vshift`, `xincr`, `x`, `xloc` are all `l_int32`) — no float
///   at all once `invangle`/`tanangle` are fixed.
///
/// # Panics
/// Panics if `binary.len() != w * h`.
fn v_shear(binary: &[u8], w: usize, h: usize, xloc: i32, radang: f32, fill: u8) -> Vec<u8> {
    assert_eq!(binary.len(), w * h, "binary buffer is not w·h");
    let (wi, hi) = (w as i32, h as i32);

    let radang = normalize_angle_for_shear(radang, MIN_DIFF_FROM_HALF_PI);
    if radang == 0.0 || f64::from(radang).tan() == 0.0 {
        return binary.to_vec();
    }

    let mut out = vec![fill; w * h];

    let sign: i32 = if radang < 0.0 { -1 } else { 1 };
    let tanangle: f32 = f64::from(radang).tan() as f32;
    let invangle: f32 = (1.0_f64 / f64::from(tanangle)).abs() as f32;
    let initxincr = (f64::from(invangle) / 2.0_f64) as i32;

    // Central band: columns [xloc-initxincr, xloc+initxincr), zero shift.
    v_shear_copy_band(&mut out, binary, wi, hi, xloc - initxincr, 2 * initxincr, 0);

    // Walk rightward (increasing x), growing vshift.
    let mut vshift = 1i32;
    let mut x = xloc + initxincr;
    while x < wi {
        let mut xincr = (f64::from(invangle) * (f64::from(vshift) + 0.5) + 0.5) as i32 - (x - xloc);
        if wi - x < xincr {
            xincr = wi - x; // reduce for last one if req'd
        }
        v_shear_copy_band(&mut out, binary, wi, hi, x, xincr, sign * vshift);
        x += xincr;
        vshift += 1;
    }

    // Walk leftward (decreasing x), vshift growing more negative.
    let mut vshift = -1i32;
    let mut x = xloc - initxincr;
    while x > 0 {
        let mut xincr = (x - xloc) - (f64::from(invangle) * (f64::from(vshift) - 0.5) + 0.5) as i32;
        if x < xincr {
            xincr = x; // reduce for last one if req'd
        }
        v_shear_copy_band(&mut out, binary, wi, hi, x - xincr, xincr, sign * vshift);
        x -= xincr;
        vshift -= 1;
    }

    out
}

/// `pixVShearCorner` (`shear.c:370-382`) — [`v_shear`] pivoted at
/// `xloc = 0` (the UL corner). The pivot the D4 sweep uses by default (and
/// the only one `pixFindSkew`'s real call chain ever reaches).
///
/// # Panics
/// Panics if `binary.len() != w * h`.
#[must_use]
pub fn v_shear_corner(binary: &[u8], w: usize, h: usize, radang: f32, fill: u8) -> Vec<u8> {
    v_shear(binary, w, h, 0, radang, fill)
}

/// `pixVShearCenter` (`shear.c:432-444`) — [`v_shear`] pivoted at
/// `xloc = w / 2` (`pixGetWidth(pixs) / 2`, integer division). Only reached
/// through `pixFindSkewSweepAndSearchScorePivot`'s explicit
/// `L_SHEAR_ABOUT_CENTER` — `pixFindSkew` itself never selects it.
///
/// # Panics
/// Panics if `binary.len() != w * h`.
#[must_use]
pub fn v_shear_center(binary: &[u8], w: usize, h: usize, radang: f32, fill: u8) -> Vec<u8> {
    let xloc = (w as i32) / 2;
    v_shear(binary, w, h, xloc, radang, fill)
}

/// `pixZero` (`pix3.c`, ~1794-1855) — true iff every pixel is background
/// (this crate's convention: byte `255`). Only the "is the whole image
/// empty" query [`find_skew_sweep_and_search_score_pivot`] needs is
/// implemented here; not a public leaf of its own (mirrors
/// `count_pixels_by_row`'s precedent — a NEW-LEAF per the manifest that
/// nonetheless stays private, being purely a prerequisite).
fn pix_zero(binary: &[u8]) -> bool {
    binary.iter().all(|&b| b == 255)
}

/// `numaGetMax` (`numafunc1.c:495-527`) — linear scan, sentinel-seeded
/// exactly as the C (`maxval = -1000000000.`), strict `>` comparison so the
/// FIRST occurrence of the maximum wins on ties. `None` for an empty slice
/// (the C errors on `numaGetCount(na)==0`).
fn numa_get_max(scores: &[f32]) -> Option<(f32, usize)> {
    if scores.is_empty() {
        return None;
    }
    let mut maxval = -1_000_000_000.0_f32;
    let mut imaxloc = 0usize;
    for (i, &val) in scores.iter().enumerate() {
        if val > maxval {
            maxval = val;
            imaxloc = i;
        }
    }
    Some((maxval, imaxloc))
}

/// `numaGetMin` (`numafunc1.c:452-484`) — the minimum-seeking mirror of
/// [`numa_get_max`] (`minval = +1000000000.`, strict `<`).
fn numa_get_min(scores: &[f32]) -> Option<(f32, usize)> {
    if scores.is_empty() {
        return None;
    }
    let mut minval = 1_000_000_000.0_f32;
    let mut iminloc = 0usize;
    for (i, &val) in scores.iter().enumerate() {
        if val < minval {
            minval = val;
            iminloc = i;
        }
    }
    Some((minval, iminloc))
}

/// `L_SHEAR_ABOUT_CORNER` / `L_SHEAR_ABOUT_CENTER` — which fixed point
/// [`find_skew_sweep_and_search_score_pivot`]'s vertical shear pivots about
/// (`skew.c`'s own doc: corner discriminates small angles better; center
/// loses less image at large angles). `pixFindSkew`'s real call chain always
/// uses `Corner`; `Center` exists only for direct callers of
/// `pixFindSkewSweepAndSearchScorePivot`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShearPivot {
    /// `L_SHEAR_ABOUT_CORNER` — pivot at the UL corner (`xloc = 0`).
    Corner,
    /// `L_SHEAR_ABOUT_CENTER` — pivot at the image center (`xloc = w / 2`).
    Center,
}

/// Dispatches to [`v_shear_corner`] or [`v_shear_center`] by [`ShearPivot`].
fn shear_pivot(
    pivot: ShearPivot,
    binary: &[u8],
    w: usize,
    h: usize,
    radang: f32,
    fill: u8,
) -> Vec<u8> {
    match pivot {
        ShearPivot::Corner => v_shear_corner(binary, w, h, radang, fill),
        ShearPivot::Center => v_shear_center(binary, w, h, radang, fill),
    }
}

/// `MinscoreThreshFactor` (`skew.c:131`) — scales `width² · height` into the
/// minimum-score threshold gating [`find_skew_sweep_and_search_score_pivot`]'s
/// confidence.
const MINSCORE_THRESH_FACTOR: f32 = 0.000_002;

/// `MinValidMaxscore` (`skew.c:126`) — below this, confidence is forced to
/// `0.0` regardless of the max/min ratio.
const MIN_VALID_MAXSCORE: f32 = 10_000.0;

/// Result of [`find_skew_sweep_and_search_score_pivot`]: `angle` (degrees —
/// "the negative of the skew angle... the angle required for deskew,
/// clockwise rotations are positive", `skew.c:369-371` verbatim), `conf`
/// (confidence, `0.0` when not trustworthy), `endscore` (the objective at
/// the returned angle — `pendscore` in the C API).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewResult {
    /// Angle required to deskew, in degrees.
    pub angle: f32,
    /// Confidence (ratio of max/min score); `0.0` when not trustworthy.
    pub conf: f32,
    /// The objective (`find_differential_square_sum`'s score) at `angle`.
    pub endscore: f32,
}

/// `pixFindSkewSweepAndSearchScorePivot` (`skew.c:665-973`) — D4, the real
/// detector `pixFindSkew` ultimately calls (NOT the standalone, quadratic-fit
/// `pixFindSkewSweep`, which sits off `pixFindSkew`'s call path entirely —
/// see the manifest's STEP 1). A coarse sweep, scored by
/// [`find_differential_square_sum`] on a [`v_shear_corner`]/
/// [`v_shear_center`]-sheared, rank-reduced copy of `binary`, its RAW
/// maximum taken via [`numa_get_max`] (no quadratic fit), followed by an
/// interval-halving binary search at a finer reduction.
///
/// `binary` is in THIS CRATE's convention (`0 = ON`); `w`/`h` are its
/// dimensions. `redsweep`/`redsearch` must each be `1 | 2 | 4 | 8`, with
/// `redsearch <= redsweep`.
///
/// # Return contract — read before treating `None` as "the angle is 0"
/// The C returns `l_ok` (`0` = OK, `1` = error) and unconditionally
/// initializes `*pangle = *pconf = *pendscore = 0.0` at entry, before any
/// validation. Two DIFFERENT outcomes both look like "zeroes" and must NOT
/// collapse into the same Rust variant (manifest STEP 1's GUARD note, STEP 4
/// item 4):
///
/// - **`None`** — the TRUE error paths only: invalid `redsweep`/`redsearch`,
///   `redsearch > redsweep`, `pixZero`-detected all-background `binary`, or a
///   downstream rank reduction failing (mirrors the C's `!pixsch || !pixsw`
///   check).
/// - **`Some(SkewResult { angle: 0.0, conf: 0.0, endscore: 0.0 })`** — "max
///   found at sweep edge". A SILENT DEGENERATE **SUCCESS** in the C (`ret` is
///   never reassigned off its initial `0`) — returning `None` here would
///   misrepresent the C's own contract.
///
/// # Precision (do not "improve" any of this)
/// - `deg2rad = (3.1415926535_f64 / 180.0_f64) as f32` — an f64 division
///   narrowed to f32 ONCE; textually distinct from `shear.c`'s `pi2` literal
///   (one fewer digit, a different purpose) — never substitute a shared
///   `PI` constant for either.
/// - `nangles = ((2.0_f64 * sweeprange) / sweepdelta + 1.0) as i32` —
///   entirely f64 (`2.` is a C double literal, forcing the whole chain).
/// - `minthresh = MINSCORE_THRESH_FACTOR * width * width * height`, computed
///   **entirely in f32, left to right** (`width`/`height` are `pixsch`'s —
///   the SEARCH-resolution image's — dimensions, promoted to f32 to match
///   the f32 literal, NOT f64) — floating multiply is not associative; do
///   not reorder, and do not compute in f64 and narrow after.
/// - The `maxscore` used for both the confidence ratio and the
///   `< MIN_VALID_MAXSCORE` gate is whatever a SINGLE mutable binding last
///   held across BOTH phases: the sweep's [`numa_get_max`] result, UNLESS the
///   binary-search loop executes at least once, in which case it is that
///   loop's last 3-way local max. The default parameters
///   (`sweepdelta=1.0`, `minbsdelta=0.01`) always run the loop at least once,
///   which is why this carry-over is easy to miss.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "faithful transcode of pixFindSkewSweepAndSearchScorePivot's full parameter list"
)]
pub fn find_skew_sweep_and_search_score_pivot(
    binary: &[u8],
    w: usize,
    h: usize,
    redsweep: u32,
    redsearch: u32,
    sweepcenter: f32,
    sweeprange: f32,
    sweepdelta: f32,
    minbsdelta: f32,
    pivot: ShearPivot,
) -> Option<SkewResult> {
    assert_eq!(binary.len(), w * h, "binary buffer is not w·h");
    let reduction_ok = |r: u32| matches!(r, 1 | 2 | 4 | 8);
    if !reduction_ok(redsweep) || !reduction_ok(redsearch) || redsearch > redsweep {
        return None;
    }

    // `deg2rad = 3.1415926535 / 180.` (skew.c) — an f64 division narrowed to
    // f32 ONCE. That literal is NOT `f64::consts::PI` either
    // (`0x400921fb54411744` vs `0x400921fb54442d18`), but as at
    // `normalize_angle_for_shear`'s `pi2`, the `as f32` discards the
    // difference: both spellings give the identical f32 (verified by bit
    // comparison). Constant rather than literal only because clippy's
    // `approx_constant` rejects the latter; the equality holds ONLY while this
    // stays narrowed to f32.
    let deg2rad = (core::f64::consts::PI / 180.0_f64) as f32;

    // Reduced image for the binary search (pixsch).
    let (sch_buf, sch_w, sch_h) = match redsearch {
        1 => (binary.to_vec(), w, h),
        2 => reduce_rank_binary_cascade(binary, w, h, [1, 0, 0, 0])?,
        4 => reduce_rank_binary_cascade(binary, w, h, [1, 1, 0, 0])?,
        _ => reduce_rank_binary_cascade(binary, w, h, [1, 1, 2, 0])?, // 8
    };
    if pix_zero(&sch_buf) {
        return None;
    }

    // Further-reduced image for the coarse sweep (pixsw). ratio is always in
    // {1,2,4,8} given the validated redsweep/redsearch above.
    let ratio = redsweep / redsearch;
    let (sw_buf, sw_w, sw_h) = match ratio {
        1 => (sch_buf.clone(), sch_w, sch_h),
        2 => reduce_rank_binary_cascade(&sch_buf, sch_w, sch_h, [1, 0, 0, 0])?,
        // NOTE: asymmetric vs redsearch==4's [1,1,0,0] above -- verified
        // against skew.c:733-737, not a transcription slip.
        4 => reduce_rank_binary_cascade(&sch_buf, sch_w, sch_h, [1, 2, 0, 0])?,
        _ => reduce_rank_binary_cascade(&sch_buf, sch_w, sch_h, [1, 2, 2, 0])?, // 8
    };

    // nangles = (l_int32)((2.*sweeprange)/sweepdelta + 1) -- f64 throughout.
    let nangles = ((2.0_f64 * f64::from(sweeprange)) / f64::from(sweepdelta) + 1.0) as i32;
    if nangles <= 0 {
        return None;
    }
    let nangles = nangles as usize;

    // ---- Sweep (coarse), on pixsw. ----
    let rangeleft = sweepcenter - sweeprange;
    let mut natheta: Vec<f32> = Vec::with_capacity(nangles);
    let mut nascore: Vec<f32> = Vec::with_capacity(nangles);
    for i in 0..nangles {
        let theta = rangeleft + (i as f32) * sweepdelta;
        let sheared = shear_pivot(pivot, &sw_buf, sw_w, sw_h, deg2rad * theta, 255);
        nascore.push(find_differential_square_sum(&sheared, sw_w, sw_h));
        natheta.push(theta);
    }

    let (mut maxscore, maxindex) = numa_get_max(&nascore)?;
    let maxangle = natheta[maxindex];

    // "max found at sweep edge" -- silent degenerate SUCCESS, not an error.
    let n = natheta.len();
    if maxindex == 0 || maxindex == n - 1 {
        return Some(SkewResult {
            angle: 0.0,
            conf: 0.0,
            endscore: 0.0,
        });
    }

    // ---- Binary search (fine), on pixsch. Fresh arrays -- the sweep's are
    // done with (mirrors the C's numaEmpty-and-reuse). ----
    let mut centerangle = maxangle;
    let mut bsearchscore = [0.0_f32; 5];
    bsearchscore[2] = find_differential_square_sum(
        &shear_pivot(pivot, &sch_buf, sch_w, sch_h, deg2rad * centerangle, 255),
        sch_w,
        sch_h,
    );
    bsearchscore[0] = find_differential_square_sum(
        &shear_pivot(
            pivot,
            &sch_buf,
            sch_w,
            sch_h,
            deg2rad * (centerangle - sweepdelta),
            255,
        ),
        sch_w,
        sch_h,
    );
    bsearchscore[4] = find_differential_square_sum(
        &shear_pivot(
            pivot,
            &sch_buf,
            sch_w,
            sch_h,
            deg2rad * (centerangle + sweepdelta),
            255,
        ),
        sch_w,
        sch_h,
    );

    let mut natheta: Vec<f32> = vec![
        centerangle,
        centerangle - sweepdelta,
        centerangle + sweepdelta,
    ];
    let mut nascore: Vec<f32> = vec![bsearchscore[2], bsearchscore[0], bsearchscore[4]];

    let mut delta = 0.5_f32 * sweepdelta;
    while delta >= minbsdelta {
        let leftcenterangle = centerangle - delta;
        bsearchscore[1] = find_differential_square_sum(
            &shear_pivot(
                pivot,
                &sch_buf,
                sch_w,
                sch_h,
                deg2rad * leftcenterangle,
                255,
            ),
            sch_w,
            sch_h,
        );
        nascore.push(bsearchscore[1]);
        natheta.push(leftcenterangle);

        let rightcenterangle = centerangle + delta;
        bsearchscore[3] = find_differential_square_sum(
            &shear_pivot(
                pivot,
                &sch_buf,
                sch_w,
                sch_h,
                deg2rad * rightcenterangle,
                255,
            ),
            sch_w,
            sch_h,
        );
        nascore.push(bsearchscore[3]);
        natheta.push(rightcenterangle);

        // Max of the CENTER THREE (indices 1..=3) only -- 0 and 4 excluded.
        maxscore = bsearchscore[1];
        let mut maxindex = 1usize;
        for (i, &score) in bsearchscore.iter().enumerate().take(4).skip(2) {
            if score > maxscore {
                maxscore = score;
                maxindex = i;
            }
        }

        let lefttemp = bsearchscore[maxindex - 1];
        let righttemp = bsearchscore[maxindex + 1];
        bsearchscore[2] = maxscore;
        bsearchscore[0] = lefttemp;
        bsearchscore[4] = righttemp;

        centerangle += delta * ((maxindex as i32 - 2) as f32);
        delta *= 0.5;
    }

    let angle = centerangle;
    let endscore = bsearchscore[2];

    let (minscore, _minloc) = numa_get_min(&nascore)?;
    let width = sch_w as f32;
    let height = sch_h as f32;
    let minthresh = MINSCORE_THRESH_FACTOR * width * width * height;
    let mut conf = if minscore > minthresh {
        maxscore / minscore
    } else {
        0.0
    };
    if centerangle > rangeleft + 2.0_f32 * sweeprange - sweepdelta
        || centerangle < rangeleft + sweepdelta
        || maxscore < MIN_VALID_MAXSCORE
    {
        conf = 0.0;
    }

    Some(SkewResult {
        angle,
        conf,
        endscore,
    })
}

/// `MinAngleToRotate` (`rotateam.c:150`) — below this magnitude (radians,
/// ~0.06°) `pixRotateAMGray`/`pixRotateAMGrayCorner` are a no-op (the input is
/// returned unchanged). The SAME name and value are independently
/// re-declared in `rotate.c` and `rotateshear.c` (three separate
/// translation-unit-local copies of one constant, not a bug) — this is
/// `rotateam.c`'s own copy, the one that governs these two functions.
const MIN_ANGLE_TO_ROTATE: f32 = 0.001;

/// `rotateAMGrayLow` (`rotateam.c:391-448`) — the area-map rotation kernel
/// about the image CENTER. A pure per-pixel loop, no cross-pixel state.
/// `grayval` fills pixels that map outside the source image (`grayval =
/// 255` for `L_BRING_IN_WHITE` in this crate's un-inverted grey convention,
/// `0` for `L_BRING_IN_BLACK`; see the module doc).
///
/// # Precision — THE trap this leaf exists to get right
/// (`rotateam.c:412-413`, `.claude/harvest/leptonica-skew-callgraph.txt`
/// STEP 4 item 3)
/// - `sina`/`cosa` are computed in **f64** (`16.0 * angle.sin()` /
///   `.cos()`), THEN narrowed to f32 **once**, on assignment.
/// - Every per-pixel term after that (`xpm`, `ypm`) is computed in **f32**
///   end-to-end — C's usual arithmetic conversions promote the `l_int32`
///   `xdif`/`ydif` operands to match the `l_float32` `cosa`/`sina`, NOT to
///   `f64`. Getting either half wrong (all-f32 `sina`/`cosa`, or promoting
///   the per-pixel loop to f64) diverges from the C oracle in the last bit
///   on a SUBSET of angles — most angles happen to match either way, which
///   is exactly why a byte-parity harness across MANY angles (not one) is
///   required to catch this.
/// - The `(l_int32)` casts on `xpm`/`ypm` truncate toward zero: plain `as
///   i32`, never `+ 0.5` (that round-half-up convention belongs to a
///   DIFFERENT function, `pixEmbedForRotation` — copying it here would be a
///   real divergence, not an improvement).
/// - `xpm >> 4` is an arithmetic (sign-extending) right shift on a possibly
///   negative `i32`; Rust's `>>` on `i32` is arithmetic by language
///   guarantee, so it transcribes directly with no adjustment.
/// - `xpm & 0x0f` / `ypm & 0x0f`: two's-complement bitwise AND, so the
///   result is always `0..=15` regardless of the operand's sign — identical
///   in C and Rust.
fn rotate_am_gray_low(grey: &[u8], w: usize, h: usize, angle: f32, grayval: u8) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let xcen = (w as i32) / 2;
    let wm2 = (w as i32) - 2;
    let ycen = (h as i32) / 2;
    let hm2 = (h as i32) - 2;
    let cosa = (16.0_f64 * f64::from(angle).cos()) as f32;
    let sina = (16.0_f64 * f64::from(angle).sin()) as f32;

    for i in 0..(h as i32) {
        let ydif = ycen - i;
        for j in 0..(w as i32) {
            let xdif = xcen - j;
            let xpm = (-(xdif as f32) * cosa - (ydif as f32) * sina) as i32;
            let ypm = (-(ydif as f32) * cosa + (xdif as f32) * sina) as i32;
            let xp = xcen + (xpm >> 4);
            let yp = ycen + (ypm >> 4);
            let xf = xpm & 0x0f;
            let yf = ypm & 0x0f;

            let out_idx = (i as usize) * w + (j as usize);
            if xp < 0 || yp < 0 || xp > wm2 || yp > hm2 {
                out[out_idx] = grayval;
                continue;
            }

            let (xpu, ypu) = (xp as usize, yp as usize);
            let v00 = (16 - xf) * (16 - yf) * i32::from(grey[ypu * w + xpu]);
            let v10 = xf * (16 - yf) * i32::from(grey[ypu * w + xpu + 1]);
            let v01 = (16 - xf) * yf * i32::from(grey[(ypu + 1) * w + xpu]);
            let v11 = xf * yf * i32::from(grey[(ypu + 1) * w + xpu + 1]);
            out[out_idx] = ((v00 + v01 + v10 + v11 + 128) / 256) as u8;
        }
    }
    out
}

/// `pixRotateAMGray` (`rotateam.c:288-317`) — rotate an 8bpp grey buffer
/// about its CENTER by `angle` radians (clockwise positive). Below
/// [`MIN_ANGLE_TO_ROTATE`] this is a no-op (returns the input unchanged,
/// matching the C wrapper's `pixClone`). Output is the SAME `w × h` as the
/// input — leptonica's area-map rotate does not expand the canvas; corners
/// that rotate off-image are cropped and `grayval`-filled (see the module
/// doc — reproducing this crop IS the parity contract, not a bug to fix).
///
/// # Panics
/// Panics if `grey.len() != w * h`.
#[must_use]
pub fn rotate_am_gray(grey: &[u8], w: usize, h: usize, angle: f32, grayval: u8) -> Vec<u8> {
    assert_eq!(grey.len(), w * h, "grey buffer is not w·h");
    if angle.abs() < MIN_ANGLE_TO_ROTATE {
        return grey.to_vec();
    }
    rotate_am_gray_low(grey, w, h, angle, grayval)
}

/// `rotateAMGrayCornerLow` (`rotateam.c:683-736`) — the UL-CORNER-pivot
/// sibling of [`rotate_am_gray_low`]: identical interpolation math, but the
/// source-position formula has NO `xcen`/`ycen` offset (`rotateam.c:708-709`
/// vs `:420-421`) — the pivot is the origin, not the image centre. The same
/// precision rules apply; see [`rotate_am_gray_low`]'s docs.
fn rotate_am_gray_corner_low(grey: &[u8], w: usize, h: usize, angle: f32, grayval: u8) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let wm2 = (w as i32) - 2;
    let hm2 = (h as i32) - 2;
    let cosa = (16.0_f64 * f64::from(angle).cos()) as f32;
    let sina = (16.0_f64 * f64::from(angle).sin()) as f32;

    for i in 0..(h as i32) {
        for j in 0..(w as i32) {
            let xpm = ((j as f32) * cosa + (i as f32) * sina) as i32;
            let ypm = ((i as f32) * cosa - (j as f32) * sina) as i32;
            let xp = xpm >> 4;
            let yp = ypm >> 4;
            let xf = xpm & 0x0f;
            let yf = ypm & 0x0f;

            let out_idx = (i as usize) * w + (j as usize);
            if xp < 0 || yp < 0 || xp > wm2 || yp > hm2 {
                out[out_idx] = grayval;
                continue;
            }

            let (xpu, ypu) = (xp as usize, yp as usize);
            let v00 = (16 - xf) * (16 - yf) * i32::from(grey[ypu * w + xpu]);
            let v10 = xf * (16 - yf) * i32::from(grey[ypu * w + xpu + 1]);
            let v01 = (16 - xf) * yf * i32::from(grey[(ypu + 1) * w + xpu]);
            let v11 = xf * yf * i32::from(grey[(ypu + 1) * w + xpu + 1]);
            out[out_idx] = ((v00 + v01 + v10 + v11 + 128) / 256) as u8;
        }
    }
    out
}

/// `pixRotateAMGrayCorner` (`rotateam.c:584-613`) — same contract as
/// [`rotate_am_gray`] but pivoting about the UL corner instead of the
/// centre.
///
/// # Panics
/// Panics if `grey.len() != w * h`.
#[must_use]
pub fn rotate_am_gray_corner(grey: &[u8], w: usize, h: usize, angle: f32, grayval: u8) -> Vec<u8> {
    assert_eq!(grey.len(), w * h, "grey buffer is not w·h");
    if angle.abs() < MIN_ANGLE_TO_ROTATE {
        return grey.to_vec();
    }
    rotate_am_gray_corner_low(grey, w, h, angle, grayval)
}

/// `DefaultSweepRange` (`skew.c:106`) — degrees, half the D6/D7 sweep's full
/// range about `sweepcenter = 0.0` (see
/// [`find_skew_sweep_and_search_score_pivot`]).
const DEFAULT_SWEEP_RANGE: f32 = 7.0;

/// `DefaultSweepDelta` (`skew.c:107`) — degrees, the coarse sweep's angle
/// step.
const DEFAULT_SWEEP_DELTA: f32 = 1.0;

/// `DefaultMinbsDelta` (`skew.c:113`) — degrees. [`deskew_general`] has no
/// `minbsdelta` PARAMETER at all (unlike
/// [`find_skew_sweep_and_search_score_pivot`], which does) — every call into
/// D4 from this leaf uses this exact constant, never a caller-supplied
/// value (verified against `skew.c:334-336`: the one call site passes
/// `DefaultMinbsDelta` literally).
const DEFAULT_MINBS_DELTA: f32 = 0.01;

/// `DefaultSweepReduction` (`skew.c:116`).
const DEFAULT_SWEEP_REDUCTION: u32 = 4;

/// `DefaultBsReduction` (`skew.c:117`).
const DEFAULT_BS_REDUCTION: u32 = 2;

/// `DefaultBinaryThreshold` (`skew.c:134`) — the fixed threshold
/// [`pix_convert_to_1`] uses when the caller passes `0`.
const DEFAULT_BINARY_THRESHOLD: i32 = 130;

/// `MinDeskewAngle` (`skew.c:120`) — DEGREES, compared directly against the
/// D4 detector's own degree-valued `angle` (no radian conversion at this
/// gate — the conversion only happens afterward, for the rotation itself).
///
/// Plain `f32` literal, not narrowed from an `f64` intermediate — unlike the
/// `deg2rad`/`pi2` sites elsewhere in this module, this one does NOT need
/// the "narrow once from f64" treatment: `0.1_f32` (Rust's direct
/// decimal-to-f32 literal parse) and `(0.1_f64 as f32)` (C's
/// decimal-to-double-then-narrow-to-float path for `static const l_float32
/// MinDeskewAngle = 0.1;`) were checked bit-for-bit (via exact rational
/// comparison against the surrounding f32 grid, not merely "commonly cited
/// as safe") and agree: both round to `0x3dcccccd`. [`MIN_ALLOWED_CONFIDENCE`]
/// was checked the same way (`0x40400000` either path) — the same
/// discipline the plan's "pi-literal rule" applies to the `deg2rad`/`pi2`
/// sites, just with the opposite conclusion (safe here, NOT safe there).
const MIN_DESKEW_ANGLE: f32 = 0.1;

/// `MinAllowedConfidence` (`skew.c:123`) — see [`MIN_DESKEW_ANGLE`]'s doc for
/// why a plain `f32` literal is verified safe here.
const MIN_ALLOWED_CONFIDENCE: f32 = 3.0;

/// `pixConvertTo1(pixs, thresh)` (`pixconv.c:3025-3064`), specialized to
/// THIS crate's always-8bpp-grey domain: for `d == 8`, `pixConvertTo1` is
/// exactly `pixConvertTo8` (a no-op, since we are already 8bpp) followed by
/// `pixThresholdToBinary(pixg, thresh)` (`grayquant.c:446-488`), whose own
/// doc comment is verbatim: "If the source pixel is less than the threshold
/// value, the dest will be 1 [ON]; otherwise, it will be 0 [background]"
/// (`grayquant.c:438-439` — strict `<`, not `<=`, confirmed by reading
/// `thresholdToBinaryLineLow`'s actual per-pixel comparison, not just the
/// doc comment). THIS CRATE's binary convention inverts leptonica's bit
/// (`0 = ON`, matching every other buffer in this module), so: `grey[i] <
/// thresh -> 0 (ON)`, else `255 (background)`.
///
/// `pixThresholdToBinary` itself rejects `thresh < 0` or (for `d == 8`)
/// `thresh > 256` as a defensive/error path, which would cascade a NULL
/// through `pixConvertTo1` and ultimately surface as [`deskew_general`]'s
/// own "D4 detector failed" outcome. NOT reproduced: every realistic caller
/// passes `thresh == 0` (→ [`DEFAULT_BINARY_THRESHOLD`]) or a value in the
/// sane `0..=255` range, no oracle fixture exercises an out-of-range value,
/// and a plain `<` comparison is well-defined — and behaviourally IDENTICAL
/// to the C for every value that would have been valid there — for any
/// `i32`, so the stricter defensive rejection is not worth forcing an extra
/// `Option` layer onto every caller of [`deskew_general`] for a path with no
/// falsifier.
fn pix_convert_to_1(grey: &[u8], thresh: i32) -> Vec<u8> {
    grey.iter()
        .map(|&g| if i32::from(g) < thresh { 0u8 } else { 255u8 })
        .collect()
}

/// Result of [`deskew_general`]: the possibly-rotated grey buffer, plus the
/// D4 detector's own `(angle, conf)`. `pixDeskewGeneral` makes `pangle`/
/// `pconf` optional out-parameters (`NULL` to skip, as `pixDeskew` does);
/// this Rust surface always returns them instead — the "`pixFindSkewAndDeskew`
/// shape" (that function IS `pixDeskewGeneral` with both requested) — since
/// they are computed internally regardless and a caller integrating this
/// into a pipeline (the `rectify.rs` composition, or a table-detection
/// front-end) will generally want to know whether/how much rotation
/// happened, not just the corrected pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct DeskewResult {
    /// The corrected grey buffer, or (below the gate, or on a D4-internal
    /// failure) the UNCHANGED original. Same `w × h` as the input either
    /// way — area-map rotation never expands the canvas (see the module
    /// doc's "CROPS" section).
    pub grey: Vec<u8>,
    /// Width, in pixels. Never changes from the input's own `w`.
    pub w: usize,
    /// Height, in pixels. Never changes from the input's own `h`.
    pub h: usize,
    /// The angle required to deskew, in DEGREES ("the negative of the skew
    /// angle... clockwise rotations are positive", `skew.c:369-371`
    /// verbatim, D4's own return contract). `0.0` when D4 failed internally
    /// (see [`deskew_general`]'s return-contract docs) — NOT necessarily
    /// `0.0` merely because the page was left unrotated: below the
    /// confidence/angle GATE, the REAL detected value is still reported,
    /// exactly as `pixDeskewGeneral` writes `*pangle` unconditionally
    /// before checking the gate.
    pub angle: f32,
    /// D4's confidence (ratio of max/min score). Same zero-vs-real
    /// distinction as `angle` above.
    pub conf: f32,
}

/// `pixDeskewGeneral` (`skew.c:289-351`) — D6, the full detect-then-correct
/// composition: binarize `grey` at a FIXED threshold for detection only (the
/// correction rotation, if any, runs on `grey` itself, never on this binary
/// copy — `skew.c:326-331` builds `pixb` purely to feed the detector, and
/// `skew.c:346-347` rotates `pixs`, not `pixb`), run the D4 sweep+search,
/// then gate on the result: below [`MIN_DESKEW_ANGLE`] or
/// [`MIN_ALLOWED_CONFIDENCE`], return the input unchanged; otherwise rotate
/// via [`rotate_am_gray`] (center pivot — `skew.c`'s own `pixRotate` call
/// uses `L_ROTATE_AREA_MAP`, and see this function's own docs below for why
/// that dispatch collapses to exactly `rotate_am_gray` in this crate's
/// always-grey domain).
///
/// `redsweep`/`sweeprange`/`sweepdelta`/`redsearch`/`thresh` all follow the
/// C's own "`0` (or `0.0`) means use the default" convention
/// (`skew.c:309-322`) — passing `0`/`0.0` for any of them is not itself an
/// error.
///
/// # GREY input only — a deliberate scope decision, not an oversight
/// The C's `pixDeskewGeneral` accepts "any depth" and special-cases
/// `depth == 1` (clone straight through, no `pixConvertTo1` call). This Rust
/// surface does not model that branch, for three compounding reasons:
/// 1. This crate has no packed-1bpp/colormap `PIX`-equivalent input type —
///    every other leaf in this module (and the wider crate: `image_input.rs`,
///    `xy_cut.rs`) already treats "the page" as an 8bpp grey byte-per-pixel
///    buffer, never a packed bitonal one.
/// 2. `.claude/harvest/oracles/skew_oracle.cpp`'s own `deskew` arm always
///    converts its input to 8bpp grey first (`read_grey` calls
///    `pixConvertTo8`) — there is no byte-parity evidence for the
///    `depth == 1` path at all, on either side.
/// 3. Most importantly: for a genuinely 1bpp `pixs`, `pixRotate`'s OWN type
///    dispatch (`rotate.c:113-196`,
///    `.claude/harvest/leptonica-skew-callgraph.txt` ARM B step 3)
///    UNCONDITIONALLY overrides `L_ROTATE_AREA_MAP` to `L_ROTATE_SHEAR` or
///    `L_ROTATE_SAMPLING` for 1bpp input — AREA_MAP is never reachable for a
///    1bpp `pixs` through this entry point at all. Both overrides route to
///    primitives this wave does NOT ship: `pixRotateShear`/`pixRotate2Shear`/
///    `pixRotate3Shear` need `pixHShear` (shear.c's OTHER axis — explicitly
///    NOT transcoded here, see [`v_shear_corner`]'s own doc), and
///    `pixRotateBySampling` is an explicit v1 non-goal in the plan.
///    Supporting `depth == 1` faithfully is therefore blocked on leaves
///    outside this task's scope, not merely skipped for convenience.
///
/// For an 8bpp grey `pixs` (this function's ENTIRE domain), none of
/// `pixRotate`'s dispatch machinery ever diverges from `pixRotateAMGray`:
/// there is no colormap concept to strip, `pixEmbedForRotation` is a no-op
/// (width=height=0, exactly as `skew.c:346-347` itself always passes), the
/// depth is already ≥ 8 so no `pixConvertTo8` promotion fires, and the
/// depth-1 type-override branch never applies. `fillval` is always `255`
/// (`skew.c` always passes `L_BRING_IN_WHITE`, and this crate's grey
/// convention is unflipped — see the module doc). So the entire ARM B
/// dispatch collapses to exactly one call: `rotate_am_gray(grey, w, h,
/// deg2rad * angle, 255)`.
///
/// # Return contract
/// `None` ONLY for `pixDeskewGeneral`'s OWN top-level validation failure:
/// `redsweep`/`redsearch` not in `{0, 1, 2, 4}`. **This is a STRICT SUBSET
/// of [`find_skew_sweep_and_search_score_pivot`]'s own `{1, 2, 4, 8}`** —
/// verified against `skew.c:309-312` and `:317-320` directly, not assumed
/// from D4's own range: `pixDeskewGeneral`/`pixDeskew`/`pixDeskewBoth`/
/// `pixFindSkewAndDeskew` all reject `8` even though the underlying detector
/// would happily accept it. See
/// `deskew_general_rejects_redsweep_8_even_though_d4_itself_accepts_it`
/// below, which pins this asymmetry as a locked contract.
///
/// `Some(DeskewResult { .. })` for every other outcome, including:
/// - D4's own internal failure (all-background image, via `pixZero`) —
///   `ret != 0` in the C. `angle`/`conf` are forced to `0.0`: the C
///   initializes `*pangle = *pconf = 0.0` at `pixDeskewGeneral`'s OWN entry
///   (`skew.c:305-306`), BEFORE calling D4 at all, and D4 itself ALSO
///   unconditionally zeroes them at its own entry
///   ([`find_skew_sweep_and_search_score_pivot`]'s own doc) — so this is the
///   value the caller observes either way. The buffer is the UNCHANGED
///   original (`return pixClone(pixs)`).
/// - The gate not clearing (`|angle| < MIN_DESKEW_ANGLE` or
///   `conf < MIN_ALLOWED_CONFIDENCE`) — the buffer is again the UNCHANGED
///   original, but `angle`/`conf` are the REAL detected values (the C writes
///   `*pangle = angle; *pconf = conf;` BEFORE this gate check, not after).
///   D4's own "max found at sweep edge" degenerate-success case
///   (`angle = 0.0, conf = 0.0`) flows through this SAME branch, since `0.0`
///   trivially fails both gate conditions — no special-casing needed.
/// - The gate clearing — the buffer is [`rotate_am_gray`]'s output.
///
/// # Panics
/// Panics if `grey.len() != w * h`.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "faithful transcode of pixDeskewGeneral's full parameter list"
)]
pub fn deskew_general(
    grey: &[u8],
    w: usize,
    h: usize,
    redsweep: u32,
    sweeprange: f32,
    sweepdelta: f32,
    redsearch: u32,
    thresh: i32,
) -> Option<DeskewResult> {
    assert_eq!(grey.len(), w * h, "grey buffer is not w·h");

    // "use 0 for default" -- resolved BEFORE validating, exactly as the C: a
    // caller-supplied 0 is never itself the error; only a nonzero value
    // outside {1,2,4} is (skew.c:309-312, :317-320).
    let redsweep = if redsweep == 0 {
        DEFAULT_SWEEP_REDUCTION
    } else {
        redsweep
    };
    if redsweep != 1 && redsweep != 2 && redsweep != 4 {
        return None;
    }
    let sweeprange = if sweeprange == 0.0 {
        DEFAULT_SWEEP_RANGE
    } else {
        sweeprange
    };
    let sweepdelta = if sweepdelta == 0.0 {
        DEFAULT_SWEEP_DELTA
    } else {
        sweepdelta
    };
    let redsearch = if redsearch == 0 {
        DEFAULT_BS_REDUCTION
    } else {
        redsearch
    };
    if redsearch != 1 && redsearch != 2 && redsearch != 4 {
        return None;
    }
    let thresh = if thresh == 0 {
        DEFAULT_BINARY_THRESHOLD
    } else {
        thresh
    };

    // deg2rad = 3.1415926535 / 180. -- the IDENTICAL literal/precision as
    // D4's own (skew.c re-declares this exact literal at both call sites:
    // line 324 here, line 708 in find_skew_sweep_and_search_score_pivot);
    // the narrowing-equivalence argument is the one recorded there and in
    // the plan's "pi-literal rule".
    let deg2rad = (core::f64::consts::PI / 180.0_f64) as f32;

    // Binarize for DETECTION only -- see this function's own doc for why
    // the correction rotation below runs on `grey`, never on `binary`.
    let binary = pix_convert_to_1(grey, thresh);

    // sweepcenter=0.0 and pivot=Corner are not parameters ANYWHERE on this
    // call chain -- pixFindSkewSweepAndSearch -> ...Score -> ...ScorePivot
    // hardcodes both at every hop (skew.c:573, 629, 632). minbsdelta is
    // likewise always DEFAULT_MINBS_DELTA (pixDeskewGeneral has no such
    // parameter to forward).
    let detected = find_skew_sweep_and_search_score_pivot(
        &binary,
        w,
        h,
        redsweep,
        redsearch,
        0.0,
        sweeprange,
        sweepdelta,
        DEFAULT_MINBS_DELTA,
        ShearPivot::Corner,
    );

    let Some(skew) = detected else {
        // D4's own true-error path (pixZero-detected all-background image;
        // never a redsweep/redsearch rejection, since we already validated
        // {1,2,4} above, a strict subset of D4's own {1,2,4,8}). Mirrors
        // skew.c's `ret != 0` branch.
        return Some(DeskewResult {
            grey: grey.to_vec(),
            w,
            h,
            angle: 0.0,
            conf: 0.0,
        });
    };
    let (angle, conf) = (skew.angle, skew.conf);

    if angle.abs() < MIN_DESKEW_ANGLE || conf < MIN_ALLOWED_CONFIDENCE {
        return Some(DeskewResult {
            grey: grey.to_vec(),
            w,
            h,
            angle,
            conf,
        });
    }

    // pixRotate(pixs, deg2rad*angle, L_ROTATE_AREA_MAP, L_BRING_IN_WHITE, 0, 0)
    // on the GREY original -- collapses to exactly this call in our
    // always-8bpp-grey domain; see this function's own doc for the full
    // dispatch-collapse argument.
    let rotated = rotate_am_gray(grey, w, h, deg2rad * angle, 255);
    Some(DeskewResult {
        grey: rotated,
        w,
        h,
        angle,
        conf,
    })
}

/// `pixDeskewBoth` (`skew.c:166-189`) — D7: [`deskew_general`] (with
/// `pixDeskew`'s own all-default parameters, i.e. `redsweep = 0`,
/// `sweeprange = 0.0`, `sweepdelta = 0.0`, `thresh = 0`, only `redsearch`
/// threaded through — `skew.c:222`'s own `pixDeskewGeneral(pixs, 0, 0.0,
/// 0.0, redsearch, 0, NULL, NULL)`) → [`rotate90_grey`] clockwise →
/// [`deskew_general`] again (same defaults, same `redsearch`) →
/// [`rotate90_grey`] counter-clockwise.
///
/// Two passes exist because the D4 differential-square-sum detector is
/// directional (most sensitive to HORIZONTAL banding — text baselines and
/// x-height lines): the 90-degree round trip lets the second pass see
/// whatever skew component was orthogonal to what the first pass could
/// detect, at the cost of two full detect-and-maybe-rotate cycles.
///
/// # Validation
/// `redsearch` must be `0` (→ [`DEFAULT_BS_REDUCTION`]) or exactly `1`, `2`,
/// or `4`. `pixDeskewBoth` validates this itself, ONCE, before ever calling
/// `pixDeskew` (`skew.c:176-179`). This Rust version does not duplicate that
/// upfront check — it forwards the raw `redsearch` straight through to BOTH
/// [`deskew_general`] calls instead, via `?`. This is bit-identical, not
/// merely convenient: since both calls use the SAME `redsearch` value, they
/// can never validate differently from each other, so a check at either
/// call site (or both) produces the identical `None`/`Some` outcome as
/// `pixDeskewBoth`'s own single upfront check would have.
///
/// # Return
/// `None` only for that validation failure. `Some((grey, w, h))` for every
/// other outcome — matching `pixDeskewBoth`'s own signature exactly: no
/// angle/conf out-parameters exist at this composition level in the C API
/// at all (unlike [`deskew_general`], which exposes them per the
/// `pixFindSkewAndDeskew` shape).
///
/// # Panics
/// Panics if `grey.len() != w * h`.
#[must_use]
pub fn deskew_both(
    grey: &[u8],
    w: usize,
    h: usize,
    redsearch: u32,
) -> Option<(Vec<u8>, usize, usize)> {
    assert_eq!(grey.len(), w * h, "grey buffer is not w·h");

    let pass1 = deskew_general(grey, w, h, 0, 0.0, 0.0, redsearch, 0)?;
    let (rot1, rw1, rh1) = rotate90_grey(&pass1.grey, pass1.w, pass1.h, 1);
    let pass2 = deskew_general(&rot1, rw1, rh1, 0, 0.0, 0.0, redsearch, 0)?;
    let (rot2, rw2, rh2) = rotate90_grey(&pass2.grey, pass2.w, pass2.h, -1);
    Some((rot2, rw2, rh2))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- D1: rotate90_grey ----------------------------------------------

    #[test]
    fn rotate90_hand_checked_2x3() {
        // w=2,h=3, row-major: row0=[A,B]=[1,2], row1=[C,D]=[3,4], row2=[E,F]=[5,6].
        #[rustfmt::skip]
        let grey = vec![
            1, 2,
            3, 4,
            5, 6,
        ];
        let (out, ow, oh) = rotate90_grey(&grey, 2, 3, 1); // clockwise
        assert_eq!((ow, oh), (3, 2));
        // Hand-derived (see the module doc's derivation): row0=[E,C,A], row1=[F,D,B].
        assert_eq!(out, vec![5, 3, 1, 6, 4, 2], "clockwise 2x3");

        let (out_ccw, ow2, oh2) = rotate90_grey(&grey, 2, 3, -1); // counter-clockwise
        assert_eq!((ow2, oh2), (3, 2));
        // Hand-derived: row0=[B,D,F], row1=[A,C,E].
        assert_eq!(out_ccw, vec![2, 4, 6, 1, 3, 5], "counter-clockwise 2x3");
    }

    #[test]
    fn rotate90_dims_swap() {
        let grey = vec![0u8; 4 * 7];
        let (_, ow, oh) = rotate90_grey(&grey, 4, 7, 1);
        assert_eq!((ow, oh), (7, 4));
    }

    #[test]
    fn rotate90_four_clockwise_turns_is_identity() {
        let (w, h) = (5usize, 3usize);
        let grey: Vec<u8> = (0..(w * h) as u8).collect();
        let (mut cur, mut cw, mut ch) = (grey.clone(), w, h);
        for _ in 0..4 {
            let (out, ow, oh) = rotate90_grey(&cur, cw, ch, 1);
            cur = out;
            cw = ow;
            ch = oh;
        }
        assert_eq!((cw, ch), (w, h));
        assert_eq!(cur, grey, "four clockwise quarter-turns is the identity");
    }

    #[test]
    fn rotate90_four_counter_clockwise_turns_is_identity() {
        let (w, h) = (5usize, 3usize);
        let grey: Vec<u8> = (0..(w * h) as u8).collect();
        let (mut cur, mut cw, mut ch) = (grey.clone(), w, h);
        for _ in 0..4 {
            let (out, ow, oh) = rotate90_grey(&cur, cw, ch, -1);
            cur = out;
            cw = ow;
            ch = oh;
        }
        assert_eq!((cw, ch), (w, h));
        assert_eq!(
            cur, grey,
            "four counter-clockwise quarter-turns is the identity"
        );
    }

    #[test]
    fn rotate90_clockwise_then_counter_clockwise_is_identity() {
        let (w, h) = (6usize, 4usize);
        let grey: Vec<u8> = (0..(w * h) as u8).collect();
        let (cw_out, cww, cwh) = rotate90_grey(&grey, w, h, 1);
        let (back, bw, bh) = rotate90_grey(&cw_out, cww, cwh, -1);
        assert_eq!((bw, bh), (w, h));
        assert_eq!(back, grey, "clockwise then counter-clockwise cancels out");
    }

    #[test]
    #[should_panic(expected = "direction must be")]
    fn rotate90_rejects_bad_direction() {
        let _ = rotate90_grey(&[0u8; 4], 2, 2, 0);
    }

    #[test]
    #[should_panic(expected = "not w·h")]
    fn rotate90_rejects_mismatched_length() {
        let _ = rotate90_grey(&[0u8; 3], 2, 2, 1);
    }

    // ---- D2: find_differential_square_sum --------------------------------

    #[test]
    fn dss_zero_for_uniform_row_density() {
        // Every row has the SAME foreground-pixel count -> every diff is 0.
        let (w, h) = (10usize, 20usize);
        let mut binary = vec![255u8; w * h]; // 255 = background/OFF.
        for y in 0..h {
            for x in (0..w).step_by(2) {
                binary[y * w + x] = 0; // 0 = ON, this crate's convention.
            }
        }
        assert_eq!(find_differential_square_sum(&binary, w, h), 0.0);
    }

    #[test]
    fn dss_positive_for_a_single_interior_differing_row() {
        let (w, h) = (20usize, 20usize);
        let mut binary = vec![255u8; w * h]; // all background.
        let spike_row = h / 2; // comfortably inside the scored band.
        for x in 0..w {
            binary[spike_row * w + x] = 0;
        }
        let sum = find_differential_square_sum(&binary, w, h);
        assert!(
            sum > 0.0,
            "a single interior differing row must contribute a positive score"
        );
    }

    #[test]
    fn dss_last_row_is_outside_the_scored_band() {
        // w=5,h=6: skiph=(0.05*5)as i32=0, skip=min(6/10,0)=0, nskip=max(0,1)=1.
        // With n=h=6 the loop is `i in [1, 5)`; the touched row indices are
        // exactly {0,1,2,3,4} (i-1 and i together) -- index 5 (the LAST row)
        // never appears as either operand, so changing only that row must
        // not move the score.
        let (w, h) = (5usize, 6usize);
        let flat = vec![255u8; w * h];
        let mut spiked_last_row = flat.clone();
        for x in 0..w {
            spiked_last_row[(h - 1) * w + x] = 0; // last row fully ON.
        }
        assert_eq!(
            find_differential_square_sum(&flat, w, h),
            find_differential_square_sum(&spiked_last_row, w, h),
            "the last row is outside the nskip=1 scored band"
        );
    }

    #[test]
    #[should_panic(expected = "not w·h")]
    fn dss_rejects_mismatched_length() {
        let _ = find_differential_square_sum(&[255u8; 3], 2, 2);
    }

    // ---- D3: v_shear_corner / v_shear_center -------------------------------
    //
    // These are structural / self-consistency checks, not oracle diffs --
    // see `.claude/harvest/oracles/skew_oracle.cpp`'s `dss`/`vshear` arms
    // (via `deskew_dump.rs`) for the real byte-parity evidence across many
    // angles. Every assertion here is chosen to hold REGARDLESS of any
    // f32/f64 precision subtlety in the shear math itself (see
    // `rotate_am_gray_uniform_field_stays_uniform`'s identical framing for
    // D5): a wrong-but-non-panicking shear could still, in principle, pass
    // some of these, which is exactly why they are a sanity net and not a
    // substitute for the oracle.

    #[test]
    fn v_shear_zero_angle_is_identity_for_both_pivots() {
        let (w, h) = (12usize, 9usize);
        let binary: Vec<u8> = (0..(w * h) as u8)
            .map(|v| if v % 3 == 0 { 0 } else { 255 })
            .collect();
        assert_eq!(
            v_shear_corner(&binary, w, h, 0.0, 255),
            binary,
            "corner, zero angle"
        );
        assert_eq!(
            v_shear_center(&binary, w, h, 0.0, 255),
            binary,
            "center, zero angle"
        );
    }

    #[test]
    fn v_shear_uniform_field_stays_uniform() {
        // Every written pixel is copied from a source that is uniformly
        // `fill`, and every unwritten pixel was initialized to `fill` too --
        // so the whole output must equal `fill` everywhere, at any angle,
        // for both pivots. This holds independent of the shear's own
        // indexing/timing correctness (it is a property of the fill value,
        // not the angle), so -- as with D5's analogous test -- it is a
        // correctness sanity check, not a substitute for the oracle diff.
        let (w, h) = (16usize, 20usize);
        let fill = 255u8;
        let binary = vec![fill; w * h];
        for &deg in &[2.0_f32, 15.0, 45.0, -30.0] {
            let rad = deg.to_radians();
            let out = v_shear_corner(&binary, w, h, rad, fill);
            assert!(
                out.iter().all(|&b| b == fill),
                "corner must stay uniform at {deg} degrees"
            );
            let out_center = v_shear_center(&binary, w, h, rad, fill);
            assert!(
                out_center.iter().all(|&b| b == fill),
                "center must stay uniform at {deg} degrees"
            );
        }
    }

    #[test]
    fn v_shear_corner_pivots_at_xloc_zero() {
        let (w, h) = (14usize, 11usize);
        let binary: Vec<u8> = (0..(w * h) as u8)
            .map(|v| if v % 5 == 0 { 0 } else { 255 })
            .collect();
        let rad = 12.0_f32.to_radians();
        assert_eq!(
            v_shear_corner(&binary, w, h, rad, 255),
            v_shear(&binary, w, h, 0, rad, 255)
        );
    }

    #[test]
    fn v_shear_center_pivots_at_width_over_two() {
        let (w, h) = (14usize, 11usize);
        let binary: Vec<u8> = (0..(w * h) as u8)
            .map(|v| if v % 5 == 0 { 0 } else { 255 })
            .collect();
        let rad = 12.0_f32.to_radians();
        assert_eq!(
            v_shear_center(&binary, w, h, rad, 255),
            v_shear(&binary, w, h, (w as i32) / 2, rad, 255)
        );
    }

    #[test]
    fn v_shear_output_keeps_input_dimensions() {
        let (w, h) = (17usize, 13usize);
        let binary = vec![255u8; w * h];
        let out = v_shear_corner(&binary, w, h, 8.0_f32.to_radians(), 255);
        assert_eq!(out.len(), w * h);
        let out_center = v_shear_center(&binary, w, h, 8.0_f32.to_radians(), 255);
        assert_eq!(out_center.len(), w * h);
    }

    #[test]
    #[should_panic(expected = "not w·h")]
    fn v_shear_corner_rejects_mismatched_length() {
        let _ = v_shear_corner(&[255u8; 3], 2, 2, 1.0, 255);
    }

    #[test]
    #[should_panic(expected = "not w·h")]
    fn v_shear_center_rejects_mismatched_length() {
        let _ = v_shear_center(&[255u8; 3], 2, 2, 1.0, 255);
    }

    // ---- D4: find_skew_sweep_and_search_score_pivot ------------------------

    #[test]
    fn find_skew_rejects_invalid_reduction_factors() {
        let binary = vec![0u8; 10 * 10];
        assert!(find_skew_sweep_and_search_score_pivot(
            &binary,
            10,
            10,
            3, // not in {1,2,4,8}
            1,
            0.0,
            5.0,
            1.0,
            0.01,
            ShearPivot::Corner,
        )
        .is_none());
        assert!(find_skew_sweep_and_search_score_pivot(
            &binary,
            10,
            10,
            1,
            2, // redsearch > redsweep
            0.0,
            5.0,
            1.0,
            0.01,
            ShearPivot::Corner,
        )
        .is_none());
    }

    #[test]
    fn find_skew_rejects_all_background_image() {
        // This crate's convention: 255 = background everywhere = pixZero.
        let binary = vec![255u8; 10 * 10];
        assert!(find_skew_sweep_and_search_score_pivot(
            &binary,
            10,
            10,
            1,
            1,
            0.0,
            5.0,
            1.0,
            0.01,
            ShearPivot::Corner,
        )
        .is_none());
    }

    #[test]
    fn find_skew_returns_degenerate_success_when_max_is_at_sweep_edge() {
        // sweeprange=0.0 forces nangles=(l_int32)((2.*0.)/sweepdelta+1)=1
        // regardless of sweepdelta, so the single sampled angle is ALWAYS
        // both the first and last element -- the "max found at sweep edge"
        // guard ALWAYS fires here, independent of the fixture's content (as
        // long as it is not all-background, which would hit the EARLIER
        // pixZero check instead).
        let (w, h) = (10usize, 10usize);
        let mut binary = vec![255u8; w * h];
        binary[0] = 0; // just needs some non-background content.
        let result = find_skew_sweep_and_search_score_pivot(
            &binary,
            w,
            h,
            1,
            1,
            0.0,
            0.0,
            1.0,
            0.01,
            ShearPivot::Corner,
        );
        assert_eq!(
            result,
            Some(SkewResult {
                angle: 0.0,
                conf: 0.0,
                endscore: 0.0,
            })
        );
    }

    #[test]
    fn find_skew_runs_to_completion_on_a_banded_fixture() {
        // Horizontal banding (2 rows on, 2 off) gives the differential
        // square sum real row-to-row signal to work with, and is far from
        // all-background. redsweep=redsearch=1 skips
        // reduce_rank_binary_cascade entirely (ratio=1 on both legs), which
        // removes a whole class of "did the reduction leave a workable
        // image" uncertainty from this smoke test.
        let (w, h) = (20usize, 24usize);
        let mut binary = vec![255u8; w * h];
        for y in 0..h {
            if y % 4 < 2 {
                for x in 0..w {
                    binary[y * w + x] = 0;
                }
            }
        }
        let result = find_skew_sweep_and_search_score_pivot(
            &binary,
            w,
            h,
            1,
            1,
            0.0,
            5.0,
            1.0,
            0.05,
            ShearPivot::Corner,
        )
        .expect("valid, non-degenerate params on a non-empty image must return Some");
        assert!(
            result.angle.is_finite(),
            "angle must be finite: {}",
            result.angle
        );
        assert!(
            result.conf >= 0.0 && result.conf.is_finite(),
            "conf must be finite and non-negative: {}",
            result.conf
        );
        assert!(
            result.endscore.is_finite(),
            "endscore must be finite: {}",
            result.endscore
        );
    }

    // ---- D5: rotate_am_gray / rotate_am_gray_corner -----------------------

    #[test]
    fn rotate_am_gray_noop_below_min_angle() {
        let grey: Vec<u8> = (0..(10 * 8) as u8).collect();
        let out = rotate_am_gray(&grey, 10, 8, 0.0, 255);
        assert_eq!(out, grey, "angle 0.0 is a no-op clone (centre)");
        let out2 = rotate_am_gray_corner(&grey, 10, 8, 0.0005, 255);
        assert_eq!(
            out2, grey,
            "|angle| < MinAngleToRotate is a no-op clone (corner)"
        );
    }

    #[test]
    fn rotate_am_gray_uniform_field_stays_uniform() {
        // Interpolating a constant field returns the same constant exactly
        // (the 16x16 sub-pixel weights always sum to 256), and picking
        // grayval == that constant makes the off-canvas fill agree too -- so
        // the WHOLE output must equal the input value everywhere, at any
        // angle, for both pivots. This holds independent of the f32/f64
        // precision trap (it is a property of the weights, not the angle),
        // so it is a correctness sanity check, not a substitute for the
        // oracle diff across many angles.
        let (w, h) = (30usize, 24usize);
        let value = 137u8;
        let grey = vec![value; w * h];
        for &deg in &[2.0_f32, 15.0, 45.0, -30.0] {
            let rad = deg.to_radians();
            let out = rotate_am_gray(&grey, w, h, rad, value);
            assert!(
                out.iter().all(|&v| v == value),
                "uniform field must stay uniform at {deg} degrees (centre)"
            );
            let out_corner = rotate_am_gray_corner(&grey, w, h, rad, value);
            assert!(
                out_corner.iter().all(|&v| v == value),
                "uniform field must stay uniform at {deg} degrees (corner)"
            );
        }
    }

    #[test]
    fn rotate_am_gray_output_keeps_input_dimensions() {
        // Leptonica's area-map rotate does not expand the canvas.
        let grey = vec![10u8; 12 * 9];
        let out = rotate_am_gray(&grey, 12, 9, 10.0_f32.to_radians(), 255);
        assert_eq!(out.len(), 12 * 9);
        let out_corner = rotate_am_gray_corner(&grey, 12, 9, 10.0_f32.to_radians(), 255);
        assert_eq!(out_corner.len(), 12 * 9);
    }

    #[test]
    #[should_panic(expected = "not w·h")]
    fn rotate_am_gray_rejects_mismatched_length() {
        let _ = rotate_am_gray(&[1u8; 3], 2, 2, 1.0, 0);
    }

    #[test]
    #[should_panic(expected = "not w·h")]
    fn rotate_am_gray_corner_rejects_mismatched_length() {
        let _ = rotate_am_gray_corner(&[1u8; 3], 2, 2, 1.0, 0);
    }

    // ---- D6: deskew_general -------------------------------------------------

    #[test]
    fn deskew_general_rejects_invalid_redsweep() {
        let grey = vec![255u8; 10 * 10];
        assert!(
            deskew_general(&grey, 10, 10, 3, 5.0, 1.0, 1, 128).is_none(),
            "redsweep=3 is not in {{0,1,2,4}}"
        );
    }

    #[test]
    fn deskew_general_rejects_invalid_redsearch() {
        let grey = vec![255u8; 10 * 10];
        assert!(
            deskew_general(&grey, 10, 10, 1, 5.0, 1.0, 3, 128).is_none(),
            "redsearch=3 is not in {{0,1,2,4}}"
        );
    }

    #[test]
    fn deskew_general_rejects_redsweep_8_even_though_d4_itself_accepts_it() {
        // skew.c:311-312 -- pixDeskewGeneral validates redsweep against
        // {1,2,4} ONLY, a STRICT subset of
        // find_skew_sweep_and_search_score_pivot's own {1,2,4,8}
        // (this module's own `matches!(r, 1 | 2 | 4 | 8)`). Pinned as a
        // locked contract, not something to "helpfully" widen later.
        let (w, h) = (32usize, 32usize);
        let mut binary = vec![255u8; w * h];
        for y in 0..h {
            if y % 4 < 2 {
                for x in 0..w {
                    binary[y * w + x] = 0;
                }
            }
        }
        assert!(
            find_skew_sweep_and_search_score_pivot(
                &binary,
                w,
                h,
                8,
                1,
                0.0,
                5.0,
                1.0,
                0.01,
                ShearPivot::Corner,
            )
            .is_some(),
            "D4 itself accepts redsweep=8 on a non-empty image"
        );

        let grey = vec![200u8; w * h]; // any grey content; only redsweep is under test.
        assert!(
            deskew_general(&grey, w, h, 8, 5.0, 1.0, 1, 128).is_none(),
            "pixDeskewGeneral's OWN validation rejects redsweep=8, unlike D4's"
        );
    }

    #[test]
    fn deskew_general_all_background_returns_unrotated_with_zero_angle_conf() {
        // Uniform WHITE page, no ink anywhere. thresh=130 (default): every
        // pixel is `255 < 130` == false, so the binarized copy is entirely
        // background -- pixZero fires inside D4, which is D4's OWN true
        // error path (None), mirroring pixDeskewGeneral's `ret != 0` branch:
        // angle/conf forced to 0.0, buffer unchanged. redsearch=1 is used
        // deliberately to bypass reduce_rank_binary_cascade entirely (its
        // own branch is a direct `binary.to_vec()`), so this assertion does
        // not depend on any behavior of the reduction cascade.
        let (w, h) = (20usize, 16usize);
        let grey = vec![255u8; w * h];
        let result = deskew_general(&grey, w, h, 1, 5.0, 1.0, 1, 130)
            .expect("valid redsweep/redsearch must not return None");
        assert_eq!(
            result.grey, grey,
            "an all-background page must come back unrotated"
        );
        assert_eq!(result.w, w);
        assert_eq!(result.h, h);
        assert_eq!(result.angle, 0.0);
        assert_eq!(result.conf, 0.0);
    }

    #[test]
    fn deskew_general_runs_to_completion_on_a_banded_fixture() {
        // Horizontal banding gives the detector real row-to-row signal
        // (same spirit as find_skew_runs_to_completion_on_a_banded_fixture's
        // own D4 smoke test) without needing to hand-verify the detector's
        // exact numeric (angle, conf) output.
        let (w, h) = (40usize, 48usize);
        let mut grey = vec![255u8; w * h];
        for y in 0..h {
            if y % 8 < 3 {
                for x in 0..w {
                    grey[y * w + x] = 0;
                }
            }
        }
        let result = deskew_general(&grey, w, h, 1, 5.0, 1.0, 1, 128)
            .expect("valid params on a non-empty image must return Some");
        assert_eq!(
            (result.w, result.h),
            (w, h),
            "area-map rotation never expands the canvas"
        );
        assert!(
            result.angle.is_finite(),
            "angle must be finite: {}",
            result.angle
        );
        assert!(
            result.conf.is_finite() && result.conf >= 0.0,
            "conf must be finite and non-negative: {}",
            result.conf
        );
        // The one direction of the gate that is unconditionally certain
        // without re-deriving the detector's own numeric output by hand:
        // BELOW the gate, pixDeskewGeneral's own branch is literally
        // `return pixClone(pixs)` -- the original, untouched.
        if result.angle.abs() < MIN_DESKEW_ANGLE || result.conf < MIN_ALLOWED_CONFIDENCE {
            assert_eq!(
                result.grey, grey,
                "below the gate must return the ORIGINAL, unrotated"
            );
        }
    }

    #[test]
    #[should_panic(expected = "not w·h")]
    fn deskew_general_rejects_mismatched_length() {
        let _ = deskew_general(&[1u8; 3], 2, 2, 1, 5.0, 1.0, 1, 128);
    }

    // ---- D7: deskew_both -----------------------------------------------------

    #[test]
    fn deskew_both_rejects_invalid_redsearch() {
        let grey = vec![255u8; 10 * 10];
        assert!(deskew_both(&grey, 10, 10, 8).is_none());
        assert!(deskew_both(&grey, 10, 10, 3).is_none());
    }

    #[test]
    fn deskew_both_all_background_round_trips_to_original_dimensions_and_content() {
        // Deliberately non-square, so a forgotten w/h swap across the two
        // rotate90 legs would actually be caught by the dimensions check
        // (a square fixture could hide that bug entirely).
        let (w, h) = (12usize, 20usize);
        let grey = vec![255u8; w * h];
        let (out, ow, oh) = deskew_both(&grey, w, h, 1)
            .expect("valid redsearch on a well-formed image must not return None");
        assert_eq!(
            (ow, oh),
            (w, h),
            "two opposite 90-degree turns must restore the original dimensions"
        );
        assert_eq!(
            out, grey,
            "an all-background page must round-trip unchanged"
        );
    }

    #[test]
    #[should_panic(expected = "not w·h")]
    fn deskew_both_rejects_mismatched_length() {
        let _ = deskew_both(&[1u8; 3], 2, 2, 1);
    }
}
