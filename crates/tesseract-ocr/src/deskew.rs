//! Deskew — page ROTATION leaves, a byte-parity transcode of leptonica's
//! `pixRotate90` (d==8 only), `pixFindDifferentialSquareSum`, and
//! `pixRotateAMGray`/`pixRotateAMGrayCorner` (`src/{rotateorth.c,skew.c,
//! rotateam.c}`, `AdaWorldAPI/leptonica` fork, v1.82.0 == the installed
//! liblept — zero ABI/version skew, the same footing the Sauvola leaf had),
//! proven byte-identical against real liblept via
//! `.claude/harvest/oracles/skew_oracle.cpp`.
//!
//! This is a DELIBERATELY PARTIAL slice of the full deskew wave
//! (`.claude/plans/deskew-wave-v1.md`, leaves D1/D2/D5 of D1-D8). NOT here:
//! the shear kernel (D3, `pixHShear`/`pixVShear`/`pixVShearCorner`/
//! `pixVShearCenter`, `shear.c`), the sweep + binary-search detector (D4,
//! `pixFindSkewSweepAndSearchScorePivot`, the thing that actually calls D3
//! during its coarse sweep), and the `pixDeskewGeneral`/`pixDeskewBoth`
//! composition (D6/D7, which calls D1 for its 90°-round-trip second pass).
//! See the plan for the full sequencing and why the leaves split this way.
//!
//! ## What's here
//! - [`rotate90_grey`] — `pixRotate90`, d==8 only (`rotateorth.c:222-232` CW /
//!   `:310-320` CCW): a pure index remap, no float math at all, lossless.
//!   Needed by the not-yet-built D7 `pixDeskewBoth`'s 90°-round-trip.
//! - [`find_differential_square_sum`] — `pixFindDifferentialSquareSum`
//!   (`skew.c:1111-1155`): the score the not-yet-built D4 sweep maximizes.
//! - [`rotate_am_gray`] / [`rotate_am_gray_corner`] — `pixRotateAMGray` /
//!   `pixRotateAMGrayCorner` (`rotateam.c:288-317` + `:391-448` low kernel;
//!   `:584-613` + `:683-736` low kernel): the grey area-map rotation the
//!   recognizer actually wants for post-deskew correction — rotating in grey
//!   preserves the antialiasing the LSTM input step depends on, whereas
//!   rotating only the binary detector buffer would throw it away.
//!
//! ## Two crate-convention notes that matter for every leaf here
//! - **Grey is raw, unflipped.** 8bpp grey buffers in this crate are
//!   leptonica-native: white background ≈ 255, dark ink ≈ 0, no inversion
//!   (matches `image_input.rs`, `xy_cut.rs`'s documented convention). So
//!   `L_BRING_IN_WHITE` fills off-canvas pixels with `255` directly — no
//!   conversion needed at the grey boundary.
//! - **Binary is flipped.** THIS CRATE'S binary buffers use `0 = ON`
//!   (ink/foreground), matching `binreduce.rs`'s documented convention — the
//!   OPPOSITE of leptonica's native `1 = ON`. [`find_differential_square_sum`]
//!   takes a buffer in THIS crate's convention; convert at the caller
//!   boundary (see [`find_differential_square_sum`]'s own docs), never
//!   silently assume leptonica's polarity.
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
}
