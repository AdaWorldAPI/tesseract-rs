//! Binary morphology — leptonica `morph.c` transcode (Batch 3C).
//!
//! Ports the **brick** morphology surface: `pixDilateBrick` / `pixErodeBrick`
//! / `pixOpenBrick` / `pixCloseBrick` (plain, NOT `pixCloseSafeBrick`). A
//! "brick" Sel is a rectangle of solid `SEL_HIT`s — every source function in
//! this module's dependency chain ultimately reduces to a rectangular
//! window OR-reduce (dilate) or AND-reduce (erode) plus, for erode, an
//! `ASYMMETRIC_MORPH_BC` boundary clear.
//!
//! Per the banked manifest `.claude/harvest/leptonica-morph-callgraph.txt`
//! (`ruff_cpp_spo::walk_free_functions` on `/tmp/leptonica-src/morph.c`):
//! `pixDilateBrick → pixDilate`, `pixErodeBrick → pixErode`,
//! `pixOpenBrick → {pixDilate, pixErode, pixOpen}`,
//! `pixCloseBrick → {pixClose, pixDilate, pixErode}`, with two non-TU
//! callees per brick function that carry algorithm (not just plumbing):
//! `selCreateBrick` and (for erode) `selFindMaxTranslations`, both resolved
//! by the follow-up harvest `.claude/harvest/leptonica-sel1-callgraph.txt`
//! (`/tmp/leptonica-src/sel1.c`) — both are LEAF/plumbing kernels, quoted
//! and closed-formed there. The inner-loop raster op itself
//! (`pixRasterop`) is resolved by `.claude/harvest/leptonica-rop-callgraph.txt`
//! (`/tmp/leptonica-src/rop.c` + `roplow.c`'s `rasteropLow` clip logic read
//! directly) — the clip-to-overlap semantics it documents are what
//! `dilate_rect`/`erode_rect` below implement per-pixel.
//!
//! ## Input/output convention
//! Same as `crate::conncomp`: `binary: &[u8]` is `w * h` bytes, one byte per
//! pixel, row-major. Foreground (ink/ON) is `byte == 0`; background is
//! `byte == 255` (the `threshold_rect_to_binary` convention).
//!
//! ## The math (derived from `morph.c`, quoted line ranges below)
//!
//! `pixDilate(pixd, pixs, sel)` (`morph.c:212-238`): `pixd` is cleared to
//! all-OFF, then for every hit `(i, j)` of the `sy x sx` sel with origin
//! `(cy, cx)`:
//! ```text
//! pixRasterop(pixd, j - cx, i - cy, w, h, PIX_SRC | PIX_DST, pixt, 0, 0);
//! ```
//! i.e. `pixd[x][y] |= pixt[x - (j - cx)][y - (i - cy)]`, out-of-bounds reads
//! contributing nothing (see `leptonica-rop-callgraph.txt`). For a brick
//! (every cell a hit) this collapses to the standard rectangular-window OR:
//! `out[x][y] = OR over i in 0..sy, j in 0..sx of src[x-(j-cx)][y-(i-cy)]`
//! (`dilate_rect` below).
//!
//! `pixErode(pixd, pixs, sel)` (`morph.c:264-309`): `pixd` is set to
//! all-ON, then for every hit: `pixd[x][y] &= pixt[x+(j-cx)][y+(i-cy)]`
//! (opposite shift direction from dilate), with out-of-bounds terms simply
//! **skipped** (no constraint — the AND stays whatever it was, i.e. `true`
//! by the all-ON initialization). Afterward (`morph.c:295-305`, the default
//! `MORPH_BC == ASYMMETRIC_MORPH_BC` branch — never reset by this crate, so
//! always taken), `selFindMaxTranslations` supplies `(xp, yp, xn, yn)` and
//! the border strips `[0,xp)`, `[w-xn,w)`, `[0,yp)`, `[h-yn,h)` are forced
//! OFF. For an all-hit rectangle sel, the closed form
//! (`leptonica-sel1-callgraph.txt`) is `xp = cx`, `xn = sx-1-cx`, `yp = cy`,
//! `yn = sy-1-cy` (`erode_rect` below).
//!
//! ## The decompose-when-separable branch (both `pixDilateBrick`/`pixErodeBrick`)
//! `morph.c:687-707` (dilate) / `:755-775` (erode): `hsize == 1 && vsize == 1`
//! short-circuits to a copy; `hsize == 1 || vsize == 1` applies ONE brick sel
//! directly (`selCreateBrick(vsize, hsize, vsize/2, hsize/2, SEL_HIT)`);
//! otherwise (both > 1) the op is done **separably**: first a horizontal sel
//! `selh = selCreateBrick(1, hsize, 0, hsize/2, SEL_HIT)`, then a vertical
//! sel `selv = selCreateBrick(vsize, 1, vsize/2, 0, SEL_HIT)` applied to the
//! horizontal result — this is mathematically exact for a rectangle
//! structuring element (rectangle erosion/dilation is always separable),
//! and crucially each of the two `pixDilate`/`pixErode` calls does its
//! **own** independent boundary clear from its own sel's
//! `selFindMaxTranslations` (so the horizontal pass only clears left/right
//! columns, the vertical pass only clears top/bottom rows).
//!
//! ## Open / close: erode/dilate-brick composition, not a fresh 2D sel
//! `pixOpenBrick` (`morph.c:807-848`) and `pixCloseBrick` (`morph.c:877-918`)
//! do NOT call `pixOpen`/`pixClose` with a single 2D brick sel in the
//! separable (`hsize>1 && vsize>1`) branch — they inline
//! erode-then-dilate (open) / dilate-then-erode (close) using the SAME
//! `selh`/`selv` pair, in the SAME horizontal-then-vertical order as
//! `pixDilateBrick`/`pixErodeBrick` themselves use. In the non-separable
//! (`hsize==1 || vsize==1`) branch they call `pixOpen`/`pixClose` directly
//! with the single full brick sel — which, since erosion/dilation with that
//! sel is exactly what `erode_brick`/`dilate_brick` compute in their own
//! `hsize==1||vsize==1` branch, is the identical sequence of rasterops.
//! Net result, provably identical to the C in both branches and the
//! `hsize==1&&vsize==1` shortcut: `pixOpenBrick(x) == dilate_brick(erode_brick(x))`
//! and `pixCloseBrick(x) == erode_brick(dilate_brick(x))`, so `open_brick`/
//! `close_brick` below are implemented by direct composition of
//! `dilate_brick`/`erode_brick` rather than re-deriving the sequence.

/// Read `on[y*w+x]` treating any out-of-range `(x, y)` as `false`
/// (background/OFF) — the dilate convention (`pixRasterop`'s clip-to-overlap
/// leaves untouched destination pixels at their `pixClearAll` default of 0).
#[inline]
fn get_or_false(on: &[bool], w: i32, h: i32, x: i32, y: i32) -> bool {
    x >= 0 && x < w && y >= 0 && y < h && on[(y * w + x) as usize]
}

/// Dilate a boolean grid by an all-hit rectangular sel `sy x sx` with origin
/// `(cy, cx)` — `morph.c:212-238`'s `pixDilate`, specialized to a brick.
fn dilate_rect(on: &[bool], w: i32, h: i32, sy: i32, sx: i32, cy: i32, cx: i32) -> Vec<bool> {
    let mut out = vec![false; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut hit = false;
            'search: for i in 0..sy {
                for j in 0..sx {
                    if get_or_false(on, w, h, x - (j - cx), y - (i - cy)) {
                        hit = true;
                        break 'search;
                    }
                }
            }
            out[(y * w + x) as usize] = hit;
        }
    }
    out
}

/// Erode a boolean grid by an all-hit rectangular sel `sy x sx` with origin
/// `(cy, cx)` — `morph.c:264-309`'s `pixErode`, specialized to a brick,
/// including the `ASYMMETRIC_MORPH_BC` (default, never reset by this crate)
/// border clear via the closed-form `selFindMaxTranslations` from
/// `.claude/harvest/leptonica-sel1-callgraph.txt`.
fn erode_rect(on: &[bool], w: i32, h: i32, sy: i32, sx: i32, cy: i32, cx: i32) -> Vec<bool> {
    let mut out = vec![true; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut all_hit = true;
            'search: for i in 0..sy {
                for j in 0..sx {
                    let xx = x + (j - cx);
                    let yy = y + (i - cy);
                    if xx >= 0 && xx < w && yy >= 0 && yy < h && !on[(yy * w + xx) as usize] {
                        all_hit = false;
                        break 'search;
                    }
                    // Out-of-range term: no constraint (rasterop clip skips
                    // it; the AND accumulator, initialized true, is
                    // unaffected).
                }
            }
            out[(y * w + x) as usize] = all_hit;
        }
    }

    // ASYMMETRIC_MORPH_BC border clear (morph.c:295-305).
    let xp = cx;
    let xn = sx - 1 - cx;
    let yp = cy;
    let yn = sy - 1 - cy;
    if xp > 0 {
        for y in 0..h {
            for x in 0..xp.min(w) {
                out[(y * w + x) as usize] = false;
            }
        }
    }
    if xn > 0 {
        for y in 0..h {
            for x in (w - xn).max(0)..w {
                out[(y * w + x) as usize] = false;
            }
        }
    }
    if yp > 0 {
        for y in 0..yp.min(h) {
            for x in 0..w {
                out[(y * w + x) as usize] = false;
            }
        }
    }
    if yn > 0 {
        for y in (h - yn).max(0)..h {
            for x in 0..w {
                out[(y * w + x) as usize] = false;
            }
        }
    }
    out
}

fn to_on(binary: &[u8]) -> Vec<bool> {
    binary.iter().map(|&b| b == 0).collect()
}

fn from_on(on: &[bool]) -> Vec<u8> {
    on.iter().map(|&v| if v { 0 } else { 255 }).collect()
}

/// `pixDilateBrick(pixd=NULL, pixs, hsize, vsize)` (`morph.c:671-710`).
///
/// `binary` is `w * h` bytes, foreground = `byte == 0`. `hsize`/`vsize` are
/// the brick Sel's width/height; the origin is at `(hsize/2, vsize/2)`
/// (`morph.c`'s Notes (2)). `hsize == 1 && vsize == 1` is a no-op copy;
/// exactly one of them `== 1` applies a single 1-D brick directly;
/// otherwise the operation is done separably (horizontal sel, then
/// vertical sel on the horizontal result) — mathematically exact for a
/// rectangle structuring element.
///
/// # Panics
/// Panics if `hsize < 1`, `vsize < 1`, or `binary.len() != w * h`.
pub fn dilate_brick(binary: &[u8], w: usize, h: usize, hsize: usize, vsize: usize) -> Vec<u8> {
    assert!(hsize >= 1 && vsize >= 1, "hsize and vsize must be >= 1");
    assert_eq!(binary.len(), w * h, "binary buffer length must be w * h");
    let on = dilate_brick_bool(&to_on(binary), w, h, hsize, vsize);
    from_on(&on)
}

fn dilate_brick_bool(on: &[bool], w: usize, h: usize, hsize: usize, vsize: usize) -> Vec<bool> {
    let (wi, hi) = (w as i32, h as i32);
    if hsize == 1 && vsize == 1 {
        return on.to_vec();
    }
    if hsize == 1 || vsize == 1 {
        let (sy, sx) = (vsize as i32, hsize as i32);
        let (cy, cx) = (sy / 2, sx / 2);
        return dilate_rect(on, wi, hi, sy, sx, cy, cx);
    }
    // Separable: horizontal selh = (1, hsize, 0, hsize/2), then
    // vertical selv = (vsize, 1, vsize/2, 0) on the result.
    let step1 = dilate_rect(on, wi, hi, 1, hsize as i32, 0, (hsize / 2) as i32);
    dilate_rect(&step1, wi, hi, vsize as i32, 1, (vsize / 2) as i32, 0)
}

/// `pixErodeBrick(pixd=NULL, pixs, hsize, vsize)` (`morph.c:739-778`). Same
/// shape and shortcuts as [`dilate_brick`], using [`erode_rect`] (which
/// includes the `ASYMMETRIC_MORPH_BC` border clear per stage).
///
/// # Panics
/// Panics if `hsize < 1`, `vsize < 1`, or `binary.len() != w * h`.
pub fn erode_brick(binary: &[u8], w: usize, h: usize, hsize: usize, vsize: usize) -> Vec<u8> {
    assert!(hsize >= 1 && vsize >= 1, "hsize and vsize must be >= 1");
    assert_eq!(binary.len(), w * h, "binary buffer length must be w * h");
    let on = erode_brick_bool(&to_on(binary), w, h, hsize, vsize);
    from_on(&on)
}

fn erode_brick_bool(on: &[bool], w: usize, h: usize, hsize: usize, vsize: usize) -> Vec<bool> {
    let (wi, hi) = (w as i32, h as i32);
    if hsize == 1 && vsize == 1 {
        return on.to_vec();
    }
    if hsize == 1 || vsize == 1 {
        let (sy, sx) = (vsize as i32, hsize as i32);
        let (cy, cx) = (sy / 2, sx / 2);
        return erode_rect(on, wi, hi, sy, sx, cy, cx);
    }
    let step1 = erode_rect(on, wi, hi, 1, hsize as i32, 0, (hsize / 2) as i32);
    erode_rect(&step1, wi, hi, vsize as i32, 1, (vsize / 2) as i32, 0)
}

/// `pixOpenBrick(pixd=NULL, pixs, hsize, vsize)` (`morph.c:807-848`) —
/// erosion followed by dilation with the same brick, `erode_brick` then
/// `dilate_brick` (see the module docs' "Open/close" section for why this
/// composition is provably identical to the C source in every branch).
///
/// # Panics
/// Panics if `hsize < 1`, `vsize < 1`, or `binary.len() != w * h`.
pub fn open_brick(binary: &[u8], w: usize, h: usize, hsize: usize, vsize: usize) -> Vec<u8> {
    assert!(hsize >= 1 && vsize >= 1, "hsize and vsize must be >= 1");
    assert_eq!(binary.len(), w * h, "binary buffer length must be w * h");
    let eroded = erode_brick_bool(&to_on(binary), w, h, hsize, vsize);
    let opened = dilate_brick_bool(&eroded, w, h, hsize, vsize);
    from_on(&opened)
}

/// `pixCloseBrick(pixd=NULL, pixs, hsize, vsize)` (`morph.c:877-918`) —
/// plain closing (NOT `pixCloseSafeBrick`): dilation followed by erosion
/// with the same brick, `dilate_brick` then `erode_brick`.
///
/// # Panics
/// Panics if `hsize < 1`, `vsize < 1`, or `binary.len() != w * h`.
pub fn close_brick(binary: &[u8], w: usize, h: usize, hsize: usize, vsize: usize) -> Vec<u8> {
    assert!(hsize >= 1 && vsize >= 1, "hsize and vsize must be >= 1");
    assert_eq!(binary.len(), w * h, "binary buffer length must be w * h");
    let dilated = dilate_brick_bool(&to_on(binary), w, h, hsize, vsize);
    let closed = erode_brick_bool(&dilated, w, h, hsize, vsize);
    from_on(&closed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_binary(on_pixels: &[(usize, usize)], w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![255_u8; w * h];
        for &(x, y) in on_pixels {
            buf[y * w + x] = 0;
        }
        buf
    }

    fn is_on(buf: &[u8], w: usize, x: usize, y: usize) -> bool {
        buf[y * w + x] == 0
    }

    /// Dilating a single pixel by a 3x3 brick (origin at (1,1)) grows it
    /// into the full 3x3 neighborhood centered on the original pixel.
    #[test]
    fn dilate_single_pixel_grows_to_brick_shape() {
        let w = 7;
        let h = 7;
        let binary = to_binary(&[(3, 3)], w, h);
        let out = dilate_brick(&binary, w, h, 3, 3);
        for y in 2..=4 {
            for x in 2..=4 {
                assert!(is_on(&out, w, x, y), "expected ON at ({x},{y})");
            }
        }
        // Outside the 3x3 neighborhood must remain OFF.
        assert!(!is_on(&out, w, 1, 1));
        assert!(!is_on(&out, w, 5, 5));
        assert!(!is_on(&out, w, 3, 1));
        assert!(!is_on(&out, w, 3, 5));
    }

    /// Eroding a fully-ON image with a 3x3 brick (cx=cy=1) leaves the
    /// interior ON but clears a 1-pixel border on every side, per
    /// ASYMMETRIC_MORPH_BC (xp=xn=yp=yn=1 for a 3x3 centered brick).
    #[test]
    fn erode_full_black_5x5_brick_3x3_clears_one_pixel_border() {
        let w = 5;
        let h = 5;
        let binary = vec![0_u8; w * h]; // fully ON (foreground)
        let out = erode_brick(&binary, w, h, 3, 3);
        for y in 0..h {
            for x in 0..w {
                let expect_on = (1..=3).contains(&x) && (1..=3).contains(&y);
                assert_eq!(
                    is_on(&out, w, x, y),
                    expect_on,
                    "mismatch at ({x},{y}): expected on={expect_on}"
                );
            }
        }
    }

    /// `open_brick == dilate_brick(erode_brick(x))` by construction; smoke
    /// test that opening removes an isolated single-pixel speck the erosion
    /// step will erase and the dilation step won't resurrect.
    #[test]
    fn open_brick_matches_erode_then_dilate_composition() {
        let w = 9;
        let h = 9;
        let binary = to_binary(&[(4, 4), (0, 0)], w, h);
        let opened = open_brick(&binary, w, h, 3, 3);
        let eroded = erode_brick(&binary, w, h, 3, 3);
        let manual = dilate_brick(&eroded, w, h, 3, 3);
        assert_eq!(opened, manual);
        // The isolated speck at (0,0) has no full 3x3 neighborhood on the
        // image (near the corner) and is small, so it is erased by erosion
        // and never resurrected by dilation.
        assert!(!is_on(&opened, w, 0, 0));
    }

    /// `close_brick == erode_brick(dilate_brick(x))` by construction.
    #[test]
    fn close_brick_matches_dilate_then_erode_composition() {
        let w = 9;
        let h = 9;
        let binary = to_binary(&[(4, 4)], w, h);
        let closed = close_brick(&binary, w, h, 3, 3);
        let dilated = dilate_brick(&binary, w, h, 3, 3);
        let manual = erode_brick(&dilated, w, h, 3, 3);
        assert_eq!(closed, manual);
    }

    #[test]
    fn hsize_vsize_one_is_identity_copy() {
        let w = 6;
        let h = 6;
        let binary = to_binary(&[(1, 1), (4, 5)], w, h);
        assert_eq!(dilate_brick(&binary, w, h, 1, 1), binary);
        assert_eq!(erode_brick(&binary, w, h, 1, 1), binary);
        assert_eq!(open_brick(&binary, w, h, 1, 1), binary);
        assert_eq!(close_brick(&binary, w, h, 1, 1), binary);
    }

    /// Even-size brick (hsize=2) has an asymmetric origin (cx = hsize/2 = 1
    /// via integer division), so xp=1, xn=hsize-1-cx=0 — the boundary clear
    /// only touches the left edge, not the right.
    #[test]
    fn even_size_brick_has_asymmetric_center() {
        let w = 6;
        let h = 6;
        let binary = vec![0_u8; w * h];
        let out = erode_brick(&binary, w, h, 2, 2);
        // Left/top column cleared (xp=yp=1), right/bottom column NOT
        // cleared by the boundary rule (xn=yn=0) for a size-2 brick.
        for y in 0..h {
            assert!(!is_on(&out, w, 0, y), "left column must be cleared");
        }
        for x in 0..w {
            assert!(!is_on(&out, w, x, 0), "top row must be cleared");
        }
        assert!(
            is_on(&out, w, w - 1, 3),
            "right column not cleared for even-size brick (xn=0)"
        );
    }
}
