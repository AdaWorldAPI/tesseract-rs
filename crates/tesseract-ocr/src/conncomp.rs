//! Connected components — leptonica `conncomp.c` transcode (Batch 3B).
//!
//! Ports the **bounding-box** path only: `pixConnCompBB` → `pixSeedfillBB`
//! (`pixSeedfill4BB` / `pixSeedfill8BB`) using the `nextOnPixelInRaster` scan
//! and Heckbert's stack-based scanline seedfill (`pushFillsegBB`/`popFillseg`).
//! The Pixa (per-component sub-image) path (`pixConnCompPixa`) is out of
//! scope — this crate only needs bounding boxes.
//!
//! Per the banked manifest `.claude/harvest/leptonica-conncomp-callgraph.txt`
//! (`ruff_cpp_spo::walk_free_functions` on `/tmp/leptonica-src/conncomp.c`):
//! `pixConnComp → pixConnCompBB → {nextOnPixelInRaster, pixSeedfillBB}`, with
//! LEAF kernels `nextOnPixelInRasterLow`, `pushFillsegBB`, `pushFillseg`,
//! `popFillseg`. `pushFillseg` (the non-BB variant, used by `pixSeedfill{4,8}`
//! for the plain-erase/count paths) is not needed here.
//!
//! ## Input convention
//! `binary: &[u8]` is `w * h` bytes, one byte per pixel, row-major. Foreground
//! (ink/ON) is `byte == 0` — the same convention `threshold_rect_to_binary`
//! (`crate::threshold`) produces (`0` = foreground/black, `255` =
//! background/white).
//!
//! ## Word-trick → per-pixel transcription notes
//! Leptonica packs the 1bpp image into 32-bit words and `nextOnPixelInRasterLow`
//! (`conncomp.c:481-540`) skip-scans whole all-zero words as a speed
//! optimization; the *raster visitation order* is: from `(xstart, ystart)`
//! to the end of row `ystart`, then every pixel of every subsequent row in
//! order. A per-pixel scan in that exact order ([`next_on_pixel_in_raster`])
//! visits the identical sequence of pixels and returns the same first-ON
//! coordinate, so it is a faithful transcode of the *behavior*, not just an
//! approximation of it. `CLEAR_DATA_BIT`/`GET_DATA_BIT` likewise operate on
//! individual bits within a word; here they become direct byte
//! reads/writes on the `on: &mut [u8]` working copy (see [`get_bit`] /
//! [`clear_bit`]) — again a 1:1 per-pixel equivalent of the bit-level
//! accessor, not an approximation. `pixSetPadBits` (zeroing the unused tail
//! bits of a packed word beyond `w`) has no counterpart here since the byte
//! buffer has no padding.
//!
//! ## The `goto skip` control flow (`pixSeedfill{4,8}BB`, `conncomp.c:659-699`
//! / `772-812`)
//! Both seedfill loops contain a `goto skip` that jumps from *outside* the
//! `do { … } while (…)` loop directly into its body, past the initial
//! right-scan-and-push, on iterations where the just-popped segment's `x1`
//! (4-conn) / `x1 - 1` (8-conn) pixel was already OFF. This is transcoded as
//! a `skip_first: bool` flag that suppresses exactly the first pass through
//! the body's leading clear+push block, then is unconditionally cleared —
//! matching the C source's one-time-only jump (every later iteration of the
//! `do`/`while` runs the full body).

/// One connected component's bounding box: leptonica `BOX` `(x, y, w, h)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnCompBox {
    /// Left edge (`box->x`).
    pub x: i32,
    /// Top edge (`box->y`).
    pub y: i32,
    /// Width (`box->w`).
    pub w: i32,
    /// Height (`box->h`).
    pub h: i32,
}

/// A pending scanline segment awaiting expansion — leptonica `struct FillSeg`
/// (`conncomp.c:104-111`).
#[derive(Debug, Clone, Copy)]
struct FillSeg {
    xleft: i32,
    xright: i32,
    y: i32,
    dy: i32,
}

/// `pushFillsegBB` (`conncomp.c:1071-1113`). The auxiliary-stack fillseg
/// recycling in the C source is a memory-reuse optimization only — `Vec`
/// allocation makes it behaviorally identical without it.
#[allow(clippy::too_many_arguments)]
fn push_fillseg_bb(
    stack: &mut Vec<FillSeg>,
    xleft: i32,
    xright: i32,
    y: i32,
    dy: i32,
    ymax: i32,
    minx: &mut i32,
    maxx: &mut i32,
    miny: &mut i32,
    maxy: &mut i32,
) {
    *minx = (*minx).min(xleft);
    *maxx = (*maxx).max(xright);
    *miny = (*miny).min(y);
    *maxy = (*maxy).max(y);

    if y + dy >= 0 && y + dy <= ymax {
        stack.push(FillSeg {
            xleft,
            xright,
            y,
            dy,
        });
    }
}

/// `popFillseg` (`conncomp.c:1187-1216`). Returns `(xleft, xright, y, dy)`
/// with `y` already advanced by `dy` ("this now points to the new line"),
/// exactly as the C out-parameters do.
fn pop_fillseg(stack: &mut Vec<FillSeg>) -> Option<(i32, i32, i32, i32)> {
    let fseg = stack.pop()?;
    Some((fseg.xleft, fseg.xright, fseg.y + fseg.dy, fseg.dy))
}

/// `GET_DATA_BIT(line, x)` per-pixel equivalent: ON iff the working copy's
/// byte at `(x, y)` is nonzero (the working copy stores `1` for foreground,
/// `0` for erased/background — see [`conn_comp_bb`]).
#[inline]
fn get_bit(on: &[u8], w: i32, x: i32, y: i32) -> bool {
    on[(y * w + x) as usize] != 0
}

/// `CLEAR_DATA_BIT(line, x)` per-pixel equivalent.
#[inline]
fn clear_bit(on: &mut [u8], w: i32, x: i32, y: i32) {
    on[(y * w + x) as usize] = 0;
}

/// `nextOnPixelInRaster` / `nextOnPixelInRasterLow` (`conncomp.c:449-540`),
/// transcoded as a direct per-pixel scan in the same raster order (see the
/// module-level word-trick note). Returns `Some((x, y))` of the first ON
/// pixel at or after `(xstart, ystart)` in raster order, or `None`.
fn next_on_pixel_in_raster(
    on: &[u8],
    w: i32,
    h: i32,
    xstart: i32,
    ystart: i32,
) -> Option<(i32, i32)> {
    for x in xstart..w {
        if get_bit(on, w, x, ystart) {
            return Some((x, ystart));
        }
    }
    for y in (ystart + 1)..h {
        for x in 0..w {
            if get_bit(on, w, x, y) {
                return Some((x, y));
            }
        }
    }
    None
}

/// `pixSeedfill4BB` (`conncomp.c:619-705`). Erases the 4-connected component
/// seeded at `(xseed, yseed)` (which must be ON) from `on`, returning its
/// bounding box. `stack` is drained to empty on return (as in the C: the
/// `while (lstackGetCount(stack) > 0)` loop only exits when empty).
fn seedfill4_bb(
    on: &mut [u8],
    w: i32,
    h: i32,
    xseed: i32,
    yseed: i32,
    stack: &mut Vec<FillSeg>,
) -> ConnCompBox {
    let xmax = w - 1;
    let ymax = h - 1;

    let mut minx = 100_000_i32;
    let mut miny = 100_000_i32;
    let mut maxx = 0_i32;
    let mut maxy = 0_i32;
    push_fillseg_bb(
        stack, xseed, xseed, yseed, 1, ymax, &mut minx, &mut maxx, &mut miny, &mut maxy,
    );
    push_fillseg_bb(
        stack,
        xseed,
        xseed,
        yseed + 1,
        -1,
        ymax,
        &mut minx,
        &mut maxx,
        &mut miny,
        &mut maxy,
    );
    minx = xseed;
    maxx = xseed;
    miny = yseed;
    maxy = yseed;

    while let Some((x1, x2, y, dy)) = pop_fillseg(stack) {
        // for (x = x1; x >= 0 && GET_DATA_BIT(line, x); x--) CLEAR_DATA_BIT(line, x);
        let mut x = x1;
        while x >= 0 && get_bit(on, w, x, y) {
            clear_bit(on, w, x, y);
            x -= 1;
        }

        // if (x >= x1) goto skip;  -- x1's pixel was already OFF, no clearing happened
        let mut skip_first = x >= x1;
        let mut xstart = 0_i32;
        if !skip_first {
            xstart = x + 1;
            if xstart < x1 - 1 {
                push_fillseg_bb(
                    stack,
                    xstart,
                    x1 - 1,
                    y,
                    -dy,
                    ymax,
                    &mut minx,
                    &mut maxx,
                    &mut miny,
                    &mut maxy,
                );
            }
            x = x1 + 1;
        }

        loop {
            if !skip_first {
                // for (; x <= xmax && GET_DATA_BIT(line, x); x++) CLEAR_DATA_BIT(line, x);
                while x <= xmax && get_bit(on, w, x, y) {
                    clear_bit(on, w, x, y);
                    x += 1;
                }
                push_fillseg_bb(
                    stack,
                    xstart,
                    x - 1,
                    y,
                    dy,
                    ymax,
                    &mut minx,
                    &mut maxx,
                    &mut miny,
                    &mut maxy,
                );
                if x > x2 + 1 {
                    push_fillseg_bb(
                        stack,
                        x2 + 1,
                        x - 1,
                        y,
                        -dy,
                        ymax,
                        &mut minx,
                        &mut maxx,
                        &mut miny,
                        &mut maxy,
                    );
                }
            }
            skip_first = false;

            // skip: for (x++; x <= x2 && x <= xmax && !GET_DATA_BIT(line, x); x++);
            x += 1;
            while x <= x2 && x <= xmax && !get_bit(on, w, x, y) {
                x += 1;
            }
            xstart = x;

            if !(x <= x2 && x <= xmax) {
                break;
            }
        }
    }

    ConnCompBox {
        x: minx,
        y: miny,
        w: maxx - minx + 1,
        h: maxy - miny + 1,
    }
}

/// `pixSeedfill8BB` (`conncomp.c:732-818`). Same shape as [`seedfill4_bb`]
/// with the 8-connectivity boundary offsets (`x1 - 1` / `x1` / `x2` instead
/// of `x1` / `x1 + 1` / `x2 + 1`) — see `conncomp.c`'s own comment that this
/// "follows Heckbert's closely, except the leak checks are changed for 8
/// connectivity."
fn seedfill8_bb(
    on: &mut [u8],
    w: i32,
    h: i32,
    xseed: i32,
    yseed: i32,
    stack: &mut Vec<FillSeg>,
) -> ConnCompBox {
    let xmax = w - 1;
    let ymax = h - 1;

    let mut minx = 100_000_i32;
    let mut miny = 100_000_i32;
    let mut maxx = 0_i32;
    let mut maxy = 0_i32;
    push_fillseg_bb(
        stack, xseed, xseed, yseed, 1, ymax, &mut minx, &mut maxx, &mut miny, &mut maxy,
    );
    push_fillseg_bb(
        stack,
        xseed,
        xseed,
        yseed + 1,
        -1,
        ymax,
        &mut minx,
        &mut maxx,
        &mut miny,
        &mut maxy,
    );
    minx = xseed;
    maxx = xseed;
    miny = yseed;
    maxy = yseed;

    while let Some((x1, x2, y, dy)) = pop_fillseg(stack) {
        // for (x = x1 - 1; x >= 0 && GET_DATA_BIT(line, x); x--) CLEAR_DATA_BIT(line, x);
        let mut x = x1 - 1;
        while x >= 0 && get_bit(on, w, x, y) {
            clear_bit(on, w, x, y);
            x -= 1;
        }

        // if (x >= x1 - 1) goto skip;
        let mut skip_first = x >= x1 - 1;
        let mut xstart = 0_i32;
        if !skip_first {
            xstart = x + 1;
            if xstart < x1 {
                push_fillseg_bb(
                    stack,
                    xstart,
                    x1 - 1,
                    y,
                    -dy,
                    ymax,
                    &mut minx,
                    &mut maxx,
                    &mut miny,
                    &mut maxy,
                );
            }
            x = x1;
        }

        loop {
            if !skip_first {
                // for (; x <= xmax && GET_DATA_BIT(line, x); x++) CLEAR_DATA_BIT(line, x);
                while x <= xmax && get_bit(on, w, x, y) {
                    clear_bit(on, w, x, y);
                    x += 1;
                }
                push_fillseg_bb(
                    stack,
                    xstart,
                    x - 1,
                    y,
                    dy,
                    ymax,
                    &mut minx,
                    &mut maxx,
                    &mut miny,
                    &mut maxy,
                );
                if x > x2 {
                    push_fillseg_bb(
                        stack,
                        x2 + 1,
                        x - 1,
                        y,
                        -dy,
                        ymax,
                        &mut minx,
                        &mut maxx,
                        &mut miny,
                        &mut maxy,
                    );
                }
            }
            skip_first = false;

            // skip: for (x++; x <= x2 + 1 && x <= xmax && !GET_DATA_BIT(line, x); x++);
            x += 1;
            while x <= x2 + 1 && x <= xmax && !get_bit(on, w, x, y) {
                x += 1;
            }
            xstart = x;

            if !(x <= x2 + 1 && x <= xmax) {
                break;
            }
        }
    }

    ConnCompBox {
        x: minx,
        y: miny,
        w: maxx - minx + 1,
        h: maxy - miny + 1,
    }
}

/// `pixConnCompBB` (`conncomp.c:306-371`) — bounding boxes of the 4- or
/// 8-connected components of a binary image, in raster-scan seed order.
///
/// `binary` is `w * h` bytes, foreground = `byte == 0` (see module docs).
/// `connectivity` must be `4` or `8`.
///
/// # Panics
/// Panics if `connectivity` is neither `4` nor `8`, or if `binary.len() !=
/// w * h`.
pub fn conn_comp_bb(binary: &[u8], w: usize, h: usize, connectivity: u32) -> Vec<ConnCompBox> {
    assert!(
        connectivity == 4 || connectivity == 8,
        "connectivity must be 4 or 8"
    );
    assert_eq!(binary.len(), w * h, "binary buffer length must be w * h");

    if w == 0 || h == 0 {
        return Vec::new();
    }

    let wi = i32::try_from(w).expect("width fits in i32");
    let hi = i32::try_from(h).expect("height fits in i32");

    // Working copy: 1 = foreground (ON), 0 = background/erased. Erasure
    // happens in place on this copy, mirroring pixCopy(NULL, pixs) + the
    // in-place CLEAR_DATA_BIT erasure in the C source.
    let mut on: Vec<u8> = binary.iter().map(|&b| u8::from(b == 0)).collect();

    // pixZero(pixs, &iszero); if (iszero) return boxaCreate(1); -- an empty
    // boxa either way, so an empty Vec is the faithful result.
    if on.iter().all(|&v| v == 0) {
        return Vec::new();
    }

    let mut boxes = Vec::new();
    let mut stack: Vec<FillSeg> = Vec::new();
    let mut xstart = 0_i32;
    let mut ystart = 0_i32;
    while let Some((x, y)) = next_on_pixel_in_raster(&on, wi, hi, xstart, ystart) {
        let bb = if connectivity == 4 {
            seedfill4_bb(&mut on, wi, hi, x, y, &mut stack)
        } else {
            seedfill8_bb(&mut on, wi, hi, x, y, &mut stack)
        };
        boxes.push(bb);

        xstart = x;
        ystart = y;
    }

    boxes
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

    #[test]
    fn single_pixel_component() {
        let w = 5;
        let h = 5;
        let binary = to_binary(&[(2, 2)], w, h);
        let boxes4 = conn_comp_bb(&binary, w, h, 4);
        assert_eq!(
            boxes4,
            vec![ConnCompBox {
                x: 2,
                y: 2,
                w: 1,
                h: 1
            }]
        );
        let boxes8 = conn_comp_bb(&binary, w, h, 8);
        assert_eq!(
            boxes8,
            vec![ConnCompBox {
                x: 2,
                y: 2,
                w: 1,
                h: 1
            }]
        );
    }

    #[test]
    fn two_diagonal_pixels_4_vs_8_connectivity() {
        let w = 4;
        let h = 4;
        let binary = to_binary(&[(1, 1), (2, 2)], w, h);

        let boxes4 = conn_comp_bb(&binary, w, h, 4);
        assert_eq!(
            boxes4.len(),
            2,
            "diagonal pixels are separate components under 4-connectivity"
        );
        assert_eq!(
            boxes4[0],
            ConnCompBox {
                x: 1,
                y: 1,
                w: 1,
                h: 1
            }
        );
        assert_eq!(
            boxes4[1],
            ConnCompBox {
                x: 2,
                y: 2,
                w: 1,
                h: 1
            }
        );

        let boxes8 = conn_comp_bb(&binary, w, h, 8);
        assert_eq!(
            boxes8.len(),
            1,
            "diagonal pixels merge into one component under 8-connectivity"
        );
        assert_eq!(
            boxes8[0],
            ConnCompBox {
                x: 1,
                y: 1,
                w: 2,
                h: 2
            }
        );
    }

    #[test]
    fn full_black_image_is_one_component() {
        let w = 6;
        let h = 4;
        let binary = vec![0_u8; w * h];
        let boxes4 = conn_comp_bb(&binary, w, h, 4);
        assert_eq!(
            boxes4,
            vec![ConnCompBox {
                x: 0,
                y: 0,
                w: w as i32,
                h: h as i32
            }]
        );
        let boxes8 = conn_comp_bb(&binary, w, h, 8);
        assert_eq!(
            boxes8,
            vec![ConnCompBox {
                x: 0,
                y: 0,
                w: w as i32,
                h: h as i32
            }]
        );
    }

    #[test]
    fn empty_image_has_no_components() {
        let w = 8;
        let h = 8;
        let binary = vec![255_u8; w * h];
        assert!(conn_comp_bb(&binary, w, h, 4).is_empty());
        assert!(conn_comp_bb(&binary, w, h, 8).is_empty());
    }
}
