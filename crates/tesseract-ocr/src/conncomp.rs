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
//!
//! ## Batch 3F₂ leaf 1 — per-component ink pixel count ([`ConnComp`])
//! [`conn_comp_areas`] extends the same seedfill walk to also return each
//! component's **ink pixel count** — the source of `BLOBNBOX::enclosed_area()`
//! (`blobbox.h:150`, `area = static_cast<int>(srcblob->area())`, backed by
//! `C_OUTLINE::area()`, `coutln.cpp:257-282`). The seedfill already visits
//! every foreground pixel of the component exactly once (each `clear_bit`
//! call clears a pixel that just tested ON, so it is never double-counted);
//! [`seedfill4_bb`]/[`seedfill8_bb`] thread a running counter through every
//! `clear_bit` call site and return it alongside the (unchanged)
//! [`ConnCompBox`] — see each function's doc for the exact call sites.
//! [`conn_comp_bb`] is refactored to be a thin `.map(|c| c.bb)` wrapper over
//! [`conn_comp_areas`], which *by construction* keeps its box output
//! byte-identical to the pre-leaf-1 implementation (also covered by an
//! explicit regression test).
//!
//! ### `enclosed_area()` provenance and a documented divergence
//! `C_OUTLINE::area()` computes area via Green's theorem walked around the
//! *crack boundary* of the component's traced outline, **plus its `children`
//! outlines' areas** (`coutln.cpp:280-282`, `total += it.data()->area()`).
//! For a simple blob this nets to exactly the ink pixel count. For a blob
//! with topology, the winding sign alternates with nesting depth: a hole
//! (a child outline, opposite winding) *subtracts* its area, and an island
//! sitting inside that hole (a grandchild outline, back to the outer
//! winding) *adds* its area back — because in Tesseract's edge tracer, a
//! hole and any island inside it are still walked as descendants of the
//! *same* `C_BLOB`/`C_OUTLINE` tree as the outer boundary.
//!
//! A flat pixel-count seedfill (this port, and leptonica's own
//! `pixConnCompPixa` + `pixCountPixels`, which the byte-parity oracle uses)
//! does **not** reproduce that "island-in-a-hole" fold-back: under 4- or
//! 8-connected seedfill, an island inside a hole never touches the
//! surrounding ring's foreground pixels (the hole's background pixels
//! separate them), so `pixConnComp` reports it as its own **separate**
//! component with its own separate `pixel_count`/bounding box, rather than
//! folding its area into the parent ring's count. This is a genuine,
//! known, and accepted divergence between "outline Green's-theorem area
//! with nested-children folding" (real Tesseract) and "flat
//! pixel-count-based area via connected-component seedfill" (this port) —
//! it manifests *only* on an island-strictly-inside-a-hole topology (an
//! isolated dot inside a letter's counter, for example), which per the
//! orchestrator's analysis is rare-to-nonexistent in Latin OCR corpora.
//! Test fixtures in this module and the byte-parity oracle deliberately
//! avoid constructing that topology; a simple hole (no island inside it,
//! e.g. the counters of "o"/"e"/"8") is unaffected and matches exactly,
//! since it involves only one level of nesting.

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

/// A connected component's bounding box plus its **ink pixel count** — the
/// `BLOBNBOX::enclosed_area()` source (see the module doc's "Batch 3F₂ leaf
/// 1" section for the exact provenance and a documented island-in-a-hole
/// divergence). Produced by [`conn_comp_areas`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnComp {
    /// The component's bounding box — byte-identical to what
    /// [`conn_comp_bb`] would report for the same component.
    pub bb: ConnCompBox,
    /// Count of foreground (`byte == 0`) pixels belonging to this
    /// component — every pixel the seedfill actually erases, counted
    /// exactly once.
    pub pixel_count: i32,
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
/// bounding box plus its ink pixel count (Batch 3F₂ leaf 1 — every
/// `clear_bit` call below clears a pixel that just tested ON, so counting
/// those calls is an exact, non-duplicating pixel tally; the box-computing
/// logic itself is untouched from the pre-leaf-1 version). `stack` is
/// drained to empty on return (as in the C: the `while
/// (lstackGetCount(stack) > 0)` loop only exits when empty).
fn seedfill4_bb(
    on: &mut [u8],
    w: i32,
    h: i32,
    xseed: i32,
    yseed: i32,
    stack: &mut Vec<FillSeg>,
) -> (ConnCompBox, i32) {
    let xmax = w - 1;
    let ymax = h - 1;
    let mut pixel_count: i32 = 0;

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
            pixel_count += 1;
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
                    pixel_count += 1;
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

    (
        ConnCompBox {
            x: minx,
            y: miny,
            w: maxx - minx + 1,
            h: maxy - miny + 1,
        },
        pixel_count,
    )
}

/// `pixSeedfill8BB` (`conncomp.c:732-818`). Same shape as [`seedfill4_bb`]
/// with the 8-connectivity boundary offsets (`x1 - 1` / `x1` / `x2` instead
/// of `x1` / `x1 + 1` / `x2 + 1`) — see `conncomp.c`'s own comment that this
/// "follows Heckbert's closely, except the leak checks are changed for 8
/// connectivity." Also returns the ink pixel count (Batch 3F₂ leaf 1) —
/// see [`seedfill4_bb`]'s doc for why counting `clear_bit` calls is exact.
fn seedfill8_bb(
    on: &mut [u8],
    w: i32,
    h: i32,
    xseed: i32,
    yseed: i32,
    stack: &mut Vec<FillSeg>,
) -> (ConnCompBox, i32) {
    let xmax = w - 1;
    let ymax = h - 1;
    let mut pixel_count: i32 = 0;

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
            pixel_count += 1;
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
                    pixel_count += 1;
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

    (
        ConnCompBox {
            x: minx,
            y: miny,
            w: maxx - minx + 1,
            h: maxy - miny + 1,
        },
        pixel_count,
    )
}

/// `pixConnCompBB` (`conncomp.c:306-371`) — bounding boxes of the 4- or
/// 8-connected components of a binary image, in raster-scan seed order.
///
/// `binary` is `w * h` bytes, foreground = `byte == 0` (see module docs).
/// `connectivity` must be `4` or `8`.
///
/// A thin wrapper over [`conn_comp_areas`] (Batch 3F₂ leaf 1) — the box
/// computation itself is unchanged, so this is byte-identical to the
/// pre-leaf-1 implementation *by construction*, not merely by observation
/// (also covered by an explicit regression test below).
///
/// # Panics
/// Panics if `connectivity` is neither `4` nor `8`, or if `binary.len() !=
/// w * h`.
pub fn conn_comp_bb(binary: &[u8], w: usize, h: usize, connectivity: u32) -> Vec<ConnCompBox> {
    conn_comp_areas(binary, w, h, connectivity)
        .into_iter()
        .map(|c| c.bb)
        .collect()
}

/// `pixConnCompBB`'s bounding-box walk, extended (Batch 3F₂ leaf 1) to also
/// report each component's ink pixel count (the `BLOBNBOX::enclosed_area()`
/// source — see the module doc). Same raster-scan seed order as
/// [`conn_comp_bb`]; `binary`/`connectivity` have the same conventions and
/// panics.
pub fn conn_comp_areas(binary: &[u8], w: usize, h: usize, connectivity: u32) -> Vec<ConnComp> {
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

    let mut comps = Vec::new();
    let mut stack: Vec<FillSeg> = Vec::new();
    let mut xstart = 0_i32;
    let mut ystart = 0_i32;
    while let Some((x, y)) = next_on_pixel_in_raster(&on, wi, hi, xstart, ystart) {
        let (bb, pixel_count) = if connectivity == 4 {
            seedfill4_bb(&mut on, wi, hi, x, y, &mut stack)
        } else {
            seedfill8_bb(&mut on, wi, hi, x, y, &mut stack)
        };
        comps.push(ConnComp { bb, pixel_count });

        xstart = x;
        ystart = y;
    }

    comps
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

    // ------------------------------------------------------------------
    // Batch 3F₂ leaf 1 — conn_comp_areas / pixel_count
    // ------------------------------------------------------------------

    /// The session-standard synthetic grey image (`conncomp_dump.rs`'s
    /// `synthetic_grey`): `((x*37 + y*11) ^ (x*y)) % 256`.
    fn session_standard_synthetic(w: usize, h: usize) -> Vec<u8> {
        let mut grey = vec![0_u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 37 + y * 11) ^ (x * y)) % 256;
                grey[y * w + x] = u8::try_from(v).expect("mod 256 fits u8");
            }
        }
        grey
    }

    /// Regression pin (mandated by the Batch 3F₂ leaf-1 spec): [`conn_comp_bb`]
    /// must stay byte-identical to `conn_comp_areas(...).map(bb)` -- true by
    /// construction here (see [`conn_comp_bb`]'s doc), but pinned as an
    /// explicit test so a future edit that re-duplicates the box logic
    /// cannot silently drift the two apart.
    #[test]
    fn conn_comp_areas_bb_matches_conn_comp_bb_on_session_synthetic() {
        let (w, h) = (24, 36);
        let grey = session_standard_synthetic(w, h);
        let otsu = crate::threshold::otsu_threshold_gray(&grey, w, 0, 0, w, h);
        let binary = crate::threshold::threshold_rect_to_binary(&grey, w, 0, 0, w, h, otsu);

        for connectivity in [4, 8] {
            let areas = conn_comp_areas(&binary, w, h, connectivity);
            let boxes_via_areas: Vec<ConnCompBox> = areas.iter().map(|c| c.bb).collect();
            let boxes_direct = conn_comp_bb(&binary, w, h, connectivity);
            assert_eq!(
                boxes_via_areas, boxes_direct,
                "conn_comp_areas(...).map(bb) must equal conn_comp_bb(...) (connectivity={connectivity})"
            );
            assert!(
                !areas.is_empty(),
                "session-standard synthetic must yield at least one component"
            );
            // Every reported pixel_count must be positive and never exceed
            // the component's own bounding-box area (a hole can only ever
            // make enclosed ink area <= bbox area for these fixtures, which
            // contain no island-in-a-hole topology -- see module doc).
            for c in &areas {
                assert!(c.pixel_count > 0);
                assert!(c.pixel_count <= c.bb.w * c.bb.h);
            }
        }
    }

    #[test]
    fn pixel_count_single_pixel_component() {
        let w = 5;
        let h = 5;
        let binary = to_binary(&[(2, 2)], w, h);
        let areas = conn_comp_areas(&binary, w, h, 8);
        assert_eq!(areas.len(), 1);
        assert_eq!(areas[0].pixel_count, 1);
        assert_eq!(
            areas[0].bb,
            ConnCompBox {
                x: 2,
                y: 2,
                w: 1,
                h: 1
            }
        );
    }

    #[test]
    fn pixel_count_l_shape() {
        // An L-shaped tromino: bbox is 2x2 (area 4) but only 3 pixels are
        // ink, so pixel_count (3) < bbox area (4) -- enclosed_area is a
        // strictly tighter measure than the bounding box for a non-filling
        // shape, exactly as filter_noise_blobs's noise-area-ratio test
        // (Batch 3F₂ leaf 2) relies on.
        let w = 4;
        let h = 4;
        let binary = to_binary(&[(1, 1), (1, 2), (2, 2)], w, h);
        let areas = conn_comp_areas(&binary, w, h, 8);
        assert_eq!(areas.len(), 1);
        assert_eq!(areas[0].pixel_count, 3);
        assert_eq!(
            areas[0].bb,
            ConnCompBox {
                x: 1,
                y: 1,
                w: 2,
                h: 2
            }
        );
        assert!(areas[0].pixel_count < areas[0].bb.w * areas[0].bb.h);
    }

    #[test]
    fn pixel_count_ring_with_hole_excludes_hole() {
        // A 3x3 ring (8 ink pixels around one background pixel in the
        // center) under 8-connectivity: this is the "simple hole, no
        // island inside it" case the module doc says matches Tesseract's
        // enclosed_area() exactly (single level of nesting, no fold-back
        // needed) -- pixel_count must be 8 (the hole's background pixel is
        // never counted), not 9 (the full bounding-box area).
        let w = 5;
        let h = 5;
        let ring: Vec<(usize, usize)> = vec![
            (1, 1),
            (2, 1),
            (3, 1),
            (1, 2),
            (3, 2),
            (1, 3),
            (2, 3),
            (3, 3),
        ];
        let binary = to_binary(&ring, w, h);
        let areas = conn_comp_areas(&binary, w, h, 8);
        assert_eq!(areas.len(), 1);
        assert_eq!(areas[0].pixel_count, 8);
        assert_eq!(
            areas[0].bb,
            ConnCompBox {
                x: 1,
                y: 1,
                w: 3,
                h: 3
            }
        );
        assert_eq!(areas[0].bb.w * areas[0].bb.h, 9);
    }
}
