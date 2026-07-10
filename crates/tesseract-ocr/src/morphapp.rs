//! Component-wise morphology + size selection — leptonica transcode
//! (`pixMorphSequenceByComponent`, `morphapp.c:198-245` +
//! `pixaMorphSequenceByComponent`, `morphapp.c:267-320`;
//! `pixSelectBySize`, `pixafunc1.c:219-277` + `pixaSelectBySize`), the last
//! two support leaves `pixGenTextblockMask` needs (v1.82.0 == the installed
//! liblept).
//!
//! ## What is transcoded, and from where
//!
//! - **[`morph_sequence_by_component`]** ⇄ `pixMorphSequenceByComponent`:
//!   extract every connected component as its OWN sub-image (the component's
//!   pixels only — NOT the bbox window's contents, so overlapping neighbors
//!   don't leak in), run the morphology sequence on each sub-image
//!   independently (the C routes through `pixMorphCompSequence`; see
//!   [`crate::morph::morph_sequence`]'s comp-equivalence note), and paint the
//!   results back at their bbox origins with OR (`PIX_PAINT`). Components
//!   whose bbox is smaller than `minw`/`minh` are **dropped entirely** — not
//!   copied through unchanged (`morphapp.c:297-312`: the gate skips the
//!   `pixaAddPix`, so the component simply never reaches the output).
//!   `minw`/`minh` of 0 are clamped to 1 (`morphapp.c:217-218`). Border
//!   effects are relative to each component's OWN frame, exactly as in the C
//!   (each sub-pix is its own little image).
//! - **[`select_by_size`]** ⇄ `pixSelectBySize`: keep exactly the connected
//!   components whose bbox satisfies the size predicate, repaint them, drop
//!   the rest. The predicate grid ([`SelectType`] × [`SelectRelation`])
//!   mirrors `L_SELECT_{WIDTH,HEIGHT,IF_EITHER,IF_BOTH}` ×
//!   `L_SELECT_IF_{LT,GT,LTE,GTE}`; the `IF_BOTH`+`GTE` cell is the one
//!   `pixGenTextblockMask` uses and the one the banked oracle pins
//!   bit-for-bit — the other cells follow the same documented semantics
//!   ("keep the components satisfying the relation", `pixafunc1.c:293-303`)
//!   but carry no oracle pin yet; they are exercised by hand-case tests only.
//!
//! ## Conventions
//!
//! Buffers use this crate's bitonal convention (`0` = ON/ink, `255` =
//! background), row-major. Connectivity is 4 or 8 (anything else → `None`,
//! the C's error).

use crate::morph::morph_sequence;

/// One extracted connected component: bbox origin + dims, and the
/// component's OWN pixels as a `w × h` sub-image (crate convention).
struct Component {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mask: Vec<u8>,
}

/// Label connected components (4/8-conn) and extract each as its own
/// sub-image — the `pixConnComp(pixs, &pixa, conn)` shape
/// (component pixels only, cropped to the component bbox). Raster-scan
/// seeded flood fill; component order is first-pixel raster order, which no
/// caller here depends on (paint-back is OR; selection is per-component).
fn extract_components(binary: &[u8], w: usize, h: usize, connectivity: u32) -> Vec<Component> {
    let mut visited = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut pixels: Vec<(usize, usize)> = Vec::new();

    for sy in 0..h {
        for sx in 0..w {
            if binary[sy * w + sx] != 0 || visited[sy * w + sx] {
                continue;
            }
            pixels.clear();
            visited[sy * w + sx] = true;
            stack.push((sx, sy));
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (sx, sy, sx, sy);
            while let Some((cx, cy)) = stack.pop() {
                pixels.push((cx, cy));
                min_x = min_x.min(cx);
                max_x = max_x.max(cx);
                min_y = min_y.min(cy);
                max_y = max_y.max(cy);
                let x0 = cx.saturating_sub(1);
                let x1 = (cx + 1).min(w - 1);
                let y0 = cy.saturating_sub(1);
                let y1 = (cy + 1).min(h - 1);
                for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        let diag = nx != cx && ny != cy;
                        if (nx == cx && ny == cy) || (connectivity == 4 && diag) {
                            continue;
                        }
                        if binary[ny * w + nx] == 0 && !visited[ny * w + nx] {
                            visited[ny * w + nx] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            let cw = max_x - min_x + 1;
            let ch = max_y - min_y + 1;
            let mut mask = vec![255u8; cw * ch];
            for &(px, py) in &pixels {
                mask[(py - min_y) * cw + (px - min_x)] = 0;
            }
            out.push(Component {
                x: min_x,
                y: min_y,
                w: cw,
                h: ch,
                mask,
            });
        }
    }
    out
}

/// Component-wise morphology sequence — `pixMorphSequenceByComponent`
/// (`morphapp.c:198-245`); see the module docs for the exact semantics
/// (own-frame morphology, OR paint-back, sub-minimum components DROPPED).
/// Returns `None` when `connectivity ∉ {4, 8}` or the sequence is invalid.
///
/// # Panics
/// Panics if `binary.len() != w * h`.
#[must_use]
pub fn morph_sequence_by_component(
    binary: &[u8],
    w: usize,
    h: usize,
    sequence: &str,
    connectivity: u32,
    minw: usize,
    minh: usize,
) -> Option<Vec<u8>> {
    if connectivity != 4 && connectivity != 8 {
        return None;
    }
    assert_eq!(binary.len(), w * h, "binary buffer length must be w * h");
    let minw = minw.max(1);
    let minh = minh.max(1);

    let mut out = vec![255u8; w * h];
    for comp in extract_components(binary, w, h, connectivity) {
        if comp.w < minw || comp.h < minh {
            continue; // dropped entirely, per the C's gate
        }
        let (res, rw, rh) = morph_sequence(&comp.mask, comp.w, comp.h, sequence)?;
        // Paint back with OR at the ORIGINAL box, clipped to it (the C
        // rasterops with the box geometry; a dimension-changing sequence
        // would be cropped to the box — none of the pageseg sequences do).
        for y in 0..comp.h.min(rh) {
            for x in 0..comp.w.min(rw) {
                if res[y * rw + x] == 0 {
                    out[(comp.y + y) * w + (comp.x + x)] = 0;
                }
            }
        }
    }
    Some(out)
}

/// Which dimension(s) [`select_by_size`]'s predicate tests —
/// `L_SELECT_{WIDTH,HEIGHT,IF_EITHER,IF_BOTH}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectType {
    /// Test the width only (height threshold ignored).
    Width,
    /// Test the height only (width threshold ignored).
    Height,
    /// Keep when EITHER dimension satisfies the relation.
    IfEither,
    /// Keep when BOTH dimensions satisfy the relation.
    IfBoth,
}

/// The comparison [`select_by_size`] applies — `L_SELECT_IF_{LT,GT,LTE,GTE}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectRelation {
    /// Keep components strictly smaller than the threshold.
    Lt,
    /// Keep components strictly larger than the threshold.
    Gt,
    /// Keep components at most the threshold.
    Lte,
    /// Keep components at least the threshold.
    Gte,
}

impl SelectRelation {
    fn holds(self, value: usize, threshold: usize) -> bool {
        match self {
            SelectRelation::Lt => value < threshold,
            SelectRelation::Gt => value > threshold,
            SelectRelation::Lte => value <= threshold,
            SelectRelation::Gte => value >= threshold,
        }
    }
}

/// The size predicate [`select_by_size`] applies — the
/// `(width, height, type, relation)` argument bundle of `pixSelectBySize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeFilter {
    /// Width threshold (ignored for [`SelectType::Height`]).
    pub width: usize,
    /// Height threshold (ignored for [`SelectType::Width`]).
    pub height: usize,
    /// Which dimension(s) the predicate tests.
    pub select_type: SelectType,
    /// The comparison applied to each tested dimension.
    pub relation: SelectRelation,
}

/// Keep exactly the connected components whose bbox satisfies the size
/// predicate — `pixSelectBySize` (`pixafunc1.c:219-277`); see the module
/// docs for the oracle-pin scope. Returns `None` when
/// `connectivity ∉ {4, 8}` (C error).
///
/// # Panics
/// Panics if `binary.len() != w * h`.
#[must_use]
pub fn select_by_size(
    binary: &[u8],
    w: usize,
    h: usize,
    connectivity: u32,
    filter: SizeFilter,
) -> Option<Vec<u8>> {
    if connectivity != 4 && connectivity != 8 {
        return None;
    }
    assert_eq!(binary.len(), w * h, "binary buffer length must be w * h");

    let mut out = vec![255u8; w * h];
    for comp in extract_components(binary, w, h, connectivity) {
        let w_ok = filter.relation.holds(comp.w, filter.width);
        let h_ok = filter.relation.holds(comp.h, filter.height);
        let keep = match filter.select_type {
            SelectType::Width => w_ok,
            SelectType::Height => h_ok,
            SelectType::IfEither => w_ok || h_ok,
            SelectType::IfBoth => w_ok && h_ok,
        };
        if keep {
            for y in 0..comp.h {
                for x in 0..comp.w {
                    if comp.mask[y * comp.w + x] == 0 {
                        out[(comp.y + y) * w + (comp.x + x)] = 0;
                    }
                }
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn by_component_runs_in_own_frame_and_drops_small() {
        // Two components: a 6×2 bar and a 1×1 speck. Sequence "d3.1" dilates
        // horizontally WITHIN each component's own frame — the bar grows to
        // its frame edges only (frame = its bbox, so no growth is visible
        // after paint-back), and the speck is dropped by minw=2.
        let (w, h) = (12usize, 5usize);
        let mut buf = vec![255u8; w * h];
        for x in 2..8 {
            buf[w + x] = 0; // bar row 1
            buf[2 * w + x] = 0; // bar row 2
        }
        buf[3 * w + 10] = 0; // speck
        let out = morph_sequence_by_component(&buf, w, h, "d3.1", 8, 2, 1).expect("valid");
        // Bar: dilation clipped to its own 6×2 frame → unchanged after paint.
        for x in 2..8 {
            assert_eq!(out[w + x], 0);
            assert_eq!(out[2 * w + x], 0);
        }
        // Speck dropped entirely.
        assert_eq!(out[3 * w + 10], 255, "sub-minw component must be dropped");
    }

    #[test]
    fn select_by_size_keeps_matching_components_only() {
        // 10×6: one 5×2 blob, one 2×1 blob.
        let (w, h) = (10usize, 6usize);
        let mut buf = vec![255u8; w * h];
        for y in 1..3 {
            for x in 1..6 {
                buf[y * w + x] = 0;
            }
        }
        buf[4 * w + 8] = 0;
        buf[4 * w + 9] = 0;

        // Keep components with w>=3 AND h>=2 → only the big blob.
        let out = select_by_size(
            &buf,
            w,
            h,
            8,
            SizeFilter {
                width: 3,
                height: 2,
                select_type: SelectType::IfBoth,
                relation: SelectRelation::Gte,
            },
        )
        .expect("valid");
        assert_eq!(out[w + 1], 0);
        assert_eq!(out[4 * w + 8], 255);

        // Keep SMALL: w<=2 (width-only, LTE) → only the little blob.
        let out = select_by_size(
            &buf,
            w,
            h,
            8,
            SizeFilter {
                width: 2,
                height: 0,
                select_type: SelectType::Width,
                relation: SelectRelation::Lte,
            },
        )
        .expect("valid");
        assert_eq!(out[w + 1], 255);
        assert_eq!(out[4 * w + 8], 0);
    }

    #[test]
    fn invalid_connectivity_is_rejected() {
        assert!(morph_sequence_by_component(&[255], 1, 1, "d3.3", 6, 0, 0).is_none());
        assert!(select_by_size(
            &[255],
            1,
            1,
            5,
            SizeFilter {
                width: 1,
                height: 1,
                select_type: SelectType::IfBoth,
                relation: SelectRelation::Gte,
            },
        )
        .is_none());
    }
}
