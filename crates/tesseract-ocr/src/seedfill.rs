//! Binary seedfill (morphological reconstruction) — leptonica transcode
//! (`pixSeedfillBinary`, `seedfill.c:247-298`, v1.82.0 == the installed
//! liblept).
//!
//! ## What is transcoded, and from where
//!
//! `pixSeedfillBinary(NULL, pixs, pixm, connectivity)` copies the seed, then
//! iterates `seedfillBinaryLow` raster/anti-raster sweeps (each sweep ORs a
//! pixel with its already-visited neighbors, then ANDs with the mask) until a
//! fixpoint, capped at `MaxIters = 40` sweep pairs (`seedfill.c:209,284-294`).
//! The fixpoint is exactly the morphological reconstruction of the mask from
//! the seed: **the union of all mask-connected components (4- or 8-conn) that
//! contain at least one `seed ∩ mask` pixel.** A seed pixel OUTSIDE the mask
//! never propagates (its own value is masked before any neighbor that could
//! read it un-masked does — traced through the raster order in the C).
//!
//! This port computes that fixpoint directly by flood fill (BFS). The one
//! semantic difference is deliberate and documented: leptonica returns the
//! PARTIAL fill if a pathologically serpentine mask needs more than
//! `MaxIters = 40` sweep pairs to converge; the BFS always returns the true
//! fixpoint. No real page-segmentation mask comes close to the cap (the
//! banked oracle fixtures — including a 20+-tile diagonal cascade — converge
//! in a handful of sweeps, and the oracle comparison would fail if the cap
//! had bound).
//!
//! ## Size-mismatch semantics (load-bearing for the halftone-mask caller)
//!
//! `pixGenerateHalftoneMask` calls this with a seed that is SMALLER than the
//! mask whenever the page dimensions are not multiples of 4 (the
//! `cascade(4,4) → expand ×4` chain floors twice). The C's `seedfillBinaryLow`
//! runs over the SEED's grid, reading the mask clipped to its own extent
//! (`hm` is passed separately "so it can clip", `seedfill.c:277`). This port
//! mirrors that: the result has the seed's dimensions; a cell is fillable iff
//! it lies inside BOTH grids and the mask is ON there. Pinned by the banked
//! oracle's `seedfill_mismatch` section (56×44 seed against a 61×47 mask).
//!
//! ## Conventions
//!
//! Buffers use this crate's bitonal convention (`0` = ON/ink, `255` =
//! background — `threshold.rs`), row-major.

/// Binary seedfill — `pixSeedfillBinary` (`seedfill.c:247-298`); see the
/// module docs for the reconstruction semantics, the `MaxIters` nuance, and
/// the size-mismatch rule. Returns a buffer of the SEED's dimensions, or
/// `None` when `connectivity ∉ {4, 8}` (C error).
///
/// # Panics
/// Panics if `seed.len() != sw * sh` or `mask.len() != mw * mh`.
#[must_use]
pub fn seedfill_binary(
    seed: &[u8],
    sw: usize,
    sh: usize,
    mask: &[u8],
    mw: usize,
    mh: usize,
    connectivity: u32,
) -> Option<Vec<u8>> {
    if connectivity != 4 && connectivity != 8 {
        return None;
    }
    assert_eq!(seed.len(), sw * sh, "seed buffer length must be sw * sh");
    assert_eq!(mask.len(), mw * mh, "mask buffer length must be mw * mh");

    let fillable = |x: usize, y: usize| x < mw && y < mh && mask[y * mw + x] == 0;

    let mut out = vec![255u8; sw * sh];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for y in 0..sh {
        for x in 0..sw {
            if seed[y * sw + x] == 0 && fillable(x, y) && out[y * sw + x] != 0 {
                out[y * sw + x] = 0;
                stack.push((x, y));
                while let Some((cx, cy)) = stack.pop() {
                    let x0 = cx.saturating_sub(1);
                    let x1 = (cx + 1).min(sw - 1);
                    let y0 = cy.saturating_sub(1);
                    let y1 = (cy + 1).min(sh - 1);
                    for ny in y0..=y1 {
                        for nx in x0..=x1 {
                            let diag = nx != cx && ny != cy;
                            if (nx == cx && ny == cy) || (connectivity == 4 && diag) {
                                continue;
                            }
                            if out[ny * sw + nx] != 0 && fillable(nx, ny) {
                                out[ny * sw + nx] = 0;
                                stack.push((nx, ny));
                            }
                        }
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

    /// Hand case: two mask blobs, seed in one — only that blob fills; a seed
    /// OUTSIDE the mask is dead (never propagates, per the C trace).
    #[test]
    fn fills_only_the_seeded_component_and_kills_outside_seeds() {
        // 7×3 mask: blob A = cols 0..3, blob B = cols 4..7 (row 1 only), gap at col 3.
        let mut mask = vec![255u8; 21];
        for x in 0..3 {
            mask[7 + x] = 0;
        }
        for x in 4..7 {
            mask[7 + x] = 0;
        }
        // Seed: one dot in blob A; one dot OUTSIDE the mask (row 0).
        let mut seed = vec![255u8; 21];
        seed[7] = 0; // (0,1) in blob A
        seed[3] = 0; // (3,0) outside the mask entirely

        let out = seedfill_binary(&seed, 7, 3, &mask, 7, 3, 4).expect("valid conn");
        let on: Vec<usize> = out
            .iter()
            .enumerate()
            .filter(|(_, &p)| p == 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(on, vec![7, 8, 9], "only blob A fills: {on:?}");
    }

    #[test]
    fn invalid_connectivity_is_an_error() {
        assert!(seedfill_binary(&[255], 1, 1, &[255], 1, 1, 6).is_none());
        assert!(seedfill_binary(&[255], 1, 1, &[255], 1, 1, 0).is_none());
    }
}
