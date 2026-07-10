//! Halftone (image-region) mask generation — leptonica transcode
//! (`pixGenerateHalftoneMask`, `pageseg.c:305-363`, v1.82.0 == the installed
//! liblept; `pixGenHalftoneMask` at `pageseg.c:280-287` is a deprecated
//! ABI-compat wrapper over the same body).
//!
//! **This is the "is it a picture?" half of the region classifier**: the mask
//! it returns covers the halftone/image regions of a binarized page, and the
//! returned text image is everything NOT under that mask — the input to the
//! textline/textblock mask generators (`pixGenTextlineMask` /
//! `pixGenTextblockMask`, same file; future leaves). Every brick it composes
//! is individually parity-proven in this crate: rank cascade + replicate
//! expansion ([`crate::binreduce`]), brick open + safe close
//! ([`crate::morph`]), binary seedfill ([`crate::seedfill`]).
//!
//! ## The transcoded chain (`pageseg.c:326-362`)
//!
//! ```text
//! seed = expand_replicate( open_brick( cascade(src, [4,4,0,0]), 5×5 ), ×4 )
//!        // "halftone parts at 8x reduction … back to 2x" — only regions
//!        // dense enough to survive rank-4 twice AND a 5×5 opening at /4
//!        // scale (i.e. a hole-free ≥20×20 core at full resolution) seed
//! mask = close_safe_brick(src, 4, 4)      // connected-region mask
//! filled = seedfill_binary(seed, mask, 4) // grow seed through the mask
//! found  = filled has any ON pixel
//! text   = src AND NOT filled             // clipped to the overlap
//! ```
//!
//! ## Size semantics (deliberate, oracle-pinned)
//!
//! When `w`/`h` are not multiples of 4, the cascade floors twice and the ×4
//! expansion lands SHORT: the returned mask has dimensions
//! `(w/4)·4 × (h/4)·4` — smaller than the input, exactly as the C's `pixd`
//! does (its seedfill result is seed-sized). The text image is full-sized:
//! the subtraction runs over the overlap (the C's clipped rasterop), so
//! input pixels beyond the mask's extent pass through unchanged. Pinned by
//! the banked oracle on a 130×117 fixture → 128×116 mask.
//!
//! ## Parity
//!
//! Proven against the REAL `pixGenerateHalftoneMask` via the banked oracle
//! (`.claude/harvest/oracles/pageseg_oracle.*`): both the `found = 0` arm
//! (a dithered block too sparse to seed — mask empty, text == input copy)
//! and the `found = 1` arm (a solid block — the real fill), every output bit
//! and both flag values identical. The oracle also pins each sub-leaf
//! separately (safe close ×3 sizes, seedfill 4/8-conn + size mismatch,
//! replicate ×3/×4) — see the tests below, which drive ALL of those
//! comparisons from the one banked dump.
//!
//! ## Conventions
//!
//! Buffers use this crate's bitonal convention (`0` = ON/ink, `255` =
//! background), row-major. The input is a binarized page (e.g. from
//! [`crate::threshold`]), assumed 150–200 ppi per the C's header comment.

use crate::binreduce::{expand_replicate, reduce_rank_binary_cascade};
use crate::morph::{close_safe_brick, morph_sequence, open_brick};
use crate::morphapp::{
    morph_sequence_by_component, select_by_size, SelectRelation, SelectType, SizeFilter,
};
use crate::seedfill::seedfill_binary;

/// `MinWidth` (`pageseg.c:90`) — inputs narrower than this are rejected.
pub const MIN_WIDTH: usize = 100;
/// `MinHeight` (`pageseg.c:91`) — inputs shorter than this are rejected.
pub const MIN_HEIGHT: usize = 100;

/// The result of [`generate_halftone_mask`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HalftoneMask {
    /// The halftone mask, `mask_w × mask_h` (this crate's `0` = ON
    /// convention). Dimensions are `(w/4)·4 × (h/4)·4` — smaller than the
    /// input when `w`/`h` are not multiples of 4 (see the module docs).
    pub mask: Vec<u8>,
    /// Mask width in pixels.
    pub mask_w: usize,
    /// Mask height in pixels.
    pub mask_h: usize,
    /// The text image (input minus mask, clipped to the overlap), always the
    /// full input `w × h`.
    pub text: Vec<u8>,
    /// `true` iff the mask has at least one ON pixel (`*phtfound` in the C).
    pub found: bool,
}

/// Generate the halftone/image-region mask of a binarized page —
/// `pixGenerateHalftoneMask` (`pageseg.c:305-363`); see the module docs for
/// the chain, the size semantics, and the parity evidence. Returns `None`
/// when `w < `[`MIN_WIDTH`]` || h < `[`MIN_HEIGHT`] (the C's MinWidth/
/// MinHeight error) or when any composed stage rejects its input (not
/// reachable once the size gate passes).
///
/// # Panics
/// Panics if `binary.len() != w * h`.
#[must_use]
pub fn generate_halftone_mask(binary: &[u8], w: usize, h: usize) -> Option<HalftoneMask> {
    assert_eq!(binary.len(), w * h, "binary buffer length must be w * h");
    if w < MIN_WIDTH || h < MIN_HEIGHT {
        return None;
    }

    // Seed for halftone parts at 8x reduction, back to 2x (pageseg.c:326-331).
    let (cascaded, cw, ch) = reduce_rank_binary_cascade(binary, w, h, [4, 4, 0, 0])?;
    let opened = open_brick(&cascaded, cw, ch, 5, 5);
    let (seed, sw, sh) = expand_replicate(&opened, cw, ch, 4, 4)?;

    // Mask for connected regions (pageseg.c:334-335).
    let region_mask = close_safe_brick(binary, w, h, 4, 4);

    // Fill seed into mask (pageseg.c:338-339). 4-connectivity, per the C.
    let filled = seedfill_binary(&seed, sw, sh, &region_mask, w, h, 4)?;
    let found = filled.contains(&0);

    // Text = input minus mask over the overlap; input passes through beyond
    // the mask's extent (the C's clipped pixSubtract rasterop). The empty-
    // mask arm is a plain copy (pixCopy, pageseg.c:352-356) — identical to
    // subtracting an empty mask, kept as one loop.
    let mut text = binary.to_vec();
    if found {
        for y in 0..h.min(sh) {
            for x in 0..w.min(sw) {
                if filled[y * sw + x] == 0 {
                    text[y * w + x] = 255;
                }
            }
        }
    }

    Some(HalftoneMask {
        mask: filled,
        mask_w: sw,
        mask_h: sh,
        text,
        found,
    })
}

/// Invert a bitonal buffer (ink ↔ background) — `pixInvert` on 1 bpp.
fn invert(binary: &[u8]) -> Vec<u8> {
    binary
        .iter()
        .map(|&p| if p == 0 { 255 } else { 0 })
        .collect()
}

/// `a AND NOT b` on same-shaped bitonal buffers — `pixSubtract` on 1 bpp
/// (equal dimensions; the clipped-overlap variant lives in
/// [`generate_halftone_mask`], which is the only mismatched-size call site).
fn subtract(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter()
        .zip(b)
        .map(|(&pa, &pb)| if pa == 0 && pb != 0 { 0 } else { 255 })
        .collect()
}

/// The result of [`gen_textline_mask`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextlineMask {
    /// The textline mask (`w × h`).
    pub mask: Vec<u8>,
    /// The vertical-whitespace mask (`w × h`) — `pixGenTextblockMask`'s
    /// second input, returned alongside exactly as the C's `*ppixvws`.
    pub vws: Vec<u8>,
    /// `true` iff the mask has at least one ON pixel (`*ptlfound`).
    pub found: bool,
}

/// Generate the textline mask + vertical-whitespace mask of a binarized,
/// deskewed, halftone-free page — `pixGenTextlineMask`
/// (`pageseg.c:389-453`):
///
/// ```text
/// pix1 = invert(src)
/// pix1 -= comp_seq(pix1, "o80.60")        // remove huge bg blocks so the
///                                          // whitespace mask can't break
///                                          // textlines at page margins
/// vws  = comp_seq(pix1, "o5.1 + o1.200")  // long vertical bg corridors
/// mask = open3x3( seq(src, "c30.1") − vws )
/// ```
///
/// Sequences run through [`morph_sequence`] — see its doc for why the
/// comp-sequence call sites are served by the same implementation (exact
/// factorization; oracle-pinned). Returns `None` when the page is smaller
/// than [`MIN_WIDTH`]`×`[`MIN_HEIGHT`] (C error) — sequence failure is
/// unreachable with these fixed strings.
///
/// # Panics
/// Panics if `binary.len() != w * h`.
#[must_use]
pub fn gen_textline_mask(binary: &[u8], w: usize, h: usize) -> Option<TextlineMask> {
    assert_eq!(binary.len(), w * h, "binary buffer length must be w * h");
    if w < MIN_WIDTH || h < MIN_HEIGHT {
        return None;
    }

    let inverted = invert(binary);
    let (big_bg, _, _) = morph_sequence(&inverted, w, h, "o80.60")?;
    let bg = subtract(&inverted, &big_bg);
    let (vws, _, _) = morph_sequence(&bg, w, h, "o5.1 + o1.200")?;

    let (closed, _, _) = morph_sequence(binary, w, h, "c30.1")?;
    let diff = subtract(&closed, &vws);
    let mask = open_brick(&diff, w, h, 3, 3);
    let found = mask.contains(&0);

    Some(TextlineMask { mask, vws, found })
}

/// Generate the textblock mask from a textline mask + vertical-whitespace
/// mask — `pixGenTextblockMask` (`pageseg.c:480-529`):
///
/// ```text
/// pix1 = seq(textline_mask, "c1.10 + o4.1")   // join lines vertically
/// (empty → None — the C returns NULL with an INFO message)
/// pix2 = by_component(pix1, "c30.30 + d3.3", 8)  // solidify per block
/// pix2 = close_safe(pix2, 10, 1)                 // small horizontal join
/// pix3 = pix2 − vws                              // reopen column corridors
/// mask = select_by_size(pix3, 25, 5, 8, IF_BOTH, GTE)  // drop noise blocks
/// ```
///
/// Returns `None` when the page is smaller than [`MIN_WIDTH`]`×`
/// [`MIN_HEIGHT`] OR the vertical join comes up empty (both are the C's
/// `NULL` returns; the oracle pins the non-empty arm via `tb_null_flag 0`).
///
/// # Panics
/// Panics if buffer lengths are not `w * h`.
#[must_use]
pub fn gen_textblock_mask(textline_mask: &[u8], vws: &[u8], w: usize, h: usize) -> Option<Vec<u8>> {
    assert_eq!(textline_mask.len(), w * h, "mask length must be w * h");
    assert_eq!(vws.len(), w * h, "vws length must be w * h");
    if w < MIN_WIDTH || h < MIN_HEIGHT {
        return None;
    }

    let (joined, _, _) = morph_sequence(textline_mask, w, h, "c1.10 + o4.1")?;
    if !joined.contains(&0) {
        return None; // "no fg pixels in textblock mask" (pageseg.c:503-507)
    }
    let solid = morph_sequence_by_component(&joined, w, h, "c30.30 + d3.3", 8, 0, 0)?;
    let closed = close_safe_brick(&solid, w, h, 10, 1);
    let carved = subtract(&closed, vws);
    select_by_size(
        &carved,
        w,
        h,
        8,
        SizeFilter {
            width: 25,
            height: 5,
            select_type: SelectType::IfBoth,
            relation: SelectRelation::Gte,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Parse a banked oracle dump: `"name w h"` + rows sections into
    /// `name → (w, h, buffer)` (crate convention: `'1'` → `0` = ink), and
    /// `"name_flag v"` lines into `name_flag → (v, 0, [])`.
    fn parse_dump(text: &str) -> HashMap<String, (usize, usize, Vec<u8>)> {
        let mut out = HashMap::new();
        let mut lines = text.lines();
        while let Some(header) = lines.next() {
            let mut it = header.split_whitespace();
            let name = it.next().expect("section name").to_string();
            if name.ends_with("_flag") {
                let v: usize = it.next().expect("flag value").parse().expect("flag");
                out.insert(name, (v, 0, Vec::new()));
                continue;
            }
            let w: usize = it.next().expect("w").parse().expect("w");
            let h: usize = it.next().expect("h").parse().expect("h");
            let mut buf = Vec::with_capacity(w * h);
            for _ in 0..h {
                let row = lines.next().expect("row");
                assert_eq!(row.len(), w, "row width in section {name}");
                buf.extend(row.bytes().map(|b| if b == b'1' { 0u8 } else { 255u8 }));
            }
            out.insert(name, (w, h, buf));
        }
        out
    }

    fn oracle() -> HashMap<String, (usize, usize, Vec<u8>)> {
        parse_dump(include_str!(
            "../../../.claude/harvest/oracles/pageseg_oracle_out.txt"
        ))
    }

    fn oracle2() -> HashMap<String, (usize, usize, Vec<u8>)> {
        parse_dump(include_str!(
            "../../../.claude/harvest/oracles/pageseg2_oracle_out.txt"
        ))
    }

    /// The 260×220 two-column text-page fixture — must match the pageseg2
    /// oracle's `fixture()` exactly.
    fn text_page_fixture() -> (Vec<u8>, usize, usize) {
        let (w, h) = (260usize, 220usize);
        let mut buf = vec![255u8; w * h];
        for (c0, c1) in [(15usize, 115usize), (155, 245)] {
            let mut yb = 20;
            while yb <= 188 {
                for y in yb..yb + 5 {
                    for x in c0..c1 {
                        if (x - c0) % 24 < 18 {
                            buf[y * w + x] = 0;
                        }
                    }
                }
                yb += 12;
            }
        }
        for y in 10..13 {
            for x in 250..253 {
                buf[y * w + x] = 0;
            }
        }
        (buf, w, h)
    }

    #[test]
    fn morph_sequences_match_liblept_incl_comp_variants() {
        let o = oracle2();
        let (buf, w, h) = text_page_fixture();
        assert_eq!(o["tl_src"], (w, h, buf.clone()), "fixture == oracle input");

        // The comp-sequence pins: the REAL pixMorphCompSequence vs OUR
        // single implementation — the exact-factorization equivalence proof.
        for (name, seq) in [
            ("seqcomp_o80_60", "o80.60"),
            ("seqcomp_o5_1_o1_200", "o5.1 + o1.200"),
            ("seq_c30_1", "c30.1"),
            ("seq_c1_10_o4_1", "c1.10 + o4.1"),
        ] {
            let (got, gw, gh) = morph_sequence(&buf, w, h, seq).expect("valid sequence");
            let (ow, oh, obuf) = &o[name];
            assert_eq!((gw, gh), (*ow, *oh), "{name} dims");
            assert_eq!(&got, obuf, "{name} pixels");
        }
    }

    #[test]
    fn by_component_and_select_by_size_match_liblept() {
        let o = oracle2();
        let (buf, w, h) = text_page_fixture();

        let got = morph_sequence_by_component(&buf, w, h, "c30.30 + d3.3", 8, 0, 0).expect("valid");
        assert_eq!(&got, &o["bycomp_c30_30_d3_3"].2, "by-component pixels");

        let got = select_by_size(
            &buf,
            w,
            h,
            8,
            SizeFilter {
                width: 25,
                height: 5,
                select_type: SelectType::IfBoth,
                relation: SelectRelation::Gte,
            },
        )
        .expect("valid");
        assert_eq!(&got, &o["selsize_25_5_both_gte"].2, "select-by-size pixels");
    }

    #[test]
    fn textline_mask_matches_liblept() {
        let o = oracle2();
        let (buf, w, h) = text_page_fixture();
        let r = gen_textline_mask(&buf, w, h).expect("big enough");
        assert_eq!(o["tl_found_flag"].0, 1);
        assert!(r.found);
        assert_eq!(&r.vws, &o["tl_vws"].2, "vertical whitespace mask");
        assert_eq!(&r.mask, &o["tl_mask"].2, "textline mask");
    }

    #[test]
    fn textblock_mask_matches_liblept() {
        let o = oracle2();
        let (buf, w, h) = text_page_fixture();
        let tl = gen_textline_mask(&buf, w, h).expect("big enough");
        assert_eq!(o["tb_null_flag"].0, 0, "oracle produced a block mask");
        let tb = gen_textblock_mask(&tl.mask, &tl.vws, w, h).expect("non-empty");
        assert_eq!(&tb, &o["tb_mask"].2, "textblock mask");
    }

    #[test]
    fn textline_and_textblock_reject_small_pages() {
        let buf = vec![255u8; 99 * 200];
        assert!(gen_textline_mask(&buf, 99, 200).is_none());
        assert!(gen_textblock_mask(&buf, &buf, 99, 200).is_none());
    }

    /// The 97×61 close-safe fixture — the binreduce oracle's formula.
    fn rf() -> (Vec<u8>, usize, usize) {
        let (w, h) = (97usize, 61usize);
        let mut buf = vec![255u8; w * h];
        for y in 0..h {
            for x in 0..w {
                if (x * 7 + y * 13) % 251 < 128 {
                    buf[y * w + x] = 0;
                }
            }
        }
        (buf, w, h)
    }

    /// The 61×47 seedfill tile-checker mask (9×7 tiles — diagonal contact,
    /// the live 4-vs-8-connectivity discriminator) + the three seed dots.
    fn sf_fixtures() -> ((Vec<u8>, usize, usize), Vec<(usize, usize)>) {
        let (w, h) = (61usize, 47usize);
        let mut mask = vec![255u8; w * h];
        for y in 0..h {
            for x in 0..w {
                if ((x / 9) + (y / 7)) % 2 == 0 {
                    mask[y * w + x] = 0;
                }
            }
        }
        ((mask, w, h), vec![(4usize, 3usize), (40, 30), (20, 10)])
    }

    fn seed_buf(dots: &[(usize, usize)], w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![255u8; w * h];
        for &(x, y) in dots {
            buf[y * w + x] = 0;
        }
        buf
    }

    /// The 130×117 composed fixtures: `dense` selects the solid-block (ht2,
    /// found=1) vs the sparse-dither (ht, found=0) halftone rect.
    fn ht_fixture(dense: bool) -> (Vec<u8>, usize, usize) {
        let (w, h) = (130usize, 117usize);
        let mut buf = vec![255u8; w * h];
        for y in 10..60 {
            for x in 8..70 {
                let on = if dense {
                    true
                } else {
                    (31 * x + 17 * y) % 7 < 5
                };
                if on {
                    buf[y * w + x] = 0;
                }
            }
        }
        for yb in [70usize, 78, 86, 94] {
            for y in yb..yb + 3 {
                for x in 75..122 {
                    if x % 5 != 0 {
                        buf[y * w + x] = 0;
                    }
                }
            }
        }
        (buf, w, h)
    }

    #[test]
    fn close_safe_brick_matches_liblept_incl_1d_arms() {
        let o = oracle();
        let (buf, w, h) = rf();
        for (hs, vs) in [(4usize, 4usize), (1, 7), (6, 1)] {
            let got = crate::morph::close_safe_brick(&buf, w, h, hs, vs);
            let (ow, oh, obuf) = &o[&format!("closesafe_{hs}_{vs}")];
            assert_eq!((w, h), (*ow, *oh));
            assert_eq!(&got, obuf, "closesafe {hs}x{vs}");
        }
    }

    #[test]
    fn seedfill_matches_liblept_and_discriminates_connectivity() {
        let o = oracle();
        let ((mask, w, h), dots) = sf_fixtures();
        // Pin the fixtures themselves against the oracle's own dumps.
        assert_eq!(o["sf_mask"], (w, h, mask.clone()));
        let seed = seed_buf(&dots, w, h);
        assert_eq!(o["sf_seed"], (w, h, seed.clone()));

        let c4 = seedfill_binary(&seed, w, h, &mask, w, h, 4).expect("c4");
        assert_eq!(&c4, &o["seedfill_c4"].2, "conn-4 fill");
        let c8 = seedfill_binary(&seed, w, h, &mask, w, h, 8).expect("c8");
        assert_eq!(&c8, &o["seedfill_c8"].2, "conn-8 fill");
        // The discriminator is real: 8-conn floods across diagonal tile
        // contacts, 4-conn cannot.
        let on = |b: &Vec<u8>| b.iter().filter(|&&p| p == 0).count();
        assert!(on(&c8) > on(&c4), "8-conn must fill strictly more");
    }

    #[test]
    fn seedfill_size_mismatch_clips_like_the_c() {
        let o = oracle();
        let ((mask, mw, mh), dots) = sf_fixtures();
        let (sw, sh) = (56usize, 44usize);
        let seed = seed_buf(&dots, sw, sh);
        let got = seedfill_binary(&seed, sw, sh, &mask, mw, mh, 4).expect("mismatch");
        let (ow, oh, obuf) = &o["seedfill_mismatch"];
        assert_eq!((sw, sh), (*ow, *oh));
        assert_eq!(&got, obuf, "seed-sized result, mask clipped");
    }

    #[test]
    fn expand_replicate_matches_the_actual_pageseg_callee() {
        let o = oracle();
        // The 9×5 esrc formula (binreduce oracle's expand fixture).
        let (w, h) = (9usize, 5usize);
        let mut buf = vec![255u8; w * h];
        for y in 0..h {
            for x in 0..w {
                if (x * 3 + y * 5) % 17 < 8 {
                    buf[y * w + x] = 0;
                }
            }
        }
        for f in [3usize, 4] {
            let (got, gw, gh) = expand_replicate(&buf, w, h, f, f).expect("factor ok");
            let (ow, oh, obuf) = &o[&format!("exprep_f{f}")];
            assert_eq!((gw, gh), (*ow, *oh), "dims factor {f}");
            assert_eq!(&got, obuf, "pixels factor {f}");
        }
    }

    #[test]
    fn halftone_mask_found_arm_matches_liblept() {
        let o = oracle();
        let (buf, w, h) = ht_fixture(true);
        assert_eq!(o["ht2_src"], (w, h, buf.clone()), "fixture == oracle input");

        let r = generate_halftone_mask(&buf, w, h).expect("big enough");
        assert_eq!(o["ht2_found_flag"].0, 1, "oracle found the halftone");
        assert!(r.found);
        let (mw, mh, mbuf) = &o["ht2_mask"];
        assert_eq!(
            (r.mask_w, r.mask_h),
            (*mw, *mh),
            "mask dims (128×116 from 130×117)"
        );
        assert_eq!(&r.mask, mbuf, "mask pixels");
        let (tw, th, tbuf) = &o["ht2_text"];
        assert_eq!((w, h), (*tw, *th));
        assert_eq!(&r.text, tbuf, "text pixels");
    }

    #[test]
    fn halftone_mask_empty_arm_matches_liblept() {
        let o = oracle();
        let (buf, w, h) = ht_fixture(false);
        assert_eq!(o["ht_src"], (w, h, buf.clone()), "fixture == oracle input");

        let r = generate_halftone_mask(&buf, w, h).expect("big enough");
        assert_eq!(o["ht_found_flag"].0, 0, "oracle found nothing");
        assert!(!r.found);
        let (mw, mh, mbuf) = &o["ht_mask"];
        assert_eq!((r.mask_w, r.mask_h), (*mw, *mh));
        assert_eq!(
            &r.mask, mbuf,
            "empty mask still dimensioned + zeroed identically"
        );
        // Empty arm: text is a verbatim copy of the input (pixCopy).
        let (_, _, tbuf) = &o["ht_text"];
        assert_eq!(&r.text, tbuf);
        assert_eq!(r.text, buf);
    }

    #[test]
    fn too_small_pages_are_rejected_like_minwidth_minheight() {
        let buf = vec![255u8; 99 * 200];
        assert!(generate_halftone_mask(&buf, 99, 200).is_none());
        let buf = vec![255u8; 200 * 99];
        assert!(generate_halftone_mask(&buf, 200, 99).is_none());
    }
}
