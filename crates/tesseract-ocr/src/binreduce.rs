//! Rank-filtered binary reduction + power-of-2 binary expansion — leptonica
//! transcode (`binreduce.c` / `binexpand.c`, v1.82.0 == the installed
//! liblept), the two leaf kernels `pixGetRegionsBinary`'s halftone/textline
//! mask generators are built from (`pageseg.c:281/389/481`; see
//! `.claude/harvest/leptonica-pageseg-inventory.md` — every OTHER brick of
//! that region classifier is already parity-green in this crate).
//!
//! ## What is transcoded, and from where
//!
//! - **`reduce_rank_binary2`** ⇄ `pixReduceRankBinary2` (`binreduce.c:227-373`).
//!   A 2× rank-threshold reduction: destination pixel ON iff the number of ON
//!   pixels in its source 2×2 block is ≥ `level` (`level ∈ 1..=4`). The C
//!   implementation computes this with word-parallel boolean identities per
//!   level — `OR/OR` (≥1), `(AND/OR) | (OR/AND)` (≥2), `(AND/OR) & (OR/AND)`
//!   (≥3), `AND/AND` (=4) — into the upper-left bit of each block, then
//!   subsamples those bits (`makeSubsampleTab2x`, `binreduce.c:390-410`). The
//!   identities are exactly the 2×2 popcount thresholds (verified case-by-case
//!   at port time), so this port computes the popcount threshold directly on
//!   the crate's 8-bit binary convention; the banked oracle pins equality with
//!   the real bit-trick path bit-for-bit.
//!   Edge semantics carried over exactly: `wd = ws/2`, `hd = hs/2` (floor —
//!   the row loop `for (i = 0; i < hs - 1; i += 2)` yields `hs/2` complete row
//!   pairs; an odd trailing source row/column is dropped), `hs ≤ 1` is an
//!   error (`None`), and `ws < 2` yields a zero-width destination which
//!   `pixCreate` rejects (`None` here).
//! - **`reduce_rank_binary_cascade`** ⇄ `pixReduceRankBinaryCascade`
//!   (`binreduce.c:152-202`): up to four cascaded 2× rank reductions; a level
//!   of `0` truncates the cascade; `level1 == 0` returns a copy (the C code
//!   warns and copies); any level `> 4` is an error (`None`).
//! - **`expand_binary_power2`** ⇄ `pixExpandBinaryPower2`
//!   (`binexpand.c:135-230`): pixel replication by `factor ∈ {1, 2, 4, 8, 16}`
//!   (`1` = copy, per the C early return); `wd = factor·w`, `hd = factor·h`.
//!   The C uses per-factor expansion lookup tables purely as a speed device —
//!   semantically each source pixel becomes a `factor × factor` block, which
//!   is what this port does directly; the oracle pins all four factors.
//!
//! ## Parity
//!
//! Proven against the REAL `liblept` 1.82.0 via the banked oracle
//! (`.claude/harvest/oracles/binreduce_oracle.cpp`, output alongside):
//! deterministic odd-dimension fixtures (97×61 reduce / 9×5 expand), all four
//! rank levels, two cascades (`(1,2,0,0)`, `(4,4,3,0)`), all four expansion
//! factors — every output bit identical (the tests below re-parse the banked
//! dump with `include_str!` and compare cell-for-cell, and also pin the
//! fixture generators themselves against the oracle's `src`/`esrc` dumps).
//!
//! ## Conventions
//!
//! Input/output binary buffers use this crate's bitonal convention
//! (`0` = foreground/ON/ink, `255` = background — `threshold.rs`), row-major,
//! `len == w·h`. Leptonica's 1 bpp `1` = ON maps to `0` here; the counts
//! oracle (`counts_oracle.*`) pinned that mapping row- and column-exactly.

/// 2× rank-threshold binary reduction — `pixReduceRankBinary2`
/// (`binreduce.c:227-373`). Destination pixel ON iff ≥ `level` of its 2×2
/// source block are ON. Returns `(buffer, wd, hd)` with `wd = w/2`,
/// `hd = h/2`, or `None` when `level ∉ 1..=4` (C error), `h ≤ 1` (C error
/// "hs must be at least 2"), or `w < 2` (zero-width destination —
/// `pixCreate(0, …)` fails in C).
#[must_use]
pub fn reduce_rank_binary2(
    binary: &[u8],
    w: usize,
    h: usize,
    level: u32,
) -> Option<(Vec<u8>, usize, usize)> {
    if !(1..=4).contains(&level) || h <= 1 || w < 2 {
        return None;
    }
    let wd = w / 2;
    let hd = h / 2;
    let mut out = vec![255u8; wd * hd];
    for yd in 0..hd {
        let y0 = 2 * yd;
        for xd in 0..wd {
            let x0 = 2 * xd;
            let mut count = 0u32;
            if binary[y0 * w + x0] == 0 {
                count += 1;
            }
            if binary[y0 * w + x0 + 1] == 0 {
                count += 1;
            }
            if binary[(y0 + 1) * w + x0] == 0 {
                count += 1;
            }
            if binary[(y0 + 1) * w + x0 + 1] == 0 {
                count += 1;
            }
            if count >= level {
                out[yd * wd + xd] = 0;
            }
        }
    }
    Some((out, wd, hd))
}

/// Up to four cascaded 2× rank reductions — `pixReduceRankBinaryCascade`
/// (`binreduce.c:152-202`). A `0` level truncates the cascade; `levels[0] == 0`
/// returns an unreduced copy (the C code warns and copies); any level `> 4`
/// is an error (`None`), as is a stage whose input is too small to reduce
/// (propagated from [`reduce_rank_binary2`]).
#[must_use]
pub fn reduce_rank_binary_cascade(
    binary: &[u8],
    w: usize,
    h: usize,
    levels: [u32; 4],
) -> Option<(Vec<u8>, usize, usize)> {
    if levels.iter().any(|&l| l > 4) {
        return None;
    }
    if levels[0] == 0 {
        return Some((binary.to_vec(), w, h));
    }
    let mut cur = (binary.to_vec(), w, h);
    for &level in &levels {
        if level == 0 {
            break;
        }
        cur = reduce_rank_binary2(&cur.0, cur.1, cur.2, level)?;
    }
    Some(cur)
}

/// Power-of-2 binary expansion by pixel replication — `pixExpandBinaryPower2`
/// (`binexpand.c:135-230`). `factor ∈ {1, 2, 4, 8, 16}` (`1` = copy, per the
/// C early return); each source pixel becomes a `factor × factor` block;
/// `None` for any other factor (C error).
#[must_use]
pub fn expand_binary_power2(
    binary: &[u8],
    w: usize,
    h: usize,
    factor: usize,
) -> Option<(Vec<u8>, usize, usize)> {
    if factor == 1 {
        return Some((binary.to_vec(), w, h));
    }
    if !matches!(factor, 2 | 4 | 8 | 16) {
        return None;
    }
    let wd = w * factor;
    let hd = h * factor;
    let mut out = vec![255u8; wd * hd];
    for y in 0..h {
        for x in 0..w {
            if binary[y * w + x] == 0 {
                for dy in 0..factor {
                    let row = (y * factor + dy) * wd;
                    out[row + x * factor..row + (x + 1) * factor].fill(0);
                }
            }
        }
    }
    Some((out, wd, hd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Parse the banked oracle dump into `name → (w, h, buffer)` in the
    /// crate's binary convention (`'1'` = ON → `0` = ink).
    fn oracle() -> HashMap<String, (usize, usize, Vec<u8>)> {
        let text = include_str!("../../../.claude/harvest/oracles/binreduce_oracle_out.txt");
        let mut out = HashMap::new();
        let mut lines = text.lines().peekable();
        while let Some(header) = lines.next() {
            let mut it = header.split_whitespace();
            let name = it.next().expect("section name").to_string();
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

    /// The reduce fixture — MUST match the oracle's `make_fixture(97, 61, 7,
    /// 13, 251, 128)` exactly (pinned by [`fixtures_match_the_oracle_dumps`]).
    fn reduce_fixture() -> (Vec<u8>, usize, usize) {
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

    /// The expand fixture — the oracle's `make_fixture(9, 5, 3, 5, 17, 8)`.
    fn expand_fixture() -> (Vec<u8>, usize, usize) {
        let (w, h) = (9usize, 5usize);
        let mut buf = vec![255u8; w * h];
        for y in 0..h {
            for x in 0..w {
                if (x * 3 + y * 5) % 17 < 8 {
                    buf[y * w + x] = 0;
                }
            }
        }
        (buf, w, h)
    }

    /// The fixture generators themselves must equal the oracle's own dumps of
    /// its inputs — otherwise every downstream comparison would be against a
    /// different image and parity would be meaningless.
    #[test]
    fn fixtures_match_the_oracle_dumps() {
        let o = oracle();
        let (buf, w, h) = reduce_fixture();
        assert_eq!(o["src"], (w, h, buf), "reduce fixture != oracle src");
        let (ebuf, ew, eh) = expand_fixture();
        assert_eq!(o["esrc"], (ew, eh, ebuf), "expand fixture != oracle esrc");
    }

    #[test]
    fn reduce_all_four_levels_match_liblept() {
        let o = oracle();
        let (buf, w, h) = reduce_fixture();
        for level in 1..=4u32 {
            let (got, gw, gh) = reduce_rank_binary2(&buf, w, h, level).expect("valid reduction");
            let (ow, oh, obuf) = &o[&format!("reduce_l{level}")];
            assert_eq!((gw, gh), (*ow, *oh), "dims at level {level}");
            assert_eq!(&got, obuf, "pixels at level {level}");
        }
    }

    #[test]
    fn cascades_match_liblept() {
        let o = oracle();
        let (buf, w, h) = reduce_fixture();

        let (got, gw, gh) =
            reduce_rank_binary_cascade(&buf, w, h, [1, 2, 0, 0]).expect("cascade 1,2");
        let (ow, oh, obuf) = &o["cascade_1_2"];
        assert_eq!((gw, gh), (*ow, *oh));
        assert_eq!(&got, obuf);

        let (got, gw, gh) =
            reduce_rank_binary_cascade(&buf, w, h, [4, 4, 3, 0]).expect("cascade 4,4,3");
        let (ow, oh, obuf) = &o["cascade_4_4_3"];
        assert_eq!((gw, gh), (*ow, *oh));
        assert_eq!(&got, obuf);
    }

    #[test]
    fn expand_all_four_factors_match_liblept() {
        let o = oracle();
        let (buf, w, h) = expand_fixture();
        for factor in [2usize, 4, 8, 16] {
            let (got, gw, gh) = expand_binary_power2(&buf, w, h, factor).expect("valid expansion");
            let (ow, oh, obuf) = &o[&format!("expand_f{factor}")];
            assert_eq!((gw, gh), (*ow, *oh), "dims at factor {factor}");
            assert_eq!(&got, obuf, "pixels at factor {factor}");
        }
    }

    /// Hand-checkable 2×2 case: one block per level threshold.
    #[test]
    fn reduce_levels_thresholds_by_popcount() {
        // 4×2 page = two 2×2 blocks: left block has 2 ON, right block has 4 ON.
        #[rustfmt::skip]
        let buf = vec![
            0, 255, 0, 0,
            255, 0, 0, 0,
        ];
        for (level, expect_left, expect_right) in
            [(1u32, 0u8, 0u8), (2, 0, 0), (3, 255, 0), (4, 255, 0)]
        {
            let (got, gw, gh) = reduce_rank_binary2(&buf, 4, 2, level).expect("valid");
            assert_eq!((gw, gh), (2, 1));
            assert_eq!(got, vec![expect_left, expect_right], "level {level}");
        }
    }

    #[test]
    fn error_and_truncation_semantics_mirror_the_c() {
        let (buf, w, h) = reduce_fixture();
        // C errors: level out of range, hs must be at least 2, zero-width dest.
        assert!(reduce_rank_binary2(&buf, w, h, 0).is_none());
        assert!(reduce_rank_binary2(&buf, w, h, 5).is_none());
        assert!(reduce_rank_binary2(&buf[..w], w, 1, 1).is_none());
        assert!(reduce_rank_binary2(&buf[..2 * h], 1, h, 1).is_none()); // w < 2
        assert!(reduce_rank_binary_cascade(&buf, w, h, [1, 5, 0, 0]).is_none());

        // level1 == 0 → unreduced copy (C warns + copies).
        let (copy, cw, ch) = reduce_rank_binary_cascade(&buf, w, h, [0, 3, 3, 3]).expect("copy");
        assert_eq!((cw, ch, &copy), (w, h, &buf));

        // Expand: factor 1 → copy; anything not in {1,2,4,8,16} → None.
        let (ebuf, ew, eh) = expand_fixture();
        let (copy, cw, ch) = expand_binary_power2(&ebuf, ew, eh, 1).expect("copy");
        assert_eq!((cw, ch, &copy), (ew, eh, &ebuf));
        assert!(expand_binary_power2(&ebuf, ew, eh, 3).is_none());
        assert!(expand_binary_power2(&ebuf, ew, eh, 32).is_none());
        assert!(expand_binary_power2(&ebuf, ew, eh, 0).is_none());
    }
}
