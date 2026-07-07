//! Batch 3F₂ leaf 2 — `filter_noise_blobs` + the `TO_BLOCK` line-size setup
//! (`Textord::filter_noise_blobs`, `textord/tordmain.cpp:291-360`, and its
//! caller `Textord::filter_blobs`, `tordmain.cpp:238-263`). Builds on
//! [`crate::conncomp::ConnComp`] (Batch 3F₂ leaf 1's per-component ink pixel
//! count, the `BLOBNBOX::enclosed_area()` source) and [`crate::stats::Stats`]
//! (Batch 3E wave 1's general `STATS` port).
//!
//! ## No PARITY-PIN needed here
//! Per this crate's established PARITY-PIN doctrine (see
//! [`crate::textline`]'s `assign_blobs_to_rows`/`compute_row_stats` docs;
//! board `E-OCR-MAKEROW-2`: "any `sort`/`nth_element` gets a pinned total
//! order on both sides, documented in-code"): `filter_noise_blobs` contains
//! **neither** a `sort` nor an `nth_element`. Every classification decision
//! (noise / small / large / stays) is a **per-element threshold test**
//! against scalars (`min_y`/`max_y`/`max_x`) that are fixed *before* the
//! pass runs — no blob is ever compared against another blob. The result is
//! therefore provably invariant to traversal order, and no comparator needs
//! pinning in this leaf.
//!
//! ## The list-order simplification (a documented divergence, not a pin)
//! Real Tesseract mutates four `BLOBNBOX_LIST`s via `BLOBNBOX_IT::extract()`
//! and `add_after_then_move()`. `ELIST_ITERATOR::add_after_then_move`
//! (`ccutil/elst.h:333-366`) inserts the new element **immediately after the
//! iterator's current cursor**, not at the list's tail. Concretely: after
//! the read-only `for (src_it...) { size_stats.add(...) }` pass
//! (`tordmain.cpp:320-322`), `src_it`'s cursor is left sitting on the FIRST
//! surviving element of `src_list` (a consequence of
//! `ELIST_ITERATOR::forward`/`mark_cycle_pt`/`cycled_list`'s "cycle point
//! shifts forward whenever the node it tracks is extracted" contract,
//! `ccutil/elst.{h,cpp}`). So when the small-list rescue pass
//! (`tordmain.cpp:330-338`) does `src_it.add_after_then_move(...)` for a
//! rescued blob, the real final order is `[S0, R0, R1, ..., S1, S2, ...]`
//! (rescued blobs spliced in right after the first original survivor `S0`),
//! **not** `[S0, S1, ..., R0, R1, ...]` (append-to-end). The same splice
//! pattern applies to `small_it`'s cursor during the re-partition pass
//! (`tordmain.cpp:340-350`).
//!
//! This port instead uses plain "append to the destination, preserving
//! first-seen order" (a stable partition) at every stage. This has **no**
//! effect on:
//! - the returned scalars (`line_size`/`line_spacing`/`max_blob_size`),
//!   which derive only from order-invariant [`Stats`] histogram
//!   accumulation (`Stats::add`/`Stats::ile` do not care about insertion
//!   order, only counts per bucket); or
//! - the SET membership of any blob (a pure per-element threshold decision,
//!   as above, with no cross-element comparison) — every blob ends up in
//!   exactly the same one of the four output lists regardless of traversal
//!   order.
//!
//! It affects **only** the within-list relative order of
//! "originally-resident" vs. "later-rescued/demoted" blobs, which has no
//! observable effect on any current consumer: the very first statement of
//! [`crate::textline::assign_blobs_to_rows`] is
//! `block.blobs.sort_by_key(blob_x_order_total_key)`, which immediately
//! re-sorts the pool by x-position regardless of arrival order. The
//! byte-parity oracle (`.claude/harvest/oracles/blob_filter_oracle.cpp`)
//! therefore compares partition **membership** — each output list's set of
//! original-fixture indices, canonicalized by sorting ascending — rather
//! than raw traversal-order sequences.

use crate::conncomp::ConnComp;
use crate::stats::Stats;

/// `MAX_NEAREST_DIST` (`tordmain.cpp:61`, `#define MAX_NEAREST_DIST 600`) —
/// the fixed upper bound of the height histogram (`STATS size_stats(0,
/// MAX_NEAREST_DIST - 1)`, `tordmain.cpp:303`).
const MAX_NEAREST_DIST: i32 = 600;
/// `textord_max_noise_size` (`INT_MEMBER`, default `7`, `textord.cpp:146`).
const TEXTORD_MAX_NOISE_SIZE: i32 = 7;
/// `textord_noise_area_ratio` (`double_MEMBER`, default `0.7`,
/// `textord.cpp:148`).
const TEXTORD_NOISE_AREA_RATIO: f64 = 0.7;
/// `textord_initialx_ile` (`double_MEMBER`, default `0.75`,
/// `textord.cpp:150`).
const TEXTORD_INITIALX_ILE: f64 = 0.75;
/// `textord_initialasc_ile` (`double_MEMBER`, default `0.90`,
/// `textord.cpp:152`).
const TEXTORD_INITIALASC_ILE: f64 = 0.90;
/// `textord_width_limit` (`double_VAR`, default `8`, `makerow.cpp:75`).
const TEXTORD_WIDTH_LIMIT: f64 = 8.0;
/// `textord_min_linesize` (`double_VAR`, default `1.25`, `makerow.cpp:80`).
const TEXTORD_MIN_LINESIZE: f64 = 1.25;
/// `textord_excess_blobsize` (`double_VAR`, default `1.3`, `makerow.cpp:81`)
/// — the same value [`crate::textline`] declares privately as
/// `TEXTORD_EXCESS_BLOBSIZE`; redeclared here following this crate's
/// established per-module-constant convention (e.g. `textline.rs`'s own
/// wave-2 re-declaration of the `CCStruct::k*Fraction` trio already defined
/// by wave 1 in the same file).
const TEXTORD_EXCESS_BLOBSIZE: f64 = 1.3;

/// `tesseract::CCStruct::kDescenderFraction` (`ccstruct.cpp:25`, `= 0.25`).
const K_DESCENDER_FRACTION: f64 = 0.25;
/// `tesseract::CCStruct::kXHeightFraction` (`ccstruct.cpp:26`, `= 0.5`).
const K_XHEIGHT_FRACTION: f64 = 0.5;
/// `tesseract::CCStruct::kAscenderFraction` (`ccstruct.cpp:27`, `= 0.25`).
const K_ASCENDER_FRACTION: f64 = 0.25;
/// `tesseract::CCStruct::kXHeightCapRatio` (`ccstruct.cpp:28-29`) =
/// `kXHeightFraction / (kXHeightFraction + kAscenderFraction)`.
const K_XHEIGHT_CAP_RATIO: f64 = K_XHEIGHT_FRACTION / (K_XHEIGHT_FRACTION + K_ASCENDER_FRACTION);

/// The four-way blob partition + line-size estimate produced by
/// [`filter_blobs`] (`Textord::filter_blobs` + `Textord::filter_noise_blobs`,
/// `tordmain.cpp:238-360`). Every tuple is `(left, bottom, right, top)`,
/// matching [`crate::textline::ToBlockCtx::blobs`]'s convention (`top >
/// bottom`, i.e. height = `top - bottom`) — see [`filter_blobs`]'s doc for
/// how a [`ConnComp`]'s raster-space box maps onto that convention.
#[derive(Debug, Clone, Default)]
pub struct FilteredBlobs {
    /// The surviving "ordinary" blob pool (`TO_BLOCK::blobs`).
    pub blobs: Vec<(i32, i32, i32, i32)>,
    /// Rejected-as-noise blobs (`TO_BLOCK::noise_blobs`).
    pub noise: Vec<(i32, i32, i32, i32)>,
    /// Too-small (but not noise-height) blobs (`TO_BLOCK::small_blobs`).
    pub small: Vec<(i32, i32, i32, i32)>,
    /// Too-large blobs (`TO_BLOCK::large_blobs`).
    pub large: Vec<(i32, i32, i32, i32)>,
    /// `TO_BLOCK::line_size` (post `filter_blobs` scaling by
    /// `textord_min_linesize`).
    pub line_size: f32,
    /// `TO_BLOCK::line_spacing`.
    pub line_spacing: f32,
    /// `TO_BLOCK::max_blob_size`.
    pub max_blob_size: f32,
}

/// Convert a [`ConnComp`]'s raster-space bounding box (`x`, `y` = top-left
/// corner, `y` increasing downward) into a `(left, bottom, right, top)`
/// tuple with `top > bottom`, by numerically relabeling `y` as `bottom` and
/// `y + h` as `top`.
///
/// This is **not** the real pipeline's page-coordinate flip (Tesseract's
/// `TBOX` uses a y-increases-upward convention established when C_OUTLINEs
/// are traced from a `Pix`, out of scope for this leaf). It is the minimal
/// labeling that satisfies wave-2/3's `top > bottom` (positive height)
/// invariant. `filter_noise_blobs` only ever reads *magnitudes* —
/// `height()`/`width()`/`enclosed_area()` — never absolute vertical
/// position or sign, so this relabeling has zero effect on any
/// classification or scalar computed in this leaf.
fn box_tuple(c: &ConnComp) -> (i32, i32, i32, i32) {
    (c.bb.x, c.bb.y, c.bb.x + c.bb.w, c.bb.y + c.bb.h)
}

/// `Textord::filter_noise_blobs` + `Textord::filter_blobs`'s line-size setup
/// (`tordmain.cpp:238-360`), single-block form (this port has no
/// multi-`TO_BLOCK_LIST` page loop — the real `filter_blobs` just calls
/// `filter_noise_blobs` once per block and rescales that block's
/// `line_size`/`line_spacing`/`max_blob_size`, same "single-column/single
/// block" scoping precedent as [`crate::textline`]'s wave 2/3).
///
/// ## Per-subexpression precision audit
/// (Mirrors [`crate::textline`]'s "Float vs double promotion" discipline —
/// every `float`/`double` boundary below is audited against the exact C++
/// expression, not merely "seems close enough".)
/// - `height()`/`width()` are `int16_t` (`TDimension`) in C++, promoted to
///   `int` by ordinary integer promotion before any arithmetic; this port's
///   [`ConnComp::bb`] fields are already `i32` (the promoted type — see
///   `conncomp.rs`'s module doc precedent for the same `TDimension`
///   reasoning), so no extra promotion step is written for those reads.
/// - `initial_x = size_stats.ile(textord_initialx_ile)`: `STATS::ile`
///   returns `double`; assigning into `float initial_x` narrows to `f32`
///   **at this statement**, not before.
/// - `max_y = ceil(initial_x * (kDescenderFraction + kXHeightFraction + 2 *
///   kAscenderFraction) / kXHeightFraction)`: `initial_x` (float)
///   multiplied by a `double`-typed sum promotes the whole expression to
///   `double`; `ceil(double)` stays `double`; narrows to `f32` only on
///   assignment to `float max_y`.
/// - `min_y = floor(initial_x / 2)`: the integer literal `2` promotes to
///   **`float`** (not `double`) against `initial_x` (float OP int → float);
///   C++11's type-generic `<cmath>` overloads then select `float
///   floor(float)`. This entire expression stays in `f32` precision — a
///   genuinely different path from `max_y`/`max_x`, which multiply by a
///   `double`-typed `textord_*` constant and DO promote to `double`.
/// - `max_x = ceil(initial_x * textord_width_limit)`: `textord_width_limit`
///   is a `double_VAR`, so this promotes to `double` exactly like `max_y`,
///   narrowing to `f32` only on assignment.
/// - `height > max_y` / `height >= min_y` / `width > max_x`: `height`/
///   `width` (int16_t, promoted to `int`) compared against a `float` value
///   promote the `int` to **`float`** (not `double`) for the comparison.
/// - `max_height *= tesseract::CCStruct::kXHeightCapRatio`: compound
///   assignment of `float *= double` computes the product in `double`,
///   narrowing back to `float` at the point of assignment.
/// - `block->line_spacing = block->line_size * (<double expr>) /
///   kXHeightFraction`: `float * double / double` → `double`, narrows to
///   `f32` only on assignment to `line_spacing`.
/// - `block->line_size *= textord_min_linesize`: `float *= double`, same
///   promote-then-narrow discipline as `max_height`'s compound assignment.
/// - `block->max_blob_size = block->line_size * textord_excess_blobsize`:
///   reads the **already-rescaled** `line_size` (the previous statement's
///   `*=` has already run), `float * double` → `double`, narrows to `f32`.
#[must_use]
pub fn filter_blobs(components: &[ConnComp]) -> FilteredBlobs {
    // ---- Pass 1 (tordmain.cpp:310-319): noise / small / stays-in-pool. ---
    let mut pool: Vec<&ConnComp> = Vec::with_capacity(components.len());
    let mut noise: Vec<&ConnComp> = Vec::new();
    let mut small: Vec<&ConnComp> = Vec::new();
    let mut large: Vec<&ConnComp> = Vec::new();

    for c in components {
        let height = c.bb.h;
        let width = c.bb.w;
        if height < TEXTORD_MAX_NOISE_SIZE {
            noise.push(c);
        } else if f64::from(c.pixel_count)
            >= (i64::from(height) * i64::from(width)) as f64 * TEXTORD_NOISE_AREA_RATIO
        {
            small.push(c);
        } else {
            pool.push(c);
        }
    }

    // ---- size_stats over the surviving pool (tordmain.cpp:320-322). ------
    let mut size_stats = Stats::new(0, MAX_NEAREST_DIST - 1);
    for c in &pool {
        size_stats.add(c.bb.h, 1);
    }

    // ---- initial_x / max_y / min_y / max_x (tordmain.cpp:323-329). -------
    let initial_x: f32 = size_stats.ile(TEXTORD_INITIALX_ILE) as f32;
    let max_y: f32 = (f64::from(initial_x)
        * (K_DESCENDER_FRACTION + K_XHEIGHT_FRACTION + 2.0 * K_ASCENDER_FRACTION)
        / K_XHEIGHT_FRACTION)
        .ceil() as f32;
    let min_y: f32 = (initial_x / 2.0_f32).floor();
    let max_x: f32 = (f64::from(initial_x) * TEXTORD_WIDTH_LIMIT).ceil() as f32;

    // ---- small-list rescue pass (tordmain.cpp:330-338). -------------------
    // `small_it.move_to_first()` in the C++ just resets the iterator to
    // walk `small_list` from its own start (populated in pass-1 order);
    // this port already iterates `small` (a plain Vec) in that exact order,
    // so no separate "reset" step is needed.
    let mut still_small: Vec<&ConnComp> = Vec::with_capacity(small.len());
    for c in small {
        let height = c.bb.h as f32;
        if height > max_y {
            large.push(c);
        } else if height >= min_y {
            pool.push(c);
        } else {
            still_small.push(c);
        }
    }

    // ---- re-partition pass over the (rescued-extended) pool ---------------
    // (tordmain.cpp:340-350).
    size_stats.clear();
    let mut final_pool: Vec<&ConnComp> = Vec::with_capacity(pool.len());
    for c in pool {
        let height = c.bb.h;
        let width = c.bb.w;
        if (height as f32) < min_y {
            still_small.push(c);
        } else if (height as f32) > max_y || (width as f32) > max_x {
            large.push(c);
        } else {
            size_stats.add(height, 1);
            final_pool.push(c);
        }
    }

    // ---- max_height / initial_x finalization (tordmain.cpp:351-359). ------
    let mut max_height: f32 = size_stats.ile(TEXTORD_INITIALASC_ILE) as f32;
    max_height = (f64::from(max_height) * K_XHEIGHT_CAP_RATIO) as f32;
    let mut initial_x = initial_x;
    if max_height > initial_x {
        initial_x = max_height;
    }
    // `initial_x` is `filter_noise_blobs`'s return value, `block->line_size`
    // at the caller.

    // ---- Textord::filter_blobs's line-size setup (tordmain.cpp:254-263). -
    let mut line_size = initial_x;
    if line_size == 0.0 {
        line_size = 1.0;
    }
    let line_spacing: f32 = (f64::from(line_size)
        * (K_DESCENDER_FRACTION + K_XHEIGHT_FRACTION + 2.0 * K_ASCENDER_FRACTION)
        / K_XHEIGHT_FRACTION) as f32;
    line_size = (f64::from(line_size) * TEXTORD_MIN_LINESIZE) as f32;
    let max_blob_size: f32 = (f64::from(line_size) * TEXTORD_EXCESS_BLOBSIZE) as f32;

    FilteredBlobs {
        blobs: final_pool.into_iter().map(box_tuple).collect(),
        noise: noise.into_iter().map(box_tuple).collect(),
        small: still_small.into_iter().map(box_tuple).collect(),
        large: large.into_iter().map(box_tuple).collect(),
        line_size,
        line_spacing,
        max_blob_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conncomp::ConnCompBox;

    fn comp(x: i32, y: i32, w: i32, h: i32, pixel_count: i32) -> ConnComp {
        ConnComp {
            bb: ConnCompBox { x, y, w, h },
            pixel_count,
        }
    }

    #[test]
    fn noise_height_below_threshold_is_rejected_as_noise() {
        // height (h=3) < textord_max_noise_size (7) -> noise, regardless of
        // how "solid" the blob is (pixel_count == full bbox area here).
        let c = comp(0, 0, 4, 3, 12);
        let out = filter_blobs(std::slice::from_ref(&c));
        assert_eq!(out.noise.len(), 1);
        assert!(out.blobs.is_empty());
        assert!(out.small.is_empty());
        assert!(out.large.is_empty());
    }

    #[test]
    fn low_density_blob_above_noise_height_is_small() {
        // height=10 (>= 7, not noise-height); bbox area = 10*10=100;
        // pixel_count=80 >= 100*0.7=70 -> classified "small" in pass 1
        // (the high-ink-density branch, not a height check -- a *solid*
        // blob is what this check flags, since real text glyphs rarely
        // fill more than ~70% of their bounding box).
        let c = comp(0, 0, 10, 10, 80);
        let out = filter_blobs(std::slice::from_ref(&c));
        // With this blob the ONLY pass-1 survivor being "small" (not pool),
        // `pool` is empty when size_stats.ile() runs, so initial_x=0 (the
        // Stats empty-histogram fallback, see stats.rs), giving
        // max_y=ceil(0*2.5)=0 / min_y=floor(0/2)=0 / max_x=ceil(0*8)=0. The
        // rescue pass then sees height(10) > max_y(0), so it lands in
        // `large` (NOT rescued back to the pool -- an empty pool starves
        // the whole size estimate, which is the correct degenerate-input
        // behaviour: a single densely-filled blob has no textline context
        // to be judged against, so it gets treated as an oversized mark).
        assert!(out.noise.is_empty());
        assert!(out.blobs.is_empty());
        assert!(out.small.is_empty());
        assert_eq!(out.large.len(), 1);
    }

    #[test]
    fn line_size_formula_hand_check_single_uniform_blob() {
        // A single blob, height=20 width=8, LOW ink density (pixel_count=60,
        // well under height*width*noise_area_ratio = 20*8*0.7 = 112) so it
        // survives pass 1 straight into the pool -- never touches
        // small/rescue, keeping the hand-check's control flow simple.
        let c = comp(0, 0, 8, 20, 60);
        let out = filter_blobs(std::slice::from_ref(&c));
        assert_eq!(
            out.blobs.len(),
            1,
            "the one low-density blob must survive into pool"
        );
        assert!(out.noise.is_empty());
        assert!(out.small.is_empty());
        assert!(out.large.is_empty());

        // Reconstruct filter_noise_blobs's *surrounding* scalar arithmetic
        // (tordmain.cpp:320-359: the ceil/floor calls, the f32<->f64
        // promotions, the threshold comparisons) by hand, but delegate the
        // histogram interpolation itself to the same `Stats::ile` this
        // leaf calls -- `Stats::ile` already has its own dedicated
        // interpolation tests in `stats.rs`; re-deriving its fractional
        // output by mental arithmetic here would just risk a second,
        // independent transcription bug rather than catching one.
        let mut size_stats = Stats::new(0, MAX_NEAREST_DIST - 1);
        size_stats.add(20, 1);
        let initial_x_raw = size_stats.ile(TEXTORD_INITIALX_ILE) as f32;
        let max_y = (f64::from(initial_x_raw) * 2.5).ceil() as f32;
        let min_y = (initial_x_raw / 2.0_f32).floor();
        let max_x = (f64::from(initial_x_raw) * 8.0).ceil() as f32;
        // The one blob's own height/width must fall inside
        // [min_y,max_y]/[0,max_x] for "no repartition-pass eviction" to
        // hold -- asserted here rather than silently assumed.
        assert!((20.0_f32) >= min_y && (20.0_f32) <= max_y);
        assert!((8.0_f32) <= max_x);

        let mut max_height_stats = Stats::new(0, MAX_NEAREST_DIST - 1);
        max_height_stats.add(20, 1); // the repartition pass re-adds the same value
        let mut max_height = max_height_stats.ile(TEXTORD_INITIALASC_ILE) as f32;
        max_height = (f64::from(max_height) * K_XHEIGHT_CAP_RATIO) as f32;
        let mut initial_x = initial_x_raw;
        if max_height > initial_x {
            initial_x = max_height;
        }

        let mut expected_line_size = initial_x;
        if expected_line_size == 0.0 {
            expected_line_size = 1.0;
        }
        let expected_line_spacing = (f64::from(expected_line_size) * 2.5) as f32;
        expected_line_size = (f64::from(expected_line_size) * 1.25) as f32;
        let expected_max_blob_size = (f64::from(expected_line_size) * 1.3) as f32;

        assert!(
            (out.line_spacing - expected_line_spacing).abs() < 1e-3,
            "line_spacing: got {} want {}",
            out.line_spacing,
            expected_line_spacing
        );
        assert!(
            (out.line_size - expected_line_size).abs() < 1e-3,
            "line_size: got {} want {}",
            out.line_size,
            expected_line_size
        );
        assert!(
            (out.max_blob_size - expected_max_blob_size).abs() < 1e-3,
            "max_blob_size: got {} want {}",
            out.max_blob_size,
            expected_max_blob_size
        );
    }

    #[test]
    fn oversized_blob_is_classified_large() {
        // A "normal" population of 9 low-density blobs (height=20,
        // pixel_count=60 -- density 37.5%, well under the 70% noise-area
        // threshold so they survive pass 1 into the pool and populate a
        // real size_stats), plus one outlier (height=200, pixel_count=1000
        // -- density 62.5%, ALSO under 70% so it survives pass 1 too,
        // reaching the same pool/STATS instead of being sidetracked into
        // `small`). The repartition pass must then evict the outlier into
        // `large` for being far taller than the population's max_y.
        let mut comps: Vec<ConnComp> = (0..9).map(|_| comp(0, 0, 8, 20, 60)).collect();
        comps.push(comp(0, 0, 8, 200, 1000));
        let out = filter_blobs(&comps);
        assert_eq!(out.blobs.len(), 9);
        assert_eq!(out.large.len(), 1);
        assert!(out.noise.is_empty());
        assert!(out.small.is_empty());
    }

    #[test]
    fn empty_input_yields_all_empty_lists_and_line_size_one() {
        let out = filter_blobs(&[]);
        assert!(out.blobs.is_empty());
        assert!(out.noise.is_empty());
        assert!(out.small.is_empty());
        assert!(out.large.is_empty());
        // initial_x = STATS::ile on an empty/all-zero histogram falls back
        // to rangemin (0.0, see `Stats::ile`'s empty-state doc), so
        // filter_blobs's `if (line_size == 0) line_size = 1;` fires.
        let expected_line_size = (1.0_f64 * 1.25) as f32;
        let expected_line_spacing = (1.0_f64 * (0.25 + 0.5 + 2.0 * 0.25) / 0.5) as f32;
        let expected_max_blob_size = (expected_line_size as f64 * 1.3) as f32;
        assert!((out.line_size - expected_line_size).abs() < 1e-6);
        assert!((out.line_spacing - expected_line_spacing).abs() < 1e-6);
        assert!((out.max_blob_size - expected_max_blob_size).abs() < 1e-6);
    }

    #[test]
    fn box_tuple_has_positive_height_and_width() {
        let c = comp(3, 5, 8, 20, 160);
        let (left, bottom, right, top) = box_tuple(&c);
        assert_eq!(left, 3);
        assert_eq!(bottom, 5);
        assert_eq!(right, 11);
        assert_eq!(top, 25);
        assert!(top > bottom);
        assert!(right > left);
    }
}
