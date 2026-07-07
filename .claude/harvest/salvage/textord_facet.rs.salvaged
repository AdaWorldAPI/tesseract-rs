//! `ccstruct`/`textord` layout data shapes — the Rust side of the P3 Batch 3D
//! byte-parity leaf, and a second sink onto the V3 SoA
//! ([`crate::facet::FacetCascade`]), alongside [`crate::network`].
//!
//! Tesseract's page-layout pipeline (`textord/` + `ccstruct/{rect,blobbox,
//! ocrrow,ocrblock,polyblk}.h`) is a family of plain **data containers** —
//! `TBOX` (a leaf geometry value type), `BLOBNBOX` (a blob + its
//! classification, linked via `ELIST_LINK`), `ROW`/`TO_ROW` (a text line, pre-
//! and post-textord), `BLOCK`/`TO_BLOCK` (a page block, pre- and
//! post-textord), and `POLY_BLOCK` (a polygonal region + its `PolyBlockType`).
//! The `ruff_cpp_spo` harvest of these 7 classes found **zero virtual
//! overrides** (`.claude/harvest/textord-class-manifest.txt`, banked in
//! `tesseract-rs`): unlike [`crate::network`]'s `Network` subclass tree, there
//! is no `classid → ClassView` *dispatch* table to resolve here — these are
//! SoA **field inventories**, not a vtable. The sink is therefore a straight
//! field→rail carving of each shape onto a [`FacetCascade`], exactly the same
//! 16-byte "classid + 12 bytes" substrate `network.rs` used for the LSTM
//! layer graph.
//!
//! # Core-First placement
//!
//! Per the Core-First doctrine this is **structure** (identity + typed
//! geometry/classification), not compute: nothing here runs textord's
//! line-fitting or blob-merging algorithms — that stays hand-ported compute
//! (mirrors the recognizer/Core split in [`crate::network`]'s docs). A
//! `TBOX`/`BLOBNBOX`/`ROW`/`TO_ROW`/`BLOCK`/`TO_BLOCK`/`POLY_BLOCK` instance
//! lands as a [`FacetCascade`]; its shape is a `classid`, never a bespoke
//! `enum TextordKind`. No parallel object model.
//!
//! # Mint reuse (no new codebook slots)
//!
//! Only 5 `0x08XX` OCR slots exist today
//! (`unicharset`/`recoder`/`charset`/`network_layer`/`textline`/`blob`/
//! `page_layout`/`page_image`/`ocr_renderer`, `crate::ogar_codebook`). This
//! batch mints **zero** new slots — it reuses the 3 that are the closest
//! semantic fit and, exactly like [`crate::network::NetworkType`] used one
//! `network_layer` canon slot for 27 layer kinds, packs the 7 layout shapes
//! two/three to a canon slot via the custom-low half:
//!
//! | canon (`0x08XX`) | custom-low ordinal | shape |
//! |---|---|---|
//! | [`BLOB_LAYOUT`] (`blob`, `0x0806`) | 0 | [`tbox_facet`] (`TBOX`) |
//! | [`BLOB_LAYOUT`] (`blob`, `0x0806`) | 1 | [`blobnbox_facet`] (`BLOBNBOX`) |
//! | [`TEXTLINE_LAYOUT`] (`textline`, `0x0805`) | 0 | [`row_facet`] (`ROW`) |
//! | [`TEXTLINE_LAYOUT`] (`textline`, `0x0805`) | 1 | [`to_row_facet`] (`TO_ROW`) |
//! | [`PAGE_LAYOUT`] (`page_layout`, `0x0807`) | 0 | [`block_facet`] (`BLOCK`) |
//! | [`PAGE_LAYOUT`] (`page_layout`, `0x0807`) | 1 | [`to_block_facet`] (`TO_BLOCK`) |
//! | [`PAGE_LAYOUT`] (`page_layout`, `0x0807`) | 2 | [`poly_block_facet`] (`POLY_BLOCK`) |
//!
//! `page_image`/`ocr_renderer` are not used by this batch (no `ccstruct`
//! data shape maps to "the source image" or "a renderer" — they stay
//! reserved for a later leaf). `TBOX` rides `blob` rather than getting its
//! own canon because every real use of a bare `TBOX` in this manifest is a
//! blob-scale geometry value (`BLOBNBOX::box`, `ROW::bound_box`, …) — the
//! closest existing mint, not a perfect one; documented here rather than
//! minting a `bounding_box` slot for a single leaf value type.
//!
//! # Byte budget discipline (12 payload bytes, 6 tiers)
//!
//! Every shape below carries strictly more scalar fields than 12 bytes can
//! hold at full precision (a `TBOX` alone is 4×`i16` = 8 bytes; a `ROW` adds 5
//! more scalars on top of its own box). Per the operator's "facet is the
//! index/routing view; full precision lives in the consumer's compute
//! struct" ruling (mirrored from `network.rs`'s `num_weights`/weights split),
//! each shape's constructor:
//!
//! 1. Keeps a small set of fields at **full precision** (the fields most
//!    load-bearing for spatial routing / classification lookups).
//! 2. **Quantizes** the rest — rounds a pixel-scale `f32` to the nearest
//!    whole pixel and narrows to `i16`/`i8`, or bit-packs small enums/flags
//!    into spare bits of a tile — with the loss documented per field on the
//!    constructor.
//! 3. **Excludes** pointers (`ColPartition*`, `BLOBNBOX*`, `BLOCK*`, …) and
//!    intrusive list links (`ELIST_LINK`/`ELIST2_LINK`) entirely: relations
//!    are `EdgeBlock`'s job, never a facet tile (same rule `network.rs`
//!    applies to the layer name string and the weight blob).
//!
//! No oracle exists yet for this batch (unlike the byte-parity leaves in
//! `tesseract-rs`) — the round-trip tests here are shape-construction tests,
//! not C++-parity tests. `examples/textord_facet_dump.rs` is the future
//! oracle seam: it prints each facet's 16 bytes as hex so a later C++ dumper
//! has something to diff against.

use crate::facet::FacetCascade;
use crate::ogar_codebook::compose_classid;

/// The `blob` canon slot (`0x0806`) hosts `TBOX` (custom `0`) and `BLOBNBOX`
/// (custom `1`) — the two blob-scale geometry/classification shapes.
pub const BLOB_LAYOUT: u16 = 0x0806;

/// The `textline` canon slot (`0x0805`) hosts `ROW` (custom `0`) and `TO_ROW`
/// (custom `1`) — the pre-/post-textord text-line shapes.
pub const TEXTLINE_LAYOUT: u16 = 0x0805;

/// The `page_layout` canon slot (`0x0807`) hosts `BLOCK` (custom `0`),
/// `TO_BLOCK` (custom `1`), and `POLY_BLOCK` (custom `2`) — the page-level
/// layout shapes.
pub const PAGE_LAYOUT: u16 = 0x0807;

/// Custom-low ordinals under [`BLOB_LAYOUT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum BlobShape {
    /// `TBOX` (`rect.h:37-323`) — a bare bounding box, no inheritance.
    Tbox = 0,
    /// `BLOBNBOX` (`blobbox.h:141-553`) — a blob's box + textord classification.
    Blobnbox = 1,
}

/// Custom-low ordinals under [`TEXTLINE_LAYOUT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum TextlineShape {
    /// `ROW` (`ocrrow.h:39-170`) — a finished text line.
    Row = 0,
    /// `TO_ROW` (`blobbox.h:555-695`) — a text line mid-textord (baseline fit
    /// in progress).
    ToRow = 1,
}

/// Custom-low ordinals under [`PAGE_LAYOUT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum PageShape {
    /// `BLOCK` (`ocrblock.h:32-205`) — a finished page block.
    Block = 0,
    /// `TO_BLOCK` (`blobbox.h:698-806`) — a page block mid-textord.
    ToBlock = 1,
    /// `POLY_BLOCK` (`polyblk.h:30-92`) — a polygonal region + its type.
    PolyBlock = 2,
}

impl BlobShape {
    /// This shape's `classid`: [`BLOB_LAYOUT`] canon, shape ordinal custom.
    #[inline]
    #[must_use]
    pub const fn classid(self) -> u32 {
        compose_classid(BLOB_LAYOUT, self as u16)
    }
}

impl TextlineShape {
    /// This shape's `classid`: [`TEXTLINE_LAYOUT`] canon, shape ordinal custom.
    #[inline]
    #[must_use]
    pub const fn classid(self) -> u32 {
        compose_classid(TEXTLINE_LAYOUT, self as u16)
    }
}

impl PageShape {
    /// This shape's `classid`: [`PAGE_LAYOUT`] canon, shape ordinal custom.
    #[inline]
    #[must_use]
    pub const fn classid(self) -> u32 {
        compose_classid(PAGE_LAYOUT, self as u16)
    }
}

// ---------------------------------------------------------------------------
// Tile-packing helpers (mirrors network.rs's `tier_u16`).
// ---------------------------------------------------------------------------

use crate::facet::FacetTier;

/// One 8:8 tile carrying an `i16`'s LE bit pattern (`(hi, lo)` of the
/// unsigned reinterpretation) — the signed-value analog of `network.rs`'s
/// `tier_u16`. Full precision, no loss.
#[inline]
const fn tier_i16(v: i16) -> FacetTier {
    let u = v as u16;
    FacetTier {
        lo: (u & 0xFF) as u8,
        hi: (u >> 8) as u8,
    }
}

/// Round an `f32` (pixel-scale) to the nearest whole pixel and saturate into
/// `i16` — the documented lossy path for wide floats: sub-pixel fraction is
/// dropped, and any magnitude beyond `i16` range clamps rather than wraps.
/// The exact `f32` is assumed to live in the out-of-line compute struct.
#[inline]
fn round_sat_i16(v: f32) -> i16 {
    if v.is_nan() {
        return 0;
    }
    v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

/// Round an `f32` or truncate an `i32` to the nearest whole unit and saturate
/// into `i8` — used for fields whose real range is small (kerning/spacing
/// gaps, pitch, font-class indices) but whose C++ storage is wider. Returns
/// the LE byte of the saturated `i8` (so callers can drop it straight into a
/// [`FacetTier`] half without an extra cast site).
#[inline]
fn sat_i8_byte(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    (v.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8) as u8
}

/// Saturate a wide integer into `i8`, returned as its LE byte (integer
/// sibling of [`sat_i8_byte`], for fields that are already integral in C++
/// — e.g. `TO_BLOCK::pitch_decision`-adjacent int fields — so no float
/// round-trip is introduced where the source has none).
#[inline]
const fn sat_i8_byte_i32(v: i32) -> u8 {
    let clamped = if v > i8::MAX as i32 {
        i8::MAX
    } else if v < i8::MIN as i32 {
        i8::MIN
    } else {
        v as i8
    };
    clamped as u8
}

// ---------------------------------------------------------------------------
// TBOX — blob(0x0806) custom 0
// ---------------------------------------------------------------------------

/// Build the [`FacetCascade`] for a `TBOX` (`rect.h:37-323`; fields
/// `bot_left`/`top_right`, each an `ICOORD` of two `TDimension = i16`,
/// `rect.h:321-322`).
///
/// **All 4 corner coordinates at full precision** — `TBOX` is *only* 4
/// scalars, so the whole value fits in 4 of the 6 tiers with zero loss. This
/// is also the byte layout every other shape's own embedded box re-uses
/// (tiers 0-3, `left, bottom, right, top` in that order) so a `ClassView`
/// that already knows how to read a `TBOX` facet reads the box-shaped prefix
/// of `BLOBNBOX`/`POLY_BLOCK` identically.
///
/// | tier | field | precision |
/// |---|---|---|
/// | 0 | `left` (`bot_left.x`) | full `i16` |
/// | 1 | `bottom` (`bot_left.y`) | full `i16` |
/// | 2 | `right` (`top_right.x`) | full `i16` |
/// | 3 | `top` (`top_right.y`) | full `i16` |
/// | 4 | reserved | — |
/// | 5 | reserved | — |
///
/// **Excluded:** nothing — every `TBOX` field is carried. Tiers 4-5 are
/// spare (documented, not zero-padding-by-accident) for a future derived
/// stat (e.g. a cached `area`/`width`/`height`) that would otherwise cost a
/// recompute on every read.
#[inline]
#[must_use]
pub const fn tbox_facet(left: i16, bottom: i16, right: i16, top: i16) -> FacetCascade {
    FacetCascade {
        facet_classid: BlobShape::Tbox.classid(),
        tiers: [
            tier_i16(left),
            tier_i16(bottom),
            tier_i16(right),
            tier_i16(top),
            FacetTier { lo: 0, hi: 0 },
            FacetTier { lo: 0, hi: 0 },
        ],
    }
}

/// Read back the 4 full-precision corners of a [`tbox_facet`] (or the
/// box-shaped prefix of [`blobnbox_facet`]/[`poly_block_facet`]) as
/// `(left, bottom, right, top)`.
#[inline]
#[must_use]
pub const fn read_tbox_tiers(f: &FacetCascade) -> (i16, i16, i16, i16) {
    (
        f.tiers[0].as_u16() as i16,
        f.tiers[1].as_u16() as i16,
        f.tiers[2].as_u16() as i16,
        f.tiers[3].as_u16() as i16,
    )
}

// ---------------------------------------------------------------------------
// BLOBNBOX — blob(0x0806) custom 1
// ---------------------------------------------------------------------------

/// Build the [`FacetCascade`] for a `BLOBNBOX` (`blobbox.h:141-553`).
///
/// | tier | field(s) | precision |
/// |---|---|---|
/// | 0 | `box.left` (`bounding_box().left()`) | full `i16` |
/// | 1 | `box.bottom` | full `i16` |
/// | 2 | `box.right` | full `i16` |
/// | 3 | `box.top` | full `i16` |
/// | 4.hi | `region_type` (3b, `BlobRegionType`, `BRT_COUNT=8`) `\|` `left_tab_type` (3b, `TabType`, `TT_VLINE=5` max) `\|` reserved(2b) | lossless (both enums fit 3 bits) |
/// | 4.lo | `right_tab_type` (3b) `\|` flags(5b): `joined`,`vert_possible`,`horz_possible`,`leader_on_left`,`leader_on_right` | lossless |
/// | 5 | `area` (`enclosed_area()`, `int32_t`) | **lossy**: saturated to `i16` |
///
/// **Full-precision box** — matches [`tbox_facet`]'s tiers 0-3 byte-for-byte,
/// so a `ClassView` shares the box-reading code between the two shapes.
///
/// **Excluded** (relations, content, or secondary classification the tier
/// budget has no room for — same "identity fingerprints point to content"
/// call `network.rs` makes for the weight blob):
/// - `cblob_ptr` (`C_BLOB*`) — content pointer, not a facet scalar.
/// - `red_box`/`reduced` — a legacy secondary "reduced" box
///   (`blobbox.h:520,558-261` in the harvested manifest) not read by the
///   recognizer path; excluded together (the flag would be meaningless
///   without the box it describes).
/// - `repeated_set_` — a dedup group id, secondary to layout routing.
/// - `neighbours_[BND_COUNT]` / `good_stroke_neighbours_[BND_COUNT]`
///   (`BLOBNBOX*` pointers + parallel bools) — these are genuine
///   *relations* between blobs; they belong on `EdgeBlock`, never packed
///   into this facet's tiles.
/// - `owner_` (`ColPartition*`) — a relation pointer, same reasoning.
/// - `base_char_top_`/`base_char_bottom_`/`baseline_y_`/`base_char_blob_`/
///   `line_crossings_` — diacritic/baseline refinements, secondary to the
///   coarse box+classification this facet indexes on.
/// - `left_rule_`/`right_rule_`/`left_crossing_rule_`/`right_crossing_rule_`
///   — rule-line proximity ints, secondary.
/// - `horz_stroke_width_`/`vert_stroke_width_`/`area_stroke_width_` — stroke
///   heuristics, secondary to geometry/classification.
/// - `flow_` (`BlobTextFlowType`) / `spt_type_` (`BlobSpecialTextType`) —
///   real classification fields, but the tier budget is exhausted by
///   `region_type`/`left_tab_type`/`right_tab_type` (the 3 the batch brief
///   named explicitly); deferred to a future wider tenant if needed.
#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn blobnbox_facet(
    left: i16,
    bottom: i16,
    right: i16,
    top: i16,
    region_type: u8,
    left_tab_type: u8,
    right_tab_type: u8,
    joined: bool,
    vert_possible: bool,
    horz_possible: bool,
    leader_on_left: bool,
    leader_on_right: bool,
    area: i32,
) -> FacetCascade {
    assert!(region_type < 8, "BlobRegionType has 8 values (BRT_COUNT)");
    assert!(left_tab_type < 6, "TabType has 6 values (TT_VLINE max)");
    assert!(right_tab_type < 6, "TabType has 6 values (TT_VLINE max)");

    let tier4_hi = ((region_type & 0x07) << 5) | ((left_tab_type & 0x07) << 2);
    let flags5 = (u8::from(joined) << 4)
        | (u8::from(vert_possible) << 3)
        | (u8::from(horz_possible) << 2)
        | (u8::from(leader_on_left) << 1)
        | u8::from(leader_on_right);
    let tier4_lo = ((right_tab_type & 0x07) << 5) | flags5;

    let area_i16 = area.clamp(i16::MIN as i32, i16::MAX as i32) as i16;

    FacetCascade {
        facet_classid: BlobShape::Blobnbox.classid(),
        tiers: [
            tier_i16(left),
            tier_i16(bottom),
            tier_i16(right),
            tier_i16(top),
            FacetTier {
                lo: tier4_lo,
                hi: tier4_hi,
            },
            tier_i16(area_i16),
        ],
    }
}

// ---------------------------------------------------------------------------
// ROW — textline(0x0805) custom 0
// ---------------------------------------------------------------------------

/// Build the [`FacetCascade`] for a `ROW` (`ocrrow.h:39-170`).
///
/// | tier | field | precision |
/// |---|---|---|
/// | 0 | `bound_box.left` | full `i16` (anchor corner) |
/// | 1 | `bound_box.top` | full `i16` (anchor corner) |
/// | 2 | `xheight` (`x_height()`) | **lossy**: rounded to nearest pixel, `i16` |
/// | 3.hi | `kerning` (`kern()`, `int32_t`) | **lossy**: saturated to `i8` |
/// | 3.lo | `spacing` (`space()`, `int32_t`) | **lossy**: saturated to `i8` |
/// | 4 | `ascrise` (`ascenders()`) | **lossy**: rounded to nearest pixel, `i16` |
/// | 5 | `descdrop` (`descenders()`) | **lossy**: rounded to nearest pixel, `i16` |
///
/// **Box is anchor-only** — only the top-left corner (`left`, `top`) is
/// kept; `right`/`bottom` (hence width/height) are dropped from the facet.
/// Unlike `BLOBNBOX`, a `ROW`'s box is exactly recomputable from its word
/// list (`recalc_bounding_box()`, `ocrrow.h:124`), so the facet only needs
/// enough of it for coarse spatial routing — the exact box is one call away
/// in the out-of-line compute struct, not lost.
///
/// **Excluded:**
/// - `bodysize` (`body_size()`) — the header documents it as
///   `xheight+ascrise` "by default" (`ocrrow.h:158-159`); redundant with the
///   two fields already carried.
/// - `has_drop_cap_`/`lmargin_`/`rmargin_` — secondary layout metadata
///   (margins are relative-to-polyblock, not this row's own geometry).
/// - `para_` (`PARA*`) — a relation pointer (paragraph membership);
///   `EdgeBlock`'s job, not a facet tile.
/// - `words` (`WERD_LIST`)/`baseline` (`QSPLINE`) — content-store / non-
///   scalar structural data, far larger than a facet tile.
#[inline]
#[must_use]
pub fn row_facet(
    box_left: i16,
    box_top: i16,
    xheight: f32,
    kerning: i32,
    spacing: i32,
    ascrise: f32,
    descdrop: f32,
) -> FacetCascade {
    FacetCascade {
        facet_classid: TextlineShape::Row.classid(),
        tiers: [
            tier_i16(box_left),
            tier_i16(box_top),
            tier_i16(round_sat_i16(xheight)),
            FacetTier {
                lo: sat_i8_byte_i32(spacing),
                hi: sat_i8_byte_i32(kerning),
            },
            tier_i16(round_sat_i16(ascrise)),
            tier_i16(round_sat_i16(descdrop)),
        ],
    }
}

// ---------------------------------------------------------------------------
// TO_ROW — textline(0x0805) custom 1
// ---------------------------------------------------------------------------

/// Build the [`FacetCascade`] for a `TO_ROW` (`blobbox.h:555-695`) — the
/// mid-textord row, before its baseline fit is finalized into a `ROW`.
///
/// | tier | field | precision |
/// |---|---|---|
/// | 0 | `m` (`line_m()`, baseline slope) | **lossy**: fixed-point ×10 000, saturated `i16` |
/// | 1 | `c` (`line_c()`, baseline offset/intercept) | **lossy**: rounded to nearest pixel, `i16` |
/// | 2 | `xheight` | **lossy**: rounded to nearest pixel, `i16` |
/// | 3 | `ascrise` | **lossy**: rounded to nearest pixel, `i16` |
/// | 4 | `descdrop` | **lossy**: rounded to nearest pixel, `i16` |
/// | 5.hi | `kern_size` (kerning analog — TO_ROW has no literal `kerning` field; `kern_size` is the average non-space gap that a finished `ROW::kerning` derives from) | **lossy**: saturated `i8` |
/// | 5.lo | `space_size` (spacing analog, same reasoning as `kern_size`) | **lossy**: saturated `i8` |
///
/// **The slope needs a fixed-point scale, not a round.** `m` is a
/// dimensionless gradient (typically a few thousandths); rounding it to the
/// nearest integer would collapse every real value to 0. Scaling by 10 000
/// before saturating to `i16` keeps 4 decimal digits of the slope
/// (±3.2767 range) — enough for any real page skew.
///
/// **`kern_size`/`space_size` stand in for "kerning/spacing"** because
/// `TO_ROW` has no field named `kerning`/`spacing` the way `ROW` does; the
/// header shows `ROW::kerning`/`ROW::spacing` come from an int16 handed to
/// the `ROW(TO_ROW*, kern, space)` constructor externally, not stored on
/// `TO_ROW` itself. `kern_size`/`space_size` (the running average gap sizes
/// textord actually measures) are the closest `TO_ROW`-native fields to that
/// concept — documented mapping, not a literal name match.
///
/// **Excluded** (pitch-fitting intermediates, thresholds, and content —
/// tier budget is fully spent on the 7 fields above):
/// - `pitch_decision`/`fixed_pitch`/`fp_space`/`fp_nonsp`/`pr_space`/
///   `pr_nonsp` — fixed-pitch-mode intermediate floats, secondary.
/// - `projection_left`/`projection_right` — secondary.
/// - `min_space`/`max_nonspace`/`space_threshold` — derived thresholds from
///   `kern_size`/`space_size`, redundant given those are already carried.
/// - `spacing` ("to next row") — redundant with `space_size`, and a
///   relation-flavoured "distance to the next row" concept better resolved
///   via row ordering than duplicated here.
/// - `y_min`/`y_max`/`initial_y_min`/`para_c`/`para_error`/`y_origin`/
///   `credibility`/`error` — line-fit auxiliary statistics, secondary to the
///   fitted `m`/`c` this facet already carries.
/// - `num_repeated_sets_`/`xheight_evidence` — secondary counts.
/// - `body_size` — same "`≈xheight+ascrise`" redundancy as `ROW::bodysize`.
/// - `all_caps`/`used_dm_model`/`merged` (bools) — small, but the tier
///   budget is spent on floats; a future revision could steal spare bits
///   from one of the `i16` tiles if these prove load-bearing.
/// - `blobs`/`rep_words`/`char_cells`/`baseline`/`projection` — content-
///   store / non-scalar structural data.
#[inline]
#[must_use]
pub fn to_row_facet(
    m: f32,
    c: f32,
    xheight: f32,
    ascrise: f32,
    descdrop: f32,
    kern_size: f32,
    space_size: f32,
) -> FacetCascade {
    let m_fixed = (m * 10_000.0)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16;

    FacetCascade {
        facet_classid: TextlineShape::ToRow.classid(),
        tiers: [
            tier_i16(m_fixed),
            tier_i16(round_sat_i16(c)),
            tier_i16(round_sat_i16(xheight)),
            tier_i16(round_sat_i16(ascrise)),
            tier_i16(round_sat_i16(descdrop)),
            FacetTier {
                lo: sat_i8_byte(space_size),
                hi: sat_i8_byte(kern_size),
            },
        ],
    }
}

// ---------------------------------------------------------------------------
// BLOCK — page_layout(0x0807) custom 0
// ---------------------------------------------------------------------------

/// Build the [`FacetCascade`] for a `BLOCK` (`ocrblock.h:32-205`). The box
/// lives on `pdblk.bounding_box()` (`PDBLK::box`, `pdblock.h`), a `TBOX`.
///
/// | tier | field | precision |
/// |---|---|---|
/// | 0 | `pdblk.box.left` | full `i16` (anchor corner) |
/// | 1 | `pdblk.box.top` | full `i16` (anchor corner) |
/// | 2 | `xheight` (`x_height()`, `int32_t`) | **lossy**: saturated `i16` |
/// | 3.hi | `pitch` (`fixed_pitch()`, stored `int16_t`) | **lossy**: saturated `i8` |
/// | 3.lo | `kerning` (`kern()`, stored `int8_t`) | **lossless** — already `i8` in C++ |
/// | 4.hi | `spacing` (`space()`, stored `int16_t`) | **lossy**: saturated `i8` |
/// | 4.lo | `font_class` (`font()`, stored `int16_t`) | **lossy**: saturated `i8` |
/// | 5.hi | `proportional` (bit 0) `\|` `right_to_left_` (bit 1) `\|` reserved(6b) | lossless (both bools) |
/// | 5.lo | reserved | — |
///
/// **Box is anchor-only**, same call as [`row_facet`] — a page has few
/// blocks, so precision matters less per-instance, but the tier budget is
/// identical regardless of instance count; kept symmetric with the row/blob
/// shapes for one shared "anchor corner" reading convention across the
/// whole batch.
///
/// **Excluded:**
/// - `filename` (`std::string`) — content-store, addressed by name/identity.
/// - `rows`/`paras_`/`c_blobs`/`rej_blobs` (`*_LIST`) — content-store /
///   relations (child rows/paragraphs/blobs belong on `EdgeBlock`).
/// - `re_rotation_`/`classify_rotation_`/`skew_` (`FCOORD`, 3×2 floats) —
///   geometry transforms for rotated pages; real fields, but the tier
///   budget is spent on the box+layout scalars above. Deferred.
/// - `median_size_` (`ICOORD`, 2×`i16`) — a derived stat (recomputable from
///   the blob list, `blobbox.h:735-736`); deferred for the same reason.
/// - `cell_over_xheight_` (`f32`) — a derived ratio, secondary.
/// - `pdblk.poly_block()` (`POLY_BLOCK*`) — a relation to a
///   [`poly_block_facet`], not a facet tile itself.
/// - `pdblk.index()` — a list position, not a stable identity.
#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn block_facet(
    box_left: i16,
    box_top: i16,
    xheight: i32,
    pitch: i16,
    kerning: i8,
    spacing: i16,
    font_class: i16,
    proportional: bool,
    right_to_left: bool,
) -> FacetCascade {
    let flags_hi = u8::from(proportional) | (u8::from(right_to_left) << 1);

    FacetCascade {
        facet_classid: PageShape::Block.classid(),
        tiers: [
            tier_i16(box_left),
            tier_i16(box_top),
            tier_i16(xheight.clamp(i16::MIN as i32, i16::MAX as i32) as i16),
            FacetTier {
                lo: kerning as u8,
                hi: sat_i8_byte(pitch as f32),
            },
            FacetTier {
                lo: sat_i8_byte(font_class as f32),
                hi: sat_i8_byte(spacing as f32),
            },
            FacetTier {
                lo: 0,
                hi: flags_hi,
            },
        ],
    }
}

// ---------------------------------------------------------------------------
// TO_BLOCK — page_layout(0x0807) custom 1
// ---------------------------------------------------------------------------

/// Build the [`FacetCascade`] for a `TO_BLOCK` (`blobbox.h:698-806`) — the
/// mid-textord block, before line/pitch decisions are finalized into a
/// `BLOCK`.
///
/// | tier | field | precision |
/// |---|---|---|
/// | 0 | `line_spacing` | **lossy**: rounded to nearest pixel, `i16` |
/// | 1 | `line_size` (font-size-in-pixels estimate, `blobbox.h:784-789`) | **lossy**: rounded, `i16` |
/// | 2 | `xheight` (median blob size) | **lossy**: rounded, `i16` |
/// | 3.hi | `kern_size` | **lossy**: saturated `i8` |
/// | 3.lo | `space_size` | **lossy**: saturated `i8` |
/// | 4 | `max_blob_size` (line-assignment cutoff) | **lossy**: rounded, `i16` |
/// | 5.hi | `pitch_decision` (`PITCH_TYPE`, 3b, `PITCH_CORR_PROP=6` max) `\|` reserved(5b) | lossless |
/// | 5.lo | `baseline_offset` (phase shift) | **lossy**: saturated `i8` |
///
/// **No box here** — `TO_BLOCK` wraps a `BLOCK*` (`block`, excluded below,
/// see `block_facet`'s own box); duplicating the box in both facets would
/// waste tier budget on a value one `EdgeBlock` hop away.
///
/// **Excluded:**
/// - `block` (`BLOCK*`) — a relation to a [`block_facet`]; `EdgeBlock`'s job.
/// - `blobs`/`underlines`/`noise_blobs`/`small_blobs`/`large_blobs`/
///   `row_list` (`*_LIST`) — content-store / relations (child blobs/rows).
/// - `key_row` (`TO_ROW*`) — a relation to a [`to_row_facet`].
/// - `fixed_pitch` — redundant with `pitch_decision`+`kern_size` once the
///   pitch mode is fixed; secondary.
/// - `min_space`/`max_nonspace` — thresholds derivable from
///   `kern_size`/`space_size`, secondary.
/// - `fp_space`/`fp_nonsp`/`pr_space`/`pr_nonsp` — pitch-mode-specific
///   intermediate floats, redundant with `kern_size`/`space_size` +
///   `pitch_decision`; tier budget is fully spent.
#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn to_block_facet(
    line_spacing: f32,
    line_size: f32,
    xheight: f32,
    kern_size: f32,
    space_size: f32,
    max_blob_size: f32,
    pitch_decision: u8,
    baseline_offset: f32,
) -> FacetCascade {
    assert!(
        pitch_decision < 7,
        "PITCH_TYPE has 7 values (PITCH_CORR_PROP max)"
    );

    FacetCascade {
        facet_classid: PageShape::ToBlock.classid(),
        tiers: [
            tier_i16(round_sat_i16(line_spacing)),
            tier_i16(round_sat_i16(line_size)),
            tier_i16(round_sat_i16(xheight)),
            FacetTier {
                lo: sat_i8_byte(space_size),
                hi: sat_i8_byte(kern_size),
            },
            tier_i16(round_sat_i16(max_blob_size)),
            FacetTier {
                lo: sat_i8_byte(baseline_offset),
                hi: (pitch_decision & 0x07) << 5,
            },
        ],
    }
}

// ---------------------------------------------------------------------------
// POLY_BLOCK — page_layout(0x0807) custom 2
// ---------------------------------------------------------------------------

/// Build the [`FacetCascade`] for a `POLY_BLOCK` (`polyblk.h:30-92`).
///
/// | tier | field | precision |
/// |---|---|---|
/// | 0 | `box.left` | full `i16` |
/// | 1 | `box.bottom` | full `i16` |
/// | 2 | `box.right` | full `i16` |
/// | 3 | `box.top` | full `i16` |
/// | 4.hi | `type` (`isA()`, `PolyBlockType`, 4b, `PT_COUNT=15`) `\|` reserved(4b) | lossless |
/// | 4.lo | reserved | — |
/// | 5 | reserved | — |
///
/// **Full-precision box, same tier order as [`tbox_facet`]** — `POLY_BLOCK`
/// has only 2 non-content fields (`box`, `type`), so unlike `BLOBNBOX` there
/// is no pressure to shrink the box to an anchor; the whole value fits
/// losslessly with 1.5 tiers to spare.
///
/// **Excluded:**
/// - `vertices` (`ICOORDELT_LIST`) — the actual polygon; content-store, far
///   larger than a facet tile and inherently variable-length.
#[inline]
#[must_use]
pub const fn poly_block_facet(
    left: i16,
    bottom: i16,
    right: i16,
    top: i16,
    ty: u8,
) -> FacetCascade {
    assert!(ty < 15, "PolyBlockType has 15 values (PT_COUNT)");
    FacetCascade {
        facet_classid: PageShape::PolyBlock.classid(),
        tiers: [
            tier_i16(left),
            tier_i16(bottom),
            tier_i16(right),
            tier_i16(top),
            FacetTier {
                lo: 0,
                hi: (ty & 0x0F) << 4,
            },
            FacetTier { lo: 0, hi: 0 },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ogar_codebook::{canonical_concept_id, classid_canon, classid_custom};

    #[test]
    fn canon_consts_match_the_codebook() {
        // Compile-lock, same discipline as network.rs's
        // `network_layer_const_matches_codebook` — a rename/renumber on
        // either side must not silently mis-route a facet_classid.
        assert_eq!(canonical_concept_id("blob"), Some(BLOB_LAYOUT));
        assert_eq!(canonical_concept_id("textline"), Some(TEXTLINE_LAYOUT));
        assert_eq!(canonical_concept_id("page_layout"), Some(PAGE_LAYOUT));
    }

    #[test]
    fn tbox_round_trips_all_four_corners_losslessly() {
        let f = tbox_facet(-100, 5, 200, 300);
        assert_eq!(classid_canon(f.facet_classid), BLOB_LAYOUT);
        assert_eq!(classid_custom(f.facet_classid), BlobShape::Tbox as u16);
        assert_eq!(f.facet_classid, BlobShape::Tbox.classid());
        assert_eq!(read_tbox_tiers(&f), (-100, 5, 200, 300));
        assert_eq!(f.to_bytes().len(), 16);
    }

    #[test]
    fn blobnbox_carries_box_plus_packed_classification() {
        let f = blobnbox_facet(
            10, 20, 30, 40, // box
            7,  // region_type = BRT_TEXT (max value, exercise the 3-bit ceiling)
            5,  // left_tab_type = TT_VLINE
            3,  // right_tab_type = TT_MAYBE_ALIGNED
            true, false, true, false, true,    // flags
            123_456, // area, will saturate to i16::MAX
        );
        assert_eq!(classid_custom(f.facet_classid), BlobShape::Blobnbox as u16);
        assert_eq!(
            read_tbox_tiers(&f),
            (10, 20, 30, 40),
            "box is full precision"
        );

        // tier4.hi: region_type(3) | left_tab_type(3) | reserved(2)
        assert_eq!(f.tiers[4].hi, (7 << 5) | (5 << 2));
        // tier4.lo: right_tab_type(3) | flags5
        let flags5: u8 = (1 << 4) | (1 << 2) | 1;
        assert_eq!(f.tiers[4].lo, (3 << 5) | flags5);

        // area saturates (documented lossy path).
        assert_eq!(f.tiers[5].as_u16() as i16, i16::MAX);
    }

    #[test]
    fn blobnbox_area_round_trips_when_in_range() {
        let f = blobnbox_facet(0, 0, 1, 1, 0, 0, 0, false, false, false, false, false, 4200);
        assert_eq!(f.tiers[5].as_u16() as i16, 4200);
    }

    #[test]
    fn row_carries_anchor_box_and_quantized_typography() {
        let f = row_facet(50, 60, 23.6, 2, 15, 7.4, 5.2);
        assert_eq!(classid_custom(f.facet_classid), TextlineShape::Row as u16);
        assert_eq!(f.tiers[0].as_u16() as i16, 50, "box anchor left");
        assert_eq!(f.tiers[1].as_u16() as i16, 60, "box anchor top");
        assert_eq!(
            f.tiers[2].as_u16() as i16,
            24,
            "xheight rounds to nearest pixel"
        );
        assert_eq!(
            f.tiers[3].hi as i8, 2,
            "kerning, lossless at this magnitude"
        );
        assert_eq!(
            f.tiers[3].lo as i8, 15,
            "spacing, lossless at this magnitude"
        );
        assert_eq!(
            f.tiers[4].as_u16() as i16,
            7,
            "ascrise rounds down from 7.4"
        );
        assert_eq!(
            f.tiers[5].as_u16() as i16,
            5,
            "descdrop rounds down from 5.2"
        );
    }

    #[test]
    fn to_row_slope_uses_fixed_point_not_a_round() {
        // A real baseline slope (~0.003) would collapse to 0 under a plain
        // round; the ×10_000 fixed-point scale is why this shape needs a
        // different helper than row_facet's plain round_sat_i16.
        let f = to_row_facet(0.0031, 42.0, 20.0, 6.0, 4.0, 3.0, 12.0);
        assert_eq!(classid_custom(f.facet_classid), TextlineShape::ToRow as u16);
        assert_eq!(f.tiers[0].as_u16() as i16, 31, "0.0031 * 10_000 = 31");
        assert_eq!(f.tiers[1].as_u16() as i16, 42, "c rounds to nearest pixel");
        assert_eq!(f.tiers[5].hi as i8, 3, "kern_size saturated");
        assert_eq!(f.tiers[5].lo as i8, 12, "space_size saturated");
    }

    #[test]
    fn block_packs_native_i8_kerning_losslessly_and_flags() {
        let f = block_facet(0, 0, 30, 40, -7, 12, 3, true, false);
        assert_eq!(classid_custom(f.facet_classid), PageShape::Block as u16);
        assert_eq!(
            f.tiers[3].lo as i8, -7,
            "kerning is already i8 in C++, lossless"
        );
        assert_eq!(f.tiers[3].hi as i8, 40, "pitch saturates");
        assert_eq!(f.tiers[4].hi as i8, 12, "spacing saturates");
        assert_eq!(f.tiers[4].lo as i8, 3, "font_class saturates");
        assert_eq!(
            f.tiers[5].hi, 0b0000_0001,
            "proportional bit set, rtl clear"
        );

        let f2 = block_facet(0, 0, 30, 40, -7, 12, 3, false, true);
        assert_eq!(
            f2.tiers[5].hi, 0b0000_0010,
            "rtl bit set, proportional clear"
        );
    }

    #[test]
    fn to_block_packs_pitch_decision_and_scalars() {
        let f = to_block_facet(18.0, 24.0, 20.0, 3.0, 10.0, 40.0, 6, -2.0);
        assert_eq!(classid_custom(f.facet_classid), PageShape::ToBlock as u16);
        assert_eq!(f.tiers[0].as_u16() as i16, 18);
        assert_eq!(f.tiers[1].as_u16() as i16, 24);
        assert_eq!(f.tiers[2].as_u16() as i16, 20);
        assert_eq!(f.tiers[3].hi as i8, 3);
        assert_eq!(f.tiers[3].lo as i8, 10);
        assert_eq!(f.tiers[4].as_u16() as i16, 40);
        assert_eq!(f.tiers[5].hi >> 5, 6, "pitch_decision = PITCH_CORR_PROP");
        assert_eq!(f.tiers[5].lo as i8, -2);
    }

    #[test]
    fn poly_block_round_trips_box_and_type() {
        let f = poly_block_facet(1, 2, 3, 4, 6); // PT_TABLE = 6
        assert_eq!(classid_custom(f.facet_classid), PageShape::PolyBlock as u16);
        assert_eq!(read_tbox_tiers(&f), (1, 2, 3, 4));
        assert_eq!(f.tiers[4].hi >> 4, 6);
    }

    #[test]
    fn distinct_shapes_under_the_same_canon_get_distinct_classids() {
        // blob canon: TBOX vs BLOBNBOX.
        let tbox = tbox_facet(0, 0, 0, 0).facet_classid;
        let blobnbox =
            blobnbox_facet(0, 0, 0, 0, 0, 0, 0, false, false, false, false, false, 0).facet_classid;
        assert_eq!(classid_canon(tbox), classid_canon(blobnbox));
        assert_ne!(tbox, blobnbox);

        // textline canon: ROW vs TO_ROW.
        let row = row_facet(0, 0, 0.0, 0, 0, 0.0, 0.0).facet_classid;
        let to_row = to_row_facet(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).facet_classid;
        assert_eq!(classid_canon(row), classid_canon(to_row));
        assert_ne!(row, to_row);

        // page_layout canon: BLOCK vs TO_BLOCK vs POLY_BLOCK, all pairwise distinct.
        let block = block_facet(0, 0, 0, 0, 0, 0, 0, false, false).facet_classid;
        let to_block = to_block_facet(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0).facet_classid;
        let poly_block = poly_block_facet(0, 0, 0, 0, 0).facet_classid;
        assert_eq!(classid_canon(block), classid_canon(to_block));
        assert_eq!(classid_canon(to_block), classid_canon(poly_block));
        assert_ne!(block, to_block);
        assert_ne!(to_block, poly_block);
        assert_ne!(block, poly_block);

        // Cross-canon: every shape across all 3 canon slots is globally distinct.
        let all = [tbox, blobnbox, row, to_row, block, to_block, poly_block];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "classid collision at ({i}, {j})");
            }
        }
    }
}
