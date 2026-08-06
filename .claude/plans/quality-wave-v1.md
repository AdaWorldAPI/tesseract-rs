# Quality Wave v1 — four scoped fixes, Opus-planned, Sonnet-implemented

> Operator directive (2026-08-06): "alles parallel oder nacheinander mit sonnet
> agents, schreib vorher den scoped plan mit opus agents." Four Opus planning
> agents each authored one spec below (verbatim); the orchestrator preflighted
> every cited anchor against the tree (PageRect pub at xy_cut.rs:95, ToBlockCtx
> noise at textline.rs:1104 with ..Default::default(), the layout.rs override
> site, decode_impl's two statement anchors, segment.rs's ToBlockCtx literal —
> all confirmed 2026-08-06).
>
> Execution model (per CLAUDE.md model policy + agent-cargo-hygiene):
> - 4 Sonnet workers, PARALLEL, shared checkout, DISJOINT file ownership:
>   W-41 -> crates/tesseract-ocr/src/dropcap.rs        (new file only)
>   W-50 -> crates/tesseract-ocr/src/grid_raster.rs    (new file only)
>   W-49 -> crates/tesseract-ocr-pdf/src/layout.rs     (sole owner this wave)
>   W-51 -> crates/tesseract-core/src/recodebeam.rs    (sole owner, additive-only)
> - Workers run NO cargo and NO git; they do not claim compilation.
> - Orchestrator wires shared files (lib.rs, textline.rs, segment.rs,
>   lstm_recognizer.rs) AFTER workers land, runs every gate centrally
>   (-p scoped, --release for recognition suites), executes every
>   disable-the-fix table, commits, pushes, PRs.
> - STOP rules baked into the specs: P-41 gate 0 (any shape-qualified .large
>   member on existing fixtures -> re-scope opt-in); P-50 fallback rule (any
>   fence movement -> DocumentOptions::grid_raster default-false).

---

# SPEC — #41 drop-cap: seam recovery (default) + loud loss signal
(authored by Opus planner; verbatim, HTML entities normalized)

## OBJECTIVE
Close the measured content loss at the drop-cap seam. Two glyphs are lost today: the ornamental initial itself (an 81x72 blob in filter_blobs' .large bucket, which no code in this crate consumes — segment.rs:146-157 builds ToBlockCtx with blobs + noise only, so filtered.large dies there) and the following l, whose ink merged into that same component so the row's ink-left starts right of it. Deliver (1) a shape-qualified drop-cap detector, (2) an x-only row-crop seam extension that recovers the merged neighbour, (3) a page-level count so the remaining loss is reported instead of silent.

## OWNED FILES
crates/tesseract-ocr/src/dropcap.rs — new, worker-owned, sole file. Pure functions + #[cfg(test)] synthetic fixtures. May call crate::{conncomp, blob_filter}.
Worker MUST NOT touch lstm_recognizer.rs, segment.rs, textline.rs, lib.rs, any test file, CLAUDE.md. No cargo, no git, no compile claims.

## OUT OF SCOPE
- Text splice (prepending the cap's own reading): glyph alone decodes "A" shrunk / "Ai" native; perturbations move it across A/Ai/Bi; prepending yields "Aiice..." vs truth "Alice..." — needs #51 dictionary disambiguation + second fixture. No shrink_to_band/glue helpers this pass.
- Widening a row crop VERTICALLY (measured harmful: "ye hewn to eet very tired of").
- Any change to make_rows / row assignment / x-height / baseline fitting.

## PUBLIC API (exact)
```rust
#[derive(Clone, Copy, Debug)]
pub struct OrdinaryScale { pub mean_h: f32, pub sd_h: f32, pub n: usize }

/// (left, bottom, right, top), y-UP page space, top > bottom.
#[derive(Clone, Copy, Debug)]
pub struct DropCap { pub bbox: (i32, i32, i32, i32), pub height_sd: f32, pub height_ratio: f32, pub aspect: f32 }

#[must_use] pub fn ordinary_scale(pool: &[(i32, i32, i32, i32)]) -> Option<OrdinaryScale>;
#[must_use] pub fn detect_drop_caps(large: &[(i32,i32,i32,i32)], pool: &[(i32,i32,i32,i32)]) -> Vec<DropCap>;
#[must_use] pub fn seam_left_extension(
    row_left: i32, row_bottom: f32, row_top: f32,
    row_spans: &[(i32, i32)], caps: &[DropCap], reach: f32,
) -> Option<i32>;
#[must_use] pub fn count_page_drop_caps(binary: &[u8], w: usize, h: usize) -> usize;

pub const MIN_POOL: usize = 8;
pub const MIN_HEIGHT_SD: f32 = 4.0;
pub const MIN_HEIGHT_RATIO: f32 = 1.8;
pub const MAX_HEIGHT_RATIO: f32 = 6.0;
pub const MIN_ASPECT: f32 = 0.25;
pub const MAX_ASPECT: f32 = 1.6;
pub const MIN_BAND_OVERLAP_FRAC: f32 = 0.5;
```

## ALGORITHM
1. ordinary_scale — mean + population sd of |top-bottom| over pool; None if pool.len() < MIN_POOL (8; mirrors noise_readmit_reach declining under 2 blobs).
2. detect_drop_caps — admit iff ALL:
   - height_sd >= 4.0 (measured cap: 8.01 SD; pool mean 25.40 sd 5.82 n 2389; 2x margin below measurement; ordinary capital ~0.5 SD).
   - height_ratio = h/mean_h in [1.8, 6.0] (measured 2.83x; floor guards uniform-type pages where sd is tiny — golden corpus is ONE font per page, gen_pages.py:201; ceiling excludes figures with >2x headroom).
   - aspect w/h in [0.25, 1.6] (measured 1.12; excludes rules/hairlines).
   - Degenerate-sd rule: sd_h <= f32::EPSILON => height_sd = INFINITY, SD test skipped, ratio window decides. (NaN compare silently DECLINES — fallback would fail wrong direction; warden Incident 4.)
   - No position test in detector (stays correct for column-opening caps); position tested once, at the seam adjacency check.
3. seam_left_extension — None unless ALL:
   - row_spans.len() >= 2 and reach.is_finite() && reach > 0.0
   - some cap vertically contains the row: overlap >= 0.5 * row band height
   - horizontally adjacent: |row_left - cap_right| <= reach
   - ext = min(reach, med_w) rounded; med_w = median row_spans width. Two neighbourhood yardsticks, smaller binds, neither absolute. (Measured winning seam 8 px; advance-half ~8-10 px at mean 25.40.)
   - new_left = max(cap_left, min(row_left, cap_right) - ext).max(0); Some only if new_left < row_left.
   - Multiple caps: largest band overlap; tie-break smallest cap_left.
4. count_page_drop_caps — conn_comp_areas(binary,w,h,8) -> y-UP flip (c.bb.y = h - (c.bb.y + c.bb.h)) -> filter_blobs -> detect_drop_caps(&f.large, &f.blobs).len().

## DEFAULT-VS-OPT-IN (warden-tested)
Seam correction + loudness: DEFAULT-ON. Text splice: not shipped.
- .large has ZERO consumers today; this creates the first two (both in wiring table).
- Default path changes only on pages carrying a drop-cap-shaped large blob. ASSUMPTION TO VERIFY AT GATE: corpus pages + resgrid are single-font renders => .large empty or shape-disqualified. If any fixture DOES carry one => escalate to DocumentOptions opt-in rather than re-pin goldens.
- Every constant is population-relative; zero pixel literals.
- new_left only decreases, bounded by cap_left and one glyph width; y-band untouched (x-only, y measured harmful).

## FALSIFIERS (all in dropcap.rs #[cfg(test)]; computed fixtures; worker must report per-test the disable performed and the observed failure text)
1. detector_admits_a_cap_and_rejects_a_rule_and_a_figure — large=[cap 72x72, rule 400x12, figure 300x260], pool mixed 18/20/22/24. Fire: exactly 1, bbox==cap. Disable: drop aspect window -> rule admitted; drop MAX_HEIGHT_RATIO -> figure admitted.
2. sd_and_ratio_guards_are_each_independently_load_bearing — (i) high-sd pool, ratio 2.0 but 3.0 SD -> declined; (ii) uniform pool, 6 SD but ratio 1.5 -> declined.
3. seam_fires_on_an_adjacent_cap_and_stays_silent_on_a_distant_row — row_left==cap_right -> Some; row_left==cap_right+4*reach -> None. Assert row_left-new_left == min(reach,med_w).round().
4. seam_never_reaches_into_the_cap_body — wide cap, large reach; assert new_left >= cap_left AND row_left-new_left <= med_w.
5. seam_requires_the_cap_to_span_the_row — spanning cap -> Some; wholly-above cap -> None.
6. seam_scales_exactly_2x_with_a_2x_layout — all coords doubled -> extension doubles exactly. Disable: any literal replaces reach/med_w.
7. synthetic_page_end_to_end — hollow-rect page (rectify.rs:485-504 pattern; 2px border ~30-40% density vs filter_blobs 0.7 density rule; ordinary rects h>=9 vs noise at blob_filter.rs:201). 4 rows x 8 rects heights 18/20/22/24 + one 72x72 hollow cap at row 0 left. PRECONDITIONS ASSERTED FIRST: filtered.large.len()==1, filtered.blobs.len()>=MIN_POOL, cap h 72 > max_y ~ 58. Then detect==1, count_page==1, seam Some for row 0, None rows 1-3 (silence twin same page).
8. uniform_pool_sd_zero_still_classifies_by_ratio — identical pool sd==0 + 3x cap -> detected, no NaN reaching a comparison; silence: 1.2x blob declined.

## ORCHESTRATOR WIRING (central, after worker lands)
W1 lib.rs: pub mod dropcap; + pub use dropcap::{...}; (~2 LoC, near line 21/46)
W2 textline.rs ToBlockCtx after pub noise (line 1104): pub large: Vec<(i32,i32,i32,i32)> + doc mirroring noise's. (~6)
W3 segment.rs seed_block ToBlockCtx literal (146-157): add large: filtered.large,  (1)
W4 lstm_recognizer.rs makerow_row_crops: (a) before for-row loop (~605): let caps = if block.large.is_empty() { Vec::new() } else { crate::dropcap::detect_drop_caps(&block.large, &block.blobs) }; (b) hoist spans+reach out of the !block.noise.is_empty() guard (649-650) so both consumers share; (c) insert seam block AFTER noise re-admission (line 671) and BEFORE the linerec.cpp:240-246 band extension (673); assign left = new_left only. (~10)
W5 lstm_recognizer.rs Document (line 136) + construction: pub drop_caps: usize; set from count_page_drop_caps(&binary, w, h) reusing existing page binarization. (~4)
W6 CLAUDE.md: supersede "deliberately NOT fixed" verdict + rung 1 in SAME commit; splice rung stays open. git show --stat prose+patch both present.

## RISKS
- One page of evidence; constants >=2x from measurement; second drop-cap page owed.
- Caption beside figure: bounded worst case <= one glyph width of figure edge.
- reach on wide tracking: bound by med_w clamp (F4).
- W5 cost: one conn_comp_areas vs ~1000ms/page — noise.

## GATE CHECKLIST (orchestrator, -p tesseract-ocr, --release)
0. Pre-check default-on assumption: dropcap_probe over corpus/pages/page_0{1..9}.pgm + resgrid.pgm; record large membership. Zero qualified => default-on provably no-op. Any hit => STOP, re-scope opt-in.
1. fmt + clippy -D warnings clean.
2. Byte-identical, NO re-pin: golden_pages, golden_lines, blocks_columns, page_bands, lab_table_grid, lab_table_columns.
3. 8+7+0 fence unchanged, exact numbers via --nocapture.
4. cargo test -p tesseract-ocr-pdf --release unchanged.
5. Per-falsifier disable report with observed red text.

---

# SPEC — #50 grid-raster inheritance (grid_raster.rs)
(authored by Opus planner; verbatim, entities normalized)

## MEASURED FIRST (corrects the task premise)
Simulated xy_cut (XyCutParams::default, incl. gutter fallback) over committed resgrid.pgm, approx Otsu 172 — ORCHESTRATOR MUST CONFIRM with examples/xy_gutter_probe.rs on the real pipeline:
1. resgrid yields 8 leaves, NOT 16 — full-HEIGHT columns spanning both grid rows (l=35/436/837/1237/1639/2039/2441/2842, w~331). The row split never happens (row gutter y=117..207 rejected by the sliver rule xy_cut.rs:577-590; neighbours' inked runs 17-22 px < min_region_px 24). Only the vertical k-way cut fires.
2. resgrid does NOT reproduce #50: both bands independently carry all 7 gutters (69-71 px vs gap_min 49). The operator's symptom came from their re-render; falsifiers must be SYNTHETIC; the 8+7+0 fence is expected to be a NO-OP.
3. On resgrid a correct raster detects 8 columns and splits NOTHING (each leaf spans exactly one column).

## OBJECTIVE
When a page's block geometry exhibits a regular column lattice, re-split any block spanning >=2 lattice columns at the lattice boundaries, BEFORE recognition — a degraded row inherits the raster the strong columns establish.

## OWNED FILES (worker)
crates/tesseract-ocr/src/grid_raster.rs — new, sole file. Pure geometry + #[cfg(test)] computed fixtures.

## OUT OF SCOPE
xy_cut.rs, lstm_recognizer.rs, lib.rs (orchestrator wires). No pixel access, no binarization, no recognition, no DocumentOptions field, no doc.v1 change, no Y-axis raster.

## PUBLIC API (exact)
```rust
use crate::xy_cut::PageRect;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnRaster {
    pub columns: Vec<(usize, usize)>,  // [left, right) ascending
    pub pitch: usize,                  // median left-to-left step
    pub cell_h: usize,                 // median height of CONFORMING blocks
}
impl ColumnRaster {
    #[must_use] pub fn boundaries(&self) -> Vec<usize>;   // gap midpoints, len == columns.len()-1
    #[must_use] pub fn spanned(&self, b: PageRect) -> usize;
}
#[must_use] pub fn detect_column_raster(blocks: &[PageRect]) -> Option<ColumnRaster>;
#[must_use] pub fn split_nonconforming(blocks: &[PageRect], r: &ColumnRaster) -> Vec<PageRect>;
```
split_nonconforming preserves input order, replacing a split block IN PLACE with sub-rects left->right (matching xy_cut's vertical-cut ordering :723-731).

## ALGORITHM
Detect — lattice fit over the MODAL-WIDTH subset (never "columns == leaves"; the merged block is itself a leaf and would poison a naive read):
1. blocks.len() >= 3 else None.
2. m = median(width). Conforming = width in [0.75m, 1.25m] (WIDTH_TOL 0.25). Need >= MIN_RASTER_COLS (3). Merged blocks (~k*pitch) and full-width headlines drop out here.
3. Sort by left; p = median(consecutive left diffs). Require p > 0 and p >= 0.9*m. MEDIAN is load-bearing: with columns 3-5 merged, diffs {p,p,4p,p} -> median still p.
4. Lattice fit: x0 = min(left); every conforming block k = round((left-x0)/p) must satisfy |left-(x0+k*p)| <= 0.15*p (POS_TOL). Require >= 3 distinct k and max(k) >= 2.
5. Occupancy: distinct k >= 0.5*(k_max+1) (OCCUPANCY_FRAC). Leniency arrives with its own evidence; stricter as more columns go missing.
6. columns[k] = (x0+k*p, x0+k*p+m); cell_h = median(conforming heights); boundaries()[k] = x0+k*p+(m+p)/2.

Apply — per block in order:
- spanned(b) = columns whose overlap with b >= 0.5*column_width.
- Split iff spanned >= 2 AND b.height >= 0.5*cell_h (SPAN_HEIGHT_FRAC — separates a merged CELL ROW from a spanning HEADLINE ~0.2 cell_h; neighbourhood-relative).
- Cut only at boundaries strictly inside (b.left, b.right); sub-rects keep b.top/b.bottom. NEVER emit a rect outside b's x-range.

Constants vs measurement: resgrid pitch deviation 0.5% (401 +- 2) -> POS_TOL 0.15 and WIDTH_TOL 0.25 are ~30x/~50x observed raggedness yet reject prose widths.

## EMPTY-CELL SEMANTICS
Emptiness = ABSENCE, exactly as xy_cut does (ink_bbox -> None -> dropped, :624-626, :659-661). Module returns sub-rects for every spanned column; ORCHESTRATOR tightens each to ink bbox vs binary page, drops ink-free. 7+1 surfaces as 7 regions, never a fabricated empty one. (Present-empty would be a doc.v1 schema change -> filed, out of scope.)

## DEFAULT-VS-OPT-IN
Default ON, no flag, but NEVER in the strip_borders (table-aware) branch — the raster would re-split the very table xy_cut_table_aware keeps whole. Mutually exclusive by construction.
Warden answers: fires only on >=3 modal-width columns + regular pitch + >=50% occupancy + spanning block >= 0.5 cell_h; fences: 8+7+0 measured no-op (prediction!), blocks_columns 2 cols < MIN_RASTER_COLS -> None, goldens single-column -> None. Failure mode is SILENCE, not a false veto.
FALLBACK RULE, stated in advance: if the orchestrator's REAL measurement moves any fence -> land behind DocumentOptions::grid_raster (default false); module unchanged, only wiring differs.

## FALSIFIERS (computed fixtures; shared builder lattice(x0, cell_w, gutter, n); boundary_k derived IN THE TEST from drawn geometry, never via boundaries() — else tautological)
1. CAN-FIRE raster_resegments_a_degraded_row_at_the_inherited_boundaries: 8 cols band A + ONE block spanning cols 0..6 in band B + nothing over col 7. Assert columns.len()==8, pitch==cell_w+gutter, output 8+7==15, the 7 sub-rects' edges equal fixture gutter midpoints EXACTLY, no rect over col 7. Disable: split returns input -> fails at 9.
2. STAY-SILENT prose two_column_prose_geometry_yields_no_raster: 2 unequal columns x 2 bands + full-width block. Anti-vacuity: assert the full-width block WOULD span both; then detect == None and output identical. Disable: MIN_RASTER_COLS=2 -> fails.
3. STAY-SILENT headline (two-sided, one test): 3-col lattice + spanning headline 0.2*cell_h + merged block 1.0*cell_h. Raster is_some with 3 columns (anti-vacuity: the GUARD declines, not inertness), headline unsplit, merged -> 3 rects. Disable: drop SPAN_HEIGHT_FRAC -> headline splits.
4. REGULARITY irregular_pitch_is_rejected_uniform_pitch_is_accepted: pitches {p,p} -> Some; {p, 2.2p} -> None. Disable: remove POS_TOL check.
5. OCCUPANCY a_lattice_supported_on_under_half_its_slots_is_rejected: k = 0,1,9 -> None.
6. ORDER split_preserves_block_order_and_inserts_sub_rects_in_place: exact Vec equality vs computed expected.
7. Degenerate: &[] and single block -> None.

## ORCHESTRATOR WIRING
1. lib.rs:38 pub mod grid_raster; :88 re-export ColumnRaster, detect_column_raster, split_nonconforming.
2. lstm_recognizer.rs private helper next to recognize_blocks_words:
   fn apply_grid_raster(blocks: Vec<PageRect>, binary: &[u8], w: usize, h: usize) -> Vec<PageRect>
   detect -> split -> tighten each rect to ink bbox vs binary (mirror xy_cut::ink_bbox), drop None. Input untouched when detection None.
3. :939 recognize_page_blocks_words_with_mode — wrap the xy_cut result; needs one binarize_page_with (Otsu 1.50 ms/page vs ~1000 ms recognition). Apply BEFORE the blocks.len() <= 1 check.
4. :1201-1205 rec_blocks — apply ONLY in the else (non-strip_borders) arm, using recog_binary (:1160-1164).
5. :1241-1245 classification blocks — same else arm, using binary (:1150). REQUIRED: build_regions assigns lines by CENTROID (:1229-1234); split recognition + unsplit classification collapses 8 cells' lines into one region (warden Incident 2: every call site).
6. Nothing else; attach_glyph_px, table gate, xy_cut_table_aware untouched.
Composition with no-content-loss guard (:1021-1039): dropping ink-free sub-rects avoids any_block_empty; if a degraded cell recognizes to nothing, whole-page pass may win — that is the guard working.

## RISKS
- The guard can discard the fix on badly degraded grids (merged garbage read outnumbering per-cell words). Detection channel: fence line_count >= 44 would collapse to ~6. Never weaken the guard.
- 8+7+0 fence prediction rests on SIMULATED binarization — the real xy_gutter_probe leaf list is the gate.
- Table interaction correct only via strip_borders exclusion — say so in module docs.
- No committed fixture reproduces #50: proven in geometry, unproven on paper — state in CLAUDE.md; a real degraded 2xN sheet owed to corpus/quality/.

## GATE CHECKLIST
fmt + clippy -p (never --all); lib tests grid_raster 7 green; run disables 1-4 confirm paired failures; xy_gutter_probe on resgrid confirms the 8-leaf simulation (else re-derive); 8+7+0 fence exact numbers via --nocapture; goldens + blocks_columns + lab_table_* unchanged; decide default-on vs DocumentOptions per fallback rule with MEASURED numbers in CLAUDE.md.

---

# SPEC — #49 header/footer render at body size after page normalization
(authored by Opus planner; verbatim)

## OBJECTIVE

A genuine headline/footer renders at body size. `doc_v1_layout`'s `_` match arm
(`crates/tesseract-ocr-pdf/src/layout.rs:1511`) catches `"text"`, `"header"`,
`"footer"` and unknown kinds alike, and at `:1558-1561` overrides every
`derive_text_metrics` `font_px` with the page-global `page_font_px`. Add a
**per-region scale factor** so a `header`/`footer` region whose own evidence is a
CLEAR outlier deviates proportionally from the page size, while every other
region stays byte-identical to today. Page size remains the base scale; the
measurement is used as a **classifier of difference**, never as a raw quantity
(CLAUDE.md § "OPERATOR DOCTRINE" (c)/(d)).

## OWNED FILE + FORBIDDEN REGIONS

Owned, exclusively: `crates/tesseract-ocr-pdf/src/layout.rs`.

**Do not touch** (all shipped PR #76/#77, all covered by existing falsifiers):
`emit_text_run` (:350), `RunFit` (:341), `MAX_JUSTIFY_SLACK_FRAC`,
`classify_justification` (:791), all `Tz`/`Tw`/justification logic,
`page_pitch_px` (:848), `page_font_px` (:892) — **including its constants**
`PITCH_TO_FONT_PX`/`MAX_FONT_TO_PITCH`, `derive_text_metrics` (:1379) body,
`text_font_size_px` (:938), the `"figure"` and `"table"` arms (:1479-1509), the
`fields` loop (:1575). No signature of an existing `pub` item changes.

## OUT OF SCOPE

- Footnotes. doc.v1's `"footer"` is **page furniture** (running foot / folio),
  not a footnote; real footnotes land inside a `"text"` block today and are
  therefore NOT covered. Closing doctrine (d) fully needs a footnote classifier
  in `structured.rs` — a different wave.
- Table cells (`TableCell.metrics`) — cells never deviate.
- Bold/italic classification (doctrine (c)'s other half).
- Changing what `page_font_px` computes.

## DESIGN

### 1. The ratio must compare LIKE WITH LIKE (load-bearing)

Contamination is a **multiplicative** bias: `attach_glyph_px`'s scan reads
neighbours through the recognition band, measured `Tf/pitch` median **1.44**, and
the same string up to **1.82×** apart between columns (CLAUDE.md § "Typography is
a PARAGRAPH property"). A ratio of *unlike* quantities — a region's band/ink
measurement over `page_font_px` (a pitch/width solve) — carries that ~1.44
systematic offset, so **every** region would read as a 1.44× headline and no
dead-band of [0.8, 1.25] could absorb it. A ratio of *like* quantities cancels it:
both numerator and denominator inherit the same bias.

⇒ The region's indicator rung **selects** its own page-level denominator. Never
cross rungs; if the matching denominator is `None`, the factor is `1.0`.

### 2. Indicator ladder (two rungs, not three)

- **Rung A — region baseline pitch.** Median consecutive baseline delta *within
  the region*, requiring `MIN_REGION_PITCH_LINES = 3` lines with `baseline`
  (>=2 deltas). Denominator: the existing `page_pitch_px(&jp)`.
  *Contamination: none.* Pitch measured sd **0.00** across all 8 columns of the
  reference page; it is the one quantity the whole normalization arc trusts. A
  multi-line headline's own leading scales with its size, so the ratio is real.
- **Rung B — region measured font.** Median over the region's lines of
  `derive_text_metrics(...).font_px`. Deliberately reuses the file's **existing**
  3-tier ladder (glyph_px*4/3 -> band fit -> none) rather than re-deriving
  band-then-glyph: that function's own doc-comment says duplicating the ladder
  "is how the two silently drift apart". Denominator: a NEW
  `page_measured_font_px_median` — the same median over **body regions only**
  (kind not in {`header`,`footer`}).

**Why a single-line header's rung-B measurement is trustworthy** — argue from the
mechanism, not from optimism: contamination is ink from an ADJACENT line entering
this line's generous recognition band. A page-furniture line sits in white space
by construction (`page_furniture` classifies it precisely because it is separated
from the body). **An isolated line has no neighbour in its band, so there is
nothing to import.** This is why the 1.82x body-vs-body spread does not bound the
header's own error.

**And the residual bias runs the SAFE way.** The rung-B denominator (body,
tight-set) *is* contaminated upward; the isolated header numerator is not. The
ratio is therefore **understated** — a genuine headline gets less amplification
than it deserves, never more. The failure mode is under-correction.

**Asymmetry, stated:** the rung-A denominator is the shipped `page_pitch_px`
unchanged (a header contributes <=2 deltas of dozens; the median absorbs them —
the exact immunity `page_pitch_is_immune_to_a_dropped_line` already pins). The
rung-B denominator is new code, has no many-sample median protection on a short
page, and so is body-only from the start.

### 3. Dead-band and clamp

`[SCALE_DEAD_BAND_LO, SCALE_DEAD_BAND_HI] = [0.80, 1.25]` -> factor exactly `1.0`.
Symmetric in log space (`1/0.80 = 1.25`), i.e. +-one conventional typographic step
(1.125/1.2/1.25). A running head set at body size but measuring differently
because its ink is cap-height-only (all-caps -> no descender) lands inside it and
is correctly normalized rather than nudged. Outside it,
`factor = ratio.clamp(SCALE_CLAMP_LO, SCALE_CLAMP_HI) = ratio.clamp(0.5, 3.0)`.

**Clamp high = 3.0**, on two independent supports: (a) the largest conventional
display step over body on a text page is ~3x; (b) the measured contamination
maximum on the reference page was **`Tf/pitch` max 3.03** — above 3.0 a
measurement is reachable by contamination alone and carries no discriminating
information. **Clamp low = 0.5**: a running foot legitimately sits at 0.7-0.85 of
body; below 0.5 is a truncated/mis-cropped line, not a design choice.

> **Honesty pin, mandatory in the doc comments:** the corpus contains no fixture
> with a real headline. These four constants are **policy pins, not
> measurements** — say so verbatim at each `const`, per the repo's "a doc-comment
> claim is not a behaviour" rule. Do not defend them; re-measure when a
> tightly-set page with a real headline lands.

### 4. Region kinds

Kind strings verified in `crates/tesseract-ocr/src/structured.rs:350-354`:
`"text" | "table" | "figure" | "header" | "footer"`.

Deviation path: **`"header"` and `"footer"` ONLY.** `"text"`, `""` (legacy
doc.v1 emits no `type`), and any unknown string -> factor `1.0`, today's exact
behaviour. This kind-gate is what makes the narrow dead-band safe: the 1.82x
body-vs-body spread cannot amplify anything, because no `text` region may
deviate whatever it measures. It also implements doctrine (d) literally —
"the enumerable exceptions" is a fixed list, not a threshold.

### 5. Placement

Untouched. The override at `:1558-1561` is `TextMetrics { font_px: f, ..m }` —
`..m` already preserves `baseline_px`. Only `font_px` scales. The HTML preview
inherits automatically via `text_font_size_px` reading `TextBlock.metrics`
(Klickwege parity), so no second call site changes.

## EXACT CHANGE SHAPE

```rust
const MIN_REGION_PITCH_LINES: usize = 3;
const SCALE_DEAD_BAND_LO: f32 = 0.80;
const SCALE_DEAD_BAND_HI: f32 = 1.25;
const SCALE_CLAMP_LO: f32 = 0.5;
const SCALE_CLAMP_HI: f32 = 3.0;

fn median_f32(v: &mut [f32]) -> Option<f32>;              // sort_by(f32::total_cmp), v[len/2]
fn region_pitch_px(lines: &[JsonLine]) -> Option<f32>;    // >=MIN_REGION_PITCH_LINES baselines
fn region_measured_font_px(lines: &[JsonLine]) -> Option<f32>;  // median of derive_text_metrics().font_px
fn page_measured_font_px_median(page: &JsonPage) -> Option<f32>; // body regions only
fn region_scale_factor(region: &JsonRegion, refs: &PageScaleRefs) -> f32; // 1.0 unless deviating

struct PageScaleRefs { pitch: Option<f32>, measured: Option<f32> }
```

`region_scale_factor`: return `1.0` unless `region.kind` is `"header"`/`"footer"`;
try rung A (`region_pitch_px` / `refs.pitch`), else rung B
(`region_measured_font_px` / `refs.measured`); denominator `None` or `<= 0.0` ->
`1.0`; ratio inside the dead band -> `1.0`; else clamp.

In `doc_v1_layout` (`:1470-1476`): build `PageScaleRefs` once per page alongside
`page_font_px`. In the `_` arm, hoist `let scale = region_scale_factor(region,
&refs);` above the line loop (once per region — the same "classified ONCE per
region" reasoning the justification call already uses at `:1516`), and change
`:1559` to `TextMetrics { font_px: f * scale, ..m }`. Nothing else in the arm
moves.

## FALSIFIERS (in `mod normalization_tests`, `JsonPage`/`JsonRegion`/`JsonLine`/`JsonWord` all `Default`)

Each test's doc comment MUST name the exact edit that makes it fail.

1. **can-fire** — 6 `"text"` regions (>=4 lines each, real baselines + xheight +
   glyph_px) + 1 single-line `"header"` whose measured font is 2x body. Assert:
   the header block's `metrics.is_some()` **and** its `font_px ~ page_font_px x
   2.0`; every body block's `font_px == page_font_px` exactly.
   **THE TRAP, state it in the comment:** if the header line omits
   `xheight`/`baseline`, `derive_text_metrics` returns `None`, the block takes the
   `TEXT_HEIGHT_TO_FONTSIZE` box-height fallback, and a 2x-tall header box
   renders bigger **with the fix deleted** — the test would pass for the wrong
   reason. The `metrics.is_some()` assertion is what forbids that.
   *Disable:* drop `* scale` -> must fail.
2. **stay-silent** — all-`"text"` page. `assert_eq!` (no epsilon) every block's
   `font_px` against `page_font_px(&jp).unwrap()`. `f * 1.0f32` is exact in IEEE,
   so equality is the correct and stronger assertion.
   *Disable:* force `region_scale_factor` to return `2.0` -> must fail.
3. **dead-band** — a `"header"` measuring 1.1x body -> `assert_eq!` page size.
   *Disable:* remove the dead-band branch -> must fail (it would return 1.1).
4. **clamp** — a `"header"` measuring 10x -> `font_px == page_font_px * 3.0`.
   *Disable:* remove the `.clamp` -> must fail.
5. **rung isolation** — a `"header"` with metrics on a page whose body lines carry
   NO metrics (rung-B denominator `None`) -> factor `1.0`.
   *Disable:* fall back to `page_font_px` as the denominator -> must fail.
6. **kind gate** — the identical 2x fixture with `kind = "text"` (and again with
   `kind = ""`) -> page size exactly. Proves the gate, not the band.

## RISKS

- **Vacuous-falsifier trap, twice-burned in this repo.** Test 1 is the exact
  shape that has failed here before (CLAUDE.md, two warning blocks). The worker must
  run the disable-the-fix edit for **every** test and report the observed failure
  message — not assert that it would fail.
- **Denominator emptiness.** A page that is ONLY header/footer has
  `page_measured_font_px_median == None` -> no deviation. Correct: there is no
  body to be different from.
- **`typography_overlay` is structurally insulated** — it builds
  `PageOcr`/`PlacedWord` -> `render_searchable_pdf`, never `doc_v1_layout`
  (`tests/typography_overlay.rs:39,262`). It cannot move. If it does, that is a
  real signal: re-measure, never widen the band to make it green.
- **Goldens** live in `tesseract-ocr` and this wave touches no file there.

## GATE CHECKLIST (orchestrator, centrally, `--release`)

1. `cargo test -p tesseract-ocr-pdf --release` — `normalization_tests` +
   `mod tests`, 0 failures.
2. `cargo test -p tesseract-ocr-pdf --release --test typography_overlay` —
   unchanged (insulation check).
3. `cargo fmt -p tesseract-ocr-pdf` + `cargo clippy -p tesseract-ocr-pdf --
   -D warnings`. **Never `--all`** (Iron rule 1).
4. Worker reports the six disable-the-fix runs with their real failure output.
5. `git show` the commit and confirm the code change and any CLAUDE.md prose
   landed in the SAME diff (the third-order failure recorded 2026-07-30).

---

# SPEC — #51 Step 1: retain per-timestep posteriors in RecodeBeamSearch
(authored by Opus planner; verbatim, entities normalized)

## OBJECTIVE
Retain a bounded, timestep-aligned top-K summary of the softmax rows decode already receives, expose read-only. Observationally inert: every existing decode result byte-identical, OFF by default. Unblocks #51 steps 2-5; implements none of them.

## OWNED FILE + ADDITIVE-ONLY
Sole file: crates/tesseract-core/src/recodebeam.rs. Sole tesseract-core worker this wave.
- NO existing line edited; every change an INSERTION between existing lines.
- compute_top_n (:584-617), decode_step (:623-686), continue_context, push_*, update_heap_if_matched, compute_code_hash, every extract_* : UNTOUCHED, zero insertions. Only decode_impl (:561-579) gains appended statements.
- No cargo, no git, no compile/test claims.
- Gate: clippy -p tesseract-core --all-targets -- -D warnings, Rust 1.97.1.

## OUT OF SCOPE
Pipeline steps 2-5; ogar/ocr/deepnsm wiring; lstm_recognizer.rs; lstm_choice_mode transcode; new contract symbols; doc.v1 emission.

## API (exact)
```rust
pub const RETAINED_TOP_K: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetainedClass { pub class: u16, pub prob: f32 }

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetainedStep {
    pub t: u32,
    /// Descending prob; ties ASCENDING class. top[..len] meaningful; tail padding prob = NEG_INFINITY.
    pub top: [RetainedClass; RETAINED_TOP_K],
    pub len: u8,
}
impl RetainedStep { #[must_use] pub fn top(&self) -> &[RetainedClass] { &self.top[..self.len as usize] } }

impl<'a> RecodeBeamSearch<'a> {
    #[must_use] pub fn retaining_posteriors(mut self) -> Self { self.retain_posteriors = true; self }
    /// EMPTY (never Option) when retention off. Index-aligned with the xcoords
    /// both extract_best_path_as_labels and _as_unichar_ids return:
    /// alternatives of char i are retained_posteriors()[xcoords[i] as usize].top().
    #[must_use] pub fn retained_posteriors(&self) -> &[RetainedStep] { &self.retained }
}
```
xcoord alignment (verified by planner): beams[t] chains t+1 nodes; extract path length == outputs.len(); extractors push raw t (:1467, :1540). So retained[j].t == j, len() == outputs.len().

K=8: code_range 111/115/4022; beam's own horizon K_BEAM_WIDTHS[0]=5 — retaining <=5 only re-reports the beam's view; 8 gives headroom, power of two. Do NOT reuse compute_top_n's heap (pins K to 5, inserts into parity-critical fn) — retention does its own pass.
Tie-break DELIBERATELY differs from the heap (insertion order): (prob desc, class asc) — reproducible consumer surface, not a transcode. Do not "fix" toward heap order.

## OPT-IN DECISION (builder, not signature)
Cost estimate (NOT measured): timesteps ~ width/3 (Mp3,3); ~72 B/step -> ~58 KB for 800-step line, one alloc/decode; ~89k f32 compares ~ tens of us vs ~110 ms/line -> <0.1%.
Opt-in wins anyway: (1) decode/decode_with_dict signatures are parity anchors with 6 call sites in lstm_recognizer.rs (:279,:315,:844,:854,:1545,:1555) — file not owned; (2) parallel decode_retaining doubles the decode surface 2->4; (3) builder makes inertness true BY CONSTRUCTION. Revisit always-on only with a real stage_timing delta.

## IMPLEMENTATION POINTS (exact insertions)
1. :63/64 — after INVALID_UNICHAR_ID: pub const RETAINED_TOP_K.
2. :439/441 — after WordResult's }: RetainedClass, RetainedStep + impl.
3. :402/403 — in struct RecodeBeamSearch after top_heap: retain_posteriors: bool, retained: Vec<RetainedStep>, (doc'd).
4. :494/495 and :529/530 — in new and new_with_dict after top_heap init: retain_posteriors: false, retained: Vec::new(),.
5. :531/533 — after new_with_dict's }: retaining_posteriors + retained_posteriors block.
6. decode_impl, exactly two appended statements:
   - after self.best_initial_dawgs.clear(); (:574): self.retained.clear(); if self.retain_posteriors { self.retained.reserve(outputs.len()); }
   - after self.decode_step(row, t); (:577) in the loop: self.retain_step(row, t);  (downstream of every parity mutation; reads only row + t).
7. :617/619 — after compute_top_n's }: private fn retain_step(&mut self, outputs: &[f32], t: usize):
   - first: if !self.retain_posteriors { return; }
   - top = [RetainedClass { class:0, prob: NEG_INFINITY }; K]; len = outputs.len().min(K) as u8 (computed from row length, never by counting sentinels);
   - iterate ascending class order; if p <= top[K-1].prob continue; else first j with p > top[j].prob, shift right (copy_within / reverse index writes, no needless_range_loop), write. Ascending + strict > yields (prob desc, class asc) free.
   - push RetainedStep { t: t as u32, top, len }.
   - Doc: NaN fails both <= and > -> silently excluded, deliberate.

## FALSIFIERS (appended at END of mod tests; committed-corpus loaders only, never /tmp)
(a) INERTNESS retention_does_not_change_any_decode_output — plain vs .retaining_posteriors() on identical rows; assert labels + unichar_ids (certs/ratings via f32::to_bits) + words equal. Both arms: passthrough_recoder(5) synthetic AND real eng rows for [91,97,92] via new_with_dict + decode_with_dict(2.25, -0.085, K_MIN_CERTAINTY). Real gate: the 11 pre-existing recodebeam + 4 dict_walker tests green with ZERO edits — say so in doc comment.
(b) FIDELITY retained_top_k_equals_an_independent_full_sort — non-monotone permutation row (7 leads, then 3, then 9); reference via DIFFERENT algorithm (full sort_by (prob desc, idx asc)); assert class sequence AND prob.to_bits()==outputs[class].to_bits().
(c) TIE-BREAK a_uniform_row_retains_the_lowest_class_indices_in_order — all equal, n>K -> exactly [0..K-1]. Fails heap/insertion-order tie-breaks.
(d) BOUNDEDNESS retention_is_timestep_aligned_and_bounded — len==rows.len(); t==j; top().len()==min(n,K) incl. passthrough n=5<8; OFF arm empty AND ON arm non-empty.
(e) DISABLE table (orchestrator executes):
  delete the off-guard -> (d) OFF-arm; index-order retention -> (b); > to >= -> (c); push only t==0 -> (d); retain_step writes top_n_flags -> (a) AND the 11 pre-existing.

## RISKS
Parity contained to two appended statements; no derives on RecodeBeamSearch so new fields can't break trait shapes. Memory bounded, clear() reuses capacity. Clippy: #[must_use], docs on all new pub items, no #[allow]. Semantic trap for step 2 (document): raw softmax top-K != beam TopN != null-forced flags (compute_top_n :616 forces null_char to Top2).

## GATE CHECKLIST
1. fmt -p tesseract-core clean. 2. clippy --all-targets -D warnings clean. 3. cargo test -p tesseract-core --release: 23/23 pre-existing + 4 new = 27/27; any pre-existing red = STOP not re-pin. 4. run disable table, every row produces the named failure, revert each. 5. cargo test -p tesseract-ocr --release goldens unchanged (default-off no-op proof).
