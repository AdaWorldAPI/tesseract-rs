# Typography & Placement Improvement Plan

> Scope: overlay/rendered font size, text/baseline placement, and table
> structure adherence — across **both** render surfaces the user named: the
> `/debug` side-by-side A|B HTML preview and the actual PDF outputs
> (searchable + structured). Recognition itself (`tesseract-ocr`'s
> byte-parity leaves) is NOT implicated — every root cause below lives in
> the consumer-side synthesis/render layer (`tesseract-ocr` structured.rs
> classification, `tesseract-ocr-pdf` layout.rs rendering), same footing as
> the double-escape/WinAnsi fixes already landed 2026-07-30.

## 1. Executive summary

**What's broken.** A real two-column scanned page renders with text that
looks roughly 25-40% of its visually-expected size, in both the `/debug`
preview and the PDF outputs. Measured precisely on one representative line:
the renderer emits a font size at **50% of box-height** (`Tf=30.5` against a
61px-tall cell) via a hardcoded fallback formula — smaller even than the
**79%-of-box-height** a correct render driven by this exact line's own
already-computed metrics would produce. The rendered baseline is also
mis-anchored ~19pt too low relative to where the real measured baseline
sits.

**Measured evidence (exact numbers, from a real 2550×3300px Alice-in-
Wonderland-style scan, `/tmp/alice_page.pgm`):**

- `doc.v1` region classification: **69 of ~72 total lines (96%)** of this
  page's ordinary flowing prose were misclassified `type=table` (2 of 5
  regions), degenerating to one cell per line (Finding B).
- The exact same misclassified line carries real, non-degenerate per-line
  typographic metrics in `doc.v1` (`xheight=22.0, ascrise=13.0,
  descdrop=-13.0, baseline=554.9`, `row_height = 22+13-(-13) = 48px`), but
  the PDF/HTML renderers **never read them for table cells** because
  `TableCell` has no `metrics` field at all — the call site hardcodes
  `None` (Finding A).
- Rendered result for that line: `Tf=30.5` = `61 × 0.5` (`box_h_pt ×
  TEXT_HEIGHT_TO_FONTSIZE`), the crude box-height fallback — **not**
  `48/61 = 79%` (what the real metrics would drive), and nowhere near the
  ~78% of box-height a reader visually expects real glyphs to occupy.
  `30.5/61 = 50%` vs `48/61 = 79%`: the crude fallback is **~37% smaller**
  than even the metrics-based render would be, on top of being globally
  wrong-shaped for typography.
- Same line's baseline: the fallback anchors the PDF pen `y` at the box's
  **bottom edge** (`bottom=574`), while the real measured baseline sits at
  `554.9` — **19.1pt too low** at this document's dpi (72, 1px≡1pt) (Finding
  C).

**The causal chain (B feeds A, A only matters because of B; C rides on the
same hardcoded `None`):**

```
Root Cause B (decide_if_table / block_is_table)
  → mislabels 69/72 ordinary prose lines as `type=table`
  → structured.rs builds TableCell{..} for each, with NO metrics field
     (TableCell has never carried typographic data — structural, not a
      missed threading step for one call site)
       │
       ▼
Root Cause A (layout.rs:480 / :765 hardcode `emit_text_run(..., None)`
  and `text_font_size_px(cell.bbox, None)` for EVERY table cell,
  misclassified or genuinely tabular alike)
  → every one of those 69/72 lines renders via the crude 0.5×box-height
     fallback instead of its own already-computed row_height metrics
       │
       ▼
Root Cause C (the `None` branch of emit_text_run also anchors the PDF pen
  at box-BOTTOM, not the measured baseline)
  → text sits both too small AND ~19pt too low, for the same 96% of the
     page, from the same one hardcoded `None`
```

So nearly the whole page's text ends up visibly shrunk and low **not**
because the font-size math is wrong in general (ordinary `TextBlock` lines
already consume real metrics correctly, per the 2026-07-28 "Measured line
metrics → true text sizing" entry in `CLAUDE.md`) but because a
classification bug (B) routes almost the entire page through a cell type (A)
that was never wired to carry the metrics that already exist, and that same
missing-metrics branch also mis-anchors the baseline (C). Fixing A closes
both the size and placement defects for every line B currently misroutes,
and is a strict superset of "fix the table-cell path in general" — it also
protects a **genuinely** correctly-classified table's cells from the same
defect.

## 2. Findings

### Finding A — table cells structurally never carry/use typographic metrics

**Provenance note.** This finding's dedicated parallel verification agent
failed to produce usable output (returned schema-satisfying but content-free
placeholder data after exhausting its retry budget). The content below is
NOT from that failed pass — it is grounded directly in the main session's
own reads of `crates/tesseract-ocr-pdf/src/layout.rs` (lines 260-420,
420-540, 620-800, 1050-1210), performed before the verification workflow
was even launched, and re-confirmed against the citations below. Recorded
here so this document's own verification chain is honest end to end.

**Claim.** `TableCell` (the struct backing every rendered table-cell run, in
both the PDF and the HTML `/debug` twin) has no field to carry per-cell
typographic metrics, and both render call sites pass a hardcoded `None`
literal for every table cell unconditionally — not merely for misclassified
regions, for *any* table cell, including a genuinely correct table.

**File:line citations:**
- `crates/tesseract-ocr-pdf/src/layout.rs:198` — `TableCell` struct
  definition, no `metrics` field.
- `crates/tesseract-ocr-pdf/src/layout.rs:284` —
  `TEXT_HEIGHT_TO_FONTSIZE = 0.5`, the crude fallback constant.
- `crates/tesseract-ocr-pdf/src/layout.rs:291-327` — `emit_text_run`, the
  `Some(m)`/`None` branch.
- `crates/tesseract-ocr-pdf/src/layout.rs:480` — `Block::Table` cell loop:
  `emit_text_run(&mut ops, cell.bbox, &cell.text, dpi, page_h_pt, None)`
  (hardcoded `None`).
- `crates/tesseract-ocr-pdf/src/layout.rs:765` — the HTML `/debug` twin:
  `text_font_size_px(cell.bbox, None)` (same hardcoded `None`).
- `crates/tesseract-ocr/src/structured.rs` — `doc_v1_layout`'s table branch
  builds `TableCell` from `region.cells`, whose JSON shape (emitted by
  `structured.rs`'s table-cell branch of `render_doc`) has never carried
  `xheight`/`ascrise`/`descdrop`/`baseline`.
- Contrast: `crates/tesseract-ocr-pdf/src/layout.rs:1147-1183` — the
  ordinary text/paragraph/header/footer branch of `doc_v1_layout` DOES
  populate `TextBlock.metrics` from `xheight`+`baseline` when present on the
  line JSON. Table cells are the one block kind excluded from this.

**Why it happens.** `extract_table_grid` (`structured.rs`) was built,
per its own design note in `CLAUDE.md`, as "pragmatic synthesis over the
proven word surface" — a `doc.v1`-native feature with no C++ oracle, added
after the per-line metrics plumbing (2026-07-28) already existed for
ordinary text blocks. The table-cell JSON shape and the `TableCell` Rust
struct were never retrofitted to also carry the metrics that the
constituent line(s) already compute internally (each cell derives from one
recognized line's `DocLineMetrics`) — so the metrics exist upstream, are
computed, and are simply never attached to the cell before it's serialized
or before the renderer consumes it.

**Blast radius.** Every table cell in every render (PDF searchable/
structured, `/debug` HTML), for **any** document — a correctly-classified
real table's cells are exactly as affected as a misclassified prose block's
"cells." On the reproduced page this is 69/72 lines (96%) because Finding B
additionally routes almost the whole page through this path, but Finding A
is a standalone defect independent of B.

### Finding B — the classification call (`table_blocks`) is unconditional; only the SPLITTING call is gated

> Fully re-verified against the current source this session, superseding an
> earlier hedged version of this finding — see the correction note below.

**Claim, now fully verified against the current source (not a pointer —
read directly, 2026-07-30 follow-up session).** `LstmRecognizer::
recognize_document_with_options` (`crates/tesseract-ocr/src/
lstm_recognizer.rs`) computes TWO independent things from `opts.
strip_borders`, and CLAUDE.md's "Ingredient 3 STRUCTURALLY LANDED" section
is correct about the FIRST but does not cover the SECOND:

1. **Whether a table's columns get fragmented into separate leaves
   (SPLITTING) — gated, exactly as documented.** Both `rec_blocks`
   (lines 1201-1205) and the classification-side `blocks` list
   (lines 1241-1245) branch `if opts.strip_borders { xy_cut_table_aware(...)
   } else { xy_cut(...) }` — plain, non-table-aware `xy_cut` for every
   caller that hasn't opted in, exactly as CLAUDE.md states.
2. **Whether a `blocks` entry gets LABELED `type=table` at all
   (CLASSIFICATION) — genuinely unconditional, no gate at all.**
   Immediately after those two gated calls, lines 1269-1272:
   ```rust
   let table_blocks: Vec<bool> = blocks
       .iter()
       .map(|&blk| Self::block_is_table(&binary, w, h, blk))
       .collect();
   ```
   This has **no `if opts.strip_borders` branch whatsoever** — every
   `blocks` entry (from EITHER the gated OR ungated `xy_cut` call above it)
   is run through `block_is_table` and the boolean result is what
   `build_regions` (`structured.rs:597-602`, which takes `table_blocks: &[bool]`
   as a plain parameter and does no classification of its own) stamps as
   `region.type`. So on the default path (`strip_borders=false`, i.e. every
   existing caller of `recognize_document`/`recognize_document_with_mode`,
   incl. the web server's `/api/v1/recognize`/`/pdf`/`/debug` routes): `blocks`
   comes from plain (non-table-aware) `xy_cut`, which still splits by ordinary
   column gutters — for `alice_page.pgm` this correctly yields two tall
   column-shaped blocks — but EACH of those two blocks is then independently
   run through the exact same unconditional `block_is_table` check, and both
   clear `decide_if_table`'s borderless threshold (they're tall,
   whitespace-generous text columns), so both get stamped `type=table`. This
   reproduces the measured JSON exactly (2/5 regions, 69/72 lines, cols=1
   each — `extract_table_grid` correctly finds no internal column structure
   within an already-column-split block of prose, which is itself further
   evidence the region was never tabular).

**File:line citations (exact, re-verified this session against the current
tree, superseding the prior hedge):**
- `crates/tesseract-ocr/src/lstm_recognizer.rs:1201-1205` — `rec_blocks`,
  gated (confirms CLAUDE.md's documented behavior).
- `crates/tesseract-ocr/src/lstm_recognizer.rs:1241-1245` — the
  classification-side `blocks` list, ALSO gated the same way (splitting
  only).
- `crates/tesseract-ocr/src/lstm_recognizer.rs:1269-1272` — **the actual
  fix site**: `table_blocks` computed via `Self::block_is_table(&binary, w,
  h, blk)` in an unconditional `.map()`, no gate.
- `crates/tesseract-ocr/src/lstm_recognizer.rs:1365-1367` — `block_is_table`'s
  definition: a thin one-line wrapper, `crate::pageseg::region_is_table
  (binary, w, h, block)`.
- `crates/tesseract-ocr/src/pageseg.rs:710` — `region_is_table`, the
  byte-parity-proven `decide_if_table` scoring core (4-condition score:
  `nhb>1`, `nvb>2`, `nvw>3`, `nvw>6`; ≥2 == table).
- `crates/tesseract-ocr/src/structured.rs:597-602` — `build_regions`'s
  signature: `table_blocks: &[bool]` arrives as a plain parameter: this
  function performs NO classification itself, it only consumes the
  already-decided booleans from line 1269-1272 above.
- `crates/tesseract-ocr/tests/lab_table_grid.rs::borderless_table_is_not_detected`
  — pins the *narrow* anti-false-positive case (a small borderless-table
  fixture correctly stays untabled); does not cover this page's large-
  multi-column-prose false positive.

**Correction note on this plan's own history.** The version of this finding
first drafted by a parallel verification agent could not complete (a
`StructuredOutput` schema retry-cap failure), so the first cut of this
document cited CLAUDE.md's operational history only, hedged as "not
independently re-derived this session — treat as pointer." The main session
then read `lstm_recognizer.rs:1135-1274` directly and confirmed the ungated
`table_blocks` computation exists exactly as originally hypothesized, with
the precise line numbers above. The original claim was right; it is now
verified, not merely plausible.

**Why it happens.** `decide_if_table`'s borderless path counts vertical
whitespace corridors (`nvw`) as one of its two "table" signals, and it
cannot structurally distinguish "many long vertical whitespace corridors
because this is a ruled/borderless table" from "many long vertical
whitespace corridors because this is ordinary prose with generous column
margins and tall line spacing." The SPLITTING call was gated behind
`strip_borders` specifically because of this ambiguity (the 8-column
resolution-grid regression); the CLASSIFICATION call
(`table_blocks`/`block_is_table`) shares the identical ambiguity but was
simply never wired to the same gate when the splitting fix landed — this is
a **gap in the Ingredient-3 fix's own scope**, not a separate, independent
defect: the fix closed the fragmentation half of the false-positive and did
not close the labeling half.

**Blast radius.** Any document with tall, whitespace-generous text
columns — large-format scans, wide-margin printed pages, magazine/newsletter
layouts — regardless of whether `strip_borders` is set, because
classification is unconditional. Directly determines whether Finding A's
defect touches "just real tables" or "96% of an ordinary page," as measured.

### Finding C — metrics-absent baseline placement anchors at box-bottom, not the true baseline

**Claim.** Confirmed by independent verification (see JSON finding
`baseline-placement-fallback` above), with one correction to the originally
stated mechanism's precision. `emit_text_run` branches on
`metrics: Option<&TextMetrics>`: with `Some(m)` the pen `y` is
`page_h_pt - px_to_pt(m.baseline_px, dpi)` (the real measured baseline) and
`Tf` is `px_to_pt(m.font_px, dpi)`; with `None` the pen `y` is
`page_h_pt - px_to_pt(bottom, dpi)` — the box's **bottom edge** — and `Tf`
is `box_h_pt * TEXT_HEIGHT_TO_FONTSIZE`. The `None` branch fires
unconditionally for every table cell (both PDF and HTML — see Finding A's
citations), so this placement bug and Finding A's size bug are two
independent-but-coincident consequences of the exact same hardcoded literal.

**File:line citations:**
- `crates/tesseract-ocr-pdf/src/layout.rs:198` (`TableCell`, no `metrics`
  field — the structural root shared with Finding A).
- `crates/tesseract-ocr-pdf/src/layout.rs:291-327` (`emit_text_run`, the
  `Some`/`None` branch).
- `crates/tesseract-ocr-pdf/src/layout.rs:318-327` (the exact
  `y_pt`/`fontsize_pt` match arms).
- `crates/tesseract-ocr-pdf/src/layout.rs:480`, `:765` (the two hardcoded
  `None` call sites, shared with Finding A).
- `crates/tesseract-ocr-pdf/src/searchable_pdf.rs:358-361`
  (`px_to_pt = px*72/dpi`).

**Measured magnitude, this exact reproduced line (dpi=72, so `px_to_pt` is
the identity):** box bbox `[368,513,1110,574]` → `bottom=574`; real measured
`baseline_px` (top-down) `=554.9`. Fallback pen `y = page_h_pt - 574`.
Metrics-based pen `y = page_h_pt - 554.9`. **Delta = -19.1** — the fallback's
`Tm` y is **19.1 units lower** on the page than the real baseline would put
it (since PDF `y` grows upward, `page_h_pt - 554.9 > page_h_pt - 574`, i.e.
the real baseline sits *higher*). At this document's dpi=72 (1px≡1pt
exactly), 19.1pt = 19.1px. At a general dpi the delta in points would be
`px_to_pt(19.1, dpi) = 19.1 × 72/dpi`.

**Correction to the originally stated mechanism.** The magnitude is **not**
simply `abs(descdrop)` (13px here) — it is `bottom - baseline_px` in
general, which is *larger* than `abs(descdrop)` because `bottom` is the OCR
recognition band's bottom (a deliberately generous "at least"
ascender-to-descender crop per the earlier 2026-07-23 text-overlap-bug fix
in this same file), not a tight visual bbox ending exactly at the descender
floor. The directional mechanism (nonzero descender ⇒ true baseline sits
above box-bottom) is correct; the exact px figure to cite is the measured
19.1, not the `descdrop` field alone.

**Why it happens.** `TextBlock` (ordinary lines) got metrics threading in
the 2026-07-28 "Measured line metrics → true text sizing" work; `TableCell`
never did, because it is a structurally separate type built later by
`extract_table_grid`'s pragmatic-synthesis path, which was never revisited
to carry the same additive metrics keys.

**Blast radius.** Identical to Finding A's — the same hardcoded `None`
drives both bugs, so any fix to Finding A that threads real `metrics` into
`TableCell` closes this finding simultaneously, for the same set of
affected lines.

## 3. Prioritized fix list

**Ordering rationale.** Fix **B before A**, even though A is the fix that
actually resolves the user's report on this page. Reasoning: (1) B is
strictly independent of A — it is a classification bug with its own
regression surface (`lab_table_grid.rs`, `quality_resolution_grid.rs`), and
fixing it first immediately shrinks A's blast radius on THIS page from
96% of lines down to whatever fraction is genuinely tabular (measured: 0%,
since neither region is a real table) — an observable, testable win on its
own, deliverable and mergeable independently. (2) Landing B first also
de-risks A's own test design: once B is fixed, A's falsifier fixture can be
built as a **genuine** table (needed anyway, since A must not regress on
real tables), rather than accidentally depending on a *misclassified* prose
line to reach the `TableCell` code path at all. (3) A is required regardless
of B's outcome — a real table with real cells still needs metrics — so
landing it second does not waste any of B's work; the two are additive, not
sequential-dependent in the code-correctness sense, only sequential-
beneficial in the review/test-design sense.

Fix C is folded into fix A's PR (see rationale in the original finding: the
`None`/`Some` branch is being touched anyway, and a two-branch function
touched twice for two independently-discovered-but-colocated bugs is a
worse review experience than one pass).

---

### Fix 1 (was Finding B) — gate the ungated `table_blocks` computation at `lstm_recognizer.rs:1269-1272`

**What changes.** The exact fix site is now verified (see Finding B's
correction note) — `table_blocks` at `crates/tesseract-ocr/src/
lstm_recognizer.rs:1269-1272` currently reads:
```rust
let table_blocks: Vec<bool> = blocks
    .iter()
    .map(|&blk| Self::block_is_table(&binary, w, h, blk))
    .collect();
```
Two candidate approaches for what it should become:

1. **Preferred, mirrors the existing precedent exactly (same function, two
   lines above):** gate this exact computation the identical way
   `rec_blocks` (1201-1205) and `blocks` (1241-1245) already are —
   ```rust
   let table_blocks: Vec<bool> = if opts.strip_borders {
       blocks.iter().map(|&blk| Self::block_is_table(&binary, w, h, blk)).collect()
   } else {
       vec![false; blocks.len()]
   };
   ```
   — OR behind a new dedicated `DocumentOptions::classify_tables` flag if
   classification should remain available independent of border-stripping
   (needs a design call — see open question below). This exactly mirrors
   the fix CLAUDE.md already documents for the splitting call, using the
   SAME conditional shape already sitting two statements above it in the
   same function, and closes the identical ambiguity by the identical
   mechanism: a caller must opt in before `nvw`-only borderless detection is
   trusted.
2. **Alternative, does not require an opt-in flag:** scale
   `decide_if_table`'s `nvw` thresholds (or add a page-size/column-count-
   aware correction) the same way the `xy_cut` gutter-threshold fix
   (2026-07-30, "the multi-column gutter bug") scaled its own absolute
   threshold against the number of columns rather than the raw page width.
   Higher-risk (touches the byte-parity-proven `decide_if_table` core
   directly — CLAUDE.md marks that function `pixDecideIfTable`-byte-parity
   GREEN against liblept, so any threshold change there needs re-validation
   against the existing oracle, not just the new falsifier) — likely not
   worth it given approach 1 is a direct precedent match at much lower risk,
   verified now to be a two-line, same-shape, same-function change.

**Recommendation: approach 1**, since it reuses an already-reviewed,
already-precedented pattern rather than touching a byte-parity leaf, and the
exact edit is now pinned to a specific 4-line block, not a general area.

**Falsifier test.** Two-sided, mirroring `lab_table_grid.rs`'s existing
style:
- **Positive (must still detect a real table):** existing
  `lab_table_grid.rs` fixtures (ruled + borderless-table-with-real-borders)
  must still classify `type=table` after the gate/threshold change — i.e.
  the fix must not turn table detection off entirely.
- **Negative (must NOT misdetect tall prose as a table):** a new fixture —
  two tall (multi-thousand-px), whitespace-generous prose columns
  (mirroring the shape of the real repro, built the Rust-fixture way per
  CLAUDE.md's "fixture generation belongs in Rust" ruling, not a committed
  Python generator) — must classify `type=text`, not `type=table`.
  Regression-named e.g. `tall_multi_column_prose_is_not_a_table`.

**Existing tests that need re-pinning.**
- `lab_table_grid.rs::borderless_table_is_not_detected` — re-verify it
  still passes unchanged (it should, since this fix narrows/gates an
  existing false-positive path, not the detection logic itself for genuine
  small fixtures).
- `quality_resolution_grid.rs::resolution_grid_holds_the_8_7_0_pattern` —
  re-run to confirm the 8+7+0 CER pattern is unaffected (this test is
  already the canonical proof that this exact false-positive class doesn't
  regress recognition quality; if approach 1 is used and the new gate
  defaults to off, this should need no change at all).
- If approach 1 introduces a new `DocumentOptions` field, every call site
  constructing `DocumentOptions` (web server routes, `tesseract-ogar`'s
  `ocr_demo`, any doctest) needs its default-value behavior confirmed
  unchanged for callers who don't opt in.

---

### Fix 2 (was Findings A + C) — thread real metrics into `TableCell` and stop hardcoding `None`

**What changes.**
1. Add `metrics: Option<TextMetrics>` to `TableCell`
   (`crates/tesseract-ocr-pdf/src/layout.rs:198`), mirroring
   `TextBlock.metrics`'s existing shape exactly.
2. In `structured.rs`'s table-cell JSON emission (the `extract_table_grid`/
   `render_doc` table branch), add the same additive per-cell keys ordinary
   lines already carry: `xheight`/`ascrise`/`descdrop`/`baseline`, sourced
   from the cell's constituent line's already-computed `DocLineMetrics`
   before it is folded into a `TableCell`/JSON cell object. This is
   additive-only JSON (new keys, existing keys untouched) — matches the
   house style already used for `plain_text`/`fields_map` (2026-07-30) and
   the original per-line metrics work (2026-07-28).
3. In `doc_v1_layout`'s table branch (`crates/tesseract-ocr-pdf/src/
   layout.rs`, ~1126-1145), read the new JSON keys the same way the
   text/paragraph branch already does (lines 1169-1183) and populate
   `TableCell.metrics`.
4. Replace the two hardcoded `None` call sites:
   - `layout.rs:480`: `emit_text_run(&mut ops, cell.bbox, &cell.text, dpi,
     page_h_pt, cell.metrics.as_ref())`.
   - `layout.rs:765`: `text_font_size_px(cell.bbox, cell.metrics.as_ref())`.
5. **Fix C rides along for free** — once `emit_text_run` receives
   `Some(m)` for a table cell, it automatically uses the real baseline
   (`m.baseline_px`) instead of box-bottom; no separate code change needed
   beyond steps 1-4.
6. Optional, narrow, low-priority cleanup for the residual genuinely-
   metrics-less case (legacy `doc.v1` lines lacking `xheight`+`baseline`,
   which after this fix is the ONLY remaining `None` caller): add a small
   fixed-fraction descender allowance to the `None` branch (e.g.
   `K_DESCENDER_ALLOWANCE_FRAC ≈ 0.15-0.2`, grounded the same way
   `TEXT_HEIGHT_TO_FONTSIZE=0.5` cites `K_XHEIGHT_FRACTION`/
   `K_ASCENDER_FRACTION`/`K_DESCENDER_FRACTION` in `textline.rs`). Worth
   bundling into this PR since the two-branch function is being touched
   anyway, but not required to resolve the user's report.

**Falsifier test.**
```rust
#[test]
fn table_cell_pen_sits_on_measured_baseline_and_real_font_size_not_box_fallback() {
    // Cell bbox height 61, with a real measured baseline 19.1px above
    // box-bottom and row_height=48 (mirrors the reproduced Alice-page
    // line exactly).
    let cell = TableCell {
        row: 0, col: 0,
        bbox: (368, 513, 1110, 574),           // top-down; height 61
        text: "ice was beginning".to_string(),
        header: false,
        metrics: Some(TextMetrics { font_px: 48.0, baseline_px: 554.9 }),
    };
    let doc = LayoutDoc {
        dpi: 72,
        pages: vec![LayoutPage {
            width: 1400, height: 700, background: None,
            blocks: vec![Block::Table(TableBlock {
                bbox: (368, 513, 1110, 2895), rows: 1, cols: 1,
                cells: vec![cell], rules: true,
            })],
        }],
    };
    let (pdf, _r) = render_pdf(&doc).expect("render");
    let content = page_content(&pdf);

    // Placement: must sit on the measured baseline, not box-bottom.
    let expected_y = 700.0 - 554.9;
    let bottom_anchored_y = 700.0 - 574.0; // the CURRENT (buggy) fallback
    assert!(ops(&content, "Tm").iter().any(|o| (num(o, 5) - expected_y).abs() < 1e-3));
    assert!(!ops(&content, "Tm").iter().any(|o| (num(o, 5) - bottom_anchored_y).abs() < 1e-3));

    // Size: must use the real measured font size (48px), not the
    // box-height×0.5 fallback (30.5pt).
    assert!(ops(&content, "Tf").iter().any(|o| (num(o, 1) - 48.0).abs() < 1e-3));
    assert!(!ops(&content, "Tf").iter().any(|o| (num(o, 1) - 30.5).abs() < 1e-3));
}
```
This test currently **fails to compile** (`TableCell` has no `metrics`
field), which is itself the falsifier — the fix is required before the
fixture can even be constructed, and once it exists it distinguishes
fixed-vs-unfixed by both the 19.1pt placement delta and the ~18pt font-size
delta measured on the real page.

**Existing tests that need re-pinning.**
- `crates/tesseract-ocr-pdf/tests/typography_overlay.rs` — currently only
  exercises `TextBlock` (crisp cell 0 of the resolution grid); should gain
  an equivalent table-cell assertion group once real table fixtures with
  ruled/borderless structure and metrics are available (depends on Fix 1
  landing first, or a synthetic genuinely-tabular fixture).
- Any existing golden-PDF/HTML snapshot tests that include a `Block::Table`
  with cells (search `tests/` for `TableBlock`/`Block::Table` fixtures) —
  their expected `Tf`/`Tm` values will shift once real metrics replace the
  0.5×box-height fallback; each must be re-measured (actual output read,
  not guessed) and re-pinned, per CLAUDE.md's established re-pinning
  discipline ("the `left` (actual) output was read and the `right`
  (expected) literal updated to match it exactly").
- `structured.rs`'s existing table-cell JSON golden-shape tests
  (`render_json_*` family) — adding new additive keys to cell JSON needs
  the same treatment `plain_text`/`fields_map` got: re-pin the two
  pre-existing golden-shape tests to include the new keys at their real
  emitted position, do not silently widen.

## 4. Non-goals / deferred

Named explicitly per the sub-agent findings and the additional session
context, so nothing is silently dropped even though it's out of this plan's
scope:

- **Table column-split threshold tuning (`extract_table_grid`'s "7×3
  vs printed 7×4" gap).** CLAUDE.md's own "Ingredient 3" section names this
  as "a genuinely different, narrower problem" from the structural
  fragmentation fix already landed — a median-word-height whitespace-gap
  heuristic that still merges two of four real columns on one fixture. Not
  a typography/placement/table-structure-*adherence* defect in the sense
  this plan addresses (metrics/classification); it's a column-count
  accuracy tuning question inside an already-correct row/cell structure.
  Filed as its own follow-up, deliberately not folded in here.
- **The genuine two-defect table gap (borderless tables score zero;
  ruled-table columns can still collapse to 1 column on some fixtures)**
  documented in CLAUDE.md's "★ TABLE EXTRACTION IS NOT READY FOR LAB
  REPORTS OR INVOICES" section — real, measured, but a **detection/column-
  splitting** defect family, not a typography-rendering defect. Overlaps
  Fix 1's classification work at the code-location level (`decide_if_table`)
  but is a distinct failure mode (missed detection vs. false-positive
  detection) with its own regression tests (`tests/lab_table_grid.rs`,
  `tests/lab_table_columns.rs`) already tracking it. Not in scope here.
- **The line-final-period/drop-cap recognition gaps** (CLAUDE.md's "Two
  findings from that page that are NOT these bugs, and NOT yet explained" —
  now the period bug is fixed, drop-cap remains open). These are
  **recognition** defects (blob-filter noise handling, line-band
  segmentation), not placement/sizing/table-structure defects — explicitly
  out of scope for a plan about rendering typography.
- **Word/box-level `ExtractBestPathAsWords` (B3-full)** and **dict-beam
  (C1)/CJK-trie (C3) accuracy layers** — named in CLAUDE.md as "accuracy
  layers, not pipeline gaps," unrelated to this plan's scope.
- **The residual metrics-less-legacy-line descender-allowance cleanup**
  (Fix 2 step 6) is explicitly optional/low-priority within this plan, not
  a full non-goal — bundle if convenient, do not block the PR on it.
- **Design question left open, not resolved here:** whether Fix 1 should
  reuse `DocumentOptions::strip_borders` directly (simplest, matches
  precedent exactly, but conflates "I want borders stripped for OCR
  cleanliness" with "I trust table classification on this kind of page,"
  which are logically separate concerns) or introduce a new dedicated flag.
  Flagged for the implementer/reviewer to decide before landing Fix 1, not
  pre-decided by this document.

## 5. How to verify this plan's own claims

Every number in §1/§2 was gathered against a real 2550×3300px scanned page
(`/tmp/alice_page.pgm`, Alice-in-Wonderland-style prose, two body-text
columns + header + footer) through the **running** `tesseract-ocr-web`
server. `/tmp` artifacts are ephemeral by convention in this repo (per
CLAUDE.md's "the proven method" section) — regenerate as needed; the routes
and extraction technique below are the durable record.

**Repro commands:**

```sh
# 1. Start the web server (release — debug recognition is ~50x slower,
#    per CLAUDE.md's repeated debug-vs-release lesson).
cargo run --release -p tesseract-ocr-web

# 2. doc.v1 JSON (region classification + per-line metrics).
curl -sS -F "file=@/tmp/alice_page.pgm" \
     http://localhost:8080/api/v1/recognize \
     -o /tmp/typog/doc_alice.json
# Inspect: jq '.pages[0].regions[] | {kind, bbox, lines: (.lines|length)}' \
#          /tmp/typog/doc_alice.json
# Inspect one line's metrics:
#   jq '.pages[0].regions[2].lines[0]' /tmp/typog/doc_alice.json

# 3. Structured PDF (the render under test).
curl -sS -F "file=@/tmp/alice_page.pgm" \
     "http://localhost:8080/pdf?mode=structured" \
     -o /tmp/typog/structured_alice.pdf

# 4. Searchable PDF (for the WinAnsi/escape-bug class of check, unrelated
#    to this plan but same extraction technique).
curl -sS -F "file=@/tmp/alice_page.pgm" \
     http://localhost:8080/pdf \
     -o /tmp/typog/searchable_alice.pdf

# 5. /debug side-by-side A|B HTML preview.
curl -sS -F "file=@/tmp/alice_page.pgm" \
     http://localhost:8080/debug \
     -o /tmp/typog/debug_alice.html

# 6. Extract the raw, uncompressed PDF content stream for direct
#    inspection of Tm/Tf/Tz operators (the technique this plan's Fact 3
#    and the 2026-07-30 WinAnsi-bug fix both used):
python3 - <<'PY'
import re, zlib
data = open("/tmp/typog/structured_alice.pdf", "rb").read()
# Find each stream ... endstream, decompress if /FlateDecode, dump raw ops.
for i, m in enumerate(re.finditer(rb'stream\r?\n(.*?)endstream', data, re.S)):
    raw = m.group(1)
    try:
        raw = zlib.decompress(raw)
    except Exception:
        pass
    open(f"/tmp/typog/structured_alice.pdf.content{i}.txt", "wb").write(raw)
PY
grep -n 'Tm\|Tf\|Tz\| re \| S' /tmp/typog/structured_alice.pdf.content0.txt | head -50
```

**What to check against the numbers in this plan:**
- doc.v1 region `kind`/`type` fields for the two large body-text regions —
  should read `text`, not `table`, once Fix 1 lands (currently: `table`,
  2 of 5 regions, 69/72 lines).
- The `Tf`/`Tm` operator pair for any line inside those regions in the
  extracted content stream — should show `Tf` close to the region's own
  `row_height` (`xheight+ascrise-descdrop`) in points, and `Tm`'s `y`
  operand close to `page_h_pt - baseline_px`, once Fix 2 lands (currently:
  `Tf = box_h_pt*0.5`, `Tm.y = page_h_pt - bottom`).
- The `/debug` HTML's rendered font-size CSS/inline-style for the same
  lines should track the PDF's `Tf` value (Klickwege parity is a repo-wide
  invariant — see the 2026-07-23 "preserving Klickwege parity" note in
  CLAUDE.md) — if the two surfaces ever diverge after this fix lands,
  that is itself a new bug, not a re-derivation of this plan's numbers.

Re-running this exact sequence against a future commit should reproduce
`69/72` misclassified lines pre-Fix-1, `0/72` (or the true tabular count)
post-Fix-1, and the `50%`/`79%` font-size-fraction split pre/post-Fix-2 —
these three numbers are the plan's own falsifiers.
