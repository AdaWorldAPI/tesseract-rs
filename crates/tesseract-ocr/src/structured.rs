//! Structured document output — a JSON DOM (`doc.v1`) over the recognizer's
//! word results, plus a typed-field harvest for invoice/form pages
//! (numeric hardening, IBAN checksum, label-proximity field extraction).
//!
//! **Consumer-side layer — NOT a Tesseract transcode.** Tesseract 5 has no
//! JSON renderer and no field-extraction stage; this module is this crate's
//! own output surface, built ON TOP of the proven pipeline (word text via
//! `ids_to_text`, boxes/confidence exactly as the TSV/hOCR renderers derive
//! them — see `renderer.rs` for those transcodes). Nothing here feeds back
//! into recognition, so no parity claim applies or is made.
//!
//! ## The JSON shape (`schema: "tesseract-rs/doc.v1"`)
//!
//! ```json
//! {
//!   "schema": "tesseract-rs/doc.v1",
//!   "pages": [{
//!     "page": 1, "width": 2480, "height": 3508,
//!     "quality": {"mean_conf": 96.10, "low_confidence": false},
//!     "regions": [{
//!       "type": "paragraph",
//!       "bbox": [l, t, r, b],
//!       "lines": [{
//!         "bbox": [l, t, r, b],
//!         "words": [{"text": "…", "bbox": [l, t, r, b], "conf": 96.5,
//!                     "leading_space": true, "numeric_norm": "250"}]
//!       }]
//!     }],
//!     "fields": [{"key": "netto", "label": "Netto:", "value": "1.250,00",
//!                  "value_cents": 125000, "bbox": [l, t, r, b],
//!                  "conf": 96.1, "checks": ["arithmetic_ok"]}],
//!     "plain_text": "…\n…",
//!     "fields_map": {"netto": "1.250,00"}
//!   }]
//! }
//! ```
//!
//! - `plain_text` and `fields_map` are Power Automate/low-code ergonomics,
//!   NOT new information — `plain_text` is every `words[].text` joined the
//!   same way [`crate::render_text`] joins them (per-word `leading_space`,
//!   a `\n` appended after every non-empty line INCLUDING the last one, an
//!   empty line contributing nothing at all — covering every line regardless
//!   of region classification), and `fields_map` is `fields` reshaped
//!   `key -> value` (last write wins on a duplicate key). Both are always
//!   present, even when empty (`""` / `{}`) — never `null`, never absent —
//!   so a no-code caller never needs a null-check before reading them. See
//!   [`render_doc`] for exactly where each is assembled.
//! - `bbox` is always top-down image coordinates `[left, top, right, bottom]`
//!   (the hOCR convention, via the same `PageIterator::BoundingBox` transcode
//!   the other renderers use).
//! - `conf` is the same 0–100 word confidence as TSV/hOCR
//!   (`ClipToRange(100 + 5·min(cert))`).
//! - `regions`: [`render_json`] emits ONE `"paragraph"` region (the plain
//!   default, byte-stable); [`render_json_with_regions`] emits CLASSIFIED
//!   regions built by [`build_regions`] from the layout stack — `"text"`
//!   (XY-cut blocks, reading order), `"table"` (a block whose leptonica
//!   `decide_if_table` score cleared the threshold), `"figure"` (halftone-mask
//!   components), `"header"`/`"footer"` (page furniture). Additive `type`
//!   values; consumers must ignore unknown ones.
//! - `quality.mean_conf` is the mean word confidence 0–100 (`null` when no
//!   words), and `low_confidence` flags a page below
//!   [`LOW_CONFIDENCE_THRESHOLD`] — the honesty signal that the input is
//!   likely handwriting / low-resolution / not printed text (`eng.lstm` is
//!   print-trained). See [`mean_word_confidence`].
//! - `numeric_norm` appears only on words the numeric hardening pass changed;
//!   `fields` only when a harvest ran. Consumers must ignore unknown keys.
//! - `glyph_px` (per-line, alongside `xheight`/`ascrise`/`descdrop`/
//!   `baseline` when present) is a DIRECT measured glyph ink-height in
//!   pixels ([`attach_glyph_px`]), replacing a statistical fit — an ink
//!   HEIGHT, not the body height the other four keys combine to. See
//!   [`attach_glyph_px`]'s doc comment for the measurement pipeline.
//!
//! ## Numeric hardening — "eine 0 kann nie ein O sein"
//!
//! In a digit-dominated token, confusable LETTERS are OCR misreads of digits
//! and are normalized: `O/o→0`, `I/l/|→1`, `Z/z→2`, `S/s→5`, `B→8`, `G→6`.
//! Guards keep legitimately-mixed identifiers untouched: GUIDs (hex+dash
//! shape), IBANs (checksum-validated instead), and any token where digits do
//! not strictly dominate letters (so `Summe`, `B8`-style codes, part numbers
//! survive). The original text is never destroyed — the normalized form goes
//! to `numeric_norm` alongside `text`.

use tesseract_core::CharSet;

use crate::renderer::LineWords;

/// One word in the structured DOM: rendered text, top-down image box,
/// 0–100 confidence, and the optional numeric-hardened form.
#[derive(Clone, Debug, PartialEq)]
pub struct DocWord {
    /// The word's text (`ids_to_text` over its unichar ids).
    pub text: String,
    /// Top-down image box `(left, top, right, bottom)`.
    pub bbox: (i32, i32, i32, i32),
    /// Word confidence 0–100 (same derivation as the TSV/hOCR renderers).
    pub conf: f32,
    /// Whether the recognizer emitted a leading space before this word.
    pub leading_space: bool,
    /// Set by [`harden_numeric_tokens`] iff the hardening changed the text.
    pub numeric_norm: Option<String>,
}

/// One recognized line: its top-down image box plus its words.
#[derive(Clone, Debug, PartialEq)]
pub struct DocLine {
    /// Top-down image box `(left, top, right, bottom)` of the line band.
    pub bbox: (i32, i32, i32, i32),
    /// Words in reading order.
    pub words: Vec<DocWord>,
    /// The row's measured typographic metrics, top-down converted (`None`
    /// when the recognition path carried none) — see [`DocLineMetrics`].
    pub metrics: Option<DocLineMetrics>,
}

/// A [`DocLine`]'s typographic metrics in the DOM's own (top-down) frame —
/// the renderer-facing conversion of
/// [`LineMetrics`](crate::renderer::LineMetrics), plus (optionally) a
/// DIRECT measured ink-height from [`attach_glyph_px`]. Emitted into
/// `doc.v1` as additive per-line keys (`xheight`/`ascrise`/`descdrop`/
/// `baseline`/`glyph_px`); consumers must ignore unknown keys, so old
/// readers are unaffected.
///
/// `xheight + ascrise - descdrop` (descdrop ≤ 0) is the full typographic
/// body height — exactly what real Tesseract sizes fonts from
/// (`LTRResultIterator::WordFontAttributes`, `ltrresultiterator.cpp:168-172`:
/// `row_height = x_height + ascenders - descenders`, converted to printer
/// points). The line's recognition-band `bbox` is deliberately TALLER than
/// this (ascender/descender slack plus `kImagePadding`), which is why sizing
/// text from the bbox alone overshoots.
///
/// [`glyph_px`](Self::glyph_px), when present, replaces that STATISTICAL fit
/// with a MEASUREMENT — see [`attach_glyph_px`]'s doc comment for the full
/// pipeline and why it exists (the fit above is unstable row-to-row: two
/// identically-printed table rows measured `24.7` vs `14.2` px through it, a
/// visible size jump in rendered output). It is an ink HEIGHT, a different
/// quantity from the body height above by a documented scale factor — see
/// the consumer (`tesseract-ocr-pdf`'s `GLYPH_PX_TO_FONT_PX`) for that
/// conversion; this module never performs it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DocLineMetrics {
    /// x-height in pixels.
    pub xheight: f32,
    /// Ascender rise above the x-height (≥ 0), pixels.
    pub ascrise: f32,
    /// Descender drop below the baseline (≤ 0), pixels.
    pub descdrop: f32,
    /// Baseline y at the line's horizontal midpoint, TOP-DOWN image pixels
    /// (`page_height - bottom_up_baseline`).
    pub baseline: f32,
    /// Measured glyph ink-height in pixels ([`attach_glyph_px`]) — `None`
    /// until that pass runs ([`DocPage::from_line_words`] alone never
    /// computes it). An ink HEIGHT, not the `xheight + ascrise - descdrop`
    /// body height the other three fields combine to — see the struct doc
    /// above.
    pub glyph_px: Option<f32>,
}

/// One page of structured output — the unit [`render_json`] serializes.
#[derive(Clone, Debug, PartialEq)]
pub struct DocPage {
    /// Page width in pixels.
    pub width: u32,
    /// Page height in pixels.
    pub height: u32,
    /// Non-empty lines, top-to-bottom (empty lines are skipped, same rule as
    /// every other renderer in this crate).
    pub lines: Vec<DocLine>,
}

impl DocPage {
    /// Build the structured DOM from recognizer line output — the same
    /// `LineWords` unit the TSV/hOCR renderers consume, converted through the
    /// same box/confidence derivations (`to_image_box`, `word_confidence`,
    /// `word_text` — see `renderer.rs` for those transcodes). Empty lines are
    /// skipped entirely.
    #[must_use]
    pub fn from_line_words(
        lines: &[LineWords],
        charset: &CharSet,
        page_w: u32,
        page_h: u32,
    ) -> Self {
        let pw = page_w as i32;
        let ph = page_h as i32;
        let doc_lines = lines
            .iter()
            .filter(|l| !l.words.is_empty())
            .map(|line| DocLine {
                metrics: line.metrics.map(|m| DocLineMetrics {
                    xheight: m.xheight,
                    ascrise: m.ascrise,
                    descdrop: m.descdrop,
                    // Bottom-up page space → the DOM's top-down frame.
                    baseline: ph as f32 - m.baseline,
                    // Never computed here -- [`attach_glyph_px`] is a
                    // separate pass (it needs the original `LineWords`
                    // char_boxes, which this DOM does not retain).
                    glyph_px: None,
                }),
                bbox: crate::renderer::to_image_box(line.line_box, pw, ph),
                words: line
                    .words
                    .iter()
                    .map(|w| DocWord {
                        text: crate::renderer::word_text(charset, w),
                        bbox: crate::renderer::to_image_box(
                            crate::renderer::union_boxes(w.char_boxes.iter().copied()),
                            pw,
                            ph,
                        ),
                        conf: crate::renderer::word_confidence(w),
                        leading_space: w.leading_space,
                        numeric_norm: None,
                    })
                    .collect(),
            })
            .collect();
        DocPage {
            width: page_w,
            height: page_h,
            lines: doc_lines,
        }
    }
}

/// Escape a string for a JSON string literal (RFC 8259 §7): `"` and `\` get
/// backslash escapes, the C0 controls use the short forms where they exist
/// (`\n` `\r` `\t` `\b` `\f`) and `\u00XX` otherwise. Everything else —
/// including all non-ASCII UTF-8 — passes through verbatim (JSON strings are
/// Unicode; no `\uXXXX` escaping of printable text).
fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Format a bbox as the JSON array `[l,t,r,b]`.
fn json_bbox(b: (i32, i32, i32, i32)) -> String {
    format!("[{},{},{},{}]", b.0, b.1, b.2, b.3)
}

/// The mean word confidence (0–100) over all words on the page, or `None`
/// when the page has no words. This is the page-level quality signal: the
/// recognizer's per-word confidence is the min over its character CTC
/// certainties (`100 + 5·min_cert`, clamped) — on clean printed text it sits
/// in the high 80s–100s; on OUT-OF-DISTRIBUTION input (handwriting,
/// low-resolution, or non-text) the softmax flattens, the certainties
/// collapse, and the mean drops sharply. It is a signal, NOT a proof — see
/// [`LOW_CONFIDENCE_THRESHOLD`].
#[must_use]
pub fn mean_word_confidence(page: &DocPage) -> Option<f32> {
    let (sum, n) = page
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .fold((0.0_f32, 0_usize), |(s, n), w| (s + w.conf, n + 1));
    (n > 0).then(|| sum / n as f32)
}

/// The mean-confidence floor below which a page is flagged `low_confidence`
/// in `doc.v1` — a **heuristic**, deliberately conservative, NOT calibrated
/// against a labelled handwriting corpus. `eng.lstm` is a PRINT-trained model
/// (Tesseract's CTC-LSTM has no handwriting support in the standard tessdata),
/// so a handwritten or otherwise unreadable page produces confidently-shaped
/// but low-certainty garbage; this floor lets a consumer surface "the model is
/// not confident — this may be handwriting / low-res / not printed text"
/// instead of returning the garbage silently. Tune per deployment; the raw
/// `mean_conf` value is always emitted so a consumer can apply its own gate.
pub const LOW_CONFIDENCE_THRESHOLD: f32 = 65.0;

/// Internal emit unit shared by both renderers: the `type` string, the
/// region bbox, and the owned line indices.
type EmitRegion<'a> = (
    &'a str,
    (i32, i32, i32, i32),
    Vec<usize>,
    Option<&'a TableGrid>,
);

/// Serialize one page (plus an optional field harvest) as a `doc.v1` JSON
/// document — see the module docs for the schema. `fields` may be empty
/// (serialized as `"fields":[]` so the key is always present and consumers
/// never need an existence check).
///
/// Confidences print with two decimals (`{:.2}`) — enough to preserve the
/// 0.5-steps the `100 + 5·cert` formula produces, without float noise.
#[must_use]
pub fn render_json(page: &DocPage, fields: &[HarvestedField]) -> String {
    // Default region synthesis: one "paragraph" over all lines — bbox = union
    // of the line boxes, same APPROX policy as TSV/hOCR's block/par rows. An
    // all-empty page emits an empty regions array. (The typed-region surface
    // is [`render_json_with_regions`]; this default keeps the plain renderer's
    // output byte-stable.)
    let default_regions: Vec<EmitRegion> = if page.lines.is_empty() {
        Vec::new()
    } else {
        let region_box = page.lines.iter().skip(1).fold(page.lines[0].bbox, |a, l| {
            let b = l.bbox;
            (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
        });
        vec![(
            "paragraph",
            region_box,
            (0..page.lines.len()).collect(),
            None,
        )]
    };
    render_doc(page, &default_regions, fields)
}

/// The kind of a classified [`DocRegion`] — the `type` value it serializes
/// as in `doc.v1`. Additive to the schema: `"paragraph"` (the
/// [`render_json`] default) and these four coexist; consumers must ignore
/// unknown values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    /// Body text (an XY-cut block's lines).
    Text,
    /// A tabular region — a [`Text`](RegionKind::Text) block whose leptonica
    /// `pixDecideIfTable` score cleared the table threshold (ruled lines /
    /// column corridors). Carries its lines like a text block.
    Table,
    /// An image / halftone region (from the halftone mask; carries no lines).
    Figure,
    /// Page-furniture header lines.
    Header,
    /// Page-furniture footer lines.
    Footer,
}

impl RegionKind {
    /// The `doc.v1` `type` string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            RegionKind::Text => "text",
            RegionKind::Table => "table",
            RegionKind::Figure => "figure",
            RegionKind::Header => "header",
            RegionKind::Footer => "footer",
        }
    }
}

/// One cell of a [`TableGrid`] — its grid position, image bbox, and the
/// concatenated text of the words that fell inside it.
///
/// `Eq` is deliberately NOT derived (unlike the rest of this struct's
/// siblings before [`TableCell::conf`] existed): `f32` has no total
/// ordering/equality (`NaN`), so a struct carrying one can only offer
/// `PartialEq`. `assert_eq!`/`==` are unaffected — only a generic `T: Eq`
/// bound would need reconsidering, and none of this crate's callers need one.
#[derive(Clone, Debug, PartialEq)]
pub struct TableCell {
    /// 0-based row (a recognized text line within the table region).
    pub row: usize,
    /// 0-based column (a whitespace-separated band).
    pub col: usize,
    /// Top-down image bbox — the union of the cell's words.
    pub bbox: (i32, i32, i32, i32),
    /// The cell's words joined in reading order.
    pub text: String,
    /// The cell's confidence (0–100, same scale as [`DocWord::conf`]) —
    /// the **MINIMUM** over the confidences of the words that fell inside
    /// it, deliberately NOT the mean.
    ///
    /// Min is the conservative aggregate on purpose: a lab/invoice VALUE
    /// cell is a single fact (`1O.5` vs `10.5`), and one misread character
    /// invalidates the whole reading — a mean would dilute exactly the
    /// signal a downstream low-confidence review gate exists to catch. A
    /// two-word cell reading "95% / 40%" confidence should report `40`, not
    /// `67.5`; a consumer gating on "is this cell trustworthy" needs the
    /// worst word, not the average one.
    pub conf: f32,
    /// `true` for the first row (the likely header).
    pub header: bool,
}

/// The reconstructed grid of a [`RegionKind::Table`] region — rows are the
/// recognized text lines, columns are the whitespace-separated bands. This is
/// doc.v1's delicate-feature **seed** for downstream structure mining
/// (`lance-graph-arm-discovery` / DeepNSM); it is pragmatic synthesis over the
/// proven word surface (like the rest of this module), NOT a Tesseract
/// transcode. Emitted inside a `"table"` region as `rows`/`cols`/`cells`.
///
/// `Eq` is not derived — see [`TableCell`]'s doc comment (this type holds
/// `Vec<TableCell>`, so the same `f32`-has-no-`Eq` reasoning cascades here).
#[derive(Clone, Debug, PartialEq)]
pub struct TableGrid {
    /// Row count (= recognized lines in the region).
    pub rows: usize,
    /// Column count (whitespace-separated bands).
    pub cols: usize,
    /// The occupied cells; empty grid positions are omitted.
    pub cells: Vec<TableCell>,
}

/// One classified page region: its kind, its top-down bbox, the indices of the
/// [`DocPage::lines`] it owns (empty for [`RegionKind::Figure`]), and — for a
/// [`RegionKind::Table`] — its reconstructed [`TableGrid`].
///
/// `Eq` is not derived — see [`TableCell`]'s doc comment (`table: Option<TableGrid>`
/// carries the same `f32`-has-no-`Eq` cascade).
#[derive(Clone, Debug, PartialEq)]
pub struct DocRegion {
    /// The region's kind (serialized as the `type` value).
    pub kind: RegionKind,
    /// Top-down image bbox `(left, top, right, bottom)`.
    pub bbox: (i32, i32, i32, i32),
    /// Indices into [`DocPage::lines`], in reading order.
    pub line_indices: Vec<usize>,
    /// The cell grid, present only for [`RegionKind::Table`] regions.
    pub table: Option<TableGrid>,
}

/// Union of two top-down boxes.
fn union_bbox(a: (i32, i32, i32, i32), b: (i32, i32, i32, i32)) -> (i32, i32, i32, i32) {
    (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
}

/// Reconstruct a [`TableGrid`] from a table region's `lines` — the delicate
/// feature that makes doc.v1 a good seed. Rows ARE the recognized lines;
/// columns come from vertical whitespace **rivers**: a separator must be
/// whitespace agreed by a MAJORITY of rows AND at least 2× a word-height wide
/// (a within-cell word space in a single row is neither, so a multi-word
/// description stays ONE cell — codex #41 P2). Each word joins the column band
/// its x-center lands in; a cell is one line's words in one column. Pragmatic
/// synthesis over the proven word surface — no border-mask or C-transcode
/// dependency, so it handles ruled and borderless tables alike. (Residual: a
/// sparse table whose single within-cell gap is both wide AND uncovered by
/// every other row stays ambiguous; a ruled table's vertical border mask would
/// settle it — future, this is word-only today.)
#[must_use]
pub fn extract_table_grid(lines: &[&DocLine]) -> TableGrid {
    let all: Vec<&DocWord> = lines.iter().flat_map(|l| l.words.iter()).collect();
    if all.is_empty() {
        return TableGrid {
            rows: lines.len(),
            cols: 0,
            cells: Vec::new(),
        };
    }

    let x0 = all.iter().map(|w| w.bbox.0).min().unwrap();
    let x1 = all.iter().map(|w| w.bbox.2).max().unwrap();
    let mut heights: Vec<i32> = all.iter().map(|w| (w.bbox.3 - w.bbox.1).max(1)).collect();
    heights.sort_unstable();
    let med_h = heights[heights.len() / 2].max(1) as usize;
    // A column separator must be WIDE (≥ 2× a word-height — within-cell word
    // spaces are ~1×) AND agreed by a MAJORITY of rows. Both guards keep a
    // multi-word description cell's internal gap from becoming a column (#41 P2).
    let gap_min = 2 * med_h;
    let width = (x1 - x0).max(1) as usize;

    // Per-x count of rows that are WHITESPACE there (no word of that row covers x).
    let mut gap_rows = vec![0u32; width];
    for line in lines {
        let mut row_covered = vec![false; width];
        for w in &line.words {
            let a = (w.bbox.0 - x0).clamp(0, x1 - x0) as usize;
            let b = (w.bbox.2 - x0).clamp(0, x1 - x0) as usize;
            for c in row_covered.iter_mut().take(b).skip(a) {
                *c = true;
            }
        }
        for (x, &cov) in row_covered.iter().enumerate() {
            if !cov {
                gap_rows[x] += 1;
            }
        }
    }
    let n_rows = lines.len() as u32;
    let support = if n_rows <= 1 { 1 } else { n_rows / 2 + 1 };

    // Cuts at the midpoint of every cross-row whitespace river wide enough to be
    // a column separator; the outer edges 0 and width bound the first/last cols.
    let mut cuts: Vec<usize> = vec![0];
    let mut gap_start: Option<usize> = None;
    for (x, &g) in gap_rows.iter().enumerate() {
        if g < support {
            if let Some(s) = gap_start.take() {
                if x - s >= gap_min {
                    cuts.push((s + x) / 2);
                }
            }
        } else if gap_start.is_none() {
            gap_start = Some(x);
        }
    }
    cuts.push(width);
    let cols = cuts.len() - 1;
    let col_of = |cx: i32| -> usize {
        let p = (cx - x0).clamp(0, width as i32 - 1) as usize;
        cuts.windows(2)
            .position(|w| p >= w[0] && p < w[1])
            .unwrap_or(cols - 1)
    };

    let mut cells = Vec::new();
    for (row, line) in lines.iter().enumerate() {
        let mut by_col: Vec<Vec<&DocWord>> = vec![Vec::new(); cols];
        for w in &line.words {
            let cx = (w.bbox.0 + w.bbox.2) / 2;
            by_col[col_of(cx)].push(w);
        }
        for (col, ws) in by_col.iter().enumerate() {
            let Some((first, rest)) = ws.split_first() else {
                continue;
            };
            let bbox = rest.iter().fold(first.bbox, |a, w| union_bbox(a, w.bbox));
            let text = ws
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            // MIN over the cell's words — see TableCell::conf's doc comment
            // for why (the same "worst word wins" rule word_confidence's own
            // per-character min already applies one level up).
            let conf = ws.iter().map(|w| w.conf).fold(f32::MAX, f32::min);
            cells.push(TableCell {
                row,
                col,
                bbox,
                text,
                conf,
                header: row == 0,
            });
        }
    }
    TableGrid {
        rows: lines.len(),
        cols,
        cells,
    }
}

/// Emit a [`TableGrid`] as the `rows`/`cols`/`cells` tail of a `"table"`
/// region object (leading comma; no surrounding braces).
fn emit_table_json(out: &mut String, grid: &TableGrid) {
    out.push_str(&format!(
        ",\"rows\":{},\"cols\":{},\"cells\":[",
        grid.rows, grid.cols
    ));
    for (i, c) in grid.cells.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"row\":{},\"col\":{},\"bbox\":{},\"text\":\"{}\",\"conf\":{:.2},\"header\":{}}}",
            c.row,
            c.col,
            json_bbox(c.bbox),
            json_escape(&c.text),
            c.conf,
            c.header
        ));
    }
    out.push(']');
}

/// Assemble typed regions from the classifier outputs:
///
/// - `header_lines` / `footer_lines` — line indices from the page-furniture
///   detector (`crate::page_furniture`).
/// - `blocks` — layout blocks in READING ORDER (e.g. `crate::xy_cut` leaves as
///   top-down `(l, t, r, b)`); each remaining line joins the FIRST block
///   containing its bbox center.
/// - `table_blocks` — parallel to `blocks`: `true` marks a block a table (its
///   region gets [`RegionKind::Table`] instead of [`RegionKind::Text`]). The
///   caller decides this on the FULL block bbox (see
///   [`crate::LstmRecognizer::recognize_document`]), not the emitted text-line
///   union — a shorter/positional slice would strip the rules and column
///   corridors `pageseg::decide_if_table` keys on. A short slice (or empty)
///   defaults every block to `Text`.
/// - `figures` — image-region bboxes (e.g. halftone-mask components,
///   `crate::pageseg`); they own no lines.
///
/// Emission order: header, blocks (with their lines, block order), a
/// catch-all `Text` region for body lines no block claimed (only if any),
/// figures, footer. Line-bearing regions get the union of their lines'
/// bboxes; empty blocks are dropped.
#[must_use]
pub fn build_regions(
    page: &DocPage,
    header_lines: &[usize],
    footer_lines: &[usize],
    blocks: &[(i32, i32, i32, i32)],
    table_blocks: &[bool],
    figures: &[(i32, i32, i32, i32)],
) -> Vec<DocRegion> {
    let mut block_members: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
    let mut header: Vec<usize> = Vec::new();
    let mut footer: Vec<usize> = Vec::new();
    let mut orphans: Vec<usize> = Vec::new();

    for (i, line) in page.lines.iter().enumerate() {
        if header_lines.contains(&i) {
            header.push(i);
            continue;
        }
        if footer_lines.contains(&i) {
            footer.push(i);
            continue;
        }
        let cx = (line.bbox.0 + line.bbox.2) / 2;
        let cy = (line.bbox.1 + line.bbox.3) / 2;
        match blocks
            .iter()
            .position(|&(l, t, r, b)| cx >= l && cx < r && cy >= t && cy < b)
        {
            Some(bi) => block_members[bi].push(i),
            None => orphans.push(i),
        }
    }

    let lines_region = |kind: RegionKind, members: &[usize]| -> Option<DocRegion> {
        let first = *members.first()?;
        let bbox = members
            .iter()
            .skip(1)
            .fold(page.lines[first].bbox, |a, &i| {
                union_bbox(a, page.lines[i].bbox)
            });
        Some(DocRegion {
            kind,
            bbox,
            line_indices: members.to_vec(),
            table: None,
        })
    };

    let mut out: Vec<DocRegion> = Vec::new();
    out.extend(lines_region(RegionKind::Header, &header));
    for (bi, members) in block_members.iter().enumerate() {
        let is_table = table_blocks.get(bi).copied().unwrap_or(false);
        let kind = if is_table {
            RegionKind::Table
        } else {
            RegionKind::Text
        };
        if let Some(mut region) = lines_region(kind, members) {
            if is_table {
                // Reconstruct the cell grid from the block's own lines — the
                // doc.v1 delicate-feature seed (rows = lines, cols = whitespace).
                let region_lines: Vec<&DocLine> = members.iter().map(|&i| &page.lines[i]).collect();
                region.table = Some(extract_table_grid(&region_lines));
            }
            out.push(region);
        }
    }
    out.extend(lines_region(RegionKind::Text, &orphans));
    out.extend(figures.iter().map(|&bbox| DocRegion {
        kind: RegionKind::Figure,
        bbox,
        line_indices: Vec::new(),
        table: None,
    }));
    out.extend(lines_region(RegionKind::Footer, &footer));
    out
}

/// Serialize with CLASSIFIED regions (see [`build_regions`]) instead of the
/// single-`"paragraph"` default. Same `doc.v1` shape; `type` takes the
/// [`RegionKind`] values.
#[must_use]
pub fn render_json_with_regions(
    page: &DocPage,
    regions: &[DocRegion],
    fields: &[HarvestedField],
) -> String {
    let mapped: Vec<EmitRegion> = regions
        .iter()
        .map(|r| {
            (
                r.kind.as_str(),
                r.bbox,
                r.line_indices.clone(),
                r.table.as_ref(),
            )
        })
        .collect();
    render_doc(page, &mapped, fields)
}

/// The shared `doc.v1` emitter over `(type, bbox, line indices)` regions —
/// both public renderers route through here so the schema cannot fork.
fn render_doc(page: &DocPage, regions: &[EmitRegion], fields: &[HarvestedField]) -> String {
    let mut out = String::new();
    out.push_str("{\"schema\":\"tesseract-rs/doc.v1\",\"pages\":[{");
    out.push_str(&format!(
        "\"page\":1,\"width\":{},\"height\":{},",
        page.width, page.height
    ));

    // Page-level quality signal (the honesty layer): mean word confidence +
    // the low-confidence flag. `mean_conf` is `null` on a page with no words.
    match mean_word_confidence(page) {
        Some(mc) => out.push_str(&format!(
            "\"quality\":{{\"mean_conf\":{:.2},\"low_confidence\":{}}},",
            mc,
            mc < LOW_CONFIDENCE_THRESHOLD
        )),
        None => out.push_str("\"quality\":{\"mean_conf\":null,\"low_confidence\":false},"),
    }

    // `plain_text` — additive, Power Automate ergonomics: the whole page as
    // ONE string, so a flow can plug OCR output straight into "Send an
    // email" / "Post a message" without an Apply-to-each over regions/lines.
    // Independent of `regions` (covers every line in `page.lines`, including
    // any orphan a region classifier didn't place) — same per-word
    // leading_space join and per-line '\n' separator as `render_text`
    // (renderer.rs): a line with no words contributes nothing (no text, no
    // separator), and every non-empty line gets a `\n` appended AFTER it,
    // including the last one.
    out.push_str("\"plain_text\":\"");
    for line in &page.lines {
        if line.words.is_empty() {
            continue;
        }
        for w in &line.words {
            // Mirrors `render_text`'s exact per-word rule (renderer.rs):
            // `if word.leading_space { push(' ') }` unconditionally, no
            // special-case for a line's first word — real recognizer output
            // already reports `leading_space: false` there, so trust the
            // signal rather than re-deriving position.
            if w.leading_space {
                out.push(' ');
            }
            out.push_str(&json_escape(&w.text));
        }
        out.push_str("\\n");
    }
    out.push_str("\",");

    out.push_str("\"regions\":[");
    for (ri, (kind, bbox, line_indices, table)) in regions.iter().enumerate() {
        if ri > 0 {
            out.push(',');
        }
        out.push_str(&format!("{{\"type\":\"{kind}\",\"bbox\":"));
        out.push_str(&json_bbox(*bbox));
        out.push_str(",\"lines\":[");
        for (li, &line_idx) in line_indices.iter().enumerate() {
            let line = &page.lines[line_idx];
            if li > 0 {
                out.push(',');
            }
            out.push_str("{\"bbox\":");
            out.push_str(&json_bbox(line.bbox));
            if let Some(m) = &line.metrics {
                // Additive per-line keys (consumers must ignore unknown
                // keys): the measured row metrics + top-down baseline, 1dp
                // (sub-pixel noise beyond that is meaningless downstream).
                out.push_str(&format!(
                    ",\"xheight\":{:.1},\"ascrise\":{:.1},\"descdrop\":{:.1},\"baseline\":{:.1}",
                    m.xheight, m.ascrise, m.descdrop, m.baseline
                ));
                if let Some(g) = m.glyph_px {
                    out.push_str(&format!(",\"glyph_px\":{g:.1}"));
                }
            }
            out.push_str(",\"words\":[");
            for (wi, w) in line.words.iter().enumerate() {
                if wi > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "{{\"text\":\"{}\",\"bbox\":{},\"conf\":{:.2},\"leading_space\":{}",
                    json_escape(&w.text),
                    json_bbox(w.bbox),
                    w.conf,
                    w.leading_space
                ));
                if let Some(norm) = &w.numeric_norm {
                    out.push_str(&format!(",\"numeric_norm\":\"{}\"", json_escape(norm)));
                }
                out.push('}');
            }
            out.push_str("]}");
        }
        out.push(']'); // close the lines array
        if let Some(grid) = table {
            emit_table_json(&mut out, grid);
        }
        out.push('}'); // close the region object
    }
    out.push_str("],");

    out.push_str("\"fields\":[");
    for (fi, f) in fields.iter().enumerate() {
        if fi > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"key\":\"{}\",\"label\":\"{}\",\"value\":\"{}\"",
            json_escape(&f.key),
            json_escape(&f.label_text),
            json_escape(&f.value)
        ));
        if let Some(cents) = f.value_cents {
            out.push_str(&format!(",\"value_cents\":{cents}"));
        }
        out.push_str(&format!(
            ",\"bbox\":{},\"conf\":{:.2},\"checks\":[",
            json_bbox(f.bbox),
            f.conf
        ));
        for (ci, c) in f.checks.iter().enumerate() {
            if ci > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"{}\"", json_escape(c)));
        }
        out.push_str("]}");
    }
    out.push(']'); // close the fields array

    // `fields_map` — additive, Power Automate ergonomics: the SAME fields as
    // above, reshaped `key -> value`. Finding one value in the `fields` array
    // needs a Filter-Array + First() expression in the no-code designer;
    // `fields_map['iban']` is a single dynamic-content lookup. Duplicate keys
    // (the harvester does not guarantee uniqueness) keep the LAST value, the
    // same "later write wins" rule a JS/Python object literal would apply.
    out.push_str(",\"fields_map\":{");
    for (fi, f) in fields.iter().enumerate() {
        if fi > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "\"{}\":\"{}\"",
            json_escape(&f.key),
            json_escape(&f.value)
        ));
    }
    out.push('}');

    out.push_str("}]}");
    out
}

// ---------------------------------------------------------------------------
// Numeric hardening — typed constraint over digit-context tokens
// ---------------------------------------------------------------------------

/// Map one confusable letter to the digit it is a misread of, if any.
/// The six classic OCR confusion pairs; anything else returns `None`.
fn confusable_digit(ch: char) -> Option<char> {
    match ch {
        'O' | 'o' => Some('0'),
        'I' | 'l' | '|' => Some('1'),
        'Z' | 'z' => Some('2'),
        'S' | 's' => Some('5'),
        'B' => Some('8'),
        'G' => Some('6'),
        _ => None,
    }
}

/// Characters that may appear inside a numeric token without disqualifying
/// it: digit separators, decimal marks, currency, sign, percent.
fn is_numeric_furniture(ch: char) -> bool {
    matches!(
        ch,
        '.' | ',' | '\'' | '-' | '+' | '%' | '€' | '$' | '£' | ' '
    )
}

/// GUID shape (`8-4-4-4-12` hex groups, any case) — such tokens legitimately
/// mix letters into digit runs and must never be "corrected".
#[must_use]
pub fn looks_like_guid(token: &str) -> bool {
    let groups: Vec<&str> = token.split('-').collect();
    groups.len() == 5
        && [8usize, 4, 4, 4, 12]
            .iter()
            .zip(&groups)
            .all(|(&len, g)| g.len() == len && g.chars().all(|c| c.is_ascii_hexdigit()))
}

/// IBAN shape: 2 ASCII letters + 2 digits + 11..=30 alphanumerics (total
/// 15..=34). Shape only — [`iban_mod97_ok`] is the actual validation.
#[must_use]
pub fn looks_like_iban(token: &str) -> bool {
    let bytes = token.as_bytes();
    (15..=34).contains(&bytes.len())
        && bytes[0].is_ascii_alphabetic()
        && bytes[1].is_ascii_alphabetic()
        && bytes[2].is_ascii_digit()
        && bytes[3].is_ascii_digit()
        && bytes[4..].iter().all(u8::is_ascii_alphanumeric)
}

/// ISO 13616 / ISO 7064 mod-97-10 IBAN checksum: move the first four chars to
/// the end, map `A..Z → 10..35`, and the resulting decimal number must be
/// `≡ 1 (mod 97)`. Case-insensitive; the input must already be shape-valid
/// ([`looks_like_iban`]) — spaces are NOT accepted here (join groups first).
#[must_use]
pub fn iban_mod97_ok(iban: &str) -> bool {
    if !looks_like_iban(iban) {
        return false;
    }
    let upper = iban.to_ascii_uppercase();
    let rearranged = format!("{}{}", &upper[4..], &upper[..4]);
    let mut rem: u32 = 0;
    for ch in rearranged.chars() {
        if let Some(d) = ch.to_digit(10) {
            rem = (rem * 10 + d) % 97;
        } else {
            // A=10 .. Z=35 — two decimal digits, folded in incrementally.
            let v = (ch as u32) - ('A' as u32) + 10;
            rem = (rem * 100 + v) % 97;
        }
    }
    rem == 1
}

/// Harden one token: if it is digit-DOMINATED (≥ 2 digits, strictly more
/// digits than letters, and every letter confusable + every other char
/// numeric furniture) — and NOT a GUID/IBAN — replace each confusable letter
/// with its digit. Returns `Some(normalized)` only when something changed.
///
/// The dominance gate is deliberately conservative: a token with as many
/// letters as digits (`B8`, `A1`) or any non-confusable letter (`Summe`,
/// `Rechnung`, part numbers like `XK-250`) is left untouched. One misread in
/// a real amount (`1.O50` → `1.050`, `2S0,00` → `250,00`) passes; a token
/// that would need half its characters "fixed" does not.
#[must_use]
pub fn harden_numeric_token(token: &str) -> Option<String> {
    if looks_like_guid(token) || looks_like_iban(token) {
        return None;
    }
    let mut digits = 0usize;
    let mut letters = 0usize;
    for ch in token.chars() {
        if ch.is_ascii_digit() {
            digits += 1;
        } else if ch.is_alphabetic() || ch == '|' {
            // A genuine (non-confusable) letter → not a numeric token.
            confusable_digit(ch)?;
            letters += 1;
        } else if !is_numeric_furniture(ch) {
            return None; // something structural (slash, colon, …) — leave it
        }
    }
    if letters == 0 || digits < 2 || digits <= letters {
        return None;
    }
    Some(
        token
            .chars()
            .map(|c| confusable_digit(c).unwrap_or(c))
            .collect(),
    )
}

/// Run [`harden_numeric_token`] over every word of a page, filling
/// [`DocWord::numeric_norm`] where the hardening fired. The original `text`
/// is never modified — consumers choose which form to trust.
pub fn harden_numeric_tokens(page: &mut DocPage) {
    for line in &mut page.lines {
        for word in &mut line.words {
            word.numeric_norm = harden_numeric_token(&word.text);
        }
    }
}

/// Parse a printed amount into cents. Handles the German and English
/// conventions: `1.250,00` / `1,250.00` (grouped + 2-digit decimals),
/// `1250,00` / `1250.00`, bare integers (`99` → `9900`), and currency/sign
/// furniture (`€ 99,50`, `-12,00`). A single separator followed by exactly
/// two digits at the end is the decimal mark; otherwise separators group
/// thousands. Returns `None` for anything that doesn't parse cleanly.
#[must_use]
pub fn parse_amount_cents(token: &str) -> Option<i64> {
    let cleaned: String = token
        .chars()
        .filter(|c| !matches!(c, '€' | '$' | '£' | ' ' | '\'' | '+'))
        .collect();
    let (neg, body) = match cleaned.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, cleaned.as_str()),
    };
    if body.is_empty()
        || !body
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == ',')
    {
        return None;
    }
    // Decide the decimal separator: the LAST '.' or ',' — but only if exactly
    // two digits follow it (printed money always has 2 decimals; a trailing
    // 3-digit group is a thousands group: "1.250" = 1250,00).
    let last_sep = body.rfind(['.', ',']);
    let (int_part, frac_part): (String, i64) = match last_sep {
        Some(pos) if body.len() - pos - 1 == 2 => {
            let frac: i64 = body[pos + 1..].parse().ok()?;
            (
                body[..pos].chars().filter(char::is_ascii_digit).collect(),
                frac,
            )
        }
        _ => (body.chars().filter(char::is_ascii_digit).collect(), 0),
    };
    if int_part.is_empty() && frac_part == 0 && body.chars().all(|c| !c.is_ascii_digit()) {
        return None;
    }
    let int_val: i64 = if int_part.is_empty() {
        0
    } else {
        int_part.parse().ok()?
    };
    let cents = int_val.checked_mul(100)?.checked_add(frac_part)?;
    Some(if neg { -cents } else { cents })
}

// ---------------------------------------------------------------------------
// Field harvest — label proximity + arithmetic cross-check
// ---------------------------------------------------------------------------

/// What kind of value a field expects — drives candidate filtering and
/// which checks run on the harvested value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    /// A money amount — candidate must satisfy [`parse_amount_cents`]
    /// (after numeric hardening).
    Amount,
    /// An identifier (invoice no, tax no, IBAN, …) — candidate is taken as
    /// text; IBAN-shaped values get the mod-97 check.
    Id,
}

/// One field to look for: a stable output key, the label spellings that
/// mark it on the page, and the expected value kind.
#[derive(Clone, Debug)]
pub struct FieldSpec {
    /// Stable key in the JSON output (`"netto"`, `"iban"`, …).
    pub key: &'static str,
    /// Lowercase label prefixes that identify this field on the page
    /// (matched against a lowercased, `:`-stripped label word).
    pub labels: &'static [&'static str],
    /// Expected value kind.
    pub kind: FieldKind,
}

/// The default German invoice field set: net/tax/gross amounts (wired to the
/// arithmetic cross-check), invoice number, tax numbers, IBAN.
#[must_use]
pub fn german_invoice_fields() -> Vec<FieldSpec> {
    vec![
        FieldSpec {
            key: "netto",
            labels: &["netto", "nettobetrag", "zwischensumme"],
            kind: FieldKind::Amount,
        },
        FieldSpec {
            key: "ust",
            labels: &[
                "ust",
                "ust.",
                "mwst",
                "mwst.",
                "umsatzsteuer",
                "mehrwertsteuer",
            ],
            kind: FieldKind::Amount,
        },
        FieldSpec {
            key: "brutto",
            labels: &[
                "brutto",
                "bruttobetrag",
                "gesamt",
                "gesamtbetrag",
                "summe",
                "endbetrag",
                "total",
            ],
            kind: FieldKind::Amount,
        },
        FieldSpec {
            key: "rechnungsnummer",
            labels: &[
                "rechnungsnr",
                "rechnungsnr.",
                "rechnungsnummer",
                "rechnung-nr",
                "re-nr",
                "re-nr.",
            ],
            kind: FieldKind::Id,
        },
        FieldSpec {
            key: "steuernummer",
            labels: &[
                "steuernr",
                "steuernr.",
                "steuernummer",
                "st-nr",
                "st-nr.",
                "ust-idnr",
                "ust-idnr.",
                "ust-id",
            ],
            kind: FieldKind::Id,
        },
        FieldSpec {
            key: "iban",
            labels: &["iban"],
            kind: FieldKind::Id,
        },
    ]
}

/// One harvested field: which spec matched, the label word as printed, the
/// value (hardened form where hardening fired), parsed cents for amounts,
/// the value word's bbox + confidence, and the validation checks that passed.
#[derive(Clone, Debug, PartialEq)]
pub struct HarvestedField {
    /// The matching [`FieldSpec::key`].
    pub key: String,
    /// The label word as printed on the page (`"Netto:"`).
    pub label_text: String,
    /// The harvested value (numeric-hardened form when it fired).
    pub value: String,
    /// Parsed cents for `Amount` fields (`None` for `Id` fields).
    pub value_cents: Option<i64>,
    /// Top-down image bbox of the value word(s).
    pub bbox: (i32, i32, i32, i32),
    /// Confidence of the value word(s) — the MINIMUM over joined words.
    pub conf: f32,
    /// Names of the checks that passed (`"iban_mod97_ok"`,
    /// `"arithmetic_ok"`). Empty = harvested but unverified.
    pub checks: Vec<String>,
}

/// Vertical overlap test: do two boxes share at least half of the shorter
/// box's height? (Same-line test for label→value pairing.)
fn same_band(a: (i32, i32, i32, i32), b: (i32, i32, i32, i32)) -> bool {
    let overlap = a.3.min(b.3) - a.1.max(b.1);
    let min_h = (a.3 - a.1).min(b.3 - b.1);
    min_h > 0 && overlap * 2 >= min_h
}

/// Normalize a printed label word for matching: lowercase, trailing `:`/`.`
/// stripped (`"Netto:"` → `"netto"`).
fn normalize_label(word: &str) -> String {
    word.trim_end_matches([':', '.']).to_ascii_lowercase()
}

/// The value form of a word: the hardened text when hardening fired, the raw
/// text otherwise.
fn value_text(w: &DocWord) -> &str {
    w.numeric_norm.as_deref().unwrap_or(&w.text)
}

/// Harvest typed fields from a page by label proximity:
///
/// 1. Find label words matching a [`FieldSpec`] (lowercased, `:`-stripped).
/// 2. Take the nearest suitable word to the RIGHT on the same line band
///    (`Amount`: must parse as an amount; `Id`: the next word). For `Id`
///    fields the value continues over following alphanumeric words (an IBAN
///    printed as `DE89 3704 …` groups) up to 34 chars.
/// 3. Amounts get [`parse_amount_cents`]; IBAN-shaped ids get
///    [`iban_mod97_ok`] → check `"iban_mod97_ok"`.
/// 4. If `netto`, `ust` and `brutto` were all harvested and
///    `netto + ust == brutto` (exact, in cents), all three get
///    `"arithmetic_ok"` — the cross-check that disambiguates which number is
///    which better than any single read.
///
/// First match per spec wins (top-to-bottom, left-to-right page order).
/// Run [`harden_numeric_tokens`] first so amounts see hardened text.
#[must_use]
pub fn harvest_fields(page: &DocPage, specs: &[FieldSpec]) -> Vec<HarvestedField> {
    let mut out: Vec<HarvestedField> = Vec::new();

    for spec in specs {
        if out.iter().any(|f| f.key == spec.key) {
            continue;
        }
        'search: for line in &page.lines {
            for (wi, word) in line.words.iter().enumerate() {
                let norm = normalize_label(&word.text);
                if !spec.labels.contains(&norm.as_str()) {
                    continue;
                }
                // Candidates: words to the right of the label, same band,
                // nearest first (reading order within the line suffices).
                let mut candidates = line.words[wi + 1..]
                    .iter()
                    .filter(|c| c.bbox.0 >= word.bbox.2 && same_band(word.bbox, c.bbox));
                match spec.kind {
                    FieldKind::Amount => {
                        for cand in candidates {
                            let text = value_text(cand);
                            if let Some(cents) = parse_amount_cents(text) {
                                out.push(HarvestedField {
                                    key: spec.key.to_string(),
                                    label_text: word.text.clone(),
                                    value: text.to_string(),
                                    value_cents: Some(cents),
                                    bbox: cand.bbox,
                                    conf: cand.conf,
                                    checks: Vec::new(),
                                });
                                break 'search;
                            }
                        }
                    }
                    FieldKind::Id => {
                        if let Some(first) = candidates.next() {
                            // Join following alnum words (IBAN groups etc.)
                            // up to the 34-char IBAN ceiling.
                            let mut value = value_text(first).to_string();
                            let mut bbox = first.bbox;
                            let mut conf = first.conf;
                            for extra in candidates {
                                let t = value_text(extra);
                                if value.len() + t.len() > 34
                                    || !t.chars().all(|c| c.is_ascii_alphanumeric())
                                {
                                    break;
                                }
                                value.push_str(t);
                                bbox = (
                                    bbox.0.min(extra.bbox.0),
                                    bbox.1.min(extra.bbox.1),
                                    bbox.2.max(extra.bbox.2),
                                    bbox.3.max(extra.bbox.3),
                                );
                                conf = conf.min(extra.conf);
                            }
                            let mut checks = Vec::new();
                            if iban_mod97_ok(&value) {
                                checks.push("iban_mod97_ok".to_string());
                            }
                            out.push(HarvestedField {
                                key: spec.key.to_string(),
                                label_text: word.text.clone(),
                                value,
                                value_cents: None,
                                bbox,
                                conf,
                                checks,
                            });
                            break 'search;
                        }
                    }
                }
            }
        }
    }

    // Arithmetic cross-check: netto + ust == brutto (exact cents).
    let cents = |key: &str| -> Option<i64> {
        out.iter()
            .find(|f| f.key == key)
            .and_then(|f| f.value_cents)
    };
    if let (Some(n), Some(u), Some(b)) = (cents("netto"), cents("ust"), cents("brutto")) {
        if n + u == b {
            for f in &mut out {
                if matches!(f.key.as_str(), "netto" | "ust" | "brutto") {
                    f.checks.push("arithmetic_ok".to_string());
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Measured glyph-ink font sizing (`glyph_px`) — a direct measurement
// replacing the statistical `xheight + ascrise - descdrop` fit
// ---------------------------------------------------------------------------
//
// `xheight + ascrise - descdrop` is a STATISTICAL FIT over a row's blob
// boxes (wave-3 `compute_block_xheight`) — measured unstable row-to-row:
// two table rows printed at the SAME point size measured `24.7` vs `14.2`
// px through it (1.74×), a visible size jump in rendered output.
// [`tesseract_core::WordResult::char_boxes`] cannot fix this directly —
// every char box in a line has EXACTLY the line-box height (they are CTC
// timestep boundaries; `y` is just the line band copied, only `x` carries
// real information) — so [`attach_glyph_px`] performs a NEW measurement
// instead: the actual ink extent inside each char box's x-span, scanned
// from the binarized page. Four stages, each a small helper below:
//
// 1. **Per-glyph ink height** ([`glyph_ink_heights`]) — for each word's
//    char boxes, the topmost-to-bottommost row of ink within the box's
//    x-span, over the line's own y-band.
// 2. **Per-line size = the 90th percentile** ([`percentile_90`],
//    [`line_raw_glyph_px`]) of those heights — NOT the median: the median
//    is content-dependent (a lowercase-heavy line reads smaller than a
//    digit-heavy one at the same point size), while p90 approximates
//    cap/ascender height, which stays far more stable across ordinary body
//    text.
// 3. **Small-sample guard** ([`GLYPH_SAMPLE_MIN`],
//    [`apply_small_sample_fallback`]) — a line with too few glyph samples
//    falls back to the median of the other sufficiently-sampled lines in
//    the same call's `lines` slice.
// 4. **Bent-paper trend normalization** ([`TREND_R2_SIGNIFICANT`],
//    [`fit_size_trend`], [`apply_trend_normalization`]) — a page
//    photographed at a slight tilt/curl varies glyph size SMOOTHLY with
//    `y`; real typography changes it in DISCRETE steps. A first-order trend
//    `size ≈ a + b·y` (the same shape `crate::rectify::fit_shear_ramp` uses
//    for shear, `slope(y) = m0 + m1·y`) is fit over the page's resolved
//    per-line sizes, and its SMOOTH component is divided out ONLY when the
//    fit is significant — a blanket per-line normalization would be WRONG:
//    this crate's own falsifier document genuinely contains a smaller
//    caption (`Kleine Schriftgröße`, p90 = 9 against a body of 12-13), and
//    flattening that would destroy real typography, not just correct camera
//    geometry.
//
// ## Why `DocLineMetrics`, not a new `DocLine` field
//
// `DocLine` is constructed via struct LITERAL (not a constructor function)
// in `crate::page_furniture`'s test helpers, outside this file's scope —
// adding a required field there would be a silent breaking change this
// file has no way to fix. [`DocLineMetrics`], by contrast, is constructed
// ONLY within this file ([`DocPage::from_line_words`] and this module's own
// tests), so it is safe to extend. The measurement is therefore attached
// only to lines that already carry a [`DocLineMetrics`] object; a line
// without one (no row-metrics pipeline ran) already falls all the way back
// to the bbox-height heuristic downstream (`tesseract-ocr-pdf`'s
// `TEXT_HEIGHT_TO_FONTSIZE`) regardless of this change.
//
// ## Units — deliberately NOT `font_px`
//
// The value this pipeline produces and [`attach_glyph_px`] writes is a raw
// ink HEIGHT (baseline to the top of the tallest ink in a glyph's own box),
// not the `xheight + ascrise - descdrop` full body height `font_px`
// elsewhere means — they are different quantities related by a scale
// factor. That conversion lives in the CONSUMER
// (`tesseract-ocr-pdf::layout::GLYPH_PX_TO_FONT_PX`), not here.
//
// ## Known residual — NOT fixed here
//
// A recognized border glyph (e.g. `|` misread from a table's vertical
// divider) spans the FULL cell height, inflating whichever line's p90 it
// lands in. `crate::pageseg::decide_if_table` already computes a de-lined
// page and discards it; wiring that de-lined page into this measurement
// (rather than the ordinary binarized page) is future work, not attempted
// here.

/// Minimum measured glyph samples a line needs before its OWN p90 is
/// trusted as-is. Below this, [`apply_small_sample_fallback`] replaces it
/// with the median of the other sufficiently-sampled lines in the same
/// call's line group (that group's "enclosing region").
///
/// Measured failure this guards against: a 7-glyph line (`"können,"`) gave
/// p90 = 24 against an enclosing body of 12-13 px — with so few samples, a
/// single unusually tall glyph (an ascender, a diacritic, or the rule-glyph
/// inflation this section's docs describe above) dominates the percentile
/// instead of being outvoted by the rest of the line. `10` is the pragmatic
/// threshold the measurement pointed to ("~10 glyph samples" per the
/// finding) — not a theoretically derived bound.
const GLYPH_SAMPLE_MIN: usize = 10;

/// Minimum variance-explained (R² — the squared Pearson correlation between
/// a line's vertical position `y` and its resolved size) before
/// [`apply_trend_normalization`] treats a page's fitted linear-in-`y` size
/// trend as real geometric distortion (a bent/tilted photograph) rather
/// than ordinary typographic variation (headings, captions, table cells at
/// arbitrary page positions).
///
/// `0.5` — over HALF the size variance across the page must be explained by
/// a straight line in `y` — is deliberately a high bar: this crate's own
/// falsifier document genuinely contains a smaller-printed caption
/// (`Kleine Schriftgröße`, p90 = 9 against a body of 12-13) that must
/// SURVIVE untouched. A single same-page step change like that contributes
/// little to R² (this module's own step-change test fixture measures
/// R² ≈ 0.003 for exactly that shape, far below this floor), whereas a
/// genuinely smooth camera-geometry gradient across MANY lines measures
/// close to `1.0`.
const TREND_R2_SIGNIFICANT: f32 = 0.5;

/// The topmost-to-bottommost row of INK (binarized `0`) within each char
/// box's x-span, scanned over the box's own y-extent converted to top-down
/// image space via [`crate::renderer::to_image_box`] (the same conversion
/// [`DocPage::from_line_words`] already applies to every box in this
/// module). A char box with no ink anywhere in its span (a boxed space, or
/// a degenerate/empty box) contributes NOTHING — not a zero-height sample,
/// which would corrupt the percentile with a value no real glyph produces.
///
/// `binary` is `page_w * page_h` bytes, row-major, top-down, `0` = ink —
/// this crate's bitonal convention throughout (`crate::xy_cut`'s module
/// docs). A `binary` shorter than `page_w * page_h`, or a non-positive
/// `page_w`/`page_h`, yields no samples (defensive; never hit on real
/// recognizer output, where `binary` is always the very page the boxes came
/// from).
fn glyph_ink_heights(
    char_boxes: &[(i32, i32, i32, i32)],
    binary: &[u8],
    page_w: i32,
    page_h: i32,
) -> Vec<f32> {
    if page_w <= 0 || page_h <= 0 {
        return Vec::new();
    }
    let bw = page_w as usize;
    let bh = page_h as usize;
    if binary.len() < bw * bh {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(char_boxes.len());
    for &cb in char_boxes {
        let (left, top, right, bottom) = crate::renderer::to_image_box(cb, page_w, page_h);
        if right <= left || bottom <= top {
            continue;
        }
        let x0 = (left as usize).min(bw);
        let x1 = (right as usize).min(bw);
        let y0 = (top as usize).min(bh);
        let y1 = (bottom as usize).min(bh);
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        let mut ink_top: Option<usize> = None;
        let mut ink_bottom: Option<usize> = None;
        for y in y0..y1 {
            let row = &binary[y * bw + x0..y * bw + x1];
            if row.contains(&0) {
                if ink_top.is_none() {
                    ink_top = Some(y);
                }
                ink_bottom = Some(y);
            }
        }
        if let (Some(t), Some(b)) = (ink_top, ink_bottom) {
            out.push((b - t + 1) as f32);
        }
    }
    out
}

/// The 90th percentile of `heights` (linear interpolation between the two
/// nearest ranks — the same convention as e.g. NumPy's default
/// `percentile`) — see this section's module doc for why p90 rather than
/// the median. Sorts `heights` in place. `None` on an empty slice.
fn percentile_90(heights: &mut [f32]) -> Option<f32> {
    if heights.is_empty() {
        return None;
    }
    heights.sort_unstable_by(f32::total_cmp);
    let rank = 0.9 * (heights.len() - 1) as f32;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f32;
    Some(heights[lo] + frac * (heights[hi] - heights[lo]))
}

/// The median of `values` (average of the two middle elements on an even
/// count). `None` on an empty slice.
fn median(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

/// One line's RAW glyph-ink measurement, before the small-sample fallback
/// and trend normalization: its p90 ink height plus how many glyph samples
/// backed it (the count [`GLYPH_SAMPLE_MIN`] gates on). `None` when the
/// line has zero measurable samples at all (never hit on real recognizer
/// output for a non-empty line, but kept total rather than panicking).
fn line_raw_glyph_px(
    line: &LineWords,
    binary: &[u8],
    page_w: i32,
    page_h: i32,
) -> Option<(f32, usize)> {
    let mut heights: Vec<f32> = Vec::new();
    for word in &line.words {
        heights.extend(glyph_ink_heights(&word.char_boxes, binary, page_w, page_h));
    }
    let n = heights.len();
    percentile_90(&mut heights).map(|p90| (p90, n))
}

/// Stage 3: replace an unreliable line's own p90 (fewer than
/// [`GLYPH_SAMPLE_MIN`] samples) with the median of the OTHER
/// sufficiently-sampled lines' p90 in the same `raw` slice — that slice's
/// "enclosing region" (a caller may scope `raw` to one block/paragraph for
/// a tighter fallback pool, or to a whole page as [`attach_glyph_px`]
/// does). A line with `None` (zero measurable samples) stays `None`. When
/// NO line in `raw` clears the threshold, an under-sampled line keeps its
/// own p90 (nothing safer to fall back to).
fn apply_small_sample_fallback(raw: &[Option<(f32, usize)>]) -> Vec<Option<f32>> {
    let fallback_pool: Vec<f32> = raw
        .iter()
        .copied()
        .filter_map(|r| match r {
            Some((p90, n)) if n >= GLYPH_SAMPLE_MIN => Some(p90),
            _ => None,
        })
        .collect();
    raw.iter()
        .copied()
        .map(|r| match r {
            Some((p90, n)) if n >= GLYPH_SAMPLE_MIN => Some(p90),
            Some((p90, _)) => Some(median(&fallback_pool).unwrap_or(p90)),
            None => None,
        })
        .collect()
}

/// A fitted `size(y) = a + b·y` trend over a page's resolved per-line
/// sizes — the SAME first-order shape `crate::rectify::fit_shear_ramp` uses
/// for shear, applied here to font size instead of baseline slope. `slope`
/// and `mean_y` are enough to remove the smooth component
/// (`size − slope·(y − mean_y)`, algebraically identical to
/// `size − (a+b·y) + mean_size` since `a + b·mean_y == mean_size` for a
/// least-squares fit through the mean point); `significant` is the
/// [`TREND_R2_SIGNIFICANT`] gate — [`apply_trend_normalization`] checks it
/// before applying the correction.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SizeTrend {
    slope: f32,
    mean_y: f32,
    significant: bool,
}

/// Least-squares fit of `size(y) = a + b·y` over `(y_center, size)`
/// samples, plus the [`TREND_R2_SIGNIFICANT`] significance gate (R², the
/// squared Pearson correlation between `y` and `size`). `None` when there
/// are fewer than 2 samples (a line needs 2 points to define a trend) or
/// every sample sits at the same `y` (degenerate — the slope is undefined,
/// and there is no height variation to measure distortion from).
fn fit_size_trend(samples: &[(f32, f32)]) -> Option<SizeTrend> {
    let n = samples.len();
    if n < 2 {
        return None;
    }
    let n_f = n as f32;
    let mean_y = samples.iter().map(|&(y, _)| y).sum::<f32>() / n_f;
    let mean_s = samples.iter().map(|&(_, s)| s).sum::<f32>() / n_f;
    let mut cov = 0.0f32;
    let mut var_y = 0.0f32;
    let mut var_s = 0.0f32;
    for &(y, s) in samples {
        let dy = y - mean_y;
        let ds = s - mean_s;
        cov += dy * ds;
        var_y += dy * dy;
        var_s += ds * ds;
    }
    if var_y <= f32::EPSILON {
        return Some(SizeTrend {
            slope: 0.0,
            mean_y,
            significant: false,
        });
    }
    let slope = cov / var_y;
    let r2 = if var_s > f32::EPSILON {
        (cov * cov) / (var_y * var_s)
    } else {
        0.0
    };
    Some(SizeTrend {
        slope,
        mean_y,
        significant: r2 >= TREND_R2_SIGNIFICANT,
    })
}

/// Stage 4: apply [`fit_size_trend`]'s smooth-component removal to every
/// `(y, size)` sample, but ONLY when the fit is [`SizeTrend::significant`]
/// — otherwise every size passes through UNCHANGED (the safe default: an
/// insignificant/step-like fit must never perturb real typography).
fn apply_trend_normalization(samples: &[(f32, f32)]) -> Vec<f32> {
    let trend = fit_size_trend(samples);
    samples
        .iter()
        .map(|&(y, size)| match trend {
            Some(t) if t.significant => size - t.slope * (y - t.mean_y),
            _ => size,
        })
        .collect()
}

/// Measure and attach a DIRECT glyph-ink font-size estimate
/// ([`DocLineMetrics::glyph_px`]) to every line in `page` that already
/// carries [`DocLineMetrics`] — i.e. came through the row-metrics pipeline
/// (`crate::renderer::LineMetrics`, wave-3 `compute_block_xheight`). Lines
/// with `metrics: None` are left untouched — see this section's module doc
/// ("Why `DocLineMetrics`, not a new `DocLine` field") for why.
///
/// Composes the four pipeline stages documented at the top of this
/// section: [`line_raw_glyph_px`] (1+2) → [`apply_small_sample_fallback`]
/// (3) → [`fit_size_trend`] + [`apply_trend_normalization`] (4).
///
/// ## Contract
///
/// `lines` MUST be the EXACT slice used to build `page` via
/// [`DocPage::from_line_words`] (same order, same empty-line filtering) —
/// only there do individual glyphs'
/// [`tesseract_core::WordResult::char_boxes`] survive; [`DocWord`] retains
/// only the union bbox. `binary` is the SAME `page_w × page_h` binarized
/// page (`crate::xy_cut::binarize_page_with`) already used elsewhere in
/// this crate's layout/table pipeline — this crate's bitonal convention
/// throughout: `0` = ink, `255` = background. On any mismatch between
/// `lines` and `page` (the wrong slice passed) this is a silent no-op — the
/// same "safe no-op on insufficient/mismatched input" convention
/// `crate::rectify::auto_rectify` uses — rather than a panic on what is,
/// from a caller's perspective, a wiring bug elsewhere.
pub fn attach_glyph_px(
    page: &mut DocPage,
    lines: &[LineWords],
    binary: &[u8],
    page_w: u32,
    page_h: u32,
) {
    let pw = page_w as i32;
    let ph = page_h as i32;
    let non_empty: Vec<&LineWords> = lines.iter().filter(|l| !l.words.is_empty()).collect();
    if non_empty.len() != page.lines.len() {
        return;
    }

    // Stages 1+2: every line's raw (p90, sample_count) -- computed for ALL
    // lines (not just the ones that will end up written, i.e. those with
    // Some(metrics)) so the stage-3/4 statistics below see the page's whole
    // real signal, not an artificially shrunk one.
    let raw: Vec<Option<(f32, usize)>> = non_empty
        .iter()
        .map(|line| line_raw_glyph_px(line, binary, pw, ph))
        .collect();
    let resolved = apply_small_sample_fallback(&raw);

    // Stage 4, over the lines that have a resolved size: each line's own
    // top-down vertical center is the `y` the trend fits against.
    let y_centers: Vec<f32> = page
        .lines
        .iter()
        .map(|l| (l.bbox.1 + l.bbox.3) as f32 / 2.0)
        .collect();
    let mut indices: Vec<usize> = Vec::new();
    let mut samples: Vec<(f32, f32)> = Vec::new();
    for (i, size) in resolved.iter().enumerate() {
        if let Some(size) = size {
            indices.push(i);
            samples.push((y_centers[i], *size));
        }
    }
    let normalized = apply_trend_normalization(&samples);

    for (idx, size) in indices.into_iter().zip(normalized) {
        if let Some(m) = &mut page.lines[idx].metrics {
            m.glyph_px = Some(size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dw(text: &str, bbox: (i32, i32, i32, i32), conf: f32) -> DocWord {
        DocWord {
            text: text.to_string(),
            bbox,
            conf,
            leading_space: false,
            numeric_norm: None,
        }
    }

    fn dl(bbox: (i32, i32, i32, i32), words: Vec<DocWord>) -> DocLine {
        DocLine {
            bbox,
            words,
            metrics: None,
        }
    }

    /// Line metrics flow: `from_line_words` converts the bottom-up
    /// [`crate::renderer::LineMetrics`] into the DOM's top-down frame
    /// (`baseline_td = page_h - baseline_up`), and `render_json` emits the
    /// additive per-line keys; a metrics-less line emits none (legacy shape
    /// byte-stable).
    #[test]
    fn render_json_emits_line_metrics_when_present() {
        use crate::renderer::{LineMetrics, LineWords};
        use tesseract_core::dawg::PermuterType;
        use tesseract_core::WordResult;
        let word = WordResult {
            unichar_ids: vec![1],
            certs: vec![-0.1],
            ratings: vec![0.1],
            char_boxes: vec![(10, 60, 30, 80)],
            permuter: PermuterType::TopChoicePerm,
            space_certainty: 0.0,
            leading_space: false,
        };
        let with_metrics = LineWords {
            words: vec![word.clone()],
            line_box: (10, 60, 90, 80),
            metrics: Some(LineMetrics {
                xheight: 10.0,
                ascrise: 4.0,
                descdrop: -3.0,
                baseline: 62.5, // bottom-up; page_h=100 → top-down 37.5
            }),
        };
        let without = LineWords {
            words: vec![word],
            line_box: (10, 20, 90, 40),
            metrics: None,
        };
        let charset =
            tesseract_core::CharSet::load_from_str("2\nNULL 0 Common 0\na 3 0 a Left a a\n")
                .expect("charset");
        let page = DocPage::from_line_words(&[with_metrics, without], &charset, 200, 100);

        assert_eq!(
            page.lines[0].metrics,
            Some(DocLineMetrics {
                xheight: 10.0,
                ascrise: 4.0,
                descdrop: -3.0,
                baseline: 37.5,
                glyph_px: None,
            }),
            "bottom-up baseline must convert to top-down"
        );
        assert_eq!(page.lines[1].metrics, None);

        let json = render_json(&page, &[]);
        assert!(
            json.contains("\"xheight\":10.0,\"ascrise\":4.0,\"descdrop\":-3.0,\"baseline\":37.5"),
            "metric keys emitted: {json}"
        );
        // The metrics-less line's object carries none of the keys.
        let second_line = json
            .split("{\"bbox\":[10,60,")
            .nth(1)
            .expect("second line obj");
        assert!(
            !second_line.starts_with("90,80],\"xheight\""),
            "metrics-less line must not emit metric keys"
        );
    }

    /// `plain_text` — Power Automate ergonomics: the whole page as ONE
    /// string. Must join EVERY line (not just the first), respect
    /// `leading_space` the same way `render_text` does, and separate lines
    /// with `\n`. Two-sided: proves the join actually happens (not just "a
    /// string is present") by checking BOTH lines' content and the
    /// separator between them.
    #[test]
    fn render_json_emits_plain_text_joining_every_line() {
        let page = DocPage {
            width: 300,
            height: 100,
            lines: vec![
                dl(
                    (10, 10, 200, 30),
                    vec![
                        dw("The", (10, 10, 40, 30), 99.0),
                        DocWord {
                            leading_space: true,
                            ..dw("dog", (45, 10, 80, 30), 99.0)
                        },
                    ],
                ),
                dl((10, 40, 200, 60), vec![dw("runs.", (10, 40, 60, 60), 99.0)]),
            ],
        };
        let json = render_json(&page, &[]);
        assert!(
            json.contains("\"plain_text\":\"The dog\\nruns.\\n\","),
            "plain_text must join both lines with a leading-space-respecting \
             per-line text, matching render_text's rule of a \\n appended \
             after EVERY non-empty line including the last: {json}"
        );
    }

    /// `plain_text` on an all-empty page is the empty string, not absent —
    /// consumers should never need a null-check before reading it.
    #[test]
    fn render_json_plain_text_is_empty_string_on_an_empty_page() {
        let page = DocPage {
            width: 100,
            height: 100,
            lines: vec![],
        };
        let json = render_json(&page, &[]);
        assert!(
            json.contains("\"plain_text\":\"\","),
            "an empty page's plain_text must be \"\", not null or absent: {json}"
        );
    }

    /// `fields_map` — Power Automate ergonomics: `fields`, reshaped
    /// `key -> value`. Real falsifier: builds two DISTINCT harvested fields
    /// and asserts BOTH keys resolve to their OWN (not swapped/duplicated)
    /// values — a implementation that emitted the same value for every key,
    /// or dropped one, would fail this.
    #[test]
    fn render_json_emits_fields_map_mirroring_fields() {
        let page = DocPage {
            width: 300,
            height: 100,
            lines: vec![],
        };
        let fields = vec![
            HarvestedField {
                key: "netto".to_string(),
                label_text: "Netto:".to_string(),
                value: "1.250,00".to_string(),
                value_cents: Some(125_000),
                bbox: (10, 10, 100, 30),
                conf: 98.0,
                checks: vec!["arithmetic_ok".to_string()],
            },
            HarvestedField {
                key: "iban".to_string(),
                label_text: "IBAN:".to_string(),
                value: "DE89370400440532013000".to_string(),
                value_cents: None,
                bbox: (10, 40, 250, 60),
                conf: 97.0,
                checks: vec!["iban_mod97_ok".to_string()],
            },
        ];
        let json = render_json(&page, &fields);
        let v: serde_json::Value =
            serde_json::from_str(&json).expect("render_json must emit valid JSON");
        let map = &v["pages"][0]["fields_map"];
        assert_eq!(
            map["netto"], "1.250,00",
            "fields_map[\"netto\"] must match the netto field's own value, not iban's: {json}"
        );
        assert_eq!(
            map["iban"], "DE89370400440532013000",
            "fields_map[\"iban\"] must match the iban field's own value, not netto's: {json}"
        );
        assert_eq!(
            map.as_object().unwrap().len(),
            2,
            "fields_map must have exactly the 2 harvested keys, no more, no fewer: {json}"
        );
    }

    /// `fields_map` on a page with no harvested fields is `{}`, not absent —
    /// the SAME "never null-check" guarantee as `plain_text`'s empty case,
    /// and what a `fields_map['x']` Power Automate expression relies on not
    /// erroring on "property of null."
    #[test]
    fn render_json_fields_map_is_empty_object_with_no_fields() {
        let page = DocPage {
            width: 100,
            height: 100,
            lines: vec![],
        };
        let json = render_json(&page, &[]);
        assert!(
            json.contains("\"fields_map\":{}"),
            "no harvested fields must emit fields_map as {{}}, not null or absent: {json}"
        );
    }

    // --- numeric hardening -------------------------------------------------

    #[test]
    fn hardening_fixes_single_misreads_in_amounts() {
        assert_eq!(harden_numeric_token("2S0,00").as_deref(), Some("250,00"));
        assert_eq!(harden_numeric_token("1.O50").as_deref(), Some("1.050"));
        assert_eq!(
            harden_numeric_token("l.250,00").as_deref(),
            Some("1.250,00")
        );
        assert_eq!(harden_numeric_token("12Z").as_deref(), Some("122"));
    }

    #[test]
    fn hardening_leaves_words_ids_and_balanced_tokens_alone() {
        // Genuine words: a non-confusable letter short-circuits.
        assert_eq!(harden_numeric_token("Summe"), None);
        assert_eq!(harden_numeric_token("Rechnung"), None);
        // Digits must strictly dominate letters: "B8" is 1:1 -- could be a
        // legitimate code, stays.
        assert_eq!(harden_numeric_token("B8"), None);
        // Pure digits: nothing to change.
        assert_eq!(harden_numeric_token("1250"), None);
        // Structural chars (slash) disqualify: dates/fractions untouched.
        assert_eq!(harden_numeric_token("1/2"), None);
        // GUID shape survives even though hex letters are digit-dominated.
        assert_eq!(
            harden_numeric_token("a1b2c3d4-0000-4111-8222-333344445555"),
            None
        );
        // IBAN shape is guarded (validated by checksum instead).
        assert_eq!(harden_numeric_token("DE89370400440532013000"), None);
    }

    #[test]
    fn harden_numeric_tokens_fills_only_changed_words() {
        let mut page = DocPage {
            width: 100,
            height: 100,
            lines: vec![dl(
                (0, 0, 100, 10),
                vec![
                    dw("Netto:", (0, 0, 30, 10), 95.0),
                    dw("2S0,00", (40, 0, 80, 10), 90.0),
                ],
            )],
        };
        harden_numeric_tokens(&mut page);
        assert_eq!(page.lines[0].words[0].numeric_norm, None);
        assert_eq!(
            page.lines[0].words[1].numeric_norm.as_deref(),
            Some("250,00")
        );
    }

    // --- IBAN --------------------------------------------------------------

    #[test]
    fn iban_mod97_accepts_the_canonical_example_and_rejects_corruption() {
        // The ISO 13616 documentation example IBAN.
        assert!(iban_mod97_ok("DE89370400440532013000"));
        // Case-insensitive.
        assert!(iban_mod97_ok("de89370400440532013000"));
        // One corrupted digit -> checksum fails.
        assert!(!iban_mod97_ok("DE89370400440532013001"));
        // Shape violations are rejected before any math.
        assert!(!iban_mod97_ok("89DE370400440532013000"));
        assert!(!iban_mod97_ok("DE8937"));
        assert!(!iban_mod97_ok(""));
    }

    // --- amount parsing ----------------------------------------------------

    #[test]
    fn parse_amount_cents_handles_german_and_english_conventions() {
        assert_eq!(parse_amount_cents("1.250,00"), Some(125_000));
        assert_eq!(parse_amount_cents("1,250.00"), Some(125_000));
        assert_eq!(parse_amount_cents("1250,00"), Some(125_000));
        assert_eq!(parse_amount_cents("1250.00"), Some(125_000));
        // A trailing 3-digit group is thousands, not decimals.
        assert_eq!(parse_amount_cents("1.250"), Some(125_000));
        assert_eq!(parse_amount_cents("12,345"), Some(1_234_500));
        assert_eq!(parse_amount_cents("0,50"), Some(50));
        assert_eq!(parse_amount_cents("99"), Some(9_900));
        assert_eq!(parse_amount_cents("€ 99,50"), Some(9_950));
        assert_eq!(parse_amount_cents("-12,00"), Some(-1_200));
        assert_eq!(parse_amount_cents("offen"), None);
        assert_eq!(parse_amount_cents(""), None);
        assert_eq!(parse_amount_cents("-"), None);
        assert_eq!(parse_amount_cents(","), None);
    }

    // --- harvest -----------------------------------------------------------

    /// A synthetic German invoice page: amounts (one with an OCR misread),
    /// an invoice number, and an IBAN printed in groups.
    fn invoice_page() -> DocPage {
        let mut page = DocPage {
            width: 600,
            height: 200,
            lines: vec![
                dl(
                    (0, 0, 400, 20),
                    vec![
                        dw("Rechnungsnr.:", (0, 0, 120, 20), 96.0),
                        dw("2024-0815", (130, 0, 220, 20), 97.0),
                    ],
                ),
                dl(
                    (0, 30, 400, 50),
                    vec![
                        dw("Netto:", (0, 30, 60, 50), 95.0),
                        // OCR misread: S for 5 -- hardening fixes it pre-parse.
                        dw("1.2S0,00", (100, 30, 200, 50), 88.0),
                    ],
                ),
                dl(
                    (0, 60, 400, 80),
                    vec![
                        dw("MwSt:", (0, 60, 60, 80), 95.0),
                        dw("237,50", (100, 60, 200, 80), 94.0),
                    ],
                ),
                dl(
                    (0, 90, 400, 110),
                    vec![
                        dw("Brutto:", (0, 90, 60, 110), 95.0),
                        dw("1.487,50", (100, 90, 200, 110), 93.0),
                    ],
                ),
                dl(
                    (0, 120, 600, 140),
                    vec![
                        dw("IBAN:", (0, 120, 50, 140), 96.0),
                        dw("DE89", (60, 120, 100, 140), 92.0),
                        dw("3704", (110, 120, 150, 140), 91.0),
                        dw("0044", (160, 120, 200, 140), 93.0),
                        dw("0532", (210, 120, 250, 140), 92.0),
                        dw("0130", (260, 120, 300, 140), 90.0),
                        dw("00", (310, 120, 330, 140), 94.0),
                    ],
                ),
            ],
        };
        harden_numeric_tokens(&mut page);
        page
    }

    #[test]
    fn harvest_extracts_typed_fields_and_cross_checks_arithmetic() {
        let page = invoice_page();
        let fields = harvest_fields(&page, &german_invoice_fields());

        let get = |key: &str| fields.iter().find(|f| f.key == key).unwrap();

        // The hardened amount parsed: 1.2S0,00 -> 1.250,00 -> 125000 cents.
        let netto = get("netto");
        assert_eq!(netto.value, "1.250,00");
        assert_eq!(netto.value_cents, Some(125_000));

        assert_eq!(get("ust").value_cents, Some(23_750));
        assert_eq!(get("brutto").value_cents, Some(148_750));

        // 125000 + 23750 == 148750 -> all three carry arithmetic_ok.
        for key in ["netto", "ust", "brutto"] {
            assert!(
                get(key).checks.iter().any(|c| c == "arithmetic_ok"),
                "{key} missing arithmetic_ok: {:?}",
                get(key).checks
            );
        }

        // Id field: taken verbatim.
        assert_eq!(get("rechnungsnummer").value, "2024-0815");

        // IBAN groups joined and checksum-verified; conf = min over groups.
        let iban = get("iban");
        assert_eq!(iban.value, "DE89370400440532013000");
        assert!(iban.checks.iter().any(|c| c == "iban_mod97_ok"));
        assert_eq!(iban.conf, 90.0);
        assert_eq!(iban.bbox, (60, 120, 330, 140));
    }

    #[test]
    fn harvest_without_arithmetic_consistency_adds_no_check() {
        let mut page = DocPage {
            width: 400,
            height: 100,
            lines: vec![
                dl(
                    (0, 0, 400, 20),
                    vec![
                        dw("Netto:", (0, 0, 60, 20), 95.0),
                        dw("100,00", (100, 0, 200, 20), 95.0),
                    ],
                ),
                dl(
                    (0, 30, 400, 50),
                    vec![
                        dw("MwSt:", (0, 30, 60, 50), 95.0),
                        dw("19,00", (100, 30, 200, 50), 95.0),
                    ],
                ),
                dl(
                    (0, 60, 400, 80),
                    vec![
                        dw("Brutto:", (0, 60, 60, 80), 95.0),
                        // WRONG total: 100 + 19 != 120.
                        dw("120,00", (100, 60, 200, 80), 95.0),
                    ],
                ),
            ],
        };
        harden_numeric_tokens(&mut page);
        let fields = harvest_fields(&page, &german_invoice_fields());
        assert!(fields
            .iter()
            .all(|f| !f.checks.iter().any(|c| c == "arithmetic_ok")));
    }

    #[test]
    fn harvest_skips_labels_with_no_parseable_value() {
        let page = DocPage {
            width: 400,
            height: 40,
            lines: vec![dl(
                (0, 0, 400, 20),
                vec![
                    dw("Netto:", (0, 0, 60, 20), 95.0),
                    dw("offen", (100, 0, 200, 20), 95.0),
                ],
            )],
        };
        let fields = harvest_fields(&page, &german_invoice_fields());
        assert!(fields.is_empty());
    }

    // --- JSON --------------------------------------------------------------

    #[test]
    fn json_escape_covers_rfc8259_musts_and_passes_unicode() {
        assert_eq!(json_escape("a\"b"), "a\\\"b");
        assert_eq!(json_escape("a\\b"), "a\\\\b");
        assert_eq!(json_escape("a\nb\tc\rd"), "a\\nb\\tc\\rd");
        assert_eq!(json_escape("\u{0008}\u{000C}"), "\\b\\f");
        assert_eq!(json_escape("\u{0001}"), "\\u0001");
        assert_eq!(json_escape("äöü€ß"), "äöü€ß");
        assert_eq!(json_escape(""), "");
    }

    #[test]
    fn render_json_golden_one_line_one_field() {
        let page = DocPage {
            width: 10,
            height: 10,
            lines: vec![dl((0, 0, 10, 10), vec![dw("a", (0, 0, 4, 10), 100.0)])],
        };
        let field = HarvestedField {
            key: "netto".to_string(),
            label_text: "Netto:".to_string(),
            value: "1,00".to_string(),
            value_cents: Some(100),
            bbox: (1, 2, 3, 4),
            conf: 99.5,
            checks: vec!["arithmetic_ok".to_string()],
        };
        let json = render_json(&page, &[field]);
        let expected = concat!(
            "{\"schema\":\"tesseract-rs/doc.v1\",\"pages\":[{",
            "\"page\":1,\"width\":10,\"height\":10,",
            "\"quality\":{\"mean_conf\":100.00,\"low_confidence\":false},",
            "\"plain_text\":\"a\\n\",",
            "\"regions\":[{\"type\":\"paragraph\",\"bbox\":[0,0,10,10],",
            "\"lines\":[{\"bbox\":[0,0,10,10],\"words\":[",
            "{\"text\":\"a\",\"bbox\":[0,0,4,10],\"conf\":100.00,\"leading_space\":false}",
            "]}]}],",
            "\"fields\":[{\"key\":\"netto\",\"label\":\"Netto:\",\"value\":\"1,00\"",
            ",\"value_cents\":100,\"bbox\":[1,2,3,4],\"conf\":99.50,",
            "\"checks\":[\"arithmetic_ok\"]}],",
            "\"fields_map\":{\"netto\":\"1,00\"}}]}",
        );
        assert_eq!(json, expected);
    }

    #[test]
    fn render_json_empty_page_keeps_stable_shape() {
        let page = DocPage {
            width: 5,
            height: 5,
            lines: vec![],
        };
        let json = render_json(&page, &[]);
        assert_eq!(
            json,
            "{\"schema\":\"tesseract-rs/doc.v1\",\"pages\":[{\"page\":1,\
             \"width\":5,\"height\":5,\"quality\":{\"mean_conf\":null,\
             \"low_confidence\":false},\"plain_text\":\"\",\"regions\":[],\
             \"fields\":[],\"fields_map\":{}}]}"
        );
    }

    #[test]
    fn low_confidence_flags_garbled_output_but_not_clean_text() {
        // Clean text: high conf → not flagged.
        let clean = DocPage {
            width: 100,
            height: 20,
            lines: vec![dl(
                (0, 0, 100, 20),
                vec![dw("Rechnung", (0, 0, 80, 20), 96.0)],
            )],
        };
        assert_eq!(mean_word_confidence(&clean), Some(96.0));
        assert!(render_json(&clean, &[]).contains("\"low_confidence\":false"));

        // Garble (e.g. handwriting): low conf across words → flagged.
        let garble = DocPage {
            width: 100,
            height: 20,
            lines: vec![dl(
                (0, 0, 100, 20),
                vec![
                    dw("xq", (0, 0, 20, 20), 41.0),
                    dw("z,", (30, 0, 50, 20), 38.0),
                ],
            )],
        };
        let mc = mean_word_confidence(&garble).unwrap();
        assert!(
            mc < LOW_CONFIDENCE_THRESHOLD,
            "garble mean {mc} must be below floor"
        );
        assert!(render_json(&garble, &[]).contains("\"low_confidence\":true"));
    }

    #[test]
    fn render_json_emits_numeric_norm_only_when_present() {
        let mut page = DocPage {
            width: 100,
            height: 20,
            lines: vec![dl(
                (0, 0, 100, 20),
                vec![dw("2S0,00", (0, 0, 60, 20), 88.0)],
            )],
        };
        harden_numeric_tokens(&mut page);
        let json = render_json(&page, &[]);
        assert!(json.contains("\"text\":\"2S0,00\""));
        assert!(json.contains("\"numeric_norm\":\"250,00\""));
    }

    // --- regions -----------------------------------------------------------

    /// Synthetic classified page: header line, two body lines in two blocks,
    /// an orphan body line outside every block, a footer line, one figure.
    #[test]
    fn build_regions_assigns_kinds_blocks_and_orphans_in_order() {
        let page = DocPage {
            width: 400,
            height: 300,
            lines: vec![
                dl((10, 5, 200, 15), vec![dw("Kopf", (10, 5, 60, 15), 95.0)]), // 0 header
                dl((10, 50, 180, 70), vec![dw("links", (10, 50, 80, 70), 95.0)]), // 1 block A
                dl(
                    (210, 50, 380, 70),
                    vec![dw("rechts", (210, 50, 300, 70), 95.0)],
                ), // 2 block B
                dl(
                    (10, 150, 180, 170),
                    vec![dw("verwaist", (10, 150, 100, 170), 95.0)],
                ), // 3 orphan
                dl(
                    (10, 280, 200, 295),
                    vec![dw("Seite", (10, 280, 60, 295), 95.0)],
                ), // 4 footer
            ],
        };
        let blocks = [(0, 40, 200, 100), (200, 40, 400, 100)];
        let figures = [(250, 150, 380, 250)];
        let regions = build_regions(&page, &[0], &[4], &blocks, &[false, false], &figures);

        let kinds: Vec<&str> = regions.iter().map(|r| r.kind.as_str()).collect();
        assert_eq!(
            kinds,
            ["header", "text", "text", "text", "figure", "footer"],
            "order: header, blocks, orphan catch-all, figures, footer"
        );
        assert_eq!(regions[0].line_indices, [0]);
        assert_eq!(regions[1].line_indices, [1]);
        assert_eq!(regions[2].line_indices, [2]);
        assert_eq!(regions[3].line_indices, [3], "orphan catch-all");
        assert!(regions[4].line_indices.is_empty(), "figures own no lines");
        assert_eq!(regions[4].bbox, (250, 150, 380, 250));
        assert_eq!(regions[5].line_indices, [4]);
        // Line-bearing region bbox = union of member line bboxes.
        assert_eq!(regions[1].bbox, (10, 50, 180, 70));
    }

    #[test]
    fn build_regions_drops_empty_blocks_and_skips_missing_sections() {
        let page = DocPage {
            width: 100,
            height: 100,
            lines: vec![dl(
                (10, 10, 90, 30),
                vec![dw("nur", (10, 10, 40, 30), 95.0)],
            )],
        };
        // Two blocks, only the first is populated; no furniture, no figures.
        let blocks = [(0, 0, 100, 50), (0, 50, 100, 100)];
        let regions = build_regions(&page, &[], &[], &blocks, &[], &[]);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].kind, RegionKind::Text);
        assert_eq!(regions[0].line_indices, [0]);
    }

    #[test]
    fn build_regions_marks_table_blocks() {
        let page = DocPage {
            width: 200,
            height: 200,
            lines: vec![
                dl((10, 10, 90, 30), vec![dw("cell", (10, 10, 40, 30), 95.0)]),
                dl(
                    (10, 110, 90, 130),
                    vec![dw("para", (10, 110, 40, 130), 95.0)],
                ),
            ],
        };
        let blocks = [(0, 0, 100, 50), (0, 100, 100, 150)];
        // First block flagged a table, second a plain text block.
        let regions = build_regions(&page, &[], &[], &blocks, &[true, false], &[]);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].kind, RegionKind::Table, "flagged block → table");
        assert_eq!(regions[0].line_indices, [0]);
        assert_eq!(regions[1].kind, RegionKind::Text, "unflagged block → text");
    }

    #[test]
    fn render_json_with_regions_emits_typed_regions() {
        let page = DocPage {
            width: 50,
            height: 50,
            lines: vec![dl((0, 0, 50, 10), vec![dw("a", (0, 0, 10, 10), 100.0)])],
        };
        let regions = vec![
            DocRegion {
                kind: RegionKind::Text,
                bbox: (0, 0, 50, 10),
                line_indices: vec![0],
                table: None,
            },
            DocRegion {
                kind: RegionKind::Figure,
                bbox: (5, 20, 45, 45),
                line_indices: vec![],
                table: None,
            },
        ];
        let json = render_json_with_regions(&page, &regions, &[]);
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("{\"type\":\"figure\",\"bbox\":[5,20,45,45],\"lines\":[]}"));
        assert!(!json.contains("\"type\":\"paragraph\""));
        // The plain renderer still emits the byte-stable default.
        let plain = render_json(&page, &[]);
        assert!(plain.contains("\"type\":\"paragraph\""));
    }

    #[test]
    fn extract_table_grid_splits_columns_by_whitespace() {
        // A 3-row × 4-column invoice-like table: Pos | Artikel | Menge | Preis.
        let rows = [
            dl(
                (10, 10, 470, 30),
                vec![
                    dw("Pos", (10, 10, 40, 30), 99.0),
                    dw("Artikel", (100, 10, 180, 30), 99.0),
                    dw("Menge", (250, 10, 320, 30), 99.0),
                    dw("Preis", (400, 10, 470, 30), 99.0),
                ],
            ),
            dl(
                (10, 40, 450, 60),
                vec![
                    dw("1", (10, 40, 25, 60), 99.0),
                    dw("Kabel", (100, 40, 160, 60), 99.0),
                    dw("2", (250, 40, 265, 60), 99.0),
                    dw("5,00", (400, 40, 450, 60), 99.0),
                ],
            ),
            dl(
                (10, 70, 445, 90),
                vec![
                    dw("2", (10, 70, 25, 90), 99.0),
                    dw("Stecker", (100, 70, 180, 90), 99.0),
                    dw("10", (250, 70, 280, 90), 99.0),
                    dw("3,50", (400, 70, 445, 90), 99.0),
                ],
            ),
        ];
        let line_refs: Vec<&DocLine> = rows.iter().collect();
        let grid = extract_table_grid(&line_refs);

        assert_eq!(grid.rows, 3, "rows are the recognized lines");
        assert_eq!(grid.cols, 4, "four whitespace-separated columns");
        assert_eq!(grid.cells.len(), 12, "3×4 fully populated");
        // Header flag is on row 0 only.
        assert_eq!(grid.cells.iter().filter(|c| c.header).count(), 4);
        assert!(grid.cells.iter().filter(|c| c.header).all(|c| c.row == 0));
        // Words land in the right cells.
        let price = grid
            .cells
            .iter()
            .find(|c| c.row == 1 && c.col == 3)
            .unwrap();
        assert_eq!(price.text, "5,00");
        let art = grid
            .cells
            .iter()
            .find(|c| c.row == 2 && c.col == 1)
            .unwrap();
        assert_eq!(art.text, "Stecker");
    }

    #[test]
    fn render_json_emits_table_cells() {
        let page = DocPage {
            width: 500,
            height: 100,
            lines: vec![
                dl(
                    (10, 10, 470, 30),
                    vec![
                        dw("Pos", (10, 10, 40, 30), 99.0),
                        dw("Preis", (400, 10, 470, 30), 99.0),
                    ],
                ),
                dl(
                    (10, 40, 450, 60),
                    vec![
                        dw("1", (10, 40, 25, 60), 99.0),
                        dw("5,00", (400, 40, 450, 60), 99.0),
                    ],
                ),
            ],
        };
        let line_refs: Vec<&DocLine> = page.lines.iter().collect();
        let grid = extract_table_grid(&line_refs);
        let regions = vec![DocRegion {
            kind: RegionKind::Table,
            bbox: (10, 10, 470, 60),
            line_indices: vec![0, 1],
            table: Some(grid),
        }];
        let json = render_json_with_regions(&page, &regions, &[]);
        assert!(json.contains("\"type\":\"table\""), "table region type");
        assert!(json.contains("\"cols\":2"), "two columns");
        assert!(json.contains("\"rows\":2"));
        assert!(json.contains("\"text\":\"5,00\""), "cell text emitted");
        assert!(json.contains("\"header\":true"), "row 0 flagged header");
    }

    #[test]
    fn extract_table_grid_keeps_multiword_cell_intact() {
        // codex #41 P2: a description cell "Kabel HDMI" with an internal word
        // gap must NOT split into two columns just because no other row happens
        // to cover that x-band. Two columns (description | price), not three.
        let rows = [
            dl(
                (10, 10, 360, 30),
                vec![
                    dw("Pos", (10, 10, 40, 30), 99.0),
                    dw("Preis", (300, 10, 360, 30), 99.0),
                ],
            ),
            dl(
                (10, 40, 350, 60),
                vec![
                    dw("Kabel", (10, 40, 60, 60), 99.0),
                    dw("HDMI", (90, 40, 140, 60), 99.0), // internal gap [60,90]=30 < 2×height(40)
                    dw("5,00", (300, 40, 350, 60), 99.0),
                ],
            ),
        ];
        let line_refs: Vec<&DocLine> = rows.iter().collect();
        let grid = extract_table_grid(&line_refs);
        assert_eq!(
            grid.cols, 2,
            "the Kabel/HDMI internal gap must NOT become a column"
        );
        let desc = grid
            .cells
            .iter()
            .find(|c| c.row == 1 && c.col == 0)
            .unwrap();
        assert_eq!(desc.text, "Kabel HDMI", "the description cell stays whole");
    }

    #[test]
    fn table_cell_conf_is_min_over_words_not_mean() {
        // Same geometry as extract_table_grid_keeps_multiword_cell_intact
        // (proven to keep "Kabel HDMI" as one cell) -- only the confidences
        // differ. 95.0 and 40.0 average to 67.5; the cell must report 40.0.
        let rows = [
            dl(
                (10, 10, 360, 30),
                vec![
                    dw("Pos", (10, 10, 40, 30), 99.0),
                    dw("Preis", (300, 10, 360, 30), 99.0),
                ],
            ),
            dl(
                (10, 40, 350, 60),
                vec![
                    dw("Kabel", (10, 40, 60, 60), 95.0),
                    dw("HDMI", (90, 40, 140, 60), 40.0), // the low-confidence word
                    dw("5,00", (300, 40, 350, 60), 99.0),
                ],
            ),
        ];
        let line_refs: Vec<&DocLine> = rows.iter().collect();
        let grid = extract_table_grid(&line_refs);

        let desc = grid
            .cells
            .iter()
            .find(|c| c.row == 1 && c.col == 0)
            .unwrap();
        assert_eq!(desc.text, "Kabel HDMI");
        assert_eq!(
            desc.conf, 40.0,
            "cell conf must be the MIN over its words (40.0), never the mean (67.5)"
        );

        // A single-word cell simply reports that word's own confidence.
        let price = grid
            .cells
            .iter()
            .find(|c| c.row == 1 && c.col == 1)
            .unwrap();
        assert_eq!(price.conf, 99.0);
    }

    #[test]
    fn render_json_emits_table_cell_conf() {
        let page = DocPage {
            width: 500,
            height: 100,
            lines: vec![
                dl(
                    (10, 10, 470, 30),
                    vec![
                        dw("Pos", (10, 10, 40, 30), 99.0),
                        dw("Preis", (400, 10, 470, 30), 99.0),
                    ],
                ),
                dl(
                    (10, 40, 450, 60),
                    vec![
                        dw("1", (10, 40, 25, 60), 88.0),
                        dw("5,00", (400, 40, 450, 60), 72.5),
                    ],
                ),
            ],
        };
        let line_refs: Vec<&DocLine> = page.lines.iter().collect();
        let grid = extract_table_grid(&line_refs);
        let regions = vec![DocRegion {
            kind: RegionKind::Table,
            bbox: (10, 10, 470, 60),
            line_indices: vec![0, 1],
            table: Some(grid),
        }];
        let json = render_json_with_regions(&page, &regions, &[]);
        // Header row cells (conf 99.0 each) and the two data cells (88.0,
        // 72.5) all round-trip at 2 decimals, matching word_confidence's own
        // formatting convention elsewhere in this module.
        assert!(json.contains("\"conf\":99.00"), "header cell conf: {json}");
        assert!(json.contains("\"conf\":88.00"), "row cell conf: {json}");
        assert!(json.contains("\"conf\":72.50"), "row cell conf: {json}");
    }

    // --- from_line_words ---------------------------------------------------

    #[test]
    fn from_line_words_converts_boxes_and_skips_empty_lines() {
        use crate::renderer::LineWords;
        use tesseract_core::dawg::PermuterType;
        use tesseract_core::WordResult;

        let charset = tesseract_core::CharSet::load_from_str(
            "3\nNULL 0 Common 0\na 3 0 a Left a a\nb 3 0 b Left b b\n",
        )
        .expect("valid unicharset");

        let word = |ids: &[i32], cert: f32, box_: (i32, i32, i32, i32)| WordResult {
            unichar_ids: ids.to_vec(),
            certs: ids.iter().map(|_| cert).collect(),
            ratings: ids.iter().map(|_| 0.0).collect(),
            char_boxes: ids.iter().map(|_| box_).collect(),
            permuter: PermuterType::TopChoicePerm,
            space_certainty: 0.0,
            leading_space: false,
        };

        let lines = vec![
            LineWords {
                words: vec![],
                line_box: (0, 0, 10, 10),
                metrics: None,
            },
            LineWords {
                // Bottom-up TBOX (0,0,4,10) on a 10-high page -> top-down (0,0,4,10).
                words: vec![word(&[1], -0.2, (0, 0, 4, 10))],
                line_box: (0, 0, 10, 10),
                metrics: None,
            },
        ];
        let page = DocPage::from_line_words(&lines, &charset, 10, 10);
        assert_eq!(page.lines.len(), 1, "empty line skipped");
        let w = &page.lines[0].words[0];
        assert_eq!(w.text, "a");
        assert_eq!(w.bbox, (0, 0, 4, 10));
        assert_eq!(w.conf, 99.0); // 100 + 5*(-0.2)
        assert_eq!(w.numeric_norm, None);
    }

    // --- Measured glyph-ink font sizing (glyph_px) --------------------------

    /// [`glyph_ink_heights`] measures the REAL ink extent, not the char
    /// box's own height: the box spans the full page height (10 rows,
    /// top-down `[0,10)`), but ink is only drawn in rows `[4,7)` — a
    /// falsifier against a naive "just return the box height" bug, which
    /// would report `10.0` here instead of the correct `3.0`. A second,
    /// all-background char box in the same call contributes NOTHING (an
    /// empty result entry, not a spurious `0.0` sample).
    #[test]
    fn glyph_ink_heights_measures_real_ink_and_skips_empty_boxes() {
        let (pw, ph) = (10i32, 10i32);
        let mut binary = vec![255u8; (pw * ph) as usize];
        // Top-down rows 4,5,6 (3 rows), columns 3,4,5 -- ink.
        for y in 4..7usize {
            for x in 3..6usize {
                binary[y * pw as usize + x] = 0;
            }
        }
        // char box 1 (TBOX left,bottom,right,top): spans the FULL page
        // height (bottom=0 -> top-down bottom=10; top=10 -> top-down
        // top=0), columns [3,6) -- to_image_box -> top-down (3,0,6,10).
        // char box 2: an entirely background column, [7,9).
        let heights = glyph_ink_heights(&[(3, 0, 6, 10), (7, 0, 9, 10)], &binary, pw, ph);
        assert_eq!(
            heights,
            vec![3.0],
            "must measure the 3-row ink extent, not the 10-row box height; \
             the background box must contribute nothing at all"
        );
    }

    /// A known, hand-computable vector: `[1..=10]`, scrambled order (proving
    /// the function sorts rather than assuming sorted input). Linear
    /// interpolation: `rank = 0.9*(10-1) = 8.1`, between index 8 (value 9)
    /// and index 9 (value 10), `frac=0.1` -> `9 + 0.1*(10-9) = 9.1`.
    #[test]
    fn percentile_90_matches_a_known_vector() {
        let mut heights = vec![7.0, 2.0, 9.0, 4.0, 10.0, 1.0, 8.0, 5.0, 3.0, 6.0];
        let got = percentile_90(&mut heights).expect("non-empty");
        assert!((got - 9.1).abs() < 1e-4, "p90 = {got}, expected ~9.1");
        assert_eq!(percentile_90(&mut []), None);
    }

    /// The measured failure the guard exists for: a 7-sample line (below
    /// [`GLYPH_SAMPLE_MIN`]) that measured an aberrant `24.0` must NOT keep
    /// that value -- it must fall back to the median of the OTHER,
    /// sufficiently-sampled lines (`{12.0, 13.0}` -> median `12.5`), while
    /// those well-sampled lines keep their own p90 untouched. A line
    /// exactly AT the threshold is trusted as-is (boundary check); a `None`
    /// (zero measurable samples) stays `None`.
    #[test]
    fn small_sample_fallback_triggers_below_threshold() {
        let raw = vec![
            Some((12.0, 12)),
            Some((24.0, 7)), // 7 < GLYPH_SAMPLE_MIN -- must not be trusted
            Some((13.0, 15)),
        ];
        let resolved = apply_small_sample_fallback(&raw);
        assert_eq!(
            resolved[0],
            Some(12.0),
            "well-sampled line keeps its own p90"
        );
        assert_eq!(
            resolved[2],
            Some(13.0),
            "well-sampled line keeps its own p90"
        );
        assert_eq!(
            resolved[1],
            Some(12.5),
            "under-sampled line must fall back to the median of the OTHER \
             lines (12.5), not its own aberrant 24.0"
        );

        // Boundary: exactly GLYPH_SAMPLE_MIN samples is trusted as-is.
        let at_threshold = vec![Some((99.0, GLYPH_SAMPLE_MIN))];
        assert_eq!(apply_small_sample_fallback(&at_threshold), vec![Some(99.0)]);

        // A line with zero measurable samples stays None.
        let with_none = vec![None, Some((10.0, 20))];
        assert_eq!(apply_small_sample_fallback(&with_none)[0], None);
    }

    /// Fixture A of the important pair: 10 lines whose size is PERFECTLY
    /// linear in `y` (`12.0 + 0.005*y`) -- the bent-paper camera-geometry
    /// signature. The fit must be judged significant (R² ≈ 1.0) and its
    /// smooth component fully divided out, so every line converges to the
    /// page's mean size regardless of its own `y`.
    #[test]
    fn trend_normalization_flattens_a_smooth_ramp() {
        let samples: Vec<(f32, f32)> = (0..10)
            .map(|i| {
                let y = (i * 100) as f32;
                (y, 12.0 + 0.005 * y)
            })
            .collect();
        let mean = samples.iter().map(|&(_, s)| s).sum::<f32>() / samples.len() as f32;
        let normalized = apply_trend_normalization(&samples);
        for (i, &v) in normalized.iter().enumerate() {
            assert!(
                (v - mean).abs() < 1e-2,
                "line {i}: a perfectly smooth ramp must flatten to ~{mean}, got {v}"
            );
        }
    }

    /// Fixture B of the important pair: 10 lines, 9 at a consistent body
    /// size (`12.0`) and ONE genuinely smaller `Kleine Schriftgröße`-style
    /// caption (`9.0`) at an ARBITRARY `y` -- uncorrelated with position
    /// (R² ≈ 0.003, far below [`TREND_R2_SIGNIFICANT`]), so the fit must be
    /// judged NOT significant and every size must survive completely
    /// UNCHANGED -- a blanket normalization would incorrectly flatten the
    /// real typographic step.
    #[test]
    fn trend_normalization_leaves_a_step_change_intact() {
        let sizes = [12.0f32, 12.0, 12.0, 12.0, 9.0, 12.0, 12.0, 12.0, 12.0, 12.0];
        let samples: Vec<(f32, f32)> = sizes
            .iter()
            .enumerate()
            .map(|(i, &s)| ((i * 100) as f32, s))
            .collect();
        let normalized = apply_trend_normalization(&samples);
        assert_eq!(
            normalized,
            sizes.to_vec(),
            "a step change uncorrelated with y must survive untouched, \
             not be smoothed toward the body size"
        );
    }

    /// End-to-end through the public entry point: builds a page with one
    /// line that already carries [`DocLineMetrics`] (the row-metrics
    /// pipeline ran) and one that does not, both with the SAME 12
    /// one-pixel-wide char boxes over an identical 6-row ink band (so every
    /// glyph sample is unambiguously `6.0` and `n=12 >= GLYPH_SAMPLE_MIN`,
    /// keeping this test focused on the metrics-gating behaviour rather
    /// than the measurement/fallback/trend arithmetic already covered
    /// above). Only the line that started with `Some(metrics)` may receive
    /// `glyph_px`; the other must stay `None` -- never fabricated.
    #[test]
    fn attach_glyph_px_only_writes_lines_that_already_carry_metrics() {
        use crate::renderer::{LineMetrics, LineWords};
        use tesseract_core::dawg::PermuterType;
        use tesseract_core::WordResult;

        // 12 one-pixel-wide char boxes (TBOX order) sharing the ink band
        // that maps to top-down image rows [10,16) -- 6 rows.
        let char_boxes: Vec<(i32, i32, i32, i32)> = (0..12).map(|i| (i, 4, i + 1, 10)).collect();
        let word = WordResult {
            unichar_ids: vec![1; 12],
            certs: vec![0.0; 12],
            ratings: vec![0.0; 12],
            char_boxes,
            permuter: PermuterType::TopChoicePerm,
            space_certainty: 0.0,
            leading_space: false,
        };
        let metrics = LineMetrics {
            xheight: 5.0,
            ascrise: 2.0,
            descdrop: -1.0,
            baseline: 5.0,
        };
        let with_metrics = LineWords {
            words: vec![word.clone()],
            line_box: (0, 4, 12, 10),
            metrics: Some(metrics),
        };
        let without_metrics = LineWords {
            words: vec![word],
            line_box: (0, 4, 12, 10),
            metrics: None,
        };
        let lines = vec![with_metrics, without_metrics];

        let (pw, ph) = (20u32, 20u32);
        let mut binary = vec![255u8; (pw * ph) as usize];
        for y in 10..16usize {
            for x in 0..12usize {
                binary[y * pw as usize + x] = 0;
            }
        }

        let charset =
            tesseract_core::CharSet::load_from_str("2\nNULL 0 Common 0\na 3 0 a Left a a\n")
                .expect("charset");
        let mut page = DocPage::from_line_words(&lines, &charset, pw, ph);
        assert_eq!(page.lines.len(), 2);
        assert!(page.lines[0].metrics.is_some());
        assert!(page.lines[1].metrics.is_none());

        attach_glyph_px(&mut page, &lines, &binary, pw, ph);

        let g0 = page.lines[0]
            .metrics
            .as_ref()
            .and_then(|m| m.glyph_px)
            .expect("line with existing metrics must receive glyph_px");
        assert!((g0 - 6.0).abs() < 1e-4, "measured ink height: {g0}");
        assert!(
            page.lines[1].metrics.is_none(),
            "a line with no pre-existing metrics must be left untouched, \
             not fabricated"
        );
    }
}
