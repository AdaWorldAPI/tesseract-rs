//! Output renderers over [`WordResult`](tesseract_core::WordResult) (P4a) —
//! the plain-text renderer (`TessTextRenderer`/`ResultIterator::GetUTF8Text`)
//! and the TSV renderer (`TessTsvRenderer`/`TessBaseAPI::GetTSVText`).
//!
//! ## What is transcoded, and from where
//!
//! - **Text join / newline logic**: `ResultIterator::IterateAndAppendUTF8TextlineText`
//!   (`ccmain/resultiterator.cpp:723-758`), the class actually returned by
//!   `TessBaseAPI::GetIterator()` (NOT the base `LTRResultIterator`, which a
//!   naive read of the class hierarchy might reach for instead — it is
//!   shadowed by this override). Per-word: `int numSpaces = preserve_interword_spaces_
//!   ? it_->word()->word->space() : (words_appended > 0);` (line 745). This
//!   port transcodes the `preserve_interword_spaces_ = true` arm — `word->space()`
//!   returns the `blanks` field (`werd.h:100-101`), which is exactly
//!   [`WordResult::leading_space`] as 0/1 (`WERD` is constructed with
//!   `leading_space` as its blank count, `recodebeam.cpp:659`). The `false`
//!   arm (libtesseract's actual default) instead inserts a space before
//!   every word but the line's first — the task spec asks for the
//!   leading-space-driven join, which is this real, named, config-gated
//!   code path (`preserve_interword_spaces` is a genuine Tesseract param),
//!   not an invention. An empty line (`Empty(RIL_WORD)`) contributes NOTHING
//!   — no text, no separator (`resultiterator.cpp:723-726`).
//! - **TSV row shape + header**: `TessTsvRenderer::BeginDocumentHandler`
//!   (`api/renderer.cpp:167-172`, the exact header string) and
//!   `TessBaseAPI::GetTSVText` (`api/baseapi.cpp:1350-1463`) for the level-1
//!   (page), level-2 (block), level-3 (par), level-4 (line), level-5 (word)
//!   row shapes, plus `AddBoxToTSV` (`baseapi.cpp:1336-1343`) for the
//!   `left\ttop\twidth\theight` box columns.
//! - **Box coordinate conversion**: `PageIterator::BoundingBoxInternal` +
//!   `PageIterator::BoundingBox` (`ccmain/pageiterator.cpp:286-370`) — a
//!   bottom-up `TBOX(left, bottom, right, top)` box is converted to the
//!   top-down image rect via `top = pix_height - box.top()`,
//!   `bottom = pix_height - box.bottom()`, `left`/`right` unchanged, each
//!   clipped to `[0, dimension]` (scale = 1, no sub-rectangle restriction —
//!   this crate has no `SetRectangle` concept, so `rect_left_ = rect_top_ = 0`
//!   and the "processing rectangle" is the whole page: **APPROX**).
//! - **Word confidence**: `LTRResultIterator::Confidence(RIL_WORD)`
//!   (`ccmain/ltrresultiterator.cpp:91-138`): `mean_certainty =
//!   best_choice->certainty()` (a single word has `certainty_count == 1`, so
//!   the `/= certainty_count` is a no-op), `return ClipToRange(100 + 5 *
//!   mean_certainty, 0.0f, 100.0f)`. `best_choice->certainty()` is the **min**
//!   over the word's per-character certainties — traced to
//!   `WERD_CHOICE::set_unichar_id` (`ccstruct/ratngs.h:436-446`:
//!   `if (certainty < certainty_) certainty_ = certainty;`, seeded from
//!   `certainty_ = FLT_MAX` in `init()`, `ratngs.h:401`) via
//!   `FakeWordFromRatings` (`ccstruct/pageres.cpp:928-949`), the exact
//!   function `RecodeBeamSearch::ExtractBestPathAsWords` calls
//!   (`recodebeam.cpp:311`) to build the `WERD_CHOICE` our own
//!   [`WordResult`](tesseract_core::WordResult) already carries the raw
//!   per-character data for. `std::to_string(float)` prints 6 decimal
//!   digits, transcoded as `{:.6}`.
//!
//! ## Placeholders — APPROX until textord (plan §P3/P4)
//!
//! This crate has no layout-analysis (textord) stage, so the following are
//! fixed rather than derived from `PageIterator::IsAtBeginningOf(RIL_BLOCK/
//! RIL_PARA)` transitions:
//! - `block_num = 1`, `par_num = 1` always (the whole page is exactly one
//!   block and one paragraph). Real Tesseract would vary these per detected
//!   layout region.
//! - The level-1 page row's box is `(0, 0, page_w, page_h)` — this crate has
//!   no `SetRectangle` sub-region restriction, so `rect_left_ = rect_top_ = 0`
//!   and `rect_width_ = page_w`, `rect_height_ = page_h` unconditionally
//!   (`baseapi.cpp:1372-1376`).
//! - The level-2/3 block/par row box is the union of every non-empty line's
//!   `line_box` (real Tesseract computes `BLOCK::restricted_bounding_box`/
//!   `ROW::para()`-grouped unions from actual layout objects we don't have).
//! - `line_num` increments only over lines that produced at least one word
//!   (an empty line is skipped entirely, exactly mirroring how a `ROW` with
//!   no recognized `WERD`s never surfaces through `PAGE_RES_IT`/`ResultIterator`
//!   in the real pipeline — there is no "gap" in the real numbering either).

use tesseract_core::{ids_to_text, CharSet, WordResult};

/// One recognized text line's word output, plus the line's own box — the
/// input unit [`render_text`]/[`render_tsv`] consume. This is the assembly
/// tier's stand-in for Tesseract's `ROW`/`PAGE_RES_IT` line object: we don't
/// have layout analysis, so the caller (whoever ran recognition per line)
/// supplies the words plus the box that was used as `line_box` when calling
/// [`tesseract_ocr::LstmRecognizer::recognize_image_file_words`](crate::LstmRecognizer::recognize_image_file_words).
///
/// `line_box` is carried explicitly (rather than re-derived from
/// `words[..].char_boxes`) so the level-4 TSV line row has a well-defined
/// box even when `words` is empty (an empty line is still skipped from
/// output — see the module docs — but a caller may still want to construct
/// the value uniformly for every segmented band).
#[derive(Clone, Debug, PartialEq)]
pub struct LineWords {
    /// The words extracted from this line (`RecodeBeamSearch::extract_best_path_as_words`
    /// output for the line), in reading order.
    pub words: Vec<WordResult>,
    /// The line's box in the same bottom-up `(left, bottom, right, top)`
    /// `TBOX`-constructor-argument order as the `line_box` parameter passed
    /// to `recognize_image_file_words` (`recodebeam.cpp:647-654`).
    pub line_box: (i32, i32, i32, i32),
    /// The row's measured typographic metrics ([`LineMetrics`]), when the
    /// line came through the makerow pipeline (which computes them as part
    /// of sizing the recognition band). `None` for paths with no row object
    /// (e.g. `recognize_image_file_words` on a bare pre-cropped line strip).
    pub metrics: Option<LineMetrics>,
}

/// A recognized row's typographic metrics — `TO_ROW`'s
/// `xheight`/`ascrise`/`descdrop` (wave-3 `compute_block_xheight` output,
/// `makerow.cpp:1279-1389`) plus the fitted baseline, in the same bottom-up
/// PAGE space as [`LineWords::line_box`].
///
/// This is the exact quantity real Tesseract's renderers size fonts from:
/// `LTRResultIterator::WordFontAttributes` (`ltrresultiterator.cpp:168-172`)
/// computes `row_height = row->x_height() + row->ascenders() -
/// row->descenders()` (descenders ≤ 0, so this is the full
/// ascender-to-descender body height) and converts it to printer points —
/// the `fontsize` the PDF renderer then emits per word
/// (`pdfrenderer.cpp:434-447`). Carrying the raw trio (rather than a
/// pre-derived font size) keeps `doc.v1` factual and lets each renderer
/// apply its own conversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineMetrics {
    /// `TO_ROW::xheight` — the x-height estimate, in pixels.
    pub xheight: f32,
    /// `TO_ROW::ascrise` — ascender rise above the x-height (≥ 0).
    pub ascrise: f32,
    /// `TO_ROW::descdrop` — descender drop below the baseline (≤ 0).
    pub descdrop: f32,
    /// The fitted baseline's y at the line's horizontal midpoint
    /// (`line_m·mid_x + parallel_c`), bottom-up PAGE space.
    pub baseline: f32,
}

impl LineMetrics {
    /// `row_height` exactly as `WordFontAttributes` computes it
    /// (`ltrresultiterator.cpp:168-169`): `x_height + ascenders -
    /// descenders`, the full typographic body height in pixels — the
    /// natural nominal font size for this row's text.
    #[must_use]
    pub fn row_height(&self) -> f32 {
        self.xheight + self.ascrise - self.descdrop
    }
}

/// Clamp `v` into `[lo, hi]` — `ClipToRange` (`ccutil/helpers.h`), used
/// throughout `PageIterator::BoundingBox`/`LTRResultIterator::Confidence`.
pub(crate) fn clip_to_range(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

/// Union of zero or more bottom-up `(left, bottom, right, top)` boxes —
/// `TBOX::bounding_union` (`ccstruct/tesseract_ocr` boxes are the fake
/// `C_BLOB`/`WERD` boxes `InitializeWord` builds, `recodebeam.cpp:645-657`).
/// An empty input yields the degenerate box `(0, 0, 0, 0)` (never hit for a
/// non-empty [`WordResult::char_boxes`] on real recognizer output, but kept
/// total rather than panicking).
pub(crate) fn union_boxes(
    boxes: impl Iterator<Item = (i32, i32, i32, i32)>,
) -> (i32, i32, i32, i32) {
    boxes.fold(
        (i32::MAX, i32::MAX, i32::MIN, i32::MIN),
        |(l, b, r, t), (l2, b2, r2, t2)| (l.min(l2), b.min(b2), r.max(r2), t.max(t2)),
    )
}

/// Convert a bottom-up `TBOX`-order box into the top-down `(left, top, right,
/// bottom)` image box — `PageIterator::BoundingBoxInternal` +
/// `PageIterator::BoundingBox` (`pageiterator.cpp:286-370`). APPROX:
/// `scale_ = 1`, `rect_left_ = rect_top_ = 0` (no `SetRectangle` sub-region or
/// scaling in this crate). This is the raw `(left, top, right, bottom)` shape
/// hOCR's `title="bbox L T R B"` prints directly; [`to_image_rect`] derives
/// the TSV `(left, top, width, height)` shape from it.
pub(crate) fn to_image_box(
    bx: (i32, i32, i32, i32),
    page_w: i32,
    page_h: i32,
) -> (i32, i32, i32, i32) {
    let (left, bottom, right, top) = bx;
    // The degenerate union_boxes() empty case maps cleanly through clip_to_range
    // to (0, page_h, 0, 0)-ish output; documented above, not special-cased here.
    let img_left = clip_to_range(left, 0, page_w);
    let img_top = clip_to_range(page_h - top, 0, page_h);
    let img_right = clip_to_range(right, img_left, page_w);
    let img_bottom = clip_to_range(page_h - bottom, img_top, page_h);
    (img_left, img_top, img_right, img_bottom)
}

/// Convert a bottom-up `TBOX`-order box into the top-down `(left, top, width,
/// height)` image rect the TSV renderer prints — see [`to_image_box`] for the
/// underlying `PageIterator::BoundingBox` transcode this derives from.
fn to_image_rect(bx: (i32, i32, i32, i32), page_w: i32, page_h: i32) -> (i32, i32, i32, i32) {
    let (left, top, right, bottom) = to_image_box(bx, page_w, page_h);
    (left, top, right - left, bottom - top)
}

/// A word's unichar ids as text — `AppendUTF8WordText` (`resultiterator.cpp:705-719`):
/// concatenate `word->BestUTF8(i, false)` for every character in the word,
/// which for our transcode is exactly [`ids_to_text`] over the word's ids.
pub(crate) fn word_text(charset: &CharSet, word: &WordResult) -> String {
    let ids: Vec<u32> = word.unichar_ids.iter().map(|&id| id as u32).collect();
    ids_to_text(charset, &ids)
}

/// A word's confidence value before final formatting —
/// `LTRResultIterator::Confidence(RIL_WORD)` (`ltrresultiterator.cpp:91-138`),
/// see the module docs for the full derivation
/// (`ClipToRange(100 + 5 * min(word certs), 0.0, 100.0)`). Shared by
/// [`render_tsv`] (formatted `{:.6}`) and [`render_hocr`]
/// (`static_cast<int>`-truncated for `x_wconf`).
pub(crate) fn word_confidence(word: &WordResult) -> f32 {
    let min_cert = word.certs.iter().copied().fold(f32::MAX, f32::min);
    (100.0 + 5.0 * min_cert).clamp(0.0, 100.0)
}

/// Render plain text — `ResultIterator::IterateAndAppendUTF8TextlineText`
/// (`resultiterator.cpp:723-758`), see the module docs for exactly which
/// code path this transcodes (the `preserve_interword_spaces_ = true` arm,
/// driven by [`WordResult::leading_space`]).
///
/// An empty line (`line.words.is_empty()`) contributes nothing at all — no
/// text, no `\n` — mirroring the real early return on `Empty(RIL_WORD)`.
/// Every non-empty line ends with exactly one `\n` (`line_separator_`,
/// default `"\n"`), including the last line in `lines` (the real renderer
/// appends `line_separator_` unconditionally per line, with no trailing
/// trim at the paragraph/page level for the single-paragraph-per-page case
/// this crate always is — see the block/par placeholders above).
#[must_use]
pub fn render_text(lines: &[LineWords], charset: &CharSet) -> String {
    render_text_with_gaps(lines, charset, &[])
}

/// Same as [`render_text`], but inserts an EXTRA blank line before each line
/// whose index is flagged in `para_break` (as produced by
/// [`detect_paragraph_gaps`]) — a paragraph separator. `para_break` shorter
/// than `lines` (including empty, [`render_text`]'s own case) is simply
/// treated as `false` for any missing index; never a panic.
///
/// Never inserts a blank line before the FIRST emitted (non-empty) line,
/// regardless of its flag — there is no preceding paragraph to separate
/// from.
#[must_use]
pub fn render_text_with_gaps(
    lines: &[LineWords],
    charset: &CharSet,
    para_break: &[bool],
) -> String {
    let mut out = String::new();
    let mut emitted_any = false;
    for (i, line) in lines.iter().enumerate() {
        if line.words.is_empty() {
            continue;
        }
        if emitted_any && para_break.get(i).copied().unwrap_or(false) {
            out.push('\n');
        }
        for word in &line.words {
            if word.leading_space {
                out.push(' ');
            }
            out.push_str(&word_text(charset, word));
        }
        out.push('\n');
        emitted_any = true;
    }
    out
}

/// Detect PARAGRAPH breaks — a vertical gap between two recognized lines
/// that is genuinely larger than the page's ordinary inter-line leading —
/// directly from the RAW binarized page. **Not a Tesseract transcode**, same
/// footing as [`crate::rectify`] and [`crate::structured`]'s `doc.v1`: no
/// oracle exists for paragraph detection here, and Tesseract's own
/// `ParagraphModel`/`DetectParagraphs` (a real, separate subsystem — script
/// direction, indentation, and spacing modelling) is not transcoded. This is
/// this crate's own, deliberately narrow synthesis: spacing alone.
///
/// # Why not the lines' own crop bands (`LineWords::line_box`)
///
/// `line_box` is the *recognition* band — `linerec.cpp:239-246`'s "at least
/// ascender-to-descender" extension plus `kImagePadding = 4` — deliberately
/// generous, so consecutive crop bands routinely TOUCH OR OVERLAP even where
/// the real image has visibly more whitespace between paragraphs (measured:
/// on a real 8-line, 2-paragraph page, every `line_box`-to-`line_box` gap
/// computed this way was **zero**, including at the true paragraph
/// boundary). This function instead scans the binarized page directly for
/// blank-row runs, then buckets each run by which pair of line CENTERS
/// (`(top+bottom)/2`, converted to raster space) it falls between — the
/// crop bands never enter the computation.
///
/// # The rule: judge a gap against its neighbours, not an absolute size
///
/// Same correction shape as `xy_cut`'s gutter fallback, `noise_readmit_reach`,
/// and the table-gutter bridge — a fixed pixel threshold cannot separate
/// "paragraph gap" from "ordinary leading" across different DPIs/point sizes,
/// but a RATIO to the page's own MEDIAN inter-line gap can. A gap `>=
/// PARAGRAPH_GAP_RATIO` times the median is flagged; measured on a real
/// canonical test page (`tesseract-ocr/test`'s `phototest.tif`), ordinary
/// gaps were 3-4px and the true paragraph gap was 10px (ratio 3.33) — a wide,
/// unambiguous margin above the 2.0 threshold.
///
/// Declines (returns all-`false`) below [`MIN_LINES_FOR_PARAGRAPH_GAPS`]
/// non-empty lines — a median of one or two samples is not a population,
/// same reasoning as `noise_readmit_reach`'s and `dropcap::ordinary_scale`'s
/// own minimum-population guards.
///
/// **Also declines when too many of the measured gaps are exactly zero**
/// (`>= `[`MAX_ZERO_GAP_FRACTION`]` of them) — measured on an UN-DESKEWED
/// rotated page (`skew_p050.pgm`, +5°): `gap_after = [0, 0, 1, 1, 3, 0]`,
/// median 1, and the lone `3` clears a naive `2.0x` ratio test — a false
/// paragraph flag between two ordinary sentences with no real break. Root
/// cause: this function's "blank across the FULL raster width" test is
/// axis-aligned, and a tilted line's ink smears across MORE raster rows at
/// its edges than at its center, so on a sufinciently rotated page most
/// inter-line gaps collapse toward zero and the few that don't are
/// quantization noise, not signal — the population itself is unreliable,
/// not merely small. A HEALTHY page (this function's true positives —
/// `phototest.tif`: `[4,3,10,3,3,3,3]`; `page_furniture.pgm`: 13 gaps, 0
/// zeros; `resgrid.pgm`: `[8,7,89,7,6]`) has a **0% zero rate**; the false
/// positive has 50%. `skew_m025.pgm` (a milder −0.25° tilt) stays fully
/// non-zero and simply never clears the ratio test — this guard changes
/// nothing there, it specifically targets the degenerate-population case.
///
/// Returns a `Vec<bool>` the same length as `lines`, `para_break[i] == true`
/// meaning "insert a blank line before line `i`" — feed directly to
/// [`render_text_with_gaps`].
#[must_use]
pub fn detect_paragraph_gaps(binary: &[u8], w: usize, h: usize, lines: &[LineWords]) -> Vec<bool> {
    const PARAGRAPH_GAP_RATIO: f32 = 2.0;
    const MIN_LINES_FOR_PARAGRAPH_GAPS: usize = 3;
    const MAX_ZERO_GAP_FRACTION: f32 = 0.3;

    let mut out = vec![false; lines.len()];
    if w == 0 || h == 0 {
        return out;
    }

    // Indices (into `lines`) of the non-empty lines, with their raster-space
    // vertical CENTER — the ordering render_text itself emits in.
    let nonempty: Vec<(usize, f32)> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.words.is_empty())
        .map(|(i, l)| {
            let (_, b, _, t) = l.line_box;
            (i, h as f32 - (b + t) as f32 / 2.0)
        })
        .collect();
    if nonempty.len() < MIN_LINES_FOR_PARAGRAPH_GAPS {
        return out;
    }

    // Whole-page blank-row runs, scanned once — a maximal run of rows with
    // no ink anywhere across the row's full width.
    let is_row_blank = |y: usize| (0..w).all(|x| binary[y * w + x] == 255);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut run_start: Option<usize> = None;
    for y in 0..h {
        if is_row_blank(y) {
            run_start.get_or_insert(y);
        } else if let Some(s) = run_start.take() {
            runs.push((s, y - s));
        }
    }
    if let Some(s) = run_start {
        runs.push((s, h - s));
    }

    // Bucket each run by the inter-line-center interval its MIDPOINT falls
    // in, keeping the longest run per interval (a paragraph gap may span
    // more than one raw blank-row run if leading/trailing rows are not
    // perfectly blank).
    let mut gap_after = vec![0usize; nonempty.len() - 1];
    for &(start, len) in &runs {
        let mid = start as f32 + len as f32 / 2.0;
        for i in 0..nonempty.len() - 1 {
            if mid >= nonempty[i].1 && mid <= nonempty[i + 1].1 {
                gap_after[i] = gap_after[i].max(len);
                break;
            }
        }
    }

    let zero_count = gap_after.iter().filter(|&&g| g == 0).count();
    if zero_count as f32 >= MAX_ZERO_GAP_FRACTION * gap_after.len() as f32 {
        return out;
    }

    let mut sorted = gap_after.clone();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2].max(1);

    for (k, &gap) in gap_after.iter().enumerate() {
        if gap as f32 >= PARAGRAPH_GAP_RATIO * median as f32 {
            let (line_idx, _) = nonempty[k + 1];
            out[line_idx] = true;
        }
    }
    out
}

/// Render Tesseract TSV — `TessTsvRenderer::BeginDocumentHandler` (header) +
/// `TessBaseAPI::GetTSVText` (body), see the module docs for exact source
/// citations and the placeholder list (`block_num`/`par_num` fixed at `1`,
/// block/par box = union of line boxes, no textord).
///
/// `page_w`/`page_h` are the full page's pixel dimensions — used both for the
/// level-1 page row's box and as the `pix_height`/`pix_width` inputs to the
/// bottom-up-to-top-down box conversion ([`to_image_rect`]) for every other
/// row.
///
/// A line with no words is skipped entirely (no level-4 row, no level-5
/// rows) — mirrors `GetTSVText`'s `if (res_it->Empty(RIL_WORD)) { ...
/// continue; }` (`baseapi.cpp:1380-1383`), under which an empty line/word
/// iterator position contributes no output at any level. If every line is
/// empty, only the header and the level-1 page row are emitted.
#[must_use]
pub fn render_tsv(lines: &[LineWords], charset: &CharSet, page_w: u32, page_h: u32) -> String {
    let page_w = page_w as i32;
    let page_h = page_h as i32;
    let mut out = String::new();
    out.push_str(
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n",
    );

    // Level 1 — page row (baseapi.cpp:1366-1376). block/par/line/word_num are
    // always 0 here; conf is always -1 (no per-page confidence).
    out.push_str(&format!(
        "1\t1\t0\t0\t0\t0\t0\t0\t{page_w}\t{page_h}\t-1\t\n"
    ));

    let non_empty: Vec<&LineWords> = lines.iter().filter(|l| !l.words.is_empty()).collect();
    if non_empty.is_empty() {
        return out;
    }

    // Levels 2/3 — block/par rows (baseapi.cpp:1391-1410). APPROX: block_num
    // = par_num = 1 fixed (no textord); box = union of every non-empty
    // line's line_box (real Tesseract unions actual BLOCK/ROW-para objects).
    let block_box = union_boxes(non_empty.iter().map(|l| l.line_box));
    let (bl, bt, bw, bh) = to_image_rect(block_box, page_w, page_h);
    out.push_str(&format!("2\t1\t1\t0\t0\t0\t{bl}\t{bt}\t{bw}\t{bh}\t-1\t\n"));
    out.push_str(&format!("3\t1\t1\t1\t0\t0\t{bl}\t{bt}\t{bw}\t{bh}\t-1\t\n"));

    // Level 4/5 — line/word rows (baseapi.cpp:1411-1460).
    let mut line_num = 0;
    for line in non_empty {
        line_num += 1;
        let (ll, lt, lw, lh) = to_image_rect(line.line_box, page_w, page_h);
        out.push_str(&format!(
            "4\t1\t1\t1\t{line_num}\t0\t{ll}\t{lt}\t{lw}\t{lh}\t-1\t\n"
        ));

        for (word_idx, word) in line.words.iter().enumerate() {
            let word_num = word_idx + 1;
            let word_box = union_boxes(word.char_boxes.iter().copied());
            let (wl, wt, ww, wh) = to_image_rect(word_box, page_w, page_h);
            // The empty-certs fallback mirrors WERD_CHOICE::init()'s
            // certainty_ = FLT_MAX sentinel (ratngs.h:401), which clips to
            // 100.0 (never hit on real output).
            let conf = word_confidence(word);
            let text = word_text(charset, word);
            out.push_str(&format!(
                "5\t1\t1\t1\t{line_num}\t{word_num}\t{wl}\t{wt}\t{ww}\t{wh}\t{conf:.6}\t{text}\n"
            ));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// hOCR renderer (D4.4)
// ---------------------------------------------------------------------------
//
// Transcodes `TessHOcrRenderer::BeginDocumentHandler`/`EndDocumentHandler`
// (`api/hocrrenderer.cpp:480-511`, the XHTML document skeleton) and
// `TessBaseAPI::GetHOCRText` (`api/hocrrenderer.cpp:119-465`, the per-page
// `ocr_page`/`ocr_carea`/`ocr_par`/`ocr_line`/`ocrx_word` div/span nesting),
// plus `HOcrEscape` (`api/baseapi.cpp:2324-2349`, the HTML-escaping used for
// every user-controlled string value: the image filename and word text).
//
// ## ID scheme (read from the C++ source, not inferred)
//
// `page_id = page_number + 1` (hOCR pages are 1-based; `page_number` itself
// is fixed at `0` in this crate — single-image, single-page renderer, no
// multi-page loop). All div/span ids embed `page_id`, then a **per-kind
// counter that is NOT reset per parent**:
// - `bcnt`/`pcnt` (block/par counters) start at 1 and increment only when
//   `last_word_in_block`/`last_word_in_para` fires (`hocrrenderer.cpp:138,
//   446-460`) — since this crate has exactly one block and one paragraph per
//   page (no textord), they are always `block_{page_id}_1` / `par_{page_id}_1`.
// - `lcnt` (line counter) starts at 1 and increments once per **closed**
//   line (`lcnt++` inside `if (last_word_in_line)`, `hocrrenderer.cpp:450`) —
//   this crate increments it once per non-empty [`LineWords`] entry, in
//   order, exactly mirroring "an empty line never surfaces" (same skip rule
//   as [`render_text`]/[`render_tsv`]).
// - `wcnt` (word counter) starts at 1 and increments **after every word**,
//   unconditionally (`hocrrenderer.cpp:446`) — critically, this is a single
//   counter running across the **entire page**, not reset per line (unlike
//   TSV's `word_num`, which restarts at 1 for every line). `word_1_1`,
//   `word_1_2`, `word_1_3`, … continue monotonically across line boundaries.
// - `scnt`/`tcnt`/`ccnt` (symbol/timestep/choice counters, for
//   `lstm_choice_mode`/`hocr_boxes`) are never surfaced: this crate has no
//   per-symbol LSTM-choice data, so those branches (`hocr_boxes`,
//   `lstm_choice_mode == 1 | 2`) are permanently the `false`/`0` arm — the
//   real default (`GetBoolVariable("hocr_font_info"/"hocr_char_boxes", ...)`
//   both default `false`; `lstm_choice_mode` defaults `0`).
//
// ## Quoting (verbatim from the source, easy to get backwards)
//
// The page div's `title` attribute is single-quoted with an embedded
// double-quoted image name (`title='image "NAME"; bbox …'`,
// `hocrrenderer.cpp:156-169`). `AddBoxTohOCR` — used for the block/par/line
// (`RIL_BLOCK`/`RIL_PARA`/`RIL_TEXTLINE`) boxes — is the **only** place using
// double quotes for the whole attribute (`title="bbox L T R B">`,
// `hocrrenderer.cpp:89-108`, per that function's own comment). The per-word
// title is built inline back in single quotes (`title='bbox L T R B;
// x_wconf C'`, `hocrrenderer.cpp:273-282`). hOCR bbox values are always
// `left top right bottom` (via [`to_image_box`]) — NOT the TSV renderer's
// `left top width height` ([`to_image_rect`]).
//
// ## Placeholders — APPROX until textord / font / language / baseline data exists
//
// In addition to the block/par/box placeholders shared with [`render_tsv`]
// (see the module docs), hOCR-specific fields this crate cannot derive yet:
// - **`x_wconf`** truncates like the real `static_cast<int>(Confidence(...))`
//   (`hocrrenderer.cpp:275`) — truncation toward zero, not rounding.
// - **Baseline + `x_size`/`x_descenders`/`x_ascenders`** (`AddBaselineCoordsTohOCR`,
//   `hocrrenderer.cpp:51-106`) are omitted entirely: this crate has no
//   textord baseline fit or row-height/ascender/descender measurement. This
//   mirrors the real renderer's own conditional omission when
//   `it->Baseline(...)` returns `false` (`hocrrenderer.cpp:65-67`) — we are
//   permanently in that "no baseline available" state, just without the
//   per-call check (there is nothing to check against).
// - **`lang`/`dir` attributes** are never emitted: this crate has no
//   per-word/per-paragraph language or writing-direction detection, so
//   `paragraph_lang`/`WordRecognitionLanguage()`/`WordDirection()` are always
//   the "nothing to say" case (`null`/`DIR_NEUTRAL` in the real source) —
//   the real renderer emits nothing for that case too (`hocrrenderer.cpp:
//   283-303`), so this is exact, not approximated.
// - **Bold/italic/font/`hocr_font_info`** (`WordFontAttributes`,
//   `hocrrenderer.cpp:267-281,308-313,379-384`) never fire — no font
//   detection in this crate; `font_info_` is fixed `false` (the
//   `TessHOcrRenderer(outputbase)` single-arg constructor's default,
//   `hocrrenderer.cpp:470-473`), so the `ocrp_font ocrp_fsize` capability
//   suffix is never appended either.
// - **`TESSERACT_VERSION_STR`** and **`scan_res`** (`GetSourceYResolution()`,
//   sourced from the image thresholder's DPI metadata, which this crate has
//   no equivalent of) are fixed placeholder constants
//   ([`HOCR_VERSION_PLACEHOLDER`], `HOCR_SCAN_RES_PLACEHOLDER`) — never
//   load-bearing, just present so the document is well-formed hOCR.
// - **No leading-space handling** (unlike [`render_text`]/[`render_tsv`]):
//   every word span unconditionally starts on its own indented line
//   (`"\n      <span class='ocrx_word'"`, `hocrrenderer.cpp:264`), exactly as
//   the C++ source does — the visual word-separation in a rendered hOCR
//   document comes from HTML whitespace-collapsing of that newline+indent,
//   not from any inserted space character, so [`WordResult::leading_space`]
//   plays no role here.

/// Escape `<`, `>`, `&`, `"`, `'` for hOCR text/attribute content — `HOcrEscape`
/// (`api/baseapi.cpp:2324-2349`). Every other byte (including all non-ASCII
/// UTF-8) passes through unchanged, exactly matching the C++ `switch` on a
/// single `char` with a `default: ret += *ptr;` arm.
fn hocr_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

/// Placeholder for `TESSERACT_VERSION_STR` (a build-time constant of the
/// libtesseract build this crate transcodes) — see the placeholder list
/// above. Never load-bearing; only appears in the `ocr-system` meta tag.
const HOCR_VERSION_PLACEHOLDER: &str = "5.5.0";

/// Placeholder for `GetSourceYResolution()` (DPI metadata this crate's image
/// pipeline does not carry) — see the placeholder list above.
const HOCR_SCAN_RES_PLACEHOLDER: i32 = 0;

/// Render a full hOCR (XHTML) document. Transcodes and composes
/// `TessHOcrRenderer::BeginDocumentHandler`, `TessBaseAPI::GetHOCRText`, and
/// `TessHOcrRenderer::EndDocumentHandler` into a single call, since this
/// crate has no multi-page renderer object to hold `font_info_`/`title_`
/// state across pages. See the module section docs above for the full ID
/// scheme, quoting rules, and placeholder list this transcodes.
///
/// `page_image_name` fills both the `<title>` element (verbatim, **not**
/// HTML-escaped — `AppendString(title())` calls no `HOcrEscape`, a faithful
/// reproduction of the real renderer's own inconsistency, not a bug
/// introduced here) and the page div's `title='image "…"'` value (escaped,
/// matching `HOcrEscape(input_file_.c_str())`, `hocrrenderer.cpp:161`).
///
/// A line with no words is skipped entirely, same rule as
/// [`render_text`]/[`render_tsv`]. If every line is empty, only the
/// `ocr_page` div (opened and immediately closed) is emitted — no
/// `ocr_carea`/`ocr_par` — mirroring the real source's `IsAtBeginningOf(RIL_BLOCK)`
/// gate never firing when `Empty(RIL_WORD)` is true for every position
/// (`hocrrenderer.cpp:203-206,209`).
#[must_use]
pub fn render_hocr(
    lines: &[LineWords],
    charset: &CharSet,
    page_w: u32,
    page_h: u32,
    page_image_name: &str,
) -> String {
    let page_w_i = page_w as i32;
    let page_h_i = page_h as i32;
    let mut out = String::new();

    // BeginDocumentHandler (hocrrenderer.cpp:480-505). font_info_ = false, so
    // the " ocrp_font ocrp_fsize" capability suffix is never appended.
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"\n");
    out.push_str("    \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n");
    out.push_str("<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n");
    out.push_str(" <head>\n");
    out.push_str("  <title>");
    // NOT escaped — see the doc comment above.
    out.push_str(page_image_name);
    out.push_str("</title>\n");
    out.push_str("  <meta http-equiv=\"Content-Type\" content=\"text/html;charset=utf-8\"/>\n");
    out.push_str(&format!(
        "  <meta name='ocr-system' content='tesseract {HOCR_VERSION_PLACEHOLDER}' />\n"
    ));
    out.push_str(
        "  <meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_dir ocrp_lang ocrp_wconf'/>\n",
    );
    out.push_str(" </head>\n");
    out.push_str(" <body>\n");

    // GetHOCRText (hocrrenderer.cpp:119-465). page_id/page_number are fixed
    // (1-based/0-based single page — no multi-page loop in this crate).
    let page_id = 1;
    let page_number = 0;
    out.push_str("  <div class='ocr_page' id='page_");
    out.push_str(&page_id.to_string());
    out.push_str("' title='image \"");
    out.push_str(&hocr_escape(page_image_name));
    out.push_str(&format!(
        "\"; bbox 0 0 {page_w} {page_h}; ppageno {page_number}; scan_res {HOCR_SCAN_RES_PLACEHOLDER} {HOCR_SCAN_RES_PLACEHOLDER}'>\n"
    ));

    let non_empty: Vec<&LineWords> = lines.iter().filter(|l| !l.words.is_empty()).collect();
    if !non_empty.is_empty() {
        // Block/par open (hocrrenderer.cpp:209-229). APPROX: bcnt = pcnt = 1
        // fixed (no textord); box = union of every non-empty line's box,
        // same placeholder policy as render_tsv's block/par row.
        let block_par_box = union_boxes(non_empty.iter().map(|l| l.line_box));
        let (bl, bt, br, bb) = to_image_box(block_par_box, page_w_i, page_h_i);
        out.push_str(&format!(
            "   <div class='ocr_carea' id='block_{page_id}_1' title=\"bbox {bl} {bt} {br} {bb}\">"
        ));
        out.push_str(&format!(
            "\n    <p class='ocr_par' id='par_{page_id}_1' title=\"bbox {bl} {bt} {br} {bb}\">"
        ));

        let last_line_idx = non_empty.len() - 1;
        let mut word_num: u32 = 1; // wcnt — global across the whole page, per hOCR (see module docs).
        for (line_idx, line) in non_empty.iter().enumerate() {
            let line_num = line_idx + 1;
            let (ll, lt, lr, lb) = to_image_box(line.line_box, page_w_i, page_h_i);
            out.push_str(&format!(
                "\n     <span class='ocr_line' id='line_{page_id}_{line_num}' title=\"bbox {ll} {lt} {lr} {lb}\">"
            ));

            let last_word_idx = line.words.len() - 1; // non-empty guaranteed by the filter above.
            for (word_idx, word) in line.words.iter().enumerate() {
                let word_box = union_boxes(word.char_boxes.iter().copied());
                let (wl, wt, wr, wb) = to_image_box(word_box, page_w_i, page_h_i);
                // static_cast<int>(Confidence(RIL_WORD)) — truncation toward
                // zero, not rounding (hocrrenderer.cpp:275).
                let conf = word_confidence(word) as i32;
                let text = hocr_escape(&word_text(charset, word));
                out.push_str(&format!(
                    "\n      <span class='ocrx_word' id='word_{page_id}_{word_num}' title='bbox {wl} {wt} {wr} {wb}; x_wconf {conf}'>{text}</span>"
                ));
                word_num += 1;

                if word_idx == last_word_idx {
                    out.push_str("\n     </span>");
                }
            }

            if line_idx == last_line_idx {
                out.push_str("\n    </p>\n");
                out.push_str("   </div>\n");
            }
        }
    }

    out.push_str("  </div>\n");

    // EndDocumentHandler (hocrrenderer.cpp:507-511).
    out.push_str(" </body>\n</html>\n");

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tesseract_core::dawg::PermuterType;

    /// A minimal charset: id 0 = `NULL` -> space (the byte-parity convention,
    /// see `tesseract_core::lib.rs`'s own `sample()` test fixture), id 1 =
    /// `a`, id 2 = `b`.
    fn test_charset() -> CharSet {
        CharSet::load_from_str("3\nNULL 0 Common 0\na 3 0 a Left a a\nb 3 0 b Left b b\n")
            .expect("valid unicharset")
    }

    fn word(ids: &[i32], leading_space: bool, cert: f32, box_: (i32, i32, i32, i32)) -> WordResult {
        WordResult {
            unichar_ids: ids.to_vec(),
            certs: ids.iter().map(|_| cert).collect(),
            ratings: ids.iter().map(|_| 0.0).collect(),
            char_boxes: ids.iter().map(|_| box_).collect(),
            permuter: PermuterType::TopChoicePerm,
            space_certainty: 0.0,
            leading_space,
        }
    }

    #[test]
    fn render_text_joins_two_lines_with_leading_space_flags() {
        // Line 1: "a" (no leading space, first word) + "b" (leading_space=true) -> "a b\n"
        let line1 = LineWords {
            words: vec![
                word(&[1], false, -0.1, (0, 0, 5, 10)),
                word(&[2], true, -0.1, (6, 0, 10, 10)),
            ],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        // Line 2: "a" (no leading space) + "b" (leading_space=false, e.g. glued) -> "ab\n"
        let line2 = LineWords {
            words: vec![
                word(&[1], false, -0.1, (0, 0, 5, 10)),
                word(&[2], false, -0.1, (6, 0, 10, 10)),
            ],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let cs = test_charset();
        assert_eq!(render_text(&[line1, line2], &cs), "a b\nab\n");
    }

    #[test]
    fn render_text_skips_empty_lines_entirely() {
        let empty = LineWords {
            words: vec![],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let line = LineWords {
            words: vec![word(&[1], false, -0.1, (0, 0, 5, 10))],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let cs = test_charset();
        // The empty line contributes NOTHING - no blank line, no separator.
        assert_eq!(render_text(&[empty.clone(), line.clone()], &cs), "a\n");
        assert_eq!(render_text(&[line, empty], &cs), "a\n");
    }

    #[test]
    fn render_tsv_golden_single_line_two_words() {
        // page 10x10. Line spans the whole page height: bottom=0, top=10 (TBOX
        // bottom-up) -> image top=10-10=0, bottom=10-0=10, height=10.
        // Word "a": char_box (0,0,4,10) -> image left=0, top=0, width=4, height=10.
        // Word "b" (leading_space, cert=-0.2): char_box (5,0,9,10) -> image
        // left=5, top=0, width=4, height=10.
        let line = LineWords {
            words: vec![
                word(&[1], false, 0.0, (0, 0, 4, 10)),
                word(&[2], true, -0.2, (5, 0, 9, 10)),
            ],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let cs = test_charset();
        let tsv = render_tsv(&[line], &cs, 10, 10);

        // conf("a") = clip(100 + 5*0.0, 0, 100) = 100.000000
        // conf("b") = clip(100 + 5*-0.2, 0, 100) = 99.000000
        let expected = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
1\t1\t0\t0\t0\t0\t0\t0\t10\t10\t-1\t\n\
2\t1\t1\t0\t0\t0\t0\t0\t10\t10\t-1\t\n\
3\t1\t1\t1\t0\t0\t0\t0\t10\t10\t-1\t\n\
4\t1\t1\t1\t1\t0\t0\t0\t10\t10\t-1\t\n\
5\t1\t1\t1\t1\t1\t0\t0\t4\t10\t100.000000\ta\n\
5\t1\t1\t1\t1\t2\t5\t0\t4\t10\t99.000000\tb\n";
        assert_eq!(tsv, expected);
    }

    #[test]
    fn render_tsv_skips_empty_lines_but_keeps_page_row() {
        let empty = LineWords {
            words: vec![],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let cs = test_charset();
        let tsv = render_tsv(&[empty], &cs, 10, 10);
        let expected = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
1\t1\t0\t0\t0\t0\t0\t0\t10\t10\t-1\t\n";
        assert_eq!(tsv, expected);
    }

    #[test]
    fn conf_edge_cases_clip_to_0_100() {
        // Very negative certainty clips to 0.0, not a negative number.
        let bad = word(&[0], false, -1000.0, (0, 0, 4, 10));
        let min_cert = bad.certs.iter().copied().fold(f32::MAX, f32::min);
        let conf = (100.0 + 5.0 * min_cert).clamp(0.0, 100.0);
        assert_eq!(conf, 0.0);

        // A positive certainty (shouldn't normally happen, but the formula is
        // unconditional) clips to 100.0, not something above 100.
        let great = word(&[0], false, 10.0, (0, 0, 4, 10));
        let min_cert = great.certs.iter().copied().fold(f32::MAX, f32::min);
        let conf = (100.0 + 5.0 * min_cert).clamp(0.0, 100.0);
        assert_eq!(conf, 100.0);

        // Empty certs (degenerate, never hit on real recognizer output):
        // mirrors WERD_CHOICE::init()'s certainty_ = FLT_MAX sentinel -> clips
        // to 100.0.
        let no_certs: Vec<f32> = Vec::new();
        let min_cert = no_certs.iter().copied().fold(f32::MAX, f32::min);
        let conf = (100.0 + 5.0 * min_cert).clamp(0.0, 100.0);
        assert_eq!(conf, 100.0);
    }

    #[test]
    fn hocr_escape_matches_the_exact_cpp_char_set() {
        // HOcrEscape (baseapi.cpp:2324-2349): exactly <, >, &, ", ' are
        // escaped; every other byte (incl. whitespace, digits, unicode)
        // passes through unchanged via the `default: ret += *ptr;` arm.
        assert_eq!(hocr_escape("<"), "&lt;");
        assert_eq!(hocr_escape(">"), "&gt;");
        assert_eq!(hocr_escape("&"), "&amp;");
        assert_eq!(hocr_escape("\""), "&quot;");
        assert_eq!(hocr_escape("'"), "&#39;");
        assert_eq!(
            hocr_escape("<a href=\"x\">Tom & Jerry's \"cat\"</a>"),
            "&lt;a href=&quot;x&quot;&gt;Tom &amp; Jerry&#39;s &quot;cat&quot;&lt;/a&gt;"
        );
        // Not in the C++ switch's cases: passes through verbatim.
        assert_eq!(hocr_escape("plain text 123"), "plain text 123");
        assert_eq!(hocr_escape(""), "");
        // Non-ASCII UTF-8 also passes through unchanged (the C++ switches on
        // a single narrow `char`, so multi-byte UTF-8 sequences never match
        // any of the five cases and fall through `default:` byte-by-byte;
        // Rust's `char`-based loop reproduces the same net effect for valid
        // UTF-8 text).
        assert_eq!(hocr_escape("café <naïve>"), "café &lt;naïve&gt;");
    }

    /// Minimal fixture reused by the hOCR tests: 2 lines, 3 words total,
    /// same box/cert shapes as [`render_tsv_golden_single_line_two_words`]
    /// plus one more line, so the numbers below are hand-computable from
    /// `to_image_box`/`word_confidence` directly.
    fn hocr_fixture() -> (Vec<LineWords>, CharSet) {
        let line1 = LineWords {
            words: vec![
                word(&[1], false, 0.0, (0, 0, 4, 10)),
                word(&[2], true, -0.2, (5, 0, 9, 10)),
            ],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let line2 = LineWords {
            words: vec![word(&[1], false, -0.1, (0, 0, 4, 10))],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        (vec![line1, line2], test_charset())
    }

    #[test]
    fn render_hocr_golden_full_document() {
        let (lines, cs) = hocr_fixture();
        let hocr = render_hocr(&lines, &cs, 10, 10, "test.pgm");

        // Boxes (page 10x10, to_image_box on bottom-up TBOX-order inputs):
        //   line1.line_box (0,0,10,10) -> (0,0,10,10); word "a" (0,0,4,10) ->
        //   (0,0,4,10); word "b" (5,0,9,10) -> (5,0,9,10).
        //   line2.line_box (0,0,10,10) -> (0,0,10,10); word "a" (0,0,4,10) ->
        //   (0,0,4,10).
        //   block/par box = union(line1.line_box, line2.line_box) = (0,0,10,10)
        //   -> (0,0,10,10).
        // Confidences: word_confidence(cert=0.0) = 100.0 -> x_wconf 100;
        //   word_confidence(cert=-0.2) = 99.0 -> x_wconf 99;
        //   word_confidence(cert=-0.1) = 99.5 -> truncated (as i32) -> 99.
        // wcnt is a single counter across the whole page: word_1_1, word_1_2
        // (line 1), word_1_3 (line 2) - NOT reset per line.
        let expected = concat!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
            "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"\n",
            "    \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n",
            "<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n",
            " <head>\n",
            "  <title>test.pgm</title>\n",
            "  <meta http-equiv=\"Content-Type\" content=\"text/html;charset=utf-8\"/>\n",
            "  <meta name='ocr-system' content='tesseract 5.5.0' />\n",
            "  <meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_dir ocrp_lang ocrp_wconf'/>\n",
            " </head>\n",
            " <body>\n",
            "  <div class='ocr_page' id='page_1' title='image \"test.pgm\"; bbox 0 0 10 10; ppageno 0; scan_res 0 0'>\n",
            "   <div class='ocr_carea' id='block_1_1' title=\"bbox 0 0 10 10\">",
            "\n    <p class='ocr_par' id='par_1_1' title=\"bbox 0 0 10 10\">",
            "\n     <span class='ocr_line' id='line_1_1' title=\"bbox 0 0 10 10\">",
            "\n      <span class='ocrx_word' id='word_1_1' title='bbox 0 0 4 10; x_wconf 100'>a</span>",
            "\n      <span class='ocrx_word' id='word_1_2' title='bbox 5 0 9 10; x_wconf 99'>b</span>",
            "\n     </span>",
            "\n     <span class='ocr_line' id='line_1_2' title=\"bbox 0 0 10 10\">",
            "\n      <span class='ocrx_word' id='word_1_3' title='bbox 0 0 4 10; x_wconf 99'>a</span>",
            "\n     </span>",
            "\n    </p>\n",
            "   </div>\n",
            "  </div>\n",
            " </body>\n",
            "</html>\n",
        );
        assert_eq!(hocr, expected);
    }

    #[test]
    fn render_hocr_title_is_not_html_escaped_but_page_bbox_title_is() {
        // AppendString(title()) calls no HOcrEscape (hocrrenderer.cpp:487) -
        // a faithful reproduction of the real renderer's own inconsistency.
        // The page div's `image "..."` value, by contrast, IS escaped
        // (hocrrenderer.cpp:161).
        let (lines, cs) = hocr_fixture();
        let hocr = render_hocr(&lines, &cs, 10, 10, "a&b.pgm");
        assert!(hocr.contains("<title>a&b.pgm</title>"));
        assert!(hocr.contains("title='image \"a&amp;b.pgm\"; bbox"));
    }

    #[test]
    fn render_hocr_skips_empty_lines_entirely() {
        let empty = LineWords {
            words: vec![],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let line = LineWords {
            words: vec![word(&[1], false, -0.1, (0, 0, 4, 10))],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let cs = test_charset();

        // The empty line contributes NOTHING - not even an id gap: the
        // surviving line is still line_1_1 / word_1_1, exactly as if the
        // empty line were never in the input slice at all.
        let with_leading_empty = render_hocr(&[empty.clone(), line.clone()], &cs, 10, 10, "x");
        let without_empty = render_hocr(std::slice::from_ref(&line), &cs, 10, 10, "x");
        assert_eq!(with_leading_empty, without_empty);
        assert!(with_leading_empty.contains("id='line_1_1'"));
        assert!(with_leading_empty.contains("id='word_1_1'"));
    }

    #[test]
    fn render_hocr_all_lines_empty_emits_only_the_page_div() {
        let empty = LineWords {
            words: vec![],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let cs = test_charset();
        let hocr = render_hocr(&[empty], &cs, 10, 10, "x");

        // No ocr_carea/ocr_par/ocr_line/ocrx_word div/span at all - just the
        // page div opened and immediately closed, mirroring
        // IsAtBeginningOf(RIL_BLOCK) never firing when every position is
        // Empty(RIL_WORD). (The `ocr-capabilities` meta tag always lists all
        // four class names regardless, so we check for the actual opening
        // tags rather than the bare substrings.)
        assert!(!hocr.contains("class='ocr_carea'"));
        assert!(!hocr.contains("class='ocr_par'"));
        assert!(!hocr.contains("class='ocr_line'"));
        assert!(!hocr.contains("class='ocrx_word'"));
        assert!(hocr.contains("  <div class='ocr_page' id='page_1'"));
        assert!(hocr.contains("  </div>\n </body>\n</html>\n"));
    }

    // -----------------------------------------------------------------
    // detect_paragraph_gaps / render_text_with_gaps — consumer-side
    // synthesis, NOT a Tesseract transcode (see the module doc). Every
    // fixture below is a synthetic binarized page + hand-built LineWords,
    // no model load needed.
    // -----------------------------------------------------------------

    /// A synthetic all-white `w x h` page with `n` horizontal ink bars
    /// (`bar_h` tall) at the given raster-space TOP y-coordinates —
    /// independent of any LineWords, mirroring the "raw pixel geometry"
    /// this function actually scans.
    fn page_with_bars(w: usize, h: usize, bar_tops: &[usize], bar_h: usize) -> Vec<u8> {
        let mut page = vec![255u8; w * h];
        for &top in bar_tops {
            for y in top..(top + bar_h).min(h) {
                for x in 0..w {
                    page[y * w + x] = 0;
                }
            }
        }
        page
    }

    /// One non-empty `LineWords` whose `line_box` TOP/BOTTOM (bottom-up,
    /// y-up page space) matches a raster bar at `[raster_top, raster_top +
    /// bar_h)` on a page of height `h`.
    fn line_at(h: usize, raster_top: usize, bar_h: usize) -> LineWords {
        let top = h as i32 - raster_top as i32;
        let bottom = h as i32 - (raster_top + bar_h) as i32;
        LineWords {
            words: vec![word(&[1], false, -0.1, (0, bottom, 10, top))],
            line_box: (0, bottom, 10, top),
            metrics: None,
        }
    }

    #[test]
    fn detect_paragraph_gaps_flags_a_gap_well_above_the_median() {
        // 5 lines, ordinary leading 10px, ONE gap of 70px (7x median) —
        // mirrors the real phototest.tif measurement (ratio 3.33) with a
        // wider margin so the fixture is unambiguous.
        let (w, h, bar_h) = (100usize, 300usize, 20usize);
        let tops = [20usize, 50, 80, 150, 180]; // raster gaps: 10,10,70,10
        let page = page_with_bars(w, h, &tops, bar_h);
        let lines: Vec<LineWords> = tops.iter().map(|&t| line_at(h, t, bar_h)).collect();

        let flags = detect_paragraph_gaps(&page, w, h, &lines);
        assert_eq!(
            flags,
            vec![false, false, false, true, false],
            "only the line after the 70px gap should be flagged"
        );
    }

    #[test]
    fn detect_paragraph_gaps_stays_silent_on_uniform_leading() {
        let (w, h, bar_h) = (100usize, 300usize, 20usize);
        let tops = [20usize, 50, 80, 110, 140]; // all raster gaps = 10px
        let page = page_with_bars(w, h, &tops, bar_h);
        let lines: Vec<LineWords> = tops.iter().map(|&t| line_at(h, t, bar_h)).collect();

        let flags = detect_paragraph_gaps(&page, w, h, &lines);
        assert_eq!(flags, vec![false; 5], "uniform leading must never flag");
    }

    #[test]
    fn detect_paragraph_gaps_declines_below_the_minimum_line_count() {
        // The REAL falsifier: with 0 or 1 non-empty lines, `gap_after` is
        // EMPTY (`nonempty.len().saturating_sub(1)` -- zero pairs to
        // compare), and computing a median from an empty slice
        // (`sorted[sorted.len() / 2]`) panics. A 2-line fixture (1 gap
        // sample) does NOT exercise this: a lone gap can never be "2x
        // itself", so the ratio test structurally cannot fire regardless of
        // the guard -- verified by disabling the guard and confirming a
        // 2-line-only fixture stays green either way. The 1-line and 0-line
        // cases below are what the guard actually protects.
        let (w, h, bar_h) = (100usize, 300usize, 20usize);

        let empty_lines: Vec<LineWords> = Vec::new();
        assert_eq!(
            detect_paragraph_gaps(&page_with_bars(w, h, &[], bar_h), w, h, &empty_lines),
            Vec::<bool>::new(),
            "zero lines must not panic"
        );

        let one_line = vec![line_at(h, 20, bar_h)];
        assert_eq!(
            detect_paragraph_gaps(&page_with_bars(w, h, &[20], bar_h), w, h, &one_line),
            vec![false],
            "one line (zero gap pairs) must not panic on an empty median"
        );

        // The originally-written 2-line case, kept as a DIFFERENT, weaker
        // guarantee: still correctly silent, but not itself proof the
        // MIN_LINES guard is load-bearing (see the note above).
        let tops = [20usize, 50];
        let page = page_with_bars(w, h, &tops, bar_h);
        let lines: Vec<LineWords> = tops.iter().map(|&t| line_at(h, t, bar_h)).collect();
        assert_eq!(detect_paragraph_gaps(&page, w, h, &lines), vec![false; 2]);
    }

    #[test]
    fn detect_paragraph_gaps_declines_on_a_degenerate_mostly_zero_population() {
        // Reproduces the measured skew_p050.pgm false positive: an
        // un-deskewed rotated page collapses MOST inter-line gaps to
        // exactly zero (tilted ink smears across raster rows), and a lone
        // small non-zero value would otherwise clear a naive ratio test.
        // Real measurement: gap_after = [0, 0, 1, 1, 3, 0] -- 3 of 6 (50%)
        // exactly zero.
        let (w, h, bar_h) = (100usize, 400usize, 30usize);
        // Bars overlap/touch except for tiny 1-3px gaps at a few positions.
        let tops = [20usize, 50, 80, 111, 142, 175, 205];
        let page = page_with_bars(w, h, &tops, bar_h);
        let lines: Vec<LineWords> = tops.iter().map(|&t| line_at(h, t, bar_h)).collect();

        // Anti-vacuity: confirm the fixture really is degenerate before
        // trusting the "declines" assertion below -- otherwise a fixture
        // that never reaches the guard at all would pass vacuously.
        let raw_gaps: Vec<i32> = tops
            .windows(2)
            .map(|p| (p[1] - p[0]) as i32 - bar_h as i32)
            .collect();
        let zero_gaps = raw_gaps.iter().filter(|&&g| g <= 0).count();
        assert!(
            zero_gaps * 2 >= raw_gaps.len(),
            "fixture must reproduce the >=50% zero-gap regime: {raw_gaps:?}"
        );

        let flags = detect_paragraph_gaps(&page, w, h, &lines);
        assert_eq!(
            flags,
            vec![false; tops.len()],
            "a degenerate mostly-zero-gap population must decline entirely, \
             even though one gap is numerically 3x another"
        );
    }

    #[test]
    fn render_text_with_gaps_inserts_exactly_one_blank_line_where_flagged() {
        let cs = test_charset();
        let l1 = LineWords {
            words: vec![word(&[1], false, -0.1, (0, 0, 10, 10))],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let l2 = LineWords {
            words: vec![word(&[2], false, -0.1, (0, 0, 10, 10))],
            line_box: (0, 0, 10, 10),
            metrics: None,
        };
        let lines = [l1, l2];

        let plain = render_text(&lines, &cs);
        assert_eq!(plain, "a\nb\n", "render_text itself must be unaffected");

        let with_gap = render_text_with_gaps(&lines, &cs, &[false, true]);
        assert_eq!(
            with_gap, "a\n\nb\n",
            "flagged second line gets a blank line before it"
        );

        // The first line is NEVER preceded by a blank line, even if flagged
        // -- there is no prior paragraph to separate from.
        let first_flagged = render_text_with_gaps(&lines, &cs, &[true, false]);
        assert_eq!(
            first_flagged, "a\nb\n",
            "a flag on the first emitted line is a no-op"
        );
    }
}
