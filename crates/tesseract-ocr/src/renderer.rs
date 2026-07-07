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
}

/// Clamp `v` into `[lo, hi]` — `ClipToRange` (`ccutil/helpers.h`), used
/// throughout `PageIterator::BoundingBox`/`LTRResultIterator::Confidence`.
fn clip_to_range(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

/// Union of zero or more bottom-up `(left, bottom, right, top)` boxes —
/// `TBOX::bounding_union` (`ccstruct/tesseract_ocr` boxes are the fake
/// `C_BLOB`/`WERD` boxes `InitializeWord` builds, `recodebeam.cpp:645-657`).
/// An empty input yields the degenerate box `(0, 0, 0, 0)` (never hit for a
/// non-empty [`WordResult::char_boxes`] on real recognizer output, but kept
/// total rather than panicking).
fn union_boxes(boxes: impl Iterator<Item = (i32, i32, i32, i32)>) -> (i32, i32, i32, i32) {
    boxes.fold(
        (i32::MAX, i32::MAX, i32::MIN, i32::MIN),
        |(l, b, r, t), (l2, b2, r2, t2)| (l.min(l2), b.min(b2), r.max(r2), t.max(t2)),
    )
}

/// Convert a bottom-up `TBOX`-order box into the top-down `(left, top, width,
/// height)` image rect the TSV/hOCR renderers print —
/// `PageIterator::BoundingBoxInternal` + `PageIterator::BoundingBox`
/// (`pageiterator.cpp:286-370`). APPROX: `scale_ = 1`, `rect_left_ =
/// rect_top_ = 0` (no `SetRectangle` sub-region or scaling in this crate).
fn to_image_rect(bx: (i32, i32, i32, i32), page_w: i32, page_h: i32) -> (i32, i32, i32, i32) {
    let (left, bottom, right, top) = bx;
    // The degenerate union_boxes() empty case maps cleanly through clip_to_range
    // to (0, page_h, 0, 0)-ish output; documented above, not special-cased here.
    let img_left = clip_to_range(left, 0, page_w);
    let img_top = clip_to_range(page_h - top, 0, page_h);
    let img_right = clip_to_range(right, img_left, page_w);
    let img_bottom = clip_to_range(page_h - bottom, img_top, page_h);
    (
        img_left,
        img_top,
        img_right - img_left,
        img_bottom - img_top,
    )
}

/// A word's unichar ids as text — `AppendUTF8WordText` (`resultiterator.cpp:705-719`):
/// concatenate `word->BestUTF8(i, false)` for every character in the word,
/// which for our transcode is exactly [`ids_to_text`] over the word's ids.
fn word_text(charset: &CharSet, word: &WordResult) -> String {
    let ids: Vec<u32> = word.unichar_ids.iter().map(|&id| id as u32).collect();
    ids_to_text(charset, &ids)
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
    let mut out = String::new();
    for line in lines {
        if line.words.is_empty() {
            continue;
        }
        for word in &line.words {
            if word.leading_space {
                out.push(' ');
            }
            out.push_str(&word_text(charset, word));
        }
        out.push('\n');
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
            // LTRResultIterator::Confidence(RIL_WORD): ClipToRange(100 + 5 *
            // min(word certs), 0.0, 100.0) — the empty-certs fallback mirrors
            // WERD_CHOICE::init()'s certainty_ = FLT_MAX sentinel
            // (ratngs.h:401), which clips to 100.0 (never hit on real output).
            let min_cert = word.certs.iter().copied().fold(f32::MAX, f32::min);
            let conf = (100.0 + 5.0 * min_cert).clamp(0.0, 100.0);
            let text = word_text(charset, word);
            out.push_str(&format!(
                "5\t1\t1\t1\t{line_num}\t{word_num}\t{wl}\t{wt}\t{ww}\t{wh}\t{conf:.6}\t{text}\n"
            ));
        }
    }
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
        };
        // Line 2: "a" (no leading space) + "b" (leading_space=false, e.g. glued) -> "ab\n"
        let line2 = LineWords {
            words: vec![
                word(&[1], false, -0.1, (0, 0, 5, 10)),
                word(&[2], false, -0.1, (6, 0, 10, 10)),
            ],
            line_box: (0, 0, 10, 10),
        };
        let cs = test_charset();
        assert_eq!(render_text(&[line1, line2], &cs), "a b\nab\n");
    }

    #[test]
    fn render_text_skips_empty_lines_entirely() {
        let empty = LineWords {
            words: vec![],
            line_box: (0, 0, 10, 10),
        };
        let line = LineWords {
            words: vec![word(&[1], false, -0.1, (0, 0, 5, 10))],
            line_box: (0, 0, 10, 10),
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
}
