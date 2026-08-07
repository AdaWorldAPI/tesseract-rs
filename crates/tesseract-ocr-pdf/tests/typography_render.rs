//! Axis C — the RENDER/TYPOGRAPHY contract, measured on the structured PDF's
//! own content stream.
//!
//! # Why this file exists
//!
//! Before it, the only integration typography fence was
//! `typography_overlay.rs`, and it has two structural blind spots:
//!
//! 1. **It asserts `Tm` only.** It reads the rendered content stream for
//!    placement and never looks at `Tf` (font size) or `Tz` (horizontal glyph
//!    scale) at all. The worst typography defect this repo has shipped —
//!    `Tz` ranging **32 % to 850 %**, letterforms alternately crushed to a
//!    third and stretched eightfold — would return tomorrow and no test would
//!    notice.
//! 2. **Its fixture cannot express the failure.** It runs on cell 0 of
//!    `resgrid`, whose ground truth is `font_px = 22` against `pitch = 30`
//!    (ratio 1.36 — generous leading). The page the defect was found on ran
//!    ~19 px pitch. `CLAUDE.md` says so in as many words: *"The committed
//!    quality fixture is structurally incapable of expressing this bug."*
//!
//! The formulas (`page_font_px`, `page_pitch_px`, `classify_justification`,
//! `RunFit`) do have unit tests — on hand-built bboxes, inside `layout.rs`.
//! What was never guarded is the **rendered result on a page where the
//! failure is expressible**. That is this file.
//!
//! # What it measures
//!
//! A tight-set multi-column page is generated (leading ≈ 1.07 × glyph height —
//! the regime where `Tf` overshoot becomes visible overlap), recognized,
//! reconstructed with [`doc_v1_layout`], rendered with [`render_pdf`], and the
//! resulting content stream is parsed:
//!
//! - **`Tz` ≡ 100 on every painted run** — painted text is never
//!   glyph-distorted. Two-sided: the assertion also proves runs were found, so
//!   it cannot pass by measuring an empty set.
//! - **No line overlap** — no painted run may set `Tf` larger than the
//!   measured baseline pitch. This is the direct expression of the original
//!   defect (`Tf`/pitch median was 1.44, and 95 % of consecutive pairs
//!   overlapped).
//! - **Column gutters stay even** — the operator's own acceptance rule:
//!   readers tolerate tight or variant leading far better than uneven
//!   horizontal gutters. Measured as the spread of per-column right edges.
//!
//! # Footing
//!
//! **NOT a byte-parity transcode.** There is no C++ side to diff against —
//! libtesseract has no structured-PDF renderer of this shape. This is a
//! quality fence over generated ground truth, the same footing as
//! `quality_resolution_grid.rs` and `typography_overlay.rs`. Every threshold
//! is a **pinned observation**; if the typography math changes on purpose the
//! numbers get re-measured and re-pinned, not defended.
//!
//! # Every assertion here is disable-verified — and the first two disables were WRONG
//!
//! | assertion | disable that turns it red |
//! |---|---|
//! | `Tz` ≡ 100 | `RunFit::Natural` returns a box-fit stretch |
//! | no overlap | `page_font_px` returns `× 2.0` |
//! | even gutters | the per-block `dx = left` translation is dropped |
//!
//! Two earlier disables passed and would have shipped a vacuous fence:
//!
//! - `PITCH_TO_FONT_PX 0.80 → 1.60` changed **nothing**, because
//!   `page_font_px` takes `min(width_solve, MAX_FONT_TO_PITCH × pitch)` and on
//!   this fixture the **width solve binds** (14.35 pt against a 15.2 pt
//!   ceiling). Nor did lifting `MAX_FONT_TO_PITCH` itself. Only inflating the
//!   final value reaches the assertion.
//! - Making `RunFit::Natural` stretch left the `Tz` test green, because the
//!   JUSTIFIED fixture never takes that branch — and `JustifyToBox` returns
//!   `Tz = 100` unconditionally (it justifies through `Tw`), so on justified
//!   text the assertion is a structural tautology. Hence
//!   [`tight_set_page`]'s `justified` flag: the `Tz` test runs the **ragged**
//!   variant, the only painted path where `Tz` could ever be non-100.
//!
//! **Turning a knob that does not bind is not a disable.** Both misses looked
//! like confirmations.

#![allow(clippy::items_after_statements, reason = "test-local geometry helpers")]

use lopdf::content::Content;
use lopdf::Document;
use tesseract_ocr::LstmRecognizer;
use tesseract_ocr_pdf::layout::{doc_v1_layout, render_pdf};
use tesseract_ocr_pdf::GreyImage;

fn corpus() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

fn load() -> Option<LstmRecognizer> {
    let c = corpus();
    if !c.join("model/eng.lstm").exists() {
        eprintln!("skipping: corpus model absent");
        return None;
    }
    LstmRecognizer::from_components(
        &std::fs::read(c.join("model/eng.lstm")).unwrap(),
        &std::fs::read_to_string(c.join("model/eng.lstm-unicharset")).unwrap(),
        &std::fs::read(c.join("model/eng.lstm-recoder")).unwrap(),
    )
    .ok()
}

/// Ground truth for [`tight_set_page`], in image pixels.
struct Truth {
    w: usize,
    h: usize,
    /// Baseline-to-baseline distance.
    pitch: usize,
    /// Ink height of one text row.
    glyph_h: usize,
    /// Left edge of each column's text.
    col_lefts: Vec<usize>,
    /// Right edge of each column's text — identical for every line in a
    /// column (justified), which is what makes gutters measurable.
    col_rights: Vec<usize>,
}

/// A hollow glyph-sized mark. Hollow, not solid: `filter_blobs` rejects
/// `pixel_count >= h*w*0.7` as "too dense to be text", which silently drops
/// the whole pool's line-size estimate to 0 (the documented `rectify.rs`
/// fixture trap).
fn mark(page: &mut [u8], w: usize, x: usize, y: usize, mw: usize, mh: usize) {
    for dy in 0..mh {
        for dx in 0..mw {
            if dy == 0 || dy == mh - 1 || dx == 0 || dx == mw - 1 {
                page[(y + dy) * w + (x + dx)] = 0;
            }
        }
    }
}

/// A **tight-set, justified, 3-column** page.
///
/// The geometry is the point. `resgrid` is `font_px 22 / pitch 30` = 1.36 and
/// therefore cannot overlap however wrong `Tf` gets; here the glyph height is
/// 15 px on a 16 px pitch (**1.07**), so any `Tf` overshoot above ~7 %
/// immediately collides with the neighbouring line — the regime the original
/// defect lived in.
///
/// Justified: every line in a column ends at exactly the same right edge, so
/// the per-column right edges are a known constant and gutter evenness is
/// directly measurable. The LAST line of each column is deliberately short —
/// real Blocksatz never stretches its last line, and a classifier that
/// includes it is the classic false negative.
fn tight_set_page(justified: bool) -> (Vec<u8>, Truth) {
    const COLS: usize = 3;
    const ROWS: usize = 14;
    const GLYPH_W: usize = 7;
    const GLYPH_H: usize = 15;
    const ADVANCE: usize = 10; // 7 px glyph + 3 px sidebearing
    const PITCH: usize = 16; // 15 px ink on a 16 px pitch => 1.07
    const MARKS_PER_LINE: usize = 18;
    const COL_W: usize = MARKS_PER_LINE * ADVANCE; // 180
    const GUTTER: usize = 40;
    const MARGIN: usize = 30;

    let w = MARGIN * 2 + COLS * COL_W + (COLS - 1) * GUTTER;
    let h = MARGIN * 2 + ROWS * PITCH;
    let mut page = vec![255u8; w * h];

    let mut col_lefts = Vec::with_capacity(COLS);
    let mut col_rights = Vec::with_capacity(COLS);
    for c in 0..COLS {
        let cx = MARGIN + c * (COL_W + GUTTER);
        col_lefts.push(cx);
        // Justified: every line but the last ends at the same measure.
        col_rights.push(cx + (MARKS_PER_LINE - 1) * ADVANCE + GLYPH_W);
        for row in 0..ROWS {
            let y = MARGIN + row * PITCH;
            // Last line short — real Blocksatz never stretches it.
            let n = match (justified, row == ROWS - 1) {
                // Justified: every line but the last ends at the measure.
                (true, true) => MARKS_PER_LINE / 2,
                (true, false) => MARKS_PER_LINE,
                _ => {
                    // Ragged: line lengths vary, so `classify_justification`
                    // returns Flattersatz and the renderer takes `RunFit::Natural`
                    // — the ONLY painted path that could ever apply `Tz`, and
                    // therefore the only one on which the Tz assertion can be
                    // falsified at all.
                    MARKS_PER_LINE - (row * 3) % 7
                }
            };
            for i in 0..n {
                mark(&mut page, w, cx + i * ADVANCE, y, GLYPH_W, GLYPH_H);
            }
        }
    }

    (
        page,
        Truth {
            w,
            h,
            pitch: PITCH,
            glyph_h: GLYPH_H,
            col_lefts,
            col_rights,
        },
    )
}

/// Every painted (`0 Tr`) text run's typography, read from the content stream.
struct PaintedRun {
    tf: f64,
    tz: f64,
    tm_x: f64,
    tm_y: f64,
}

fn painted_runs(pdf: &[u8]) -> Vec<PaintedRun> {
    let doc = Document::load_mem(pdf).unwrap();
    let pages = doc.get_pages();
    let &page_id = pages.get(&1).unwrap();
    let content = Content::decode(&doc.get_page_content(page_id).unwrap()).unwrap();

    let num = |o: &lopdf::Object| -> f64 {
        o.as_float()
            .map(f64::from)
            .or_else(|_| o.as_i64().map(|v| v as f64))
            .unwrap_or(0.0)
    };

    // The PDF text state is PERSISTENT across runs, so each operator's value
    // carries forward until re-set. Tracking it (rather than reading only the
    // operands adjacent to a Tj) is what makes a leaked Tz or a never-reset Tw
    // visible instead of invisible.
    let (mut tf, mut tz, mut tr) = (0.0f64, 100.0f64, 0.0f64);
    let (mut tx, mut ty) = (0.0f64, 0.0f64);
    let mut out = Vec::new();
    for op in &content.operations {
        match op.operator.as_str() {
            "Tf" => tf = num(&op.operands[1]),
            "Tz" => tz = num(&op.operands[0]),
            "Tr" => tr = num(&op.operands[0]),
            "Tm" => {
                tx = num(&op.operands[4]);
                ty = num(&op.operands[5]);
            }
            // `0 Tr` == painted; the invisible searchable layer is `3 Tr`
            // and is deliberately NOT measured here (it keeps `Tz` by design).
            "Tj" | "TJ" if tr.abs() < 0.5 => out.push(PaintedRun {
                tf,
                tz,
                tm_x: tx,
                tm_y: ty,
            }),
            _ => {}
        }
    }
    out
}

fn render(truth: &Truth, page: &[u8]) -> Option<Vec<u8>> {
    let r = load()?;
    let doc = r
        .recognize_document(page, truth.w, truth.h, None, None)
        .expect("recognize");
    let raster = GreyImage {
        data: page.to_vec(),
        w: truth.w,
        h: truth.h,
    };
    let layout = doc_v1_layout(&doc.json, &[raster]).expect("doc.v1 -> layout");
    let (pdf, _report) = render_pdf(&layout).expect("render");
    Some(pdf)
}

#[test]
fn painted_text_is_never_glyph_distorted() {
    let (page, truth) = tight_set_page(false);
    let Some(pdf) = render(&truth, &page) else {
        return;
    };
    let runs = painted_runs(&pdf);
    assert!(
        runs.len() >= 10,
        "fixture must produce painted runs to measure ({} found)",
        runs.len()
    );
    let bad: Vec<f64> = runs
        .iter()
        .map(|r| r.tz)
        .filter(|tz| (tz - 100.0).abs() > 0.01)
        .collect();
    assert!(
        bad.is_empty(),
        "painted text must render at Tz=100; {} of {} runs distorted (e.g. {:?})",
        bad.len(),
        runs.len(),
        &bad[..bad.len().min(6)]
    );
}

#[test]
fn no_painted_line_overlaps_its_neighbour() {
    let (page, truth) = tight_set_page(true);
    let Some(pdf) = render(&truth, &page) else {
        return;
    };
    let runs = painted_runs(&pdf);
    assert!(runs.len() >= 10, "fixture must produce painted runs");

    // Baseline pitch AS RENDERED: the spread of distinct Tm.y values within a
    // column. Measured from the output, not assumed from the fixture, so the
    // assertion survives a deliberate scale change.
    let mut ys: Vec<f64> = runs.iter().map(|r| r.tm_y).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ys.dedup_by(|a, b| (*a - *b).abs() < 0.5);
    let mut deltas: Vec<f64> = ys.windows(2).map(|p| p[1] - p[0]).collect();
    deltas.retain(|d| *d > 1.0);
    assert!(
        deltas.len() >= 5,
        "need several distinct baselines to measure a pitch ({} found)",
        deltas.len()
    );
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pitch = deltas[deltas.len() / 2];

    // Apparatus check: px == pt at 72 dpi, so the pitch read out of the
    // content stream must be the pitch that was DRAWN. Without this the
    // overlap assertion below could pass by measuring the wrong quantity —
    // a null result is a claim about the apparatus until proven otherwise.
    assert!(
        (pitch - truth.pitch as f64).abs() <= 2.0,
        "measured baseline pitch {pitch:.2} pt should match the drawn {} px \
         (px == pt at 72 dpi) — the reader is not reading what was rendered",
        truth.pitch
    );

    let over: Vec<f64> = runs.iter().map(|r| r.tf).filter(|tf| *tf > pitch).collect();
    assert!(
        over.is_empty(),
        "no painted run may be taller than the {pitch:.2} pt baseline pitch; \
         {} of {} overshoot (e.g. {:?}) — this is the Tf/pitch 1.44 defect",
        over.len(),
        runs.len(),
        &over[..over.len().min(6)]
    );
}

#[test]
fn column_gutters_stay_even_across_the_page() {
    let (page, truth) = tight_set_page(true);
    let Some(pdf) = render(&truth, &page) else {
        return;
    };
    let runs = painted_runs(&pdf);
    assert!(runs.len() >= 10, "fixture must produce painted runs");

    // Cluster run origins into columns by the fixture's known left edges.
    let mut per_col: Vec<Vec<f64>> = vec![Vec::new(); truth.col_lefts.len()];
    for r in &runs {
        let (mut best, mut bd) = (0usize, f64::MAX);
        for (i, &l) in truth.col_lefts.iter().enumerate() {
            let d = (r.tm_x - l as f64).abs();
            if d < bd {
                bd = d;
                best = i;
            }
        }
        per_col[best].push(r.tm_x);
    }
    let populated = per_col.iter().filter(|c| !c.is_empty()).count();
    assert!(
        populated >= 2,
        "gutter evenness needs at least two populated columns, got {populated}"
    );

    // Each column's runs must start at that column's known left edge. A
    // column whose text creeps rightwards is exactly the uneven-gutter
    // complaint: "inter-column gutters varied 6-12% with text length".
    let mut origins: Vec<f64> = Vec::new();
    for (i, xs) in per_col.iter().enumerate() {
        if xs.is_empty() {
            continue;
        }
        let want = truth.col_lefts[i] as f64;
        let worst = xs.iter().map(|x| (x - want).abs()).fold(0.0f64, f64::max);
        assert!(
            worst <= truth.glyph_h as f64,
            "column {i} runs must start at its measure ({want}); worst drift {worst:.1} px"
        );
        let mut v = xs.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        origins.push(v[v.len() / 2]);
    }

    // The gutters themselves. A column's justified right edge is a known
    // constant, so every gutter on this page is the SAME width by
    // construction: `col_lefts[i+1] - col_rights[i]`. What the render must
    // preserve is therefore the SPACING between column origins — and the
    // operator's acceptance rule is that unevenness here is what readers
    // actually notice, ahead of tight leading.
    let expected_gutters: Vec<usize> = truth
        .col_rights
        .iter()
        .zip(truth.col_lefts.iter().skip(1))
        .map(|(r, l)| l - r)
        .collect();
    let expected_spacing: Vec<f64> = truth
        .col_lefts
        .windows(2)
        .map(|p| (p[1] - p[0]) as f64)
        .collect();
    let measured: Vec<f64> = origins.windows(2).map(|p| p[1] - p[0]).collect();

    assert_eq!(
        measured.len(),
        expected_spacing.len(),
        "every column must be populated to judge gutter evenness"
    );
    for (i, (m, e)) in measured.iter().zip(&expected_spacing).enumerate() {
        assert!(
            (m - e).abs() <= 4.0,
            "column {i}->{} spacing {m:.1} px vs drawn {e:.1} px \
             (gutters {expected_gutters:?})",
            i + 1
        );
    }
    let lo = measured.iter().cloned().fold(f64::MAX, f64::min);
    let hi = measured.iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        hi - lo <= 4.0,
        "gutters must stay even; measured spacings {measured:?}, spread {:.1} px",
        hi - lo
    );
}
