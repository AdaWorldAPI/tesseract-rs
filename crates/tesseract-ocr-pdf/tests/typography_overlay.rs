//! **Typography-fidelity + overlay CI** — the step-by-step checks the
//! measured-metrics work (doc.v1 `xheight`/`ascrise`/`descdrop`/`baseline`)
//! makes possible, each assessed against EXACT ground truth because the
//! fixture is generated (`corpus/gen/gen_resolution_grid.py` — authored
//! text, known font, known layout; see `resgrid.gt.json`).
//!
//! Runs on the CRISP cell only (cell 0 of the resolution grid) so the whole
//! target stays fast; the resolution *ladder* is the sibling quality fence
//! (`tesseract-ocr/tests/quality_resolution_grid.rs`).
//!
//! Step by step, what is assessed and against what:
//!
//! 1. **Font size** — measured `LineMetrics` per line: `xheight` must sit in
//!    the typographic band for the KNOWN 22px DejaVu Sans render, and
//!    `row_height = xheight + ascrise - descdrop` (the exact quantity real
//!    Tesseract sizes fonts from, `ltrresultiterator.cpp:168-172`) must
//!    bracket the known body height.
//! 2. **Type spacing** — consecutive measured baselines must reproduce the
//!    generator's line pitch (30px) within tight tolerance; per-line word
//!    counts must match the authored text (word segmentation = horizontal
//!    spacing read correctly).
//! 3. **Placement** — each measured baseline within a few px of the known
//!    baseline y; line bbox left edge at the known text x; first/last word
//!    x-spans near the known spans.
//! 4. **Original overlay** — build the searchable PDF from the SAME doc.v1
//!    words over the original raster and decode its content stream: every
//!    invisible text run's `Tm` must sit at its word's bbox coordinates
//!    (px == pt at 72 dpi), and every word bbox must actually cover INK in
//!    the original image (≥ 5 dark pixels) — the gross-misplacement
//!    falsifier that the historical 3× word-box compression would have
//!    failed loudly.

use std::path::Path;

use lopdf::content::Content;
use lopdf::Document;
use tesseract_core::DictLite;
use tesseract_ocr::LstmRecognizer;
use tesseract_ocr_pdf::{render_searchable_pdf, GreyImage, PageOcr, PlacedWord};

fn corpus() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

struct DocWordOut {
    text: String,
    bbox: (u32, u32, u32, u32), // top-down (l, t, r, b), cell-crop space
}

struct DocLineOut {
    bbox: (i64, i64, i64, i64),
    xheight: f64,
    ascrise: f64,
    descdrop: f64,
    baseline: f64,
    words: Vec<DocWordOut>,
}

#[test]
fn typography_and_overlay_match_generated_ground_truth() {
    let c = corpus();
    if !c.join("model/eng.lstm").exists() || !c.join("quality/resgrid.pgm").exists() {
        eprintln!("skipping: corpus model/quality fixtures absent");
        return;
    }
    let lstm = std::fs::read(c.join("model/eng.lstm")).unwrap();
    let uni = std::fs::read_to_string(c.join("model/eng.lstm-unicharset")).unwrap();
    let rec = std::fs::read(c.join("model/eng.lstm-recoder")).unwrap();
    let r = LstmRecognizer::from_components(&lstm, &uni, &rec).unwrap();
    let dict = match (
        std::fs::read(c.join("model/eng.lstm-word-dawg")),
        std::fs::read(c.join("model/eng.lstm-punc-dawg")),
        std::fs::read(c.join("model/eng.lstm-number-dawg")),
    ) {
        (Ok(w), Ok(p), Ok(n)) => DictLite::from_components(&w, &p, &n).ok(),
        _ => None,
    };

    // ── Crop the crisp cell 0 out of the grid ────────────────────────────
    let bytes = std::fs::read(c.join("quality/resgrid.pgm")).unwrap();
    let (grid, gw, _gh) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    let gt: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(c.join("quality/resgrid.gt.json")).unwrap())
            .unwrap();
    let (cw, ch) = (
        gt["cell_w"].as_u64().unwrap() as usize,
        gt["cell_h"].as_u64().unwrap() as usize,
    );
    let (cx, cy) = (
        gt["cells"][0]["x"].as_u64().unwrap() as usize,
        gt["cells"][0]["y"].as_u64().unwrap() as usize,
    );
    let mut cell = Vec::with_capacity(cw * ch);
    for y in cy..cy + ch {
        cell.extend_from_slice(&grid[y * gw + cx..y * gw + cx + cw]);
    }

    let doc = r
        .recognize_document(&cell, cw, ch, dict.as_ref(), None)
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&doc.json).unwrap();

    // Collect lines (any region kind), top-down order by bbox top.
    let mut lines: Vec<DocLineOut> = Vec::new();
    for region in v["pages"][0]["regions"].as_array().unwrap() {
        for line in region["lines"]
            .as_array()
            .map(|x| x.as_slice())
            .unwrap_or(&[])
        {
            let bb = line["bbox"].as_array().unwrap();
            let words = line["words"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| {
                    let wb = w["bbox"].as_array().unwrap();
                    DocWordOut {
                        text: w["text"].as_str().unwrap_or_default().to_string(),
                        bbox: (
                            wb[0].as_i64().unwrap().max(0) as u32,
                            wb[1].as_i64().unwrap().max(0) as u32,
                            wb[2].as_i64().unwrap().max(0) as u32,
                            wb[3].as_i64().unwrap().max(0) as u32,
                        ),
                    }
                })
                .collect();
            lines.push(DocLineOut {
                bbox: (
                    bb[0].as_i64().unwrap(),
                    bb[1].as_i64().unwrap(),
                    bb[2].as_i64().unwrap(),
                    bb[3].as_i64().unwrap(),
                ),
                xheight: line["xheight"].as_f64().expect("metrics emitted"),
                ascrise: line["ascrise"].as_f64().expect("metrics emitted"),
                descdrop: line["descdrop"].as_f64().expect("metrics emitted"),
                baseline: line["baseline"].as_f64().expect("metrics emitted"),
                words,
            });
        }
    }
    lines.sort_by_key(|l| l.bbox.1);

    let gt_lines = gt["lines"].as_array().unwrap();
    assert_eq!(
        lines.len(),
        gt_lines.len(),
        "must find exactly the {} authored lines",
        gt_lines.len()
    );

    let font_px = gt["font_px"].as_f64().unwrap(); // 22
    let pitch = gt["pitch"].as_f64().unwrap(); // 30
    let text_x = gt_lines[0]["x"].as_f64().unwrap(); // 10

    for (i, (line, gt_line)) in lines.iter().zip(gt_lines).enumerate() {
        // ── 1. Font size ────────────────────────────────────────────────
        // DejaVu Sans x-height ≈ 0.55 em; allow the measurement band
        // [0.40, 0.65]·font_px. row_height (ascender-to-descender ink) must
        // bracket the body height: [0.7, 1.2]·font_px.
        assert!(
            line.xheight >= 0.40 * font_px && line.xheight <= 0.65 * font_px,
            "line {i}: measured xheight {} outside [{}, {}]",
            line.xheight,
            0.40 * font_px,
            0.65 * font_px
        );
        let row_height = line.xheight + line.ascrise - line.descdrop;
        assert!(
            row_height >= 0.7 * font_px && row_height <= 1.2 * font_px,
            "line {i}: row_height {row_height} outside [{}, {}]",
            0.7 * font_px,
            1.2 * font_px
        );

        // ── 3. Placement ────────────────────────────────────────────────
        let gt_baseline = gt_line["baseline_y"].as_f64().unwrap();
        assert!(
            (line.baseline - gt_baseline).abs() <= 3.0,
            "line {i}: baseline {} vs ground truth {gt_baseline} (>3px off)",
            line.baseline
        );
        assert!(
            (line.bbox.0 as f64 - text_x).abs() <= 6.0,
            "line {i}: left edge {} vs known text x {text_x}",
            line.bbox.0
        );
        // Word segmentation == authored words (also part of step 2: spacing
        // read correctly means spaces split exactly the authored words).
        let gt_words = gt_line["words"].as_array().unwrap();
        assert_eq!(
            line.words.len(),
            gt_words.len(),
            "line {i}: word count {} vs authored {}",
            line.words.len(),
            gt_words.len()
        );
        // First/last word x-spans near the known spans.
        let first_x0 = gt_words[0]["x0"].as_f64().unwrap();
        let last_x1 = gt_words[gt_words.len() - 1]["x1"].as_f64().unwrap();
        assert!(
            (line.words[0].bbox.0 as f64 - first_x0).abs() <= 6.0,
            "line {i}: first word left {} vs known {first_x0}",
            line.words[0].bbox.0
        );
        assert!(
            (line.words.last().unwrap().bbox.2 as f64 - last_x1).abs() <= 8.0,
            "line {i}: last word right {} vs known {last_x1}",
            line.words.last().unwrap().bbox.2
        );
    }

    // ── 2. Type spacing: measured baseline pitch == generator pitch ─────
    for pair in lines.windows(2) {
        let d = pair[1].baseline - pair[0].baseline;
        assert!(
            (d - pitch).abs() <= 2.0,
            "baseline pitch {d} vs generator pitch {pitch} (>2px off)"
        );
    }

    // ── 4. Original overlay ─────────────────────────────────────────────
    // 4a. Every word bbox must cover real INK in the original raster.
    let all_words: Vec<&DocWordOut> = lines.iter().flat_map(|l| l.words.iter()).collect();
    for w in &all_words {
        let (l, t, rr, b) = w.bbox;
        let mut dark = 0usize;
        for y in (t as usize)..(b as usize).min(ch) {
            for x in (l as usize)..(rr as usize).min(cw) {
                if cell[y * cw + x] < 128 {
                    dark += 1;
                }
            }
        }
        assert!(
            dark >= 5,
            "word {:?} bbox {:?} covers almost no ink ({dark} dark px) — \
             the overlay is not over the original glyphs",
            w.text,
            w.bbox
        );
    }

    // 4b. The searchable PDF's invisible runs sit at the word bboxes:
    // Tm.x == bbox.left, Tm.y == page_h - bbox.bottom (px == pt at 72 dpi).
    let page = PageOcr {
        grey: GreyImage {
            data: cell.clone(),
            w: cw,
            h: ch,
        },
        words: all_words
            .iter()
            .map(|w| PlacedWord {
                text: w.text.clone(),
                box_: w.bbox,
            })
            .collect(),
    };
    let (pdf, _report) = render_searchable_pdf(&[page], 72).unwrap();
    let docp = Document::load_mem(&pdf).unwrap();
    let pages = docp.get_pages();
    let &page_id = pages.get(&1).unwrap();
    let content = Content::decode(&docp.get_page_content(page_id).unwrap()).unwrap();

    let num = |o: &lopdf::Object| -> f64 {
        o.as_float()
            .map(f64::from)
            .or_else(|_| o.as_i64().map(|v| v as f64))
            .unwrap()
    };
    let mut tms: Vec<(f64, f64)> = Vec::new();
    for op in &content.operations {
        if op.operator == "Tm" {
            tms.push((num(&op.operands[4]), num(&op.operands[5])));
        }
    }
    // One Tm per non-degenerate word, in emission order == doc.v1 order.
    assert_eq!(
        tms.len(),
        all_words.len(),
        "expected one Tm per word in the invisible layer"
    );
    for (w, (tx, ty)) in all_words.iter().zip(&tms) {
        let (l, _, _, b) = w.bbox;
        assert!(
            (tx - f64::from(l)).abs() <= 0.51,
            "word {:?}: Tm.x {tx} vs bbox left {l}",
            w.text
        );
        let expect_y = ch as f64 - f64::from(b);
        assert!(
            (ty - expect_y).abs() <= 0.51,
            "word {:?}: Tm.y {ty} vs page_h - bbox bottom {expect_y}",
            w.text
        );
    }
}
