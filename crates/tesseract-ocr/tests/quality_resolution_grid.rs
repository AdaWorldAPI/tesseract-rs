//! **The 8+7+0 quality fence** — a CI regression gate on end-to-end
//! recognition quality across a resolution ladder, scored by **Levenshtein
//! character error rate** (the metric the published on-device-OCR
//! comparisons use; a bag-of-words score cannot see degradation *inside* a
//! word, where the print-trained LSTM's characteristic failure is confident
//! confusable substitution).
//!
//! Fixture: `corpus/quality/resgrid.pgm` + `resgrid.gt.json`, generated
//! deterministically by `corpus/gen/gen_resolution_grid.py` (authored text,
//! DejaVu, license-clean — mirrors the public "resolution testset" sheet's
//! SHAPE without depending on any third-party image). 8x2 grid of the same
//! paragraph at descending effective resolution; constant geometry, so the
//! sidecar carries exact ground truth.
//!
//! **On temporal context.** The LSTM is recurrent, so a character's
//! recognition depends on what precedes it *within its line* — cell scores
//! are only comparable because every cell is recognized from its OWN crop
//! (the block-aware path segments the grid into per-cell blocks first), so
//! no cell inherits another cell's hidden state. The ladder therefore
//! measures resolution alone, not read-order luck. If the grid were read as
//! full-width rows spanning all 8 columns (the pre-`recognize_page_blocks_words`
//! behaviour), a sharp left cell's context would carry INTO a blurred right
//! cell and mask its true difficulty — which is also why the line-count
//! assertion below is part of the fence, not a nicety.
//!
//! The pinned pattern (measured, then fenced) — **8 + 7 + 0**:
//!
//! - **top row, all 8 cells:** CER <= 0.05 (measured 0.000),
//! - **bottom row, 7 of 8 (cells 8-14):** CER <= 0.05 (measured 0.000
//!   through 0.023 — the ladder's last legible rung),
//! - **cell 15: DEAD** — CER >= 0.5 (measured 0.814: it decodes to
//!   confident garbage, the documented confusable failure mode). This bound
//!   is two-sided ON PURPOSE: if the engine ever climbs past this cliff the
//!   test fails UPWARD and tells us to re-pin the ladder, exactly like a
//!   byte-parity anchor going stale.

use std::path::Path;

use tesseract_core::DictLite;
use tesseract_ocr::LstmRecognizer;

fn corpus() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

#[test]
fn resolution_grid_holds_the_8_7_0_pattern() {
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

    let bytes = std::fs::read(c.join("quality/resgrid.pgm")).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    let gt: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(c.join("quality/resgrid.gt.json")).unwrap())
            .unwrap();

    let doc = r
        .recognize_document(&grey, w, h, dict.as_ref(), None)
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&doc.json).unwrap();

    // Ground-truth text (identical for every cell).
    let gt_text: String = gt["lines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["text"].as_str().unwrap())
        .collect::<Vec<_>>()
        .join(" ");

    // Bucket recognized words per cell by bbox centre.
    let cells = gt["cells"].as_array().unwrap();
    let (cw, ch) = (
        gt["cell_w"].as_i64().unwrap(),
        gt["cell_h"].as_i64().unwrap(),
    );
    let mut per_cell: Vec<Vec<String>> = vec![Vec::new(); cells.len()];
    let mut line_count = 0usize;
    for page in v["pages"].as_array().unwrap() {
        for region in page["regions"].as_array().unwrap() {
            for line in region["lines"]
                .as_array()
                .map(|x| x.as_slice())
                .unwrap_or(&[])
            {
                line_count += 1;
                for word in line["words"]
                    .as_array()
                    .map(|x| x.as_slice())
                    .unwrap_or(&[])
                {
                    let bb = word["bbox"].as_array().unwrap();
                    let cx = (bb[0].as_i64().unwrap() + bb[2].as_i64().unwrap()) / 2;
                    let cy = (bb[1].as_i64().unwrap() + bb[3].as_i64().unwrap()) / 2;
                    for (i, cell) in cells.iter().enumerate() {
                        let (x, y) = (cell["x"].as_i64().unwrap(), cell["y"].as_i64().unwrap());
                        if cx >= x && cx < x + cw && cy >= y && cy < y + ch {
                            per_cell[i]
                                .push(word["text"].as_str().unwrap_or_default().to_string());
                        }
                    }
                }
            }
        }
    }

    // Character error rate per cell (Levenshtein / ground-truth length).
    let cer: Vec<f64> = per_cell
        .iter()
        .map(|ws| {
            let got = ws.join(" ");
            levenshtein(&gt_text, &got) as f64 / gt_text.chars().count() as f64
        })
        .collect();
    eprintln!(
        "per-cell CER: {}",
        cer.iter()
            .map(|c| format!("{c:.3}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // Multi-column reading order: 16 cells × 6 lines each. A merged
    // across-the-gutter reading would produce ~12 full-width lines instead.
    assert!(
        line_count >= 44,
        "expected ~48 per-cell lines (16 cells x 3), got {line_count} \
         — full-width merged reading detected"
    );

    // ── The 8+7+0 pattern ──────────────────────────────────────────────
    const LEGIBLE: f64 = 0.05;
    const DEAD: f64 = 0.5;
    // 8: the whole top row.
    for (i, e) in cer.iter().enumerate().take(8) {
        assert!(
            *e <= LEGIBLE,
            "top-row cell {i} regressed: CER {e:.3} > {LEGIBLE}"
        );
    }
    // 7: bottom row cells 8-14 (the ladder's last legible rung).
    for (i, e) in cer.iter().enumerate().take(15).skip(8) {
        assert!(
            *e <= LEGIBLE,
            "bottom-row cell {i} regressed: CER {e:.3} > {LEGIBLE}"
        );
    }
    // 0: cell 15 is past the cliff. Two-sided (see the module docs).
    assert!(
        cer[15] >= DEAD,
        "cell 15 unexpectedly legible: CER {:.3} < {DEAD} — the engine \
         improved past the pinned cliff; re-tune LADDER in \
         corpus/gen/gen_resolution_grid.py and re-pin this fence",
        cer[15]
    );
}

/// Levenshtein edit distance (two-row DP) — the CER numerator.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}
