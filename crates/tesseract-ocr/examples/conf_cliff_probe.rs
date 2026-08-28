//! Per-CELL confidence (mean/min/max) alongside the already-pinned per-cell
//! CER on `resgrid.pgm`, to test whether LSTM confidence degrades smoothly
//! or cliffs — independent of whether CER itself has already cliffed.
//! Reuses `quality_resolution_grid.rs`'s exact cell bucketing so the two
//! numbers line up on the identical cell definitions.
//!
//! **This is the source of record for
//! `tesseract-paperless::consistency::LOW_CONFIDENCE_THRESHOLD`'s doc
//! comment** (cell 12's floor, cell 15's mean). Re-run and re-derive both
//! if the model, fixture, or ladder change — do not hand-edit the numbers
//! in that doc comment without re-running this probe.
//!
//! Finding: confidence cliffs BEFORE CER does. Cell 13 (min_conf 93.09) is
//! still perfectly recognized (CER 0.000) — the LSTM is measurably less
//! certain before it is measurably wrong.

use std::path::Path;
use tesseract_core::DictLite;
use tesseract_ocr::LstmRecognizer;

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn main() {
    let c = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
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

    let gt_text: String = gt["lines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["text"].as_str().unwrap())
        .collect::<Vec<_>>()
        .join(" ");

    let cells = gt["cells"].as_array().unwrap();
    let (cw, ch) = (
        gt["cell_w"].as_i64().unwrap(),
        gt["cell_h"].as_i64().unwrap(),
    );
    // (text, conf) per word, per cell.
    let mut per_cell: Vec<Vec<(String, f32)>> = vec![Vec::new(); cells.len()];
    for page in v["pages"].as_array().unwrap() {
        for region in page["regions"].as_array().unwrap() {
            for line in region["lines"]
                .as_array()
                .map(|x| x.as_slice())
                .unwrap_or(&[])
            {
                for word in line["words"]
                    .as_array()
                    .map(|x| x.as_slice())
                    .unwrap_or(&[])
                {
                    let bb = word["bbox"].as_array().unwrap();
                    let cx = (bb[0].as_i64().unwrap() + bb[2].as_i64().unwrap()) / 2;
                    let cy = (bb[1].as_i64().unwrap() + bb[3].as_i64().unwrap()) / 2;
                    let conf = word["conf"].as_f64().unwrap_or(0.0) as f32;
                    for (i, cell) in cells.iter().enumerate() {
                        let (x, y) = (cell["x"].as_i64().unwrap(), cell["y"].as_i64().unwrap());
                        if cx >= x && cx < x + cw && cy >= y && cy < y + ch {
                            per_cell[i].push((
                                word["text"].as_str().unwrap_or_default().to_string(),
                                conf,
                            ));
                        }
                    }
                }
            }
        }
    }

    println!(
        "{:>4} {:>9} {:>9} {:>9} {:>7} {:>6}  words",
        "cell", "mean_conf", "min_conf", "max_conf", "CER", "nwords"
    );
    for (i, words) in per_cell.iter().enumerate() {
        let got = words
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let cer = levenshtein(&gt_text, &got) as f64 / gt_text.chars().count() as f64;
        if words.is_empty() {
            println!(
                "{i:>4} {:>9} {:>9} {:>9} {cer:>7.3} {:>6}  (empty)",
                "-", "-", "-", 0
            );
            continue;
        }
        let confs: Vec<f32> = words.iter().map(|(_, c)| *c).collect();
        let mean = confs.iter().sum::<f32>() / confs.len() as f32;
        let min = confs.iter().cloned().fold(f32::MAX, f32::min);
        let max = confs.iter().cloned().fold(f32::MIN, f32::max);
        println!(
            "{i:>4} {mean:>9.2} {min:>9.2} {max:>9.2} {cer:>7.3} {:>6}",
            confs.len()
        );
    }
}
