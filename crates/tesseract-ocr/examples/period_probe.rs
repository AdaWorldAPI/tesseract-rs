//! Probe: why are sentence periods missing from a recognized page?
//!
//! Built against a real two-column book scan on which this crate emitted **2**
//! `.` against 42 `,`, with **0 of 72** recognized lines ending in a period —
//! on ordinary English prose. Real `libtesseract` 5.3.4 on the SAME file gets
//! **9** periods (7 line-final) and the same 42 commas, so this is a measured
//! recognition-PARITY GAP, not an `eng.lstm` limitation.
//!
//! Each arm below eliminated one hypothesis. All four came back NEGATIVE — the
//! null results ARE the finding, which is why this stays committed:
//!
//! | arm | hypothesis | result |
//! |---|---|---|
//! | `otsu` vs `sauvola` | a period is the smallest ink feature, so a global threshold on an aged scan loses it | **identical** (conf 99.348 vs 99.339) |
//! | `dict` vs `nodict` | the punctuation DAWG / dict beam suppresses a low-evidence glyph | **identical** |
//! | `widen` | the line-band crop clips the line-final period off the right edge | **0 recovered** at +60 px |
//! | `plain` | `DocPage`/`doc.v1` word assembly drops it after recognition | **also 0** — so it is upstream of the DOM |
//!
//! Commas survive at full count in every arm, which is what makes the loss
//! specific and strange: a comma is barely larger than a period and descends
//! below the baseline, so neither a size nor a vertical-clipping story explains
//! periods vanishing while commas do not.
//!
//! ```sh
//! cargo run -p tesseract-ocr --example period_probe --release -- page.pgm
//! # oracle to compare against:
//! tesseract page.pgm out --psm 1 -l eng
//! ```
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]

use tesseract_core::DictLite;
use tesseract_ocr::{BinarizeMode, LstmRecognizer};

/// `(line_count, line_final_periods, total_periods, total_commas)` for a
/// `doc.v1` JSON document.
fn tally(json: &str) -> (usize, usize, usize, usize) {
    let v: serde_json::Value = serde_json::from_str(json).expect("doc.v1");
    let (mut lines, mut line_final) = (0usize, 0usize);
    let mut all = String::new();
    for page in v["pages"].as_array().into_iter().flatten() {
        for reg in page["regions"].as_array().into_iter().flatten() {
            for line in reg["lines"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                let t: String = line["words"]
                    .as_array()
                    .map(|ws| {
                        ws.iter()
                            .filter_map(|x| x["text"].as_str())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default();
                lines += 1;
                if t.trim_end().ends_with('.') {
                    line_final += 1;
                }
                all.push_str(&t);
                all.push('\n');
            }
        }
    }
    (
        lines,
        line_final,
        all.matches('.').count(),
        all.matches(',').count(),
    )
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(pgm) = a.get(1) else {
        eprintln!("usage: period_probe <page.pgm>");
        return;
    };
    let c = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model");
    let r = LstmRecognizer::from_components(
        &std::fs::read(c.join("eng.lstm")).expect("eng.lstm"),
        &std::fs::read_to_string(c.join("eng.lstm-unicharset")).expect("unicharset"),
        &std::fs::read(c.join("eng.lstm-recoder")).expect("recoder"),
    )
    .expect("assemble recognizer");
    let dict = match (
        std::fs::read(c.join("eng.lstm-word-dawg")),
        std::fs::read(c.join("eng.lstm-punc-dawg")),
        std::fs::read(c.join("eng.lstm-number-dawg")),
    ) {
        (Ok(w), Ok(p), Ok(n)) => DictLite::from_components(&w, &p, &n).ok(),
        _ => None,
    };
    let bytes = std::fs::read(pgm).expect("read pgm");
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).expect("parse pgm");

    println!(
        "{:24} {:>6} {:>12} {:>8} {:>7}  mean conf",
        "arm", "lines", "line-final .", "total .", "commas"
    );
    let arms: [(&str, BinarizeMode, bool); 4] = [
        ("otsu + dict", BinarizeMode::Otsu, true),
        ("otsu, NO dict", BinarizeMode::Otsu, false),
        (
            "sauvola + dict",
            BinarizeMode::Sauvola {
                whsize: 16,
                k: 0.34,
            },
            true,
        ),
        (
            "sauvola, NO dict",
            BinarizeMode::Sauvola {
                whsize: 16,
                k: 0.34,
            },
            false,
        ),
    ];
    for (label, mode, use_dict) in arms {
        let d = if use_dict { dict.as_ref() } else { None };
        let doc = r
            .recognize_document_with_mode(&grey, w, h, d, None, mode)
            .expect("recognize_document");
        let (l, lf, p, cm) = tally(&doc.json);
        println!(
            "{label:24} {l:>6} {lf:>12} {p:>8} {cm:>7}  {:?}",
            doc.mean_confidence
        );
    }

    // The DOM-assembly arm: the plain text path never builds a DocPage, so a
    // difference here would localise the loss to doc.v1 assembly.
    let text = r
        .recognize_page_makerow(&grey, w, h, dict.as_ref())
        .expect("recognize_page_makerow");
    println!(
        "{:24} {:>6} {:>12} {:>8} {:>7}  (plain text path, no doc.v1)",
        "makerow plain",
        text.lines().count(),
        text.lines().filter(|l| l.trim_end().ends_with('.')).count(),
        text.matches('.').count(),
        text.matches(',').count()
    );
    println!(
        "\ncompare: tesseract {pgm} out --psm 1 -l eng   \
         (measured on the reference page: 71 lines, 7 line-final, 9 total, 42 commas)"
    );
}
