//! Probe: count trailing periods in recognized lines, Otsu vs Sauvola.
//!
//! Built to test — and DISPROVE — the hypothesis that the missing sentence
//! periods on a real two-column scan were a binarization loss (a period is the
//! smallest ink feature on a page, so a global Otsu threshold on an aged scan
//! was the obvious suspect). Measured on that page: **Otsu and Sauvola are
//! identical** — 0 of 72 lines end in a period under BOTH, mean confidence
//! 99.348 vs 99.339. The adaptive binarizer that rescues faded pages elsewhere
//! in this crate does nothing here, so the loss is downstream of thresholding.
//!
//! Kept committed because the null result is the finding: it closes off the
//! binarization branch of the search, and re-running it is how the next
//! session confirms that still holds. See `CLAUDE.md`'s "Two findings from
//! that page that are NOT these bugs" for the open state.
#![allow(clippy::print_stdout, reason = "diagnostic CLI")]
use tesseract_core::DictLite;
use tesseract_ocr::{BinarizeMode, LstmRecognizer};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let c = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/model");
    let r = LstmRecognizer::from_components(
        &std::fs::read(c.join("eng.lstm")).unwrap(),
        &std::fs::read_to_string(c.join("eng.lstm-unicharset")).unwrap(),
        &std::fs::read(c.join("eng.lstm-recoder")).unwrap(),
    )
    .unwrap();
    let dict = match (
        std::fs::read(c.join("eng.lstm-word-dawg")),
        std::fs::read(c.join("eng.lstm-punc-dawg")),
        std::fs::read(c.join("eng.lstm-number-dawg")),
    ) {
        (Ok(w), Ok(p), Ok(n)) => DictLite::from_components(&w, &p, &n).ok(),
        _ => None,
    };
    let bytes = std::fs::read(&a[1]).unwrap();
    let (grey, w, h) = tesseract_ocr::image_input::parse_pgm(&bytes).unwrap();
    for (label, mode) in [
        ("otsu", BinarizeMode::Otsu),
        (
            "sauvola",
            BinarizeMode::Sauvola {
                whsize: 16,
                k: 0.34,
            },
        ),
    ] {
        let doc = r
            .recognize_document_with_mode(&grey, w, h, dict.as_ref(), None, mode)
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&doc.json).unwrap();
        let (mut lines, mut with_period, mut regions) = (0usize, 0usize, 0usize);
        let mut sample: Vec<String> = Vec::new();
        for page in v["pages"].as_array().unwrap() {
            for reg in page["regions"].as_array().unwrap() {
                regions += 1;
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
                        with_period += 1;
                    }
                    if sample.len() < 6 {
                        sample.push(t);
                    }
                }
            }
        }
        println!("{label:8} regions={regions:2} lines={lines:3} lines_ending_in_period={with_period:3} conf={:?}", doc.mean_confidence);
        for s in &sample {
            println!("          {s}");
        }
    }
}
