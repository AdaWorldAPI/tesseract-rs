//! **The falsifying A/B probe** — Sauvola vs Otsu binarization under UNEVEN
//! ILLUMINATION, the failure mode Sauvola's local mean/std is supposed to
//! survive and a single global Otsu threshold cannot (see `binarize.rs`'s
//! module docs and the `xy_cut::BinarizeMode` doc comments). The existing
//! rendered corpus (`corpus/pages/*.pgm`, `corpus/quality/resgrid.pgm`) is
//! cleanly and evenly lit, so that advantage is invisible there;
//! `corpus/gen/gen_uneven_light.py` degrades a real corpus page with a
//! multiplicative illumination field (a linear gradient and a radial
//! vignette, each at two strengths) to give the two modes something to
//! actually disagree about.
//!
//! ```sh
//! python3 corpus/gen/gen_uneven_light.py   # writes corpus/quality/uneven_*.pgm
//! cargo run -p tesseract-ocr --example binarize_ab
//! ```
//!
//! ## An important, CODE-PROVEN finding this probe surfaces
//!
//! [`LstmRecognizer::recognize_document_with_mode`]'s `binarize_mode`
//! parameter (added alongside this probe — see `xy_cut.rs` /
//! `lstm_recognizer.rs`) reaches ONLY the region/table classification pass
//! (`region_figures` / `block_is_table` / the layout `xy_cut` call feeding
//! `build_regions`) — **not** the actual text-line finding/recognition
//! (`recognize_page_blocks_words`, which runs its own separate, always-Otsu
//! binarization via `segment::segment_rows`, untouched by this parameter).
//! Concretely: `Document::word_count`, `Document::line_count`, and
//! `Document::mean_confidence` are computed from a `page` value built
//! BEFORE `binarize_mode` is ever consulted, so those fields — and,
//! transitively, the recognized WORD TEXT the CER column below is computed
//! from — are **provably mode-independent** for any input, by construction,
//! not by chance. Expect the `word_count` / `mean_conf` / `cer` columns
//! below to read IDENTICAL for Otsu vs Sauvola on every fixture. That
//! identity IS a measured falsifying result, not a bug in this probe: it
//! means today's wiring cannot make Sauvola improve recognized TEXT quality
//! on an unevenly-lit page — only how such a page's regions get
//! CLASSIFIED. The `ink_frac` column (computed directly from
//! [`binarize_page_with`](tesseract_ocr::xy_cut::binarize_page_with),
//! entirely independent of the OCR pipeline) is the column that DOES show
//! Sauvola's real effect: the fraction of page pixels classified as ink,
//! which a global Otsu split inflates sharply once the dimmed region's
//! background drops toward — or past — the point the WHOLE page's
//! histogram places its one threshold, while Sauvola holds near the clean
//! page's own fraction throughout.

use std::fs;
use std::path::{Path, PathBuf};

use tesseract_core::DictLite;
use tesseract_ocr::xy_cut::binarize_page_with;
use tesseract_ocr::{parse_pgm, BinarizeMode, LstmRecognizer};

/// The corpus root, resolved relative to this crate's manifest dir — the
/// same convention `golden_bench.rs` / `golden_report.rs` use.
fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Prints a diagnostic to stderr and exits 1 — a measurement that silently
/// produced a partial/misleading number would be worse than one that stops
/// (mirroring `golden_bench.rs::fail`).
fn fail(context: &str, err: impl std::fmt::Display) -> ! {
    eprintln!("error: {context}: {err}");
    std::process::exit(1);
}

/// The fixtures this probe measures, in report order: (label, filename under
/// `corpus/quality/`, from `gen_uneven_light.py`). `clean` MUST be first —
/// its own Otsu recognition is the CER reference text for every row (per
/// the brief: "the clean page's Otsu reading is the reference text — that
/// is the ground truth here, not an external transcript").
const FIXTURES: &[(&str, &str)] = &[
    ("clean", "uneven_clean.pgm"),
    ("linear_060", "uneven_linear_060.pgm"),
    ("linear_085", "uneven_linear_085.pgm"),
    ("vignette_060", "uneven_vignette_060.pgm"),
    ("vignette_085", "uneven_vignette_085.pgm"),
];

/// The two modes under test. The Sauvola window (`whsize = 16`) comfortably
/// fits the 512x720 source page (well above its `2*whsize+3` minimum);
/// `k = 0.34` is the typical production constant used throughout
/// `binarize.rs`'s own tests.
const MODES: &[(&str, BinarizeMode)] = &[
    ("otsu", BinarizeMode::Otsu),
    (
        "sauvola",
        BinarizeMode::Sauvola {
            whsize: 16,
            k: 0.34,
        },
    ),
];

/// Levenshtein edit distance (two-row DP) — the CER numerator, copied from
/// `tests/quality_resolution_grid.rs`'s helper of the same name so this
/// probe scores CER exactly the way the CI quality fence does.
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

/// Concatenate every recognized word's text, in doc.v1 traversal order
/// (`pages -> regions -> lines -> words`) — the same traversal
/// `tests/quality_resolution_grid.rs` uses to bucket words per cell,
/// simplified here to a flat space-joined string for CER scoring.
fn extract_text(doc_json: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(doc_json).expect("parse doc.v1 JSON");
    let mut words = Vec::new();
    for page in v["pages"].as_array().map(|x| x.as_slice()).unwrap_or(&[]) {
        for region in page["regions"]
            .as_array()
            .map(|x| x.as_slice())
            .unwrap_or(&[])
        {
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
                    if let Some(t) = word["text"].as_str() {
                        words.push(t.to_string());
                    }
                }
            }
        }
    }
    words.join(" ")
}

/// Fraction of page pixels classified as ink (`0`, this crate's foreground
/// convention) under `mode` — computed directly from
/// [`binarize_page_with`], entirely independent of the OCR pipeline (see
/// the module docs' finding on why this is the column that actually moves).
fn ink_fraction(grey: &[u8], w: usize, h: usize, mode: BinarizeMode) -> f64 {
    let binary = binarize_page_with(grey, w, h, mode);
    let ink = binary.iter().filter(|&&p| p == 0).count();
    ink as f64 / binary.len() as f64
}

fn main() {
    let root = corpus_root();
    let model_dir = root.join("model");
    let quality_dir = root.join("quality");

    let lstm_path = model_dir.join("eng.lstm");
    let uni_path = model_dir.join("eng.lstm-unicharset");
    let rec_path = model_dir.join("eng.lstm-recoder");

    let lstm =
        fs::read(&lstm_path).unwrap_or_else(|e| fail(&format!("read {}", lstm_path.display()), e));
    let uni = fs::read_to_string(&uni_path)
        .unwrap_or_else(|e| fail(&format!("read {}", uni_path.display()), e));
    let rec =
        fs::read(&rec_path).unwrap_or_else(|e| fail(&format!("read {}", rec_path.display()), e));
    let recognizer = LstmRecognizer::from_components(&lstm, &uni, &rec)
        .unwrap_or_else(|e| fail("LstmRecognizer::from_components", e));

    // Dict is optional -- a missing/corrupt DAWG degrades gracefully to
    // None, the same rule `state.rs` (tesseract-ocr-web) and the quality
    // fence test both use.
    let dict = match (
        fs::read(model_dir.join("eng.lstm-word-dawg")),
        fs::read(model_dir.join("eng.lstm-punc-dawg")),
        fs::read(model_dir.join("eng.lstm-number-dawg")),
    ) {
        (Ok(w), Ok(p), Ok(n)) => DictLite::from_components(&w, &p, &n).ok(),
        _ => None,
    };

    // Load every fixture up front; fail loudly (with a hint) if
    // gen_uneven_light.py hasn't been run yet.
    let pages: Vec<(&str, Vec<u8>, usize, usize)> = FIXTURES
        .iter()
        .map(|&(label, filename)| {
            let path = quality_dir.join(filename);
            let bytes = fs::read(&path).unwrap_or_else(|e| {
                fail(
                    &format!(
                        "read {} -- run `python3 corpus/gen/gen_uneven_light.py` first",
                        path.display()
                    ),
                    e,
                )
            });
            let (grey, w, h) = parse_pgm(&bytes)
                .unwrap_or_else(|e| fail(&format!("parse_pgm {}", path.display()), e));
            (label, grey, w, h)
        })
        .collect();

    // The CER reference: the clean page's OWN Otsu recognition (per the
    // brief -- ground truth is measured here, not an external transcript).
    let (clean_label, clean_grey, clean_w, clean_h) = &pages[0];
    assert_eq!(
        *clean_label, "clean",
        "FIXTURES[0] must be the clean baseline"
    );
    let clean_doc = recognizer
        .recognize_document_with_mode(
            clean_grey,
            *clean_w,
            *clean_h,
            dict.as_ref(),
            None,
            BinarizeMode::Otsu,
        )
        .unwrap_or_else(|e| fail("recognize_document_with_mode(clean, otsu)", e));
    let reference_text = extract_text(&clean_doc.json);
    let reference_len = reference_text.chars().count().max(1); // guards a div-by-zero

    println!("# binarize_ab — Sauvola vs Otsu under uneven illumination\n");
    println!("reference (clean, otsu) text: {reference_text:?}\n");
    println!("| fixture | mode | word_count | mean_conf | cer | ink_frac |");
    println!("|---|---|---|---|---|---|");

    // Per-mode running sums for the closing summary line.
    let mut cer_sum = vec![0.0f64; MODES.len()];
    let mut ink_sum = vec![0.0f64; MODES.len()];
    let mut n_rows = vec![0usize; MODES.len()];

    for (label, grey, w, h) in &pages {
        for (mi, &(mode_label, mode)) in MODES.iter().enumerate() {
            let doc = recognizer
                .recognize_document_with_mode(grey, *w, *h, dict.as_ref(), None, mode)
                .unwrap_or_else(|e| {
                    fail(
                        &format!("recognize_document_with_mode({label}, {mode_label})"),
                        e,
                    )
                });
            let text = extract_text(&doc.json);
            let cer = levenshtein(&reference_text, &text) as f64 / reference_len as f64;
            let ink = ink_fraction(grey, *w, *h, mode);
            let conf = doc
                .mean_confidence
                .map_or_else(|| "n/a".to_string(), |c| format!("{c:.2}"));
            println!(
                "| {label} | {mode_label} | {} | {conf} | {cer:.4} | {ink:.4} |",
                doc.word_count
            );
            cer_sum[mi] += cer;
            ink_sum[mi] += ink;
            n_rows[mi] += 1;
        }
    }

    println!();
    for (mi, &(mode_label, _)) in MODES.iter().enumerate() {
        let n = n_rows[mi].max(1) as f64;
        println!(
            "summary[{mode_label}]: mean_cer={:.4} mean_ink_frac={:.4} (n={})",
            cer_sum[mi] / n,
            ink_sum[mi] / n,
            n_rows[mi]
        );
    }
}
