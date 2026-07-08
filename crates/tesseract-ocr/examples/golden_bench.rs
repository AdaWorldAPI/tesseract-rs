//! D6.3 -- tesseract-rs (our) perf bench: loads the eng LSTM model + dict
//! exactly like the golden tests, then times `recognize_page_makerow` across
//! the whole `corpus/pages/*.pgm` corpus. Report-only, no gate; the CLI-side
//! counterpart is `corpus/gen/run_cli_golden.py --bench`.
//!
//! Methodology note: unlike the CLI-side bench (which spawns an isolated
//! `tesseract` process per page per run, so it times each page independently
//! and keeps a PER-PAGE best-of-3), this bench loads the model ONCE and runs
//! [`PASSES`] full passes over the WHOLE corpus in the same process. It
//! reports the single full pass with the lowest TOTAL wall time -- the
//! natural "best of N runs" unit when the corpus, not the process, is what's
//! being repeated.
//!
//! ```sh
//! cargo run -p tesseract-ocr --release --example golden_bench
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tesseract_core::DictLite;
use tesseract_ocr::{parse_pgm, LstmRecognizer};

/// The corpus root, resolved relative to this crate's manifest dir -- the
/// same convention `golden_report.rs` uses.
fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
}

/// Prints a diagnostic to stderr and exits 1. A bench that silently produced
/// a partial/misleading number would be worse than one that stops.
fn fail(context: &str, err: impl std::fmt::Display) -> ! {
    eprintln!("error: {context}: {err}");
    std::process::exit(1);
}

/// The number of full-corpus timed passes (after 1 untimed warm-up pass).
const PASSES: usize = 3;

fn main() {
    let root = corpus_root();
    let model_dir = root.join("model");
    let pages_dir = root.join("pages");

    let lstm_path = model_dir.join("eng.lstm");
    let uni_path = model_dir.join("eng.lstm-unicharset");
    let rec_path = model_dir.join("eng.lstm-recoder");
    let word_path = model_dir.join("eng.lstm-word-dawg");
    let punc_path = model_dir.join("eng.lstm-punc-dawg");
    let number_path = model_dir.join("eng.lstm-number-dawg");

    let lstm =
        fs::read(&lstm_path).unwrap_or_else(|e| fail(&format!("read {}", lstm_path.display()), e));
    let uni = fs::read_to_string(&uni_path)
        .unwrap_or_else(|e| fail(&format!("read {}", uni_path.display()), e));
    let rec =
        fs::read(&rec_path).unwrap_or_else(|e| fail(&format!("read {}", rec_path.display()), e));
    let recognizer = LstmRecognizer::from_components(&lstm, &uni, &rec)
        .unwrap_or_else(|e| fail("LstmRecognizer::from_components", e));

    let word =
        fs::read(&word_path).unwrap_or_else(|e| fail(&format!("read {}", word_path.display()), e));
    let punc =
        fs::read(&punc_path).unwrap_or_else(|e| fail(&format!("read {}", punc_path.display()), e));
    let number = fs::read(&number_path)
        .unwrap_or_else(|e| fail(&format!("read {}", number_path.display()), e));
    let dict = DictLite::from_components(&word, &punc, &number)
        .unwrap_or_else(|e| fail("DictLite::from_components", e));

    let mut page_paths: Vec<PathBuf> = match fs::read_dir(&pages_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("pgm"))
            .collect(),
        Err(e) => fail(&format!("read_dir {}", pages_dir.display()), e),
    };
    page_paths.sort();
    if page_paths.is_empty() {
        eprintln!("error: no *.pgm pages found under {}", pages_dir.display());
        std::process::exit(1);
    }

    // Pre-load + pre-parse every page ONCE; the timed loop below only pays
    // for recognize_page_makerow itself, never file IO or PGM parsing.
    let pages: Vec<(String, Vec<u8>, usize, usize)> = page_paths
        .iter()
        .map(|path| {
            let bytes =
                fs::read(path).unwrap_or_else(|e| fail(&format!("read {}", path.display()), e));
            let (grey, w, h) = parse_pgm(&bytes)
                .unwrap_or_else(|e| fail(&format!("parse_pgm {}", path.display()), e));
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            (name, grey, w, h)
        })
        .collect();

    // 1 untimed warm-up pass over the whole corpus.
    for (name, grey, w, h) in &pages {
        if let Err(e) = recognizer.recognize_page_makerow(grey, *w, *h, Some(&dict)) {
            fail(&format!("recognize_page_makerow (warm-up) page={name}"), e);
        }
    }

    // PASSES timed full-corpus passes; keep every pass's per-page times so
    // the BEST PASS (lowest total) can be selected afterward.
    let mut passes: Vec<Vec<f64>> = Vec::with_capacity(PASSES);
    for _ in 0..PASSES {
        let mut pass_ms = Vec::with_capacity(pages.len());
        for (name, grey, w, h) in &pages {
            let start = Instant::now();
            if let Err(e) = recognizer.recognize_page_makerow(grey, *w, *h, Some(&dict)) {
                fail(&format!("recognize_page_makerow page={name}"), e);
            }
            pass_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        passes.push(pass_ms);
    }

    let best_idx = passes
        .iter()
        .map(|p| p.iter().sum::<f64>())
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best = &passes[best_idx];
    let total_ms: f64 = best.iter().sum();

    println!(
        "## D6.3 — tesseract-rs perf bench (recognize_page_makerow, best of {PASSES} full-corpus passes)"
    );
    println!();
    println!("| page | best-pass (ms) |");
    println!("|---|---|");
    for (i, (name, ..)) in pages.iter().enumerate() {
        println!("| {name} | {:.2} |", best[i]);
    }
    println!();
    if total_ms > 0.0 {
        println!(
            "pages/sec (best pass, {} pages / {:.3} s): {:.3}",
            pages.len(),
            total_ms / 1000.0,
            pages.len() as f64 / (total_ms / 1000.0)
        );
    }

    let vmhwm_line = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|l| l.starts_with("VmHWM:"))
                .map(str::trim)
                .map(str::to_string)
        });
    match vmhwm_line {
        Some(line) => println!("peak RSS: {line}"),
        None => println!("peak RSS: VmHWM unavailable"),
    }
}
