//! `tesseract-ocr-pdf` — D5.3-skeleton orchestrator binary.
//!
//! Per page: does it have an extractable text layer (D5.1)? Print it, no
//! OCR. Otherwise (image-only page): D5.2 (rasterization via `pdfium-render`
//! or equivalent) is not yet wired, so this binary reports a documented
//! `NotImplemented` to stderr and moves on — the OCR arm itself already
//! works end-to-end for raw grey images, demonstrated by the `.pgm` input
//! mode below.
//!
//! ```sh
//! # PDF mode (text-layer fast path; raster fallback stubbed):
//! tesseract-ocr-pdf document.pdf --data-dir /tmp
//!
//! # Direct-image mode (bypasses PDF entirely, demos the OCR arm E2E):
//! tesseract-ocr-pdf /tmp/line36.pgm --data-dir /tmp
//! ```
//!
//! `--data-dir DIR` looks for `eng.lstm`, `eng.lstm-unicharset`,
//! `eng.lstm-recoder`, and (optionally, for the dict-beam path)
//! `eng.lstm-word-dawg` / `eng.lstm-punc-dawg` / `eng.lstm-number-dawg`.
#![allow(
    clippy::print_stdout,
    reason = "the orchestrator's job is to print results"
)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use tesseract_ocr_pdf::{extract_text_layer, OcrPipeline};

fn parse_args() -> (PathBuf, PathBuf) {
    let mut input = None;
    let mut data_dir = PathBuf::from("/tmp");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--data-dir" {
            if let Some(dir) = args.next() {
                data_dir = PathBuf::from(dir);
            }
        } else if input.is_none() {
            input = Some(PathBuf::from(arg));
        }
    }
    let input = input.unwrap_or_else(|| {
        eprintln!("usage: tesseract-ocr-pdf <pdf-or-pgm> [--data-dir DIR]");
        std::process::exit(2);
    });
    (input, data_dir)
}

fn load_pipeline(data_dir: &Path) -> Result<OcrPipeline, Box<dyn std::error::Error>> {
    let lstm = data_dir.join("eng.lstm");
    let unicharset = data_dir.join("eng.lstm-unicharset");
    let recoder = data_dir.join("eng.lstm-recoder");
    let word_dawg = data_dir.join("eng.lstm-word-dawg");
    let punc_dawg = data_dir.join("eng.lstm-punc-dawg");
    let number_dawg = data_dir.join("eng.lstm-number-dawg");
    let have_dict = word_dawg.exists() && punc_dawg.exists() && number_dawg.exists();
    let pipeline = OcrPipeline::from_data_paths(
        &lstm,
        &unicharset,
        &recoder,
        have_dict.then_some(word_dawg.as_path()),
        have_dict.then_some(punc_dawg.as_path()),
        have_dict.then_some(number_dawg.as_path()),
    )?;
    Ok(pipeline)
}

/// Direct `.pgm` input: bypasses PDF handling entirely, feeds the grey image
/// straight into the OCR arm. Demonstrates D5.3's OCR half is already wired
/// for raw image input, independent of the PDF/raster front end.
fn run_pgm(path: &Path, data_dir: &Path) -> ExitCode {
    let pipeline = match load_pipeline(data_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: loading OCR data from {}: {e}", data_dir.display());
            return ExitCode::FAILURE;
        }
    };
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: reading {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let (grey, w, h) = match tesseract_ocr::parse_pgm(&bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: parsing {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    match pipeline.ocr_grey_page(&grey, w, h) {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: OCR failed on {}: {e}", path.display());
            ExitCode::FAILURE
        }
    }
}

/// PDF input: D5.1 text-layer fast path per page; image-only pages hit the
/// D5.2-pending raster stub.
fn run_pdf(path: &Path, _data_dir: &Path) -> ExitCode {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: reading {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let pages = match extract_text_layer(&bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: parsing PDF {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let mut any_text = false;
    for (i, page) in pages.iter().enumerate() {
        match page {
            Some(text) => {
                any_text = true;
                println!("--- page {} (text layer) ---", i + 1);
                println!("{text}");
            }
            None => {
                eprintln!(
                    "page {}: image-only (no text layer) — raster fallback NOT IMPLEMENTED \
                     (D5.2/pdfium pending); see .claude/plans/pdf-to-text-ocr-v1.md Phase 5",
                    i + 1
                );
            }
        }
    }
    if any_text {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "warning: no page in {} yielded a text layer",
            path.display()
        );
        ExitCode::SUCCESS
    }
}

fn main() -> ExitCode {
    let (input, data_dir) = parse_args();
    let is_pgm = input
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("pgm"));
    if is_pgm {
        run_pgm(&input, &data_dir)
    } else {
        run_pdf(&input, &data_dir)
    }
}
